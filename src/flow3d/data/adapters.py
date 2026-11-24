from __future__ import annotations

import glob
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Any

import numpy as np

from flow3d.contracts import Flow3DBatch, Flow3DConditions, PhysicsContractError, SampleProvenance
from flow3d.physics.boundary import evaluate_cylinder_boundary_conditions
from flow3d.data.interventions import HISTORY_FRAMES, paired_branch_report, validated_intervention
from flow3d.data.normalization import Flow3DNormalizer, si_roundtrip_diagnostics
from flow3d.data.schema import (
    convert_h5_6channel_to_field_and_geometry,
    denormalize_star_to_si,
    transpose_yzx_to_zyx_fields,
)


@dataclass(frozen=True)
class TrajectorySegmentRecord:
    segment_id: str
    num_frames: int
    dt_phys_s: float
    t_start_s: float
    t_end_s: float
    file_names: tuple[str, ...]


def detect_temporal_rank_degeneracy(field_t4zyx: np.ndarray, rank_threshold: int = 2, rtol: float = 1e-4) -> tuple[bool, int]:
    """Flag endpoint-linear evolution for review and report the sampled temporal rank.

    Two oscillating spatial modes also have rank two. Low rank alone is never
    evidence of interpolation. This inexpensive diagnostic does not certify provenance.
    """
    f = np.asarray(field_t4zyx)
    if f.ndim != 5:
        raise PhysicsContractError("rank diagnostic requires (T,4,Z,Y,X)")
    return _rank_diagnostics(f[:, :, ::4, ::4, ::4], rank_threshold, rtol)


def _rank_diagnostics(samples: np.ndarray, threshold: int, rtol: float) -> tuple[bool, int]:
    a = np.asarray(samples, dtype=np.float64).reshape(len(samples), -1)
    if len(a) < 5:
        return False, max(0, len(a) - 1)
    # Solid samples may contain NaNs. Use only columns valid across time.
    a = a[:, np.isfinite(a).all(axis=0)]
    if not a.shape[1]:
        return False, 0
    diff = np.diff(a, axis=0)
    # The Gram matrix avoids materializing a tall singular-vector matrix.
    values = np.sqrt(np.maximum(np.linalg.eigvalsh(diff @ diff.T), 0.0))
    rank = int(np.sum(values > values[-1] * rtol))
    if not 0 < rank <= threshold:
        return False, rank
    delta = a[-1] - a[0]
    line = a[0] + np.linspace(0, 1, len(a))[:, None] * delta
    # Scale by endpoint displacement, not a potentially large stationary mean.
    linear = np.linalg.norm(a - line) <= rtol * np.sqrt(len(a)) * np.linalg.norm(delta)
    return bool(linear), rank


def load_c04_benchmark_npz(
    npz_path: str | Path,
    case_id: str = "C04",
    ancestor_id: str = "c04_re126k_anchor",
    trajectory_id: str = "c04_s1_volume_sequence",
    normalize: bool = True,
    *,
    expected_sha256: str | None = None,
) -> tuple[Flow3DBatch, dict[str, Any]]:
    """Decode an upstream C04 SI volume. Only an independently pinned canonical reference is verified."""
    p = Path(npz_path)
    with p.open("rb") as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
        sha = digest.hexdigest()
        if expected_sha256 is not None and (
            not isinstance(expected_sha256, str) or sha != expected_sha256.lower()
        ):
            raise PhysicsContractError("C04 file SHA-256 differs from the independently supplied reference")
        stream.seek(0)
        with np.load(stream, allow_pickle=False) as d:
            required = {"fields", "times", "solid_mask", "sdf", "metadata_json"}
            if not required.issubset(d.files):
                raise PhysicsContractError(
                    f"C04 archive missing upstream keys: {sorted(required - set(d.files))}"
                )
            meta = json.loads(str(d["metadata_json"]))
            fields, times = d["fields"], np.asarray(d["times"], dtype=np.float64)
            solid_mask, sdf_m = d["solid_mask"], np.asarray(d["sdf"], dtype=np.float32)
    if not isinstance(meta, dict) or fields.dtype != np.float32 or fields.ndim != 5 or fields.shape[1] != 4:
        raise PhysicsContractError("C04 needs metadata and float32 [T,4,Z,Y,X] fields")
    if fields.shape[0] < 2 or times.shape != (fields.shape[0],):
        raise PhysicsContractError("C04 times must match the frame count (at least two)")
    expected = {
        "case_id": "C04",
        "is_3d": True,
        "channels": ["u", "v", "w", "p"],
        "units": ["m/s", "m/s", "m/s", "Pa"],
        "layout": "(T, 4, Nz, Ny, Nx)",
        "spatial_shape": list(fields.shape[2:]),
        "num_frames": len(times),
    }
    for key, value in expected.items():
        if meta.get(key) != value:
            raise PhysicsContractError(f"C04 metadata {key} disagrees with the upstream SI volume schema")
    if case_id != "C04" or meta.get("ground_truth_verified") is False or meta.get("ground_truth_warning"):
        raise PhysicsContractError("C04 identity or source evidence conflicts with the reference")
    dt_s = _constant_step(times, "C04 time", len(times))
    for key, value in (("time_start_s", times[0]), ("time_end_s", times[-1])):
        if key not in meta or not np.isclose(meta[key], value, atol=1e-6, rtol=1e-6):
            raise PhysicsContractError(f"C04 timestamps differ from metadata {key}")
    diameter_m = meta.get("cylinder_diameter_m")
    spacing_m = meta.get("cell_spacing_m")
    if (
        diameter_m is None
        or spacing_m is None
        or not np.isclose(diameter_m, 0.3)
        or not np.isclose(spacing_m, 0.01875)
    ):
        raise PhysicsContractError("C04 reference requires D=0.3 m and h=0.01875 m")
    for key, value in (
        ("reynolds_number", 126167.05),
        ("Re", 126167.05),
        ("u0_m_s", 0.5),
        ("U0_m_s", 0.5),
        ("kinematic_viscosity_m2_s", 1.1889e-6),
    ):
        if key in meta and not np.isclose(meta[key], value, rtol=1e-4, atol=0):
            raise PhysicsContractError(f"C04 declared {key} conflicts with benchmark conditions")
    if (
        solid_mask.shape != fields.shape[2:]
        or sdf_m.shape != solid_mask.shape
        or not np.isfinite(sdf_m).all()
    ):
        raise PhysicsContractError("C04 geometry shape or SDF finiteness is invalid")
    if not np.all((solid_mask == 0) | (solid_mask == 1)) or not np.array_equal(solid_mask > 0.5, sdf_m <= 0):
        raise PhysicsContractError("C04 solid_mask must be binary and agree with the SDF sign")
    verified = expected_sha256 is not None
    if verified:
        if fields.shape != (4, 4, 64, 96, 192) or not np.allclose(
            times, [0, 0.06, 0.12, 0.18], rtol=0, atol=1e-6
        ):
            raise PhysicsContractError(
                "pinned C04 reference requires four canonical ROI frames at 0/.06/.12/.18 s"
            )
        if ancestor_id != "c04_re126k_anchor" or trajectory_id != "c04_s1_volume_sequence":
            raise PhysicsContractError("cannot relabel pinned C04 lineage")
        # Upstream samples a circular cylinder at (x,y)=(0,0) on a cell-centred interior ROI.
        x = -0.60 + (np.arange(192) + 0.5) * spacing_m
        y = -0.90 + (np.arange(96) + 0.5) * spacing_m
        reference_sdf = np.hypot(x[None, :], y[:, None]) - diameter_m / 2
        if not np.allclose(sdf_m, reference_sdf[None], atol=2e-6, rtol=1e-6):
            raise PhysicsContractError("C04 ROI coordinates or cylinder SDF differ from the canonical grid")
    # Flip polarity: solid_mask (True=solid) -> fluid_mask (1.0=fluid, 0.0=solid)
    fluid_mask = 1.0 - (solid_mask > 0.5).astype(np.float32)
    fields[:, :, fluid_mask <= 0.5] = 0.0
    # Dimensionless SDF/D
    sdf_over_d = (sdf_m / diameter_m).astype(np.float32)
    geometry = np.stack([fluid_mask, sdf_over_d], axis=0)[None, ...]  # (1, 2, Z, Y, X)

    if normalize:
        normalizer = Flow3DNormalizer()
        _, b_stats = normalizer.normalize(fields[None, ...], fluid_mask)
    else:
        b_stats = None
    u0_m_s = 0.5
    cond = Flow3DConditions(
        reynolds_number=126167.05,
        u0_m_s=u0_m_s,
        diameter_m=diameter_m,
        viscosity_m2_s=1.1889e-6,
        dt_phys_s=dt_s,
        text_prompt="3D cylinder wake Re=126167 C04 baseline",
        known_mask={"reynolds_number": True, "u0_m_s": True, "diameter_m": True, "dt_phys_s": True},
        valid_mask={"reynolds_number": True, "u0_m_s": True, "diameter_m": True, "dt_phys_s": True},
    )
    provs = [
        SampleProvenance(
            case_id=case_id,
            geometry_id="cylinder_3d_D0p3m",
            trajectory_id=trajectory_id,
            ancestor_id=ancestor_id,
            segment_id="seg_0",
            time_idx=int(i),
            time_phys_s=float(t),
            time_ctu=float(t * u0_m_s / diameter_m),
            ground_truth_verified=verified,
            tier1_stability_status="BENCHMARK_REFERENCE_4FRAME" if verified else "UNKNOWN",
            pressure_gauge="solver_outlet_relative_zero_pa" if verified else SampleProvenance.pressure_gauge,
            boundary_condition_id="c04_solver_velocity_inlet_pressure_outlet_slip_farfield_periodic_span_noslip_cylinder"
            if verified
            else SampleProvenance.boundary_condition_id,
        )
        for i, t in enumerate(times)
    ]
    batch = Flow3DBatch(
        field=fields[None, ...],
        geometry=geometry,
        b=b_stats if normalize else None,
        spacing=(spacing_m, spacing_m, spacing_m),
        conditions=(cond,),
        provenance=tuple(provs),
    )
    summary = {
        "format": "c04_benchmark_npz",
        "sha256": sha,
        "ground_truth_verified": verified,
        "status": "VERIFIED_4FRAME_BENCHMARK" if verified else "UNVERIFIED",
        "verification_reason": "pinned content and benchmark schema match"
        if verified
        else "no independent reference SHA-256 supplied",
        "path": str(p),
        "num_frames": int(fields.shape[0]),
        "spatial_shape_zyx": list(fields.shape[2:]),
        "dt_phys_s": dt_s,
        "dt_ctu": cond.resolved_dt_ctu,
        "solid_cells": int((fluid_mask <= 0.5).sum()),
        "fluid_cells": int((fluid_mask > 0.5).sum()),
        "sdf_over_d_min": float(sdf_over_d.min()),
        "sdf_over_d_max": float(sdf_over_d.max()),
        "supports_stage2_L4_window": bool(verified and fields.shape[0] >= 5),
    }
    return batch, summary


def inspect_and_segment_c04_shedding_directory(
    dir_path: str | Path,
    reference_geometry_npz: str | Path | None = None,
    max_frames_to_load: int = 8,
) -> dict[str, Any]:
    """Inspect and segment `video_c04_shedding/*.npz` (82 files, YZX (96,64,192), solid NaNs, non-uniform dt)."""
    dpath = Path(dir_path)
    if not dpath.exists():
        raise FileNotFoundError(f"shedding directory not found: {dpath}")
    files = sorted(glob.glob(str(dpath / "*.npz")))
    if not files:
        raise PhysicsContractError(f"no .npz files found in {dpath}")

    meta_rows: list[tuple[float, int, str]] = []
    for fpath in files:
        with np.load(fpath, allow_pickle=False) as d:
            req = {"u", "v_span", "w_cross", "p", "time_s", "step"}
            if not req.issubset(set(d.files)):
                raise PhysicsContractError(f"unexpected keys in {fpath}: {d.files}")
            meta_rows.append((float(d["time_s"]), int(d["step"]), fpath))

    # Sort by physical time and step
    meta_rows.sort(key=lambda r: (r[0], r[1]))

    # Identify duplicate timestamps and partition into constant-dt segments
    dedup_rows: list[tuple[float, int, str]] = []
    duplicate_timestamps: list[dict[str, Any]] = []
    for row in meta_rows:
        if dedup_rows and abs(row[0] - dedup_rows[-1][0]) < 1e-7:
            duplicate_timestamps.append({
                "time_s": row[0],
                "kept_file": os.path.basename(dedup_rows[-1][2]),
                "dropped_duplicate_file": os.path.basename(row[2]),
            })
        else:
            dedup_rows.append(row)

    segments: list[TrajectorySegmentRecord] = []
    if len(dedup_rows) >= 2:
        cur_files = [os.path.basename(dedup_rows[0][2]), os.path.basename(dedup_rows[1][2])]
        cur_t_start = dedup_rows[0][0]
        cur_t_end = dedup_rows[1][0]
        cur_dt = round(cur_t_end - cur_t_start, 4)
        seg_idx = 0
        for i in range(2, len(dedup_rows)):
            dt_i = round(dedup_rows[i][0] - dedup_rows[i - 1][0], 4)
            if abs(dt_i - cur_dt) <= 1e-3:
                cur_files.append(os.path.basename(dedup_rows[i][2]))
                cur_t_end = dedup_rows[i][0]
            else:
                segments.append(
                    TrajectorySegmentRecord(
                        segment_id=f"seg_{seg_idx}_dt{cur_dt:.2f}s",
                        num_frames=len(cur_files),
                        dt_phys_s=cur_dt,
                        t_start_s=cur_t_start,
                        t_end_s=cur_t_end,
                        file_names=tuple(cur_files),
                    )
                )
                seg_idx += 1
                cur_files = [os.path.basename(dedup_rows[i - 1][2]), os.path.basename(dedup_rows[i][2])]
                cur_t_start = dedup_rows[i - 1][0]
                cur_t_end = dedup_rows[i][0]
                cur_dt = dt_i
        segments.append(
            TrajectorySegmentRecord(
                segment_id=f"seg_{seg_idx}_dt{cur_dt:.2f}s",
                num_frames=len(cur_files),
                dt_phys_s=cur_dt,
                t_start_s=cur_t_start,
                t_end_s=cur_t_end,
                file_names=tuple(cur_files),
            )
        )

    # Load up to `max_frames_to_load` from the first longest constant-dt segment to verify transpose & NaN scrubbing
    best_seg = max(segments, key=lambda s: s.num_frames)
    sample_names = best_seg.file_names[: min(max_frames_to_load, len(best_seg.file_names))]
    loaded_frames: list[np.ndarray] = []
    solid_nan_counts: list[int] = []
    fluid_mask_zyx: np.ndarray | None = None

    for fname in sample_names:
        with np.load(dpath / fname, allow_pickle=False) as d:
            u_raw = np.asarray(d["u"], dtype=np.float32)
            v_span_raw = np.asarray(d["v_span"], dtype=np.float32)
            w_cross_raw = np.asarray(d["w_cross"], dtype=np.float32)
            p_raw = np.asarray(d["p"], dtype=np.float32)
            frame_zyx = transpose_yzx_to_zyx_fields(u_raw, w_cross_raw, v_span_raw, p_raw)
            nan_mask_u = ~np.isfinite(frame_zyx[0])
            solid_nan_counts.append(int(nan_mask_u.sum()))
            if fluid_mask_zyx is None:
                if reference_geometry_npz is not None and Path(reference_geometry_npz).exists():
                    with np.load(reference_geometry_npz, allow_pickle=False) as ref_d:
                        fluid_mask_zyx = (1.0 - (np.asarray(ref_d["solid_mask"]) > 0.5).astype(np.float32))
                else:
                    # Solid obstacle cells are simultaneously NaN across all 4 physical channels
                    all_ch_nan = np.all(~np.isfinite(frame_zyx), axis=0)
                    fluid_mask_zyx = (~all_ch_nan).astype(np.float32)
            valid_fluid = fluid_mask_zyx > 0.5
            # Strictly check finiteness inside valid fluid domain BEFORE scrubbing solid NaNs
            if not np.all(np.isfinite(frame_zyx[:, valid_fluid])):
                raise PhysicsContractError(
                    f"shedding snapshot '{fname}' contains NaN/Inf inside the valid fluid domain"
                )
            frame_zyx[:, ~valid_fluid] = 0.0
            loaded_frames.append(frame_zyx)

    assert fluid_mask_zyx is not None
    normalizer = Flow3DNormalizer()
    stacked = np.stack(loaded_frames, axis=0)
    _, b_sample = normalizer.normalize(stacked, fluid_mask_zyx)

    return {
        "format": "c04_shedding_npz_directory",
        "path": str(dpath),
        "total_files": len(files),
        "unique_timestamp_files": len(dedup_rows),
        "duplicate_timestamps": duplicate_timestamps,
        "raw_shape_yzx": [96, 64, 192],
        "standardized_shape_zyx": [64, 96, 192],
        "velocity_channel_mapping": {"X_u": "u", "Y_v": "w_cross", "Z_w": "v_span"},
        "solid_nan_count_per_frame": solid_nan_counts[0] if solid_nan_counts else 0,
        "constant_dt_segments": [
            {
                "segment_id": s.segment_id,
                "num_frames": s.num_frames,
                "dt_phys_s": s.dt_phys_s,
                "t_start_s": s.t_start_s,
                "t_end_s": s.t_end_s,
                "max_rollout_from_L4": max(0, s.num_frames - 4),
            }
            for s in segments
        ],
        "longest_constant_dt_segment": {
            "segment_id": best_seg.segment_id,
            "num_frames": best_seg.num_frames,
            "dt_phys_s": best_seg.dt_phys_s,
            "max_rollout_from_L4": max(0, best_seg.num_frames - 4),
        },
        "sample_b_stats_mean": b_sample.mean(axis=0).tolist(),
    }


_H5_CHANNELS = ["u_star", "v_cross_star", "w_span_star", "p_star", "sdf_star", "solid_mask"]


class _ExcludedTrajectory(PhysicsContractError):
    """Dataset authority has explicitly excluded this candidate."""


class _UnverifiedTrajectory(PhysicsContractError):
    """The available evidence does not authorize admission."""


def _audit_files(directory: Path) -> tuple[dict, dict, dict]:
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))

    def sidecar(name: str) -> dict:
        p = directory / name
        return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}

    return (
        manifest,
        sidecar("multi_re_tier1_certification_summary.json"),
        sidecar("extraction_and_rampool_assessment.json"),
    )


def _source_key(path: str) -> str:
    # Upstream records the original solver path on Windows even on Linux mirrors.
    return PureWindowsPath(path).as_posix().casefold()


def _admit_manifest_row(row: dict, manifest: dict, tier: dict, audit: dict) -> tuple[str, str]:
    name, cid = row["file_name"], row["case_id"]
    if not isinstance(name, str) or not name or PureWindowsPath(name).name != name or not cid:
        raise PhysicsContractError("invalid trajectory identity or file name")
    if sum(r.get("file_name") == name for r in manifest["trajectories"]) != 1:
        raise PhysicsContractError("ambiguous duplicate trajectory filename in manifest")
    integrity = audit.get("ram_pool_audit", {}).get("integrity_checks")
    interpolation_source = (
        str(integrity.get("slot_2_to_26_provenance", "")) if isinstance(integrity, dict) else ""
    )
    flagged_interpolation = name in interpolation_source
    if "ground_truth_verified" not in row or type(row["ground_truth_verified"]) is not bool:
        raise _UnverifiedTrajectory("missing or malformed ground_truth_verified")
    if row["ground_truth_verified"] is False or row.get("ground_truth_warning"):
        reason = f"unverified ground truth: {row.get('ground_truth_warning') or cid}"
        if flagged_interpolation:
            reason += f"; RAM-pool audit: {interpolation_source}"
        raise _ExcludedTrajectory(reason)
    if row.get("training_eligible") is False or row.get("quarantine_status"):
        raise _ExcludedTrajectory("manifest excludes this trajectory from training")
    for q in manifest.get("quarantined_trajectories", []):
        if (isinstance(q, str) and q in (name, cid)) or (
            isinstance(q, dict) and (q.get("file_name") == name or q.get("case_id") == cid)
        ):
            raise _ExcludedTrajectory("trajectory is on the manifest quarantine list")
    source = row.get("source_model")
    if not isinstance(source, str) or not source.strip():
        raise _UnverifiedTrajectory("missing source_model lineage")
    solver = _source_key(source)
    for key, q in manifest.get("quarantined_solvers", {}).items():
        if (
            key == cid
            or _source_key(key) in solver.split("/")
            or (q.get("case_dir") and solver.startswith(_source_key(q["case_dir"]).rstrip("/") + "/"))
        ):
            raise _ExcludedTrajectory("trajectory derives from a quarantined solver")
    if not isinstance(integrity, dict):
        raise _UnverifiedTrajectory("missing RAM-pool integrity audit")
    if flagged_interpolation:
        raise _ExcludedTrajectory("RAM-pool audit identifies synthetic interpolation")
    cases = [
        c
        for c in tier.get("cases", [])
        if c.get("case_dir")
        and solver.startswith(_source_key(c["case_dir"]).rstrip("/") + "/")
        and np.isclose(float(c["target_Re"]), float(row["reynolds_number"]), rtol=0.01)
    ]
    if len(cases) != 1 or not cases[0].get("status"):
        raise _UnverifiedTrajectory("no unique Tier-1 certification for source_model and Re")
    status = cases[0]["status"]
    if status == "INVALID_CONFIG_MISMATCH":
        raise _ExcludedTrajectory("Tier-1 reports a solver configuration mismatch")
    # A solver's exports cannot acquire separate ancestors by renaming their files.
    aliases = {
        r.get("ancestor_id")
        for r in manifest["trajectories"]
        if r.get("source_model") and _source_key(r["source_model"]) == solver and r.get("ancestor_id")
    }
    if len(aliases) > 1:
        raise PhysicsContractError("conflicting ancestor_id for the same source model")
    ancestor = next(iter(aliases), solver)
    if not isinstance(ancestor, str) or ancestor.strip().casefold() in {"", "none", "unknown", "null"}:
        raise _UnverifiedTrajectory("unknown ancestor identity")
    return ancestor, status


def _positive_number(row: dict, attrs: Any, key: str) -> float:
    value = float(row[key])
    if not np.isfinite(value) or value <= 0 or not np.isclose(value, float(attrs[key]), rtol=1e-5, atol=0):
        raise PhysicsContractError(f"invalid or conflicting {key} in manifest and HDF5")
    return value


def _constant_step(axis: np.ndarray, key: str, length: int) -> float:
    if axis.shape != (length,) or length < 2 or not np.isfinite(axis).all():
        raise PhysicsContractError(f"invalid {key} axis")
    ds = np.diff(axis.astype(np.float64))
    step = float(ds.mean())
    # Exported coordinates and absolute timestamps are rounded to float32.
    tol = 2 * np.finfo(np.float32).eps * max(np.abs(axis).max(), abs(step))
    if np.any(ds <= 0) or not np.allclose(ds, step, atol=tol, rtol=1e-5):
        raise PhysicsContractError(f"{key} is not increasing with constant spacing")
    return step


def _has_h5_signature(path: Path) -> bool:
    # An HDF5 user block may precede the signature at an offset of 2**N >= 512.
    size = path.stat().st_size
    with path.open("rb") as stream:
        offset = 0
        while offset + 8 <= size:
            stream.seek(offset)
            if stream.read(8) == b"\x89HDF\r\n\x1a\n":
                return True
            offset = max(512, offset * 2)
    return False


def _grid_spacing(f: Any, shape: tuple, diameter: float) -> list[float]:
    spacing = []
    for idx, (key, star) in enumerate(
        (("z_span_m", "z_star"), ("y_cross_m", "y_star"), ("x_stream_m", "x_star"))
    ):
        axis = f["coordinates"][key][:]
        spacing.append(_constant_step(axis, key, shape[2 + idx]))
        if star in f["coordinates"]:
            h_star = _constant_step(f["coordinates"][star][:], star, shape[2 + idx])
            if not np.isclose(h_star * diameter, spacing[-1], rtol=1e-4):
                raise PhysicsContractError(f"{star} spacing conflicts with SI coordinates")
    return spacing


def _time_axes(f: Any, n: int, row: dict, speed: float, diameter: float):
    ts, ctu, steps = (f[k][:] for k in ("time_physical_s", "time_nondim_ctu", "step_indices"))
    dt = _constant_step(ts, "time_physical_s", n)
    dt_star = _constant_step(ctu, "time_nondim_ctu", n)
    if (
        steps.shape != (n,)
        or not np.issubdtype(steps.dtype, np.integer)
        or not np.all(np.diff(steps.astype(np.int64)) > 0)
    ):
        raise PhysicsContractError("invalid step_indices")
    if not np.allclose(ctu, ts.astype(np.float64) * speed / diameter, rtol=1e-5, atol=1e-6):
        raise PhysicsContractError("physical and CTU time axes disagree")
    for key, value in (
        ("dt_phys_mean_s", dt),
        ("dt_ctu_mean", dt_star),
        ("t_phys_min_s", ts[0]),
        ("t_phys_max_s", ts[-1]),
        ("t_ctu_min", ctu[0]),
        ("t_ctu_max", ctu[-1]),
    ):
        if (key in row or key.startswith("dt_")) and not np.isclose(row[key], value, rtol=1e-4, atol=1e-7):
            raise PhysicsContractError(f"HDF5 time axis differs from manifest {key}")
    return ts, ctu, steps, dt, dt_star


def _inspect_h5(
    path: Path,
    row: dict,
    ancestor: str,
    tier: str,
    *,
    load: bool = False,
    normalize: bool = False,
    sdf_clip: float | None = None,
) -> tuple[dict, Flow3DBatch | None]:
    if not path.is_file():
        raise _UnverifiedTrajectory("HDF5 file missing")
    if not _has_h5_signature(path):
        raise PhysicsContractError("invalid HDF5 signature")
    try:
        import h5py
    except ImportError as exc:
        raise _UnverifiedTrajectory(
            "HDF5 signature verified; install h5py to verify tensor contents"
        ) from exc

    with h5py.File(path, "r") as f:
        attrs = f.attrs
        if (
            "ground_truth_verified" in attrs
            and not (
                isinstance(attrs["ground_truth_verified"], (bool, np.bool_))
                and bool(attrs["ground_truth_verified"])
            )
        ) or attrs.get("ground_truth_warning"):
            raise PhysicsContractError("HDF5 attributes flag unverified ground truth")
        if _source_key(str(attrs["source_model"])) != _source_key(row["source_model"]):
            raise PhysicsContractError("HDF5 source_model differs from manifest")
        if row.get("file_size_bytes") is not None and row["file_size_bytes"] != path.stat().st_size:
            raise PhysicsContractError("HDF5 size differs from manifest")
        data = f["flow_fields"]
        if not isinstance(data, h5py.Dataset) or data.dtype != np.dtype("float32"):
            raise PhysicsContractError("flow_fields must be a float32 HDF5 dataset")
        if "channel_units" in attrs:
            units = [k.decode() if isinstance(k, bytes) else str(k) for k in attrs["channel_units"]]
            if units != ["1", "1", "1", "1", "1", "binary"]:
                raise PhysicsContractError("unexpected channel_units for nondimensional HDF5 fields")
        shape = tuple(data.shape)
        if (
            len(shape) != 5
            or shape[1] != 6
            or tuple(row["tensor_shape"]) != shape
            or row["num_steps"] != shape[0]
        ):
            raise PhysicsContractError("flow_fields shape differs from manifest or expected T,6,Z,Y,X")
        channels = [k.decode() if isinstance(k, bytes) else str(k) for k in attrs["tensor_channels"]]
        if channels != _H5_CHANNELS or row["channels"] != _H5_CHANNELS:
            raise PhysicsContractError("unexpected HDF5 or manifest channel order")
        re = _positive_number(row, attrs, "reynolds_number")
        speed = _positive_number(row, attrs, "u0_m_s")
        diameter = _positive_number(row, attrs, "cylinder_diameter_m")
        viscosity = _positive_number(row, attrs, "kinematic_viscosity_m2_s")
        density = _positive_number(row, attrs, "density_kg_m3")
        gauge = float(attrs.get("p_inf_pa", row.get("p_inf_pa", 0.0)))
        if not np.isfinite(gauge) or ("p_inf_pa" in row and gauge != float(row["p_inf_pa"])):
            raise PhysicsContractError("invalid or conflicting reference pressure")
        spacing = _grid_spacing(f, shape, diameter)
        n = shape[0]
        ts, ctu, steps, dt, dt_star = _time_axes(f, n, row, speed, diameter)
        condition_mask = dict.fromkeys(
            ("reynolds_number", "u0_m_s", "diameter_m", "viscosity_m2_s", "dt_phys_s", "dt_ctu"), True
        )
        intervention = validated_intervention(row, ts)
        boundary_id = row.get(
            "boundary_condition_id",
            attrs.get("boundary_condition_id", SampleProvenance.boundary_condition_id),
        )
        pressure_gauge = row.get(
            "pressure_gauge", attrs.get("pressure_gauge", SampleProvenance.pressure_gauge)
        )
        digest = hashlib.sha256() if intervention is not None else None
        action_metadata = {}
        actions = ()
        if intervention is not None:
            actions = tuple(float(v) for v in intervention["values"])
            condition_mask["action_schedule"] = True
            action_metadata = {
                "exogenous": intervention["exogenous"],
                "action_channel": intervention["channel"],
                "action_unit": intervention["unit"],
                "action_times_s": ts.tolist(),
                "action_known_at_time_s": intervention["known_at_time_s"],
            }
            for key in ("boundary_condition_id", "pressure_gauge"):
                label = row.get(key)
                if not isinstance(label, str) or not label.strip() or label != attrs.get(key, label):
                    raise PhysicsContractError(f"missing or conflicting intervention {key}")
            digest.update(json.dumps([boundary_id, pressure_gauge]).encode("utf-8"))
            # Canonical little-endian header binds units, spatial origin, horizon and timeline.
            digest.update(np.asarray(shape, dtype="<i8").tobytes())
            for values in (
                [re, speed, diameter, viscosity, density, gauge],
                ts,
                ctu,
                *(f["coordinates"][k][:] for k in ("z_span_m", "y_cross_m", "x_stream_m")),
            ):
                digest.update(np.ascontiguousarray(values, dtype="<f8"))
        cond = Flow3DConditions(
            action_schedule=actions,
            extra_metadata=action_metadata,
            known_mask=condition_mask.copy(),
            valid_mask=condition_mask,
            reynolds_number=re,
            u0_m_s=speed,
            diameter_m=diameter,
            viscosity_m2_s=viscosity,
            dt_phys_s=dt,
            dt_ctu=dt_star,
            text_prompt=f"3D cylinder wake Re={re:g}",
        )
        samples = []
        geometry = None
        field = np.empty((1, n, 4, *shape[2:]), np.float32) if load else None
        stats = np.empty((1, n, 6), np.float32) if load and normalize else None
        normalizer = Flow3DNormalizer()
        roundtrips = []
        for i in range(n):
            frame, geom = convert_h5_6channel_to_field_and_geometry(data[i], sdf_clip=None)
            if geometry is None:
                geometry = geom
                valid = geom[0] > 0.5
                if not valid.any():
                    raise PhysicsContractError("zero fluid cells")
                if (geom[1, valid] < 0).any() or (geom[1, ~valid] > 0).any():
                    raise PhysicsContractError("SDF sign disagrees with the declared solid mask")
            elif not np.array_equal(geometry, geom):
                raise PhysicsContractError(f"geometry changes in frame {i}")
            if not np.isfinite(frame[:, valid]).all():
                raise PhysicsContractError(f"non-finite fluid value at frame {i}")
            frame[:, ~valid] = 0.0
            samples.append(frame[:, ::4, ::4, ::4].copy())
            history_digest = digest is not None and i < HISTORY_FRAMES
            check_roundtrip = i == 0 or i == n - 1
            if field is not None or history_digest or check_roundtrip:
                si = denormalize_star_to_si(frame, speed, density, gauge)
                si[:, ~valid] = 0.0
                if i == 0:
                    first_roi = evaluate_cylinder_boundary_conditions(si, geom[0], geom[1], tuple(spacing)).to_dict()
                if check_roundtrip:
                    normalized, frame_stats = normalizer.normalize_frame(si, valid)
                    restored = normalizer.denormalize_frame(normalized, frame_stats, valid)
                    errors = si_roundtrip_diagnostics(si, restored, valid)
                    if not errors["passed"]:
                        raise PhysicsContractError(f"SI normalization roundtrip failed at frame {i}: {errors}")
                    roundtrips.append({"frame_index": i, **errors})
                if history_digest:
                    if i == 0:
                        digest.update(np.ascontiguousarray(geometry, dtype="<f4"))
                    digest.update(np.ascontiguousarray(si, dtype="<f4"))
                if field is not None:
                    field[0, i] = si
                if stats is not None:
                    if not check_roundtrip:
                        frame_stats = normalizer.fit_frame(si, valid)
                    stats[0, i] = frame_stats.as_array()
    linear, rank = _rank_diagnostics(np.stack(samples), 2, 1e-4)
    if linear:
        raise _UnverifiedTrajectory("endpoint-linear samples; review interpolation and source provenance")
    details = {
        "tensor_key": "flow_fields",
        "first_frame_boundary_diagnostics": first_roi,
        "effective_temporal_rank": rank,
        "low_temporal_rank": rank <= 2,
        "si_normalizer_roundtrips": roundtrips,
        "spacing_m": spacing,
        "spacing_star": [h / diameter for h in spacing],
        "field_units": ["m/s", "m/s", "m/s", "Pa"],
        "dt_phys_mean_s": dt,
        "dt_ctu_mean": dt_star,
        "t_start_s": float(ts[0]),
        "t_end_s": float(ts[-1]),
        "tier1_limit_cycle_status": tier,
        "ancestor_id": ancestor,
        "tier1_stability_passed": tier == "PASS",
        "num_steps": n,
        "max_rollout_from_L4": max(0, n - 4),
        "path": str(path),
    }
    if intervention is not None:
        details["intervention"] = {**intervention, "history_and_grid_sha256": digest.hexdigest()}
    if field is None:
        return details, None
    if sdf_clip is not None:
        if not np.isfinite(sdf_clip) or sdf_clip <= 0:
            raise PhysicsContractError("sdf_clip must be finite and positive")
        geometry[1] = np.clip(geometry[1], -sdf_clip, sdf_clip)
    provenance = tuple(
        SampleProvenance(
            case_id=row["case_id"],
            ancestor_id=ancestor,
            trajectory_id=path.stem,
            geometry_id=row.get("geometry_id", f"cylinder_3d_D{diameter:g}m".replace(".", "p")),
            segment_id="seg_0",
            time_idx=int(step),
            time_phys_s=float(t),
            time_ctu=float(c),
            ground_truth_verified=True,
            tier1_stability_status=tier,
            pressure_gauge=pressure_gauge,
            boundary_condition_id=boundary_id,
        )
        for step, t, c in zip(steps, ts, ctu)
    )
    return details, Flow3DBatch(
        field=field,
        geometry=geometry[None, ...],
        b=stats,
        spacing=tuple(spacing),
        conditions=(cond,),
        provenance=provenance,
    )


def inspect_world_model_dataset_manifest(dataset_dir: str | Path) -> dict[str, Any]:
    """Admit manifest candidates only after provenance, timestamp, geometry and payload checks."""
    directory = Path(dataset_dir)
    manifest, tier, audit = _audit_files(directory)
    eligible, unverified, quarantined = [], [], []
    for key, q in manifest.get("quarantined_solvers", {}).items():
        quarantined.append(
            {
                "case_id": key,
                "reason": f"solver Re mismatch: nominal={q.get('nominal_re_from_name')} vs actual={q.get('actual_comsol_re')}",
                "status": "QUARANTINED_SOLVER_MISMATCH",
            }
        )
    failures = 0
    for row in manifest.get("trajectories", []):
        name = row.get("file_name")
        record = {k: row.get(k) for k in ("case_id", "file_name", "reynolds_number", "u0_m_s", "num_steps")}
        record["file_exists_on_disk"] = bool(isinstance(name, str) and (directory / name).is_file())
        try:
            ancestor, status = _admit_manifest_row(row, manifest, tier, audit)
            record["tier1_limit_cycle_status"] = status
            details, _ = _inspect_h5(directory / name, row, ancestor, status)
        except _ExcludedTrajectory as exc:
            quarantined.append(dict(record, status="QUARANTINED", quarantine_reasons=[str(exc)]))
        except _UnverifiedTrajectory as exc:
            failures += 1
            unverified.append(dict(record, status="UNVERIFIED", reason=str(exc)))
        except (OSError, ValueError, TypeError, KeyError) as exc:
            failures += 1
            quarantined.append(dict(record, status="QUARANTINED", quarantine_reasons=[str(exc)]))
        else:
            eligible.append(
                dict(
                    record,
                    **details,
                    status="ELIGIBLE",
                    raw_h5_channels=row["channels"],
                    requires_ch4_ch5_swap_and_mask_flip=True,
                )
            )
    max_h = max((t["max_rollout_from_L4"] for t in eligible), default=0)
    return {
        "format": "world_model_multi_re_hdf5_manifest",
        "path": str(directory),
        "eligible_trajectory_count": len(eligible),
        "eligible_trajectories": eligible,
        "unverified_count": len(unverified),
        "unverified_trajectories": unverified,
        "quarantined_count": len(quarantined),
        "quarantined_trajectories": quarantined,
        "required_candidate_failure_count": failures,
        "intervention_pairs": paired_branch_report(manifest.get("trajectories", []), eligible),
        "tier1_passed_trajectory_count": sum(t["tier1_stability_passed"] for t in eligible),
        "max_supported_rollout_horizon_from_L4": max_h,
        "h100_rollout_status": "WINDOW_AVAILABLE" if max_h >= 100 else "INSUFFICIENT_VERIFIED_FRAMES",
    }


def load_world_model_h5_trajectory(
    h5_path: str | Path,
    case_id: str | None = None,
    ancestor_id: str | None = None,
    reynolds_number: float | None = None,
    u0_m_s: float | None = None,
    diameter_m: float | None = None,
    viscosity_m2_s: float | None = None,
    dt_phys_s: float | None = None,
    dt_ctu: float | None = None,
    sdf_clip: float | None = 8.0,
    check_rank_degeneracy: bool = True,
    normalize: bool = True,
) -> tuple[Flow3DBatch, dict[str, Any]]:
    """Load the certified source trajectory in SI. Overrides assert metadata agreement."""
    path = Path(h5_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    manifest, tier, audit = _audit_files(path.parent)
    rows = [r for r in manifest.get("trajectories", []) if r.get("file_name") == path.name]
    if len(rows) != 1:
        raise PhysicsContractError("expected one matching manifest trajectory")
    row = rows[0]
    ancestor, status = _admit_manifest_row(row, manifest, tier, audit)
    if (case_id is not None and case_id != row["case_id"]) or (
        ancestor_id is not None and ancestor_id != ancestor
    ):
        raise PhysicsContractError("provided identity differs from audited lineage")
    for key, value in (
        ("reynolds_number", reynolds_number),
        ("u0_m_s", u0_m_s),
        ("cylinder_diameter_m", diameter_m),
        ("kinematic_viscosity_m2_s", viscosity_m2_s),
        ("dt_phys_mean_s", dt_phys_s),
        ("dt_ctu_mean", dt_ctu),
    ):
        if value is not None and not np.isclose(value, row[key], rtol=1e-4, atol=0):
            raise PhysicsContractError(f"provided {key} differs from manifest")
    # check_rank_degeneracy remains accepted for callers; the shared admission check is mandatory.
    details, batch = _inspect_h5(
        path, row, ancestor, status, load=True, normalize=normalize, sdf_clip=sdf_clip
    )
    assert batch is not None
    return batch, details
