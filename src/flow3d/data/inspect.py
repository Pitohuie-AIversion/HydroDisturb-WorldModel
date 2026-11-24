from __future__ import annotations

from pathlib import Path
from typing import Any
from zipfile import BadZipFile

import numpy as np

from flow3d.contracts import PhysicsContractError
from flow3d.data.adapters import (
    inspect_and_segment_c04_shedding_directory,
    inspect_world_model_dataset_manifest,
    load_c04_benchmark_npz,
)
from flow3d.data.normalization import Flow3DNormalizer, si_roundtrip_diagnostics
from flow3d.data.schema import load_yaml_or_json_config
from flow3d.physics.boundary import evaluate_cylinder_boundary_conditions
from flow3d.physics.operators import build_full_3d_stencil_mask, div_h


def _g3_window_support(num_frames: int, dt_ctu: float) -> dict[str, Any]:
    """Window counts and direct DFT bin spacing on one verified constant-step trajectory."""
    history = 4
    horizons = []
    for forecast in (10, 20, 45, 50, 100):
        starts = max(0, num_frames - history - forecast + 1)
        # Elapsed time from the last history frame to the last prediction.
        # The direct DFT uses the H forecast samples, ignoring history and any zero padding.
        duration = forecast * dt_ctu if starts else None
        horizons.append({
            "forecast_frames": forecast,
            "available_starts": starts,
            "forecast_duration_ctu": duration,
            "dft_delta_st": 1.0 / duration if duration is not None else None,
        })
    return {"history_frames": history, "horizons": horizons}


def inspect_single_source(path: str | Path, *, expected_sha256: str | None = None) -> dict[str, Any]:
    """Inspect a single 3D CFD data path (C04 NPZ file, shedding NPZ directory, or HDF5 dataset directory)."""
    p = Path(path)
    if not p.exists():
        return {"path": str(p), "status": "MISSING", "error": f"path does not exist: {p}"}

    if p.is_file() and p.suffix.lower() == ".npz":
        try:
            batch, summary = load_c04_benchmark_npz(p, expected_sha256=expected_sha256)
        except (ValueError, TypeError, OSError, BadZipFile, EOFError) as exc:
            return {"format": "c04_benchmark_npz", "path": str(p), "status": "INVALID", "error": str(exc)}
        f0 = batch.field[0, 0]
        fluid_mask = batch.geometry[0, 0]
        sdf_over_d = batch.geometry[0, 1]
        div_field = div_h(f0[0], f0[1], f0[2], batch.spacing, mask=fluid_mask)
        stencil_valid = build_full_3d_stencil_mask(fluid_mask) > 0.5
        all_fluid = fluid_mask > 0.5
        bc_diag = evaluate_cylinder_boundary_conditions(f0, fluid_mask, sdf_over_d, batch.spacing)
        summary["stencil_valid_fluid_cells"] = int(stencil_valid.sum())
        summary["physical_divergence_rms_s_inv"] = (
            float(np.sqrt(np.mean(div_field[stencil_valid] ** 2))) if stencil_valid.any() else None
        )
        summary["physical_divergence_mae_s_inv"] = (
            float(np.mean(np.abs(div_field[stencil_valid]))) if stencil_valid.any() else None
        )
        summary["all_fluid_divergence_rms_s_inv"] = float(np.sqrt(np.mean(div_field[all_fluid] ** 2)))
        summary["all_fluid_divergence_mae_s_inv"] = float(np.mean(np.abs(div_field[all_fluid])))
        summary["boundary_diagnostics"] = bc_diag.to_dict()
        if summary["ground_truth_verified"]:
            normalizer = Flow3DNormalizer()
            summary["si_normalizer_roundtrips"] = []
            for i in (0, batch.field.shape[1] - 1):
                frame = batch.field[0, i]
                normalized, stats = normalizer.normalize_frame(frame, fluid_mask)
                restored = normalizer.denormalize_frame(normalized, stats, fluid_mask)
                diagnostic = si_roundtrip_diagnostics(frame, restored, fluid_mask)
                summary["si_normalizer_roundtrips"].append({"frame_index": i, **diagnostic})
                if not diagnostic["passed"]:
                    summary.update(
                        status="INVALID",
                        ground_truth_verified=False,
                        verification_reason=f"SI normalization roundtrip failed at frame {i}",
                    )
        return summary

    if p.is_dir():
        if (p / "manifest.json").exists():
            report = inspect_world_model_dataset_manifest(p)
            for row in report["eligible_trajectories"]:
                row["g3_window_support"] = _g3_window_support(row["num_steps"], row["dt_ctu_mean"])
            return report
        npz_files = list(p.glob("*.npz"))
        if npz_files:
            return inspect_and_segment_c04_shedding_directory(p)

    raise PhysicsContractError(f"unsupported or unknown 3D CFD dataset format at: {p}")


def run_g0_inspection_from_config(config_path: str | Path) -> dict[str, Any]:
    """Run G0 data inspection across all configured local 3D CFD data sources."""
    cfg = load_yaml_or_json_config(config_path)
    sources = cfg.get("sources", {})
    reports: dict[str, Any] = {}
    for name, src_path in sources.items():
        if isinstance(src_path, str):
            pin = cfg.get("c04_benchmark_sha256") if name == "c04_benchmark_4frame_npz" else None
            reports[name] = inspect_single_source(src_path, expected_sha256=pin)
    return {
        "schema_version": cfg.get("schema_version", "flow3d.data.v1"),
        "config_path": str(config_path),
        "sources": reports,
    }
