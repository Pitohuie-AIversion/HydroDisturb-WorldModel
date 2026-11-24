import json
import sys

import numpy as np
import pytest

from flow3d.cli.inspect_data import main as inspect_main
from flow3d.contracts import PhysicsContractError
from flow3d.data.adapters import (
    detect_temporal_rank_degeneracy,
    inspect_world_model_dataset_manifest,
    load_world_model_h5_trajectory,
)
from flow3d.data.inspect import _g3_window_support, inspect_single_source


def _write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def test_g1_rejects_changed_frames_or_hdf5_after_training(trajectory, monkeypatch):
    import copy
    from pathlib import Path
    import runpy

    pytest.importorskip("torch")
    pytest.importorskip("matplotlib")
    from flow3d.data.dataset import Flow3DH5Dataset, stage1_split_identity

    path, _ = trajectory
    datasets = Flow3DH5Dataset.create_splits(path.parent)
    identity = stage1_split_identity(*datasets)
    summary = {"config": {"training": {}}, "split_identity": identity}
    g1 = runpy.run_path(str(Path(__file__).resolve().parents[2] / "scripts/evaluate_g1_ab_heads.py"))

    # Checking frame membership must not fetch test tensors or evaluate either model.
    monkeypatch.setattr(Flow3DH5Dataset, "__getitem__", lambda *args: pytest.fail("unexpected frame read"))
    test_ds, record = g1["_held_out_split"](path.parent, summary, summary)
    assert record["status"] == "verified"
    assert len(test_ds) == len(identity["splits"]["test"]) > 0

    other = copy.deepcopy(summary)
    other["split_identity"]["splits"]["test"][0][-1] -= 1
    with pytest.raises(ValueError, match="different or unrecorded"):
        g1["_held_out_split"](path.parent, summary, other)
    with pytest.raises(ValueError, match="frame IDs differ"):
        g1["_held_out_split"](path.parent, other, other)

    with path.open("ab") as stream:
        stream.write(b"changed bytes after training")
    with pytest.raises(ValueError, match="HDF5 contents"):
        g1["_held_out_split"](path.parent, summary, summary)


def test_g1_selects_by_voxel_pooled_normalized_reconstruction():
    from pathlib import Path
    import runpy

    torch = pytest.importorskip("torch")
    pytest.importorskip("matplotlib")
    g1 = runpy.run_path(str(Path(__file__).resolve().parents[2] / "scripts/evaluate_g1_ab_heads.py"))
    items = []
    for i in range(2):
        geom = torch.ones(2, 8, 8, 8)
        geom[0, :i + 1] = 0
        items.append(dict(norm_field=torch.zeros(4, 8, 8, 8), phys_field=torch.zeros(4, 8, 8, 8),
                          geometry=geom, spacing_star=(1., 1., 1.), spacing_m=(1., 1., 1.), b=torch.ones(6),
                          case_id="fixture", reynolds_number=100, u0_m_s=1, step_index=i))

    class Predictions:
        def __init__(self, errors):
            self.errors = iter(errors)

        def __call__(self, norm, geom, **kwargs):
            offset = torch.tensor(next(self.errors)).reshape(1, 4, 1, 1, 1)
            pred = torch.where(geom[:, :1] > 0.5, norm + offset, 1000.)
            return {"norm_fields": pred, "phys_fields": torch.zeros_like(pred)}

    evaluate = g1["_evaluate_model_on_dataset"]
    select = g1["_select_head"]
    frames_a, _ = evaluate(Predictions([[1, 2, 2, 0], [0, 0, 0, 0]]), items, torch.device("cpu"))
    frames_b, _ = evaluate(Predictions([[0, 0, 0, 2], [0, 0, 0, 2]]), items, torch.device("cpu"))
    weights = {"reconstruction_weight": 0.5, "pressure_weight": 2.}
    selected = select(frames_a, frames_b, weights, weights)
    assert selected["scores"] == pytest.approx({"head_a": 0.5 * 9 * 7 / 13, "head_b": 8.})
    assert selected["winner"] == "head_a" and selected["weight_source"] == "shared_validation_weights"
    selected = select(frames_a, frames_b, weights, {})
    assert selected["scores"] == pytest.approx({"head_a": 9 * 7 / 13, "head_b": 4.})
    assert selected["winner"] == "head_b" and selected["weight_source"] == "common_unit_weights"
    assert select(frames_a, frames_a, weights, weights)["winner"] == "tie"


def _manifest(directory):
    channels = ["u_star", "v_cross_star", "w_span_star", "p_star", "sdf_star", "solid_mask"]
    row = {
        "case_id": "run_a",
        "source_model": "E:/solver/run/a.mph",
        "file_name": "a.h5",
        "reynolds_number": 100,
        "u0_m_s": 2.0,
        "cylinder_diameter_m": 0.3,
        "density_kg_m3": 10.0,
        "kinematic_viscosity_m2_s": 0.006,
        "ground_truth_verified": True,
        "channels": channels,
        "tensor_shape": [12, 6, 8, 8, 8],
        "num_steps": 12,
        "dt_phys_mean_s": 0.04,
        "dt_ctu_mean": 0.04 * 2 / 0.3,
    }
    _write_json(directory / "manifest.json", {"trajectories": [row]})
    _write_json(
        directory / "multi_re_tier1_certification_summary.json",
        {"cases": [{"case_dir": "e:/SOLVER/run", "target_Re": 100, "status": "UNSTABLE"}]},
    )
    _write_json(
        directory / "extraction_and_rampool_assessment.json", {"ram_pool_audit": {"integrity_checks": {}}}
    )
    return row


@pytest.fixture
def trajectory(tmp_path):
    h5py = pytest.importorskip("h5py")
    row = _manifest(tmp_path)
    path = tmp_path / "a.h5"
    rng = np.random.default_rng(520)
    fields = rng.normal(size=row["tensor_shape"]).astype("f")
    fields[:, 4] = 1
    fields[:, 5] = 0
    fields[:, 5, 0, 0, 0] = 1
    fields[:, 4, 0, 0, 0] = -1
    fields[:, :4, 0, 0, 0] = np.nan
    with h5py.File(path, "w") as f:
        for key in (
            "u0_m_s",
            "reynolds_number",
            "cylinder_diameter_m",
            "kinematic_viscosity_m2_s",
            "density_kg_m3",
            "source_model",
        ):
            f.attrs[key] = row[key]
        f.attrs["tensor_channels"] = row["channels"]
        f.attrs["p_inf_pa"] = 5
        f["flow_fields"] = fields
        f["time_physical_s"] = 4.1 + np.arange(12) * 0.04
        f["time_nondim_ctu"] = f["time_physical_s"][:] * 2 / 0.3
        f["step_indices"] = np.arange(110, 122)
        for i, key in enumerate(("z_span_m", "y_cross_m", "x_stream_m")):
            f[f"coordinates/{key}"] = np.arange(8) * 0.01 * (i + 1)
    return path, fields


def test_geometry_physical_time_lineage_and_si(trajectory):
    path, raw = trajectory
    report = inspect_world_model_dataset_manifest(path.parent)
    assert report["eligible_trajectory_count"] == 1
    assert report["tier1_passed_trajectory_count"] == 0
    batch, details = load_world_model_h5_trajectory(path)
    np.testing.assert_allclose(batch.spacing, [0.01, 0.02, 0.03])
    np.testing.assert_allclose(details["spacing_star"], np.array(batch.spacing) / 0.3)
    np.testing.assert_allclose(batch.field[0, 0, :3, 1, 1, 1], raw[0, :3, 1, 1, 1] * 2)
    assert batch.field[0, 0, 3, 1, 1, 1] == pytest.approx(raw[0, 3, 1, 1, 1] * 40 + 5)
    assert (batch.field[0, :, :, 0, 0, 0] == 0).all()
    assert batch.b[0, 0, 3] == pytest.approx(np.nanmean(raw[0, 3]) * 40 + 5)
    assert batch.conditions[0].dt_phys_s == pytest.approx(0.04)
    assert batch.provenance[0].time_phys_s == pytest.approx(4.1)
    assert batch.provenance[0].time_ctu == pytest.approx(4.1 * 2 / 0.3)
    assert batch.provenance[0].time_idx == 110
    assert batch.provenance[0].ancestor_id == "e:/solver/run/a.mph"
    assert batch.provenance[0].tier1_stability_status == "UNSTABLE"
    with pytest.raises(PhysicsContractError, match="dt_phys_mean_s"):
        load_world_model_h5_trajectory(path, dt_phys_s=0.02)


def test_missing_and_signature_only_are_not_eligible(tmp_path, monkeypatch):
    _manifest(tmp_path)
    assert inspect_world_model_dataset_manifest(tmp_path)["unverified_count"] == 1
    (tmp_path / "a.h5").write_bytes(b"wrong")
    report = inspect_world_model_dataset_manifest(tmp_path)
    assert report["eligible_trajectory_count"] == 0 and report["quarantined_count"] == 1
    (tmp_path / "a.h5").write_bytes(b"\x89HDF\r\n\x1a\n")
    monkeypatch.setitem(sys.modules, "h5py", None)
    report = inspect_world_model_dataset_manifest(tmp_path)
    assert report["eligible_trajectory_count"] == 0
    assert "signature verified" in report["unverified_trajectories"][0]["reason"]


@pytest.mark.parametrize(
    "bad", ["ground_truth", "lineage", "Re", "time_axis", "axes", "channel_order", "mask", "fluid_nan"]
)
def test_corruption_rejected_by_inspector_and_loader(trajectory, bad):
    import h5py

    path, _ = trajectory
    with h5py.File(path, "r+") as f:
        if bad == "ground_truth":
            p = path.parent / "manifest.json"
            m = json.loads(p.read_text())
            m["trajectories"][0]["ground_truth_verified"] = False
            _write_json(p, m)
        elif bad == "lineage":
            f.attrs["source_model"] = "a different solver"
        elif bad == "Re":
            f.attrs["reynolds_number"] = 200
        elif bad == "time_axis":
            f["time_physical_s"][5] += 0.01
        elif bad == "axes":
            f["coordinates/z_span_m"][2] += 0.03
        elif bad == "channel_order":
            f.attrs["tensor_channels"] = list(reversed(f.attrs["tensor_channels"]))
        elif bad == "mask":
            f["flow_fields"][8, 5, 1, 1, 1] = np.nan
        elif bad == "fluid_nan":
            f["flow_fields"][9, 2, 1, 1, 1] = np.nan
    assert inspect_world_model_dataset_manifest(path.parent)["eligible_trajectory_count"] == 0
    with pytest.raises(PhysicsContractError):
        load_world_model_h5_trajectory(path)


def test_oscillatory_rank_two_eligible_and_linear_requires_review(trajectory):
    import h5py

    path, raw = trajectory
    a, b = np.nan_to_num(raw[:2, :4])
    phase = np.arange(12)[:, None, None, None, None] * 0.3
    wave = np.cos(phase) * a + np.sin(phase) * b
    assert detect_temporal_rank_degeneracy(wave) == (False, 2)
    with h5py.File(path, "r+") as f:
        f["flow_fields"][:, :4] = wave
    report = inspect_world_model_dataset_manifest(path.parent)
    assert report["eligible_trajectories"][0]["low_temporal_rank"]
    with h5py.File(path, "r+") as f:
        f["flow_fields"][:, :4] = a + np.linspace(0, 1, 12)[:, None, None, None, None] * (b - a)
    report = inspect_world_model_dataset_manifest(path.parent)
    assert report["eligible_trajectory_count"] == 0
    assert "endpoint-linear" in report["unverified_trajectories"][0]["reason"]
    with pytest.raises(PhysicsContractError, match="endpoint-linear"):
        load_world_model_h5_trajectory(path, check_rank_degeneracy=False)


def test_partial_failure_count_preserves_known_exclusions(trajectory):
    path, _ = trajectory
    mp = path.parent / "manifest.json"
    original = json.loads(mp.read_text())["trajectories"][0]
    excluded = {**original, "ground_truth_verified": False, "file_name": "excluded.h5"}
    missing = {**original, "file_name": "missing.h5"}
    _write_json(mp, {"trajectories": [original, excluded]})
    assert inspect_main(["--path", str(path.parent), "--strict"]) == 0
    _write_json(mp, {"trajectories": [original, excluded, missing]})
    report = inspect_single_source(path.parent)
    assert report["eligible_trajectory_count"] == 1
    assert report["quarantined_count"] == 1
    assert report["unverified_count"] == 1
    assert report["required_candidate_failure_count"] == 1
    assert inspect_main(["--path", str(path.parent), "--strict"]) == 1
    assert report["eligible_trajectories"][0]["tier1_limit_cycle_status"] == "UNSTABLE"
    # This twelve-frame fixture has no ten-step forecast after four history frames.
    assert report["eligible_trajectories"][0]["g3_window_support"]["horizons"][0] == {
        "forecast_frames": 10,
        "available_starts": 0,
        "forecast_duration_ctu": None,
        "dft_delta_st": None,
    }
    assert all("g3_window_support" not in row for row in report["unverified_trajectories"])


def test_g3_window_start_counts_and_direct_dft_bin_spacing():
    support = _g3_window_support(50, 0.1)
    assert support["history_frames"] == 4
    rows = support["horizons"]
    assert [row["forecast_frames"] for row in rows] == [10, 20, 45, 50, 100]
    assert [row["available_starts"] for row in rows] == [37, 27, 2, 0, 0]
    for row in rows[:3]:
        duration = row["forecast_frames"] * 0.1
        assert row["forecast_duration_ctu"] == pytest.approx(duration)
        assert row["dft_delta_st"] == pytest.approx(1 / duration)
    assert all(row["forecast_duration_ctu"] is None and row["dft_delta_st"] is None for row in rows[3:])
    assert _g3_window_support(104, 0.1)["horizons"][-1]["available_starts"] == 1


def _branch_pair(path):
    import shutil

    import h5py

    mp = path.parent / "manifest.json"
    source = json.loads(mp.read_text())["trajectories"][0]
    branch = path.with_name("pulse.h5")
    shutil.copyfile(path, branch)
    with h5py.File(branch, "r+") as f:
        ts = f["time_physical_s"][:].tolist()
        f["flow_fields"][5:, 0, 1, 1, 1] += 1.0
    rows = []
    for name, step, filename in (("control", 0.0, path.name), ("pulse", 0.2, branch.name)):
        spec = {
            "pair_id": "experiment",
            "branch_id": name,
            "channel": "inlet_velocity_offset",
            "unit": "m/s",
            "times_s": ts,
            "values": [0.0] * 4 + [step] * (len(ts) - 4),
            "known_at_time_s": ts[3],
            "exogenous": True,
        }
        rows.append(
            {
                **source,
                "case_id": name,
                "file_name": filename,
                "ancestor_id": "initial_run",
                "boundary_condition_id": "fixed_inlet_pressure_outlet",
                "pressure_gauge": "outlet_mean",
                "intervention": spec,
            }
        )
    _write_json(mp, {"trajectories": rows})
    return branch, rows


def test_preflight_certifies_matched_schedules_without_split_leakage(trajectory):
    from flow3d.data.splits import group_split_by_provenance

    path, _ = trajectory
    assert (
        inspect_world_model_dataset_manifest(path.parent)["intervention_pairs"]["available_pair_count"] == 0
    )
    branch, _ = _branch_pair(path)
    report = inspect_world_model_dataset_manifest(path.parent)["intervention_pairs"]
    assert report["available_pairs"] == [
        {
            "pair_id": "experiment",
            "branch_ids": ["control", "pulse"],
            "file_names": ["a.h5", "pulse.h5"],
            "ancestor_id": "initial_run",
            "horizon": 8,
        }
    ]
    control, _ = load_world_model_h5_trajectory(path)
    pulse, _ = load_world_model_h5_trajectory(branch)
    cond = pulse.conditions[0]
    assert cond.known_mask["action_schedule"] and cond.valid_mask["action_schedule"]
    assert cond.to_dict()["extra_metadata"]["exogenous"] is True
    assert cond.action_schedule == (0.0,) * 4 + (0.2,) * 8
    assert cond.extra_metadata["action_times_s"] == [p.time_phys_s for p in pulse.provenance]
    assert cond.extra_metadata["action_unit"] == "m/s"
    restored = type(cond)(**json.loads(json.dumps(cond.to_dict())))
    assert restored.extra_metadata == cond.extra_metadata
    assert list(restored.action_schedule) == list(cond.action_schedule)
    with pytest.raises(PhysicsContractError, match="at least 3"):
        group_split_by_provenance([*control.provenance, *pulse.provenance])


@pytest.mark.parametrize(
    "mismatch",
    [
        "history",
        "grid",
        "physics",
        "times",
        "channel",
        "boundary_condition_id",
        "pressure_gauge",
        "future",
        "duplicate_id",
        "history_action",
        "endogenous",
        "late",
        "schedule_length",
        "schedule_time",
        "unknown_ancestor",
    ],
)
def test_intervention_rejects_unmatched_and_unavailable_schedules(trajectory, mismatch):
    import h5py

    path, _ = trajectory
    branch, rows = _branch_pair(path)
    pulse = rows[1]["intervention"]
    with h5py.File(branch, "r+") as f:
        if mismatch == "history":
            f["flow_fields"][3, 1, 2, 1, 1] += 0.5
        elif mismatch == "grid":
            f["coordinates/z_span_m"][:] += 0.25
        elif mismatch == "physics":
            f.attrs["density_kg_m3"] = rows[1]["density_kg_m3"] = 20
        elif mismatch == "times":
            f["time_physical_s"][:] += 1.0
            f["time_nondim_ctu"][:] += 2.0 / 0.3
            pulse["times_s"] = f["time_physical_s"][:].tolist()
        elif mismatch in {"pressure_gauge", "boundary_condition_id"}:
            rows[1][mismatch] = "different_reference"
        elif mismatch == "channel":
            pulse["unit"] = "Pa"
        elif mismatch == "future":
            pulse["values"] = [0.0] * len(pulse["times_s"])
        elif mismatch == "duplicate_id":
            pulse["branch_id"] = "control"
        elif mismatch == "history_action":
            pulse["values"][2] = 0.3
        elif mismatch == "endogenous":
            pulse["exogenous"] = False
        elif mismatch == "late":
            pulse["known_at_time_s"] = pulse["times_s"][4]
        elif mismatch == "schedule_length":
            pulse["values"].pop()
        elif mismatch == "schedule_time":
            pulse["times_s"][1] += 0.01
        elif mismatch == "unknown_ancestor":
            del rows[1]["ancestor_id"]
    _write_json(path.parent / "manifest.json", {"trajectories": rows})
    report = inspect_world_model_dataset_manifest(path.parent)
    assert report["intervention_pairs"]["available_pair_count"] == 0
    assert len(report["intervention_pairs"]["unmatched_pairs"]) == 1
    if mismatch in {"endogenous", "late", "schedule_length", "schedule_time", "unknown_ancestor"}:
        with pytest.raises(PhysicsContractError):
            load_world_model_h5_trajectory(branch)


def test_c04_and_hdf5_same_si_stats_and_physical_readout(trajectory):
    import h5py

    torch = pytest.importorskip("torch")
    from flow3d.data.adapters import load_c04_benchmark_npz
    from flow3d.data.normalization import Flow3DNormalizer
    from flow3d.models.heads import PhysicalReadout3D

    path, raw = trajectory
    with h5py.File(path, "r+") as f:
        times = f["time_physical_s"][:]
        # Express the HDF5 fixture in the benchmark's reference units and isotropic grid.
        physics = {"u0_m_s": 0.5, "reynolds_number": 126167.05, "kinematic_viscosity_m2_s": 1.1889e-6}
        for key, value in physics.items():
            f.attrs[key] = value
        f["time_nondim_ctu"][:] = times * 0.5 / 0.3
        for axis in ("z_span_m", "y_cross_m", "x_stream_m"):
            f[f"coordinates/{axis}"][:] = np.arange(8) * 0.01875
    manifest_path = path.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["trajectories"][0].update(**physics, dt_ctu_mean=0.04 * 0.5 / 0.3)
    _write_json(manifest_path, manifest)
    cert = path.parent / "multi_re_tier1_certification_summary.json"
    authority = json.loads(cert.read_text())
    authority["cases"][0]["target_Re"] = physics["reynolds_number"]
    _write_json(cert, authority)
    # Independent dimensional reference; density=10 kg/m3, pressure offset=5 Pa.
    si = raw[:, :4].copy()
    si[:, :3] *= 0.5
    si[:, 3] = si[:, 3] * 2.5 + 5
    npz = path.with_suffix(".npz")
    metadata = {
        "case_id": "C04",
        "is_3d": True,
        "channels": ["u", "v", "w", "p"],
        "units": ["m/s", "m/s", "m/s", "Pa"],
        "layout": "(T, 4, Nz, Ny, Nx)",
        "spatial_shape": list(si.shape[2:]),
        "num_frames": len(times),
        "cylinder_diameter_m": 0.3,
        "cell_spacing_m": 0.01875,
        "time_start_s": times[0],
        "time_end_s": times[-1],
    }
    np.savez(
        npz,
        fields=si,
        times=times,
        sdf=0.3 * raw[0, 4],
        solid_mask=raw[0, 5],
        metadata_json=json.dumps(metadata),
    )
    benchmark, unpinned = load_c04_benchmark_npz(npz)
    assert unpinned["status"] == "UNVERIFIED" and not benchmark.provenance[0].ground_truth_verified
    h5, _ = load_world_model_h5_trajectory(path)
    np.testing.assert_allclose(h5.field, benchmark.field)
    np.testing.assert_allclose(h5.b, benchmark.b)
    np.testing.assert_allclose(h5.geometry, benchmark.geometry)
    np.testing.assert_allclose(h5.spacing_over_d(), benchmark.spacing_over_d())
    normalized, _ = Flow3DNormalizer().normalize(benchmark.field, benchmark.geometry[0, 0])
    actual = PhysicalReadout3D()(
        torch.from_numpy(normalized), torch.from_numpy(h5.b), torch.from_numpy(h5.geometry[:, :1])
    )
    np.testing.assert_allclose(actual.detach().numpy(), benchmark.field, rtol=1e-5, atol=1e-6)


def test_stage1_single_frame_dataset_preserves_solver_timeline(trajectory):
    pytest.importorskip("torch")
    from flow3d.data.dataset import Flow3DH5Dataset, collate_flow3d_stage1

    path, _ = trajectory
    dataset = Flow3DH5Dataset(path.parent, eligible_files=[path.name], split="train", purge_gap_frames=4)
    first = dataset[0]
    assert first["step_index"] == 110
    assert first["time_physical_s"] == pytest.approx(4.1)
    assert first["time_nondim_ctu"] == pytest.approx(4.1 * 2 / 0.3)
    assert collate_flow3d_stage1([first])["meta"][0]["time_physical_s"] == pytest.approx(4.1)


def test_dataset_groups_related_branches_by_ancestor(trajectory):
    import shutil

    import h5py

    pytest.importorskip("torch")
    from flow3d.data.dataset import Flow3DH5Dataset

    path, _ = trajectory
    _, rows = _branch_pair(path)
    template = {k: v for k, v in rows[0].items() if k != "intervention"}
    cert_path = path.parent / "multi_re_tier1_certification_summary.json"
    cert = json.loads(cert_path.read_text())
    for name in ("run_b", "run_c"):
        file = path.with_name(f"{name}.h5")
        shutil.copyfile(path, file)
        model = f"E:/solver/{name}/{name}.mph"
        with h5py.File(file, "r+") as f:
            f.attrs["source_model"] = model
        rows.append(
            {**template, "case_id": name, "file_name": file.name, "source_model": model, "ancestor_id": name}
        )
        cert["cases"].append({"case_dir": f"E:/solver/{name}", "target_Re": 100, "status": "UNSTABLE"})
    _write_json(cert_path, cert)
    _write_json(path.parent / "manifest.json", {"trajectories": rows})
    args = {
        "dataset_dir": path.parent,
        "eligible_files": [r["file_name"] for r in rows],
        "as_torch_tensors": False,
    }
    with pytest.raises(PhysicsContractError, match="group_by_ancestor"):
        Flow3DH5Dataset(**args, split="train", split_strategy="temporal_block")
    owners = []
    for part in ("train", "val", "test"):
        ds = Flow3DH5Dataset(**args, split=part, split_strategy="group_by_ancestor")
        owners.append({r["case_id"] for r in ds})
    assert {"control", "pulse"} in owners
    assert len(set.union(*owners)) == 4 and sum(map(len, owners)) == 4


def test_forecast_clis_train_decode_score_and_predict_without_ground_truth(trajectory, request):
    import os
    from pathlib import Path
    import subprocess

    torch = pytest.importorskip("torch")
    pytest.importorskip("yaml")
    from flow3d.data.schema import load_yaml_or_json_config

    path, raw = trajectory
    repo = Path(__file__).resolve().parents[2]
    cfg = load_yaml_or_json_config(repo / "configs/flow3d/stage1.yaml")
    cfg["grid"].update(spatial_shape_zyx=[8, 8, 8])
    cfg["encoder"].update(base_channels=8, channel_mults=[1] * 4, latent_channels=8, gn_groups=1)
    cfg["decoder"].update(base_channels=8, channel_mults=[1] * 4, gn_groups=1)
    cfg["latent_norm"]["num_groups"] = 1
    cfg["spatial_transformer"].update(embed_dim=8, num_heads=1, mlp_dim=16)
    cfg["runtime"]["cpu_threads"] = 1
    cfg["training"].update(scheduler="cosine", train_ratio=0.4, val_ratio=0.3, test_ratio=0.3, purge_gap_frames=1)
    config_path = path.parent / "stage1.json"
    _write_json(config_path, cfg)
    checkpoint = path.parent / "checkpoint"
    subprocess.run(
        [sys.executable, str(repo / "scripts/train_stage1_flow3d.py"), "--dataset-dir", str(path.parent),
         "--config", str(config_path), "--save-dir", str(checkpoint), "--eligible-files", path.name,
         "--epochs", "1", "--batch-size", "2", "--num-workers", "0", "--no-amp", "--no-tensorboard"],
        cwd=repo, env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
        check=True,
    )

    from flow3d.cli.forecast import main as forecast_main

    original_threads = torch.get_num_threads()
    request.addfinalizer(lambda: torch.set_num_threads(original_threads))
    torch.set_num_threads(1)
    out = path.parent / "forecast"
    forecast_main(["run", "--checkpoint-dir", str(checkpoint), "--dataset-dir", str(path.parent),
                   "--out-dir", str(out), "--history", "2", "--horizon", "1", "--epochs", "1", "--device", "cpu"])
    report = json.loads((out / "rollout_report.json").read_text())
    assert report["split_frame_counts"] == {"train": 4, "val": 3, "test": 3}
    assert report["training"]["train_windows"] == 2 and report["test_windows"] == 1
    assert len(report["records"]) == 1
    row = report["records"][0]
    assert (row["origin_step"], row["target_step"]) == (120, 121)
    assert set(row["scores"]) == {"mode_res", "delta_z", "persistence"}
    # Independently recover persistence error from the original nondimensional HDF5 fields.
    fluid = raw[0, 5] == 0
    diff = raw[11, :4] - raw[10, :4]
    velocity_rmse = np.sqrt(np.mean(np.sum((2.0 * diff[:3, fluid]) ** 2, axis=0)))
    pressure_rmse = 40.0 * np.sqrt(np.mean(diff[3, fluid] ** 2))
    base = row["scores"]["persistence"]
    assert base["velocity_vector_rmse_m_s"] == pytest.approx(velocity_rmse)
    assert base["pressure_rmse_pa"] == pytest.approx(pressure_rmse)
    with np.load(out / "forecast_example.npz") as example:
        sample = {name: example[name] for name in example.files}
    assert sample["fields"].shape == (1, 4, 8, 8, 8)
    assert sample["channels"].tolist() == ["u", "v", "w", "p"]
    assert sample["units"].tolist() == ["m/s", "m/s", "m/s", "Pa"]
    np.testing.assert_array_equal(sample["fluid_mask"], fluid)
    np.testing.assert_allclose(sample["spacing_zyx_m"], [0.01, 0.02, 0.03])
    np.testing.assert_allclose(sample["time_physical_s"], [4.1 + 11 * 0.04])
    np.testing.assert_allclose(sample["time_nondim_ctu"], sample["time_physical_s"] * 2 / 0.3)
    assert np.isfinite(sample["fields"]).all()
    generated = sample["fields"][0]
    vel_error = generated[:3, fluid] - raw[11, :3][:, fluid] * 2.0
    p_error = generated[3, fluid] - (raw[11, 3][fluid] * 40.0 + 5.0)
    assert row["scores"]["mode_res"]["velocity_vector_rmse_m_s"] == pytest.approx(
        np.sqrt(np.mean(np.sum(vel_error**2, axis=0)))
    )
    assert row["scores"]["mode_res"]["pressure_rmse_pa"] == pytest.approx(np.sqrt(np.mean(p_error**2)))

    val_out = path.parent / "validation_forecast"
    forecast_main(["run", "--cache-manifest", report["cache_manifest"], "--dataset-dir", str(path.parent),
                   "--out-dir", str(val_out), "--eval-split", "val", "--origin-stride", "10",
                   "--history", "2", "--train-horizon", "2", "--horizon", "1", "--epochs", "1", "--no-example"])
    validation = json.loads((val_out / "rollout_report.json").read_text())
    assert validation["eval_split"] == "val" and "test_windows" not in validation
    assert validation["eligible_windows"] == validation["val_windows"] == 1
    assert validation["training"]["rollout_steps"] == 2 and validation["horizon"] == 1
    assert validation["records"][0]["target_step"] < row["origin_step"]
    assert not (val_out / "forecast_example.npz").exists()

    # Standalone inference must also work after the CFD source has gone offline.
    path.rename(path.with_suffix(".offline"))
    standalone = out / "standalone.npz"
    prediction_args = ["predict", "--cache-manifest", report["cache_manifest"], "--model", str(out / "forecaster.pt"),
                       "--trajectory", row["trajectory_id"], "--horizon", "1", "--device", "cpu"]
    forecast_main([*prediction_args, "--origin-step", str(row["origin_step"]), "--output", str(standalone)])
    with np.load(standalone) as predicted:
        assert set(predicted.files) == set(sample)
        for key, value in sample.items():
            np.testing.assert_array_equal(predicted[key], value)
    # At the last cached step neither a true future field nor a future latent exists.
    beyond_cache = out / "future.npz"
    forecast_main([*prediction_args, "--origin-step", "121", "--output", str(beyond_cache)])
    with np.load(beyond_cache) as future:
        assert np.isfinite(future["fields"]).all()
        np.testing.assert_allclose(future["time_physical_s"], [4.1 + 12 * 0.04])
