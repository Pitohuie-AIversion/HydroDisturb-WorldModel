from dataclasses import replace
import hashlib
import json

import numpy as np
import pytest

from flow3d.cli.inspect_data import _has_strict_failure
from flow3d.contracts import PhysicsContractError, SampleProvenance
from flow3d.data.adapters import load_c04_benchmark_npz
from flow3d.data.inspect import inspect_single_source, run_g0_inspection_from_config
from flow3d.data.schema import DatasetSchema3D
from flow3d.data.splits import temporal_block_split_with_purge
from flow3d.physics.boundary import evaluate_cylinder_boundary_conditions


def _write_npz(path, *, canonical=False, changes=None, times=(0, 0.06, 0.12, 0.18)):
    shape = (64, 96, 192) if canonical else (3, 5, 7)
    z, y, x = shape
    yy = -0.9 + (np.arange(y) + 0.5) * 0.01875
    xx = -0.6 + (np.arange(x) + 0.5) * 0.01875
    sdf = np.broadcast_to((np.hypot(yy[:, None], xx[None, :]) - 0.15).astype("f"), shape)
    meta = {
        "case_id": "C04",
        "is_3d": True,
        "channels": ["u", "v", "w", "p"],
        "units": ["m/s", "m/s", "m/s", "Pa"],
        "layout": "(T, 4, Nz, Ny, Nx)",
        "spatial_shape": [z, y, x],
        "num_frames": len(times),
        "cell_spacing_m": 0.01875,
        "cylinder_diameter_m": 0.30,
        "time_start_s": times[0],
        "time_end_s": times[-1],
    }
    meta.update(changes or {})
    fields = np.zeros((len(times), 4, *shape), np.float32)
    np.savez_compressed(
        path,
        fields=fields,
        times=np.array(times, dtype="f"),
        solid_mask=sdf <= 0,
        sdf=sdf,
        metadata_json=json.dumps(meta),
    )
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_c04_pinning_cli_and_config(tmp_path):
    # Generated from the upstream C04 writer's cell-centred grid and metadata schema.
    p = tmp_path / "canonical.npz"
    pin = _write_npz(p, canonical=True)
    batch, decoded = load_c04_benchmark_npz(p, normalize=False)
    assert not batch.provenance[0].ground_truth_verified
    assert _has_strict_failure(decoded)
    config = tmp_path / "data.json"
    config.write_text(
        json.dumps({"sources": {"c04_benchmark_4frame_npz": str(p)}, "c04_benchmark_sha256": pin})
    )
    report = run_g0_inspection_from_config(config)
    row = report["sources"]["c04_benchmark_4frame_npz"]
    assert row["status"] == "VERIFIED_4FRAME_BENCHMARK" and row["ground_truth_verified"]
    assert row["sha256"] == pin and not row["supports_stage2_L4_window"]
    assert not _has_strict_failure(report)
    assert _has_strict_failure(inspect_single_source(p, expected_sha256="0" * 64))


@pytest.mark.parametrize(
    "changes,times,pinned",
    [
        ({"case_id": "Re5K"}, [0, 0.06, 0.12, 0.18], False),
        ({"channels": ["u", "w", "v", "p"]}, [0, 0.06, 0.12, 0.18], False),
        ({"units": ["1"] * 4}, [0, 0.06, 0.12, 0.18], False),
        ({"reynolds_number": 5001, "u0_m_s": 0.01982}, [0, 0.06, 0.12, 0.18], False),
        ({}, [0, 0.01, 0.02, 1, 2], False),
        ({}, [0, 0.06, 0.12, 0.18], True),  # A trusted digest cannot waive the canonical reference shape.
    ],
)
def test_bad_npz_cannot_acquire_reference_identity(tmp_path, changes, times, pinned):
    p = tmp_path / "unknown.npz"
    pin = _write_npz(p, changes=changes, times=times)
    report = inspect_single_source(p, expected_sha256=pin if pinned else None)
    assert report["status"] == "INVALID" and _has_strict_failure(report)


def test_six_face_transport_si_and_solid_mask():
    shape = (3, 4, 5)
    field = np.zeros((4, *shape))
    field[0, :, :, -1] = 2
    field[1, :, 0, :] = 3
    field[2, -1, :, :] = 4
    fluid = np.ones(shape)
    fluid[0, 0, -1] = 0
    field[:3, 0, 0, -1] = 99
    diag = evaluate_cylinder_boundary_conditions(field, fluid, np.ones(shape), (2.0, 3.0, 5.0))
    assert diag.roi_face_flux_m3_s == {
        "x_min": 0,
        "x_max": 2 * (3 * 4 - 1) * 6,
        "y_min": -3 * (3 * 5 - 1) * 10,
        "y_max": 0,
        "z_min": 0,
        "z_max": 4 * (4 * 5) * 15,
    }
    assert diag.roi_net_outward_flux_m3_s == sum(diag.roi_face_flux_m3_s.values())
    assert "net_streamwise_flux_diff" not in diag.to_dict()
    schema = DatasetSchema3D().to_dict()
    assert schema["sampled_outer_faces"] == "interior_roi_cell_center_traces"
    assert schema["solver_boundaries"]["z_span"] == "periodic"


def _series(n):
    return [
        SampleProvenance(
            case_id="c",
            trajectory_id="t",
            ancestor_id="solver",
            geometry_id="g",
            segment_id="s",
            ground_truth_verified=True,
            time_phys_s=5 + 0.04 * t,
            time_ctu=50 + 0.1 * t,
            time_idx=t + 600,
        )
        for t in range(n)
    ]


def test_temporal_split_only_emits_complete_windows():
    s = temporal_block_split_with_purge(_series(100), history_frames=4, forecast_frames=10)
    blocks = (s.train_indices, s.val_indices, s.test_indices)
    for starts, block in zip(s.window_starts, blocks):
        assert starts and starts[-1] + 13 == block[-1]
        assert all(set(range(i, i + 14)) <= set(block) for i in starts)
    assert min(blocks[1]) - max(blocks[0]) - 1 == min(blocks[2]) - max(blocks[1]) - 1 == 14
    single = temporal_block_split_with_purge(
        _series(14), history_frames=4, forecast_frames=10, ratios=(1, 0, 0)
    )
    assert single.window_starts == ((0,), (), ())


@pytest.mark.parametrize(
    "problem",
    ["too_short", "tiny_gap", "mixed_segment", "unverified", "bad_dt", "index_gap", "unknown_id", "ratios"],
)
def test_temporal_split_refuses_unverifiable_windows(problem):
    provenance = _series(50 if problem == "too_short" else 200)
    args = {"history_frames": 4, "forecast_frames": 10}
    if problem == "tiny_gap":
        args["purge_gap_frames"] = 1
    elif problem == "ratios":
        args["ratios"] = (1, -0.1, 0.1)
    elif problem not in ("too_short",):
        changes = {
            "mixed_segment": {"segment_id": "other"},
            "unverified": {"ground_truth_verified": False},
            "bad_dt": {"time_phys_s": 8.45721},
            "index_gap": {"time_idx": 91},
            "unknown_id": {"ancestor_id": "unknown"},
        }
        provenance[87] = replace(provenance[87], **changes[problem])
    with pytest.raises(PhysicsContractError):
        temporal_block_split_with_purge(provenance, **args)


def test_roi_matching_means_and_flux_do_not_hide_local_velocity_and_pressure_errors():
    shape = (2, 2, 3)
    reference = np.zeros((4, *shape))
    predicted = reference.copy()
    # Opposite errors on x-max cancel in both its normal-velocity mean and its net flux.
    predicted[0, 0, :, -1], predicted[0, 1, :, -1] = 3, -3
    predicted[1, 0, :, -1], predicted[1, 1, :, -1] = 4, -4
    predicted[3, 0, :, -1], predicted[3, 1, :, -1] = 9, -9
    m = np.ones(shape)
    m[:, :, 0] = 0
    predicted[:, :, :, 0] = np.nan
    diag = evaluate_cylinder_boundary_conditions(predicted, m, np.ones(shape), reference_4ch=reference)
    assert diag.roi_x_max_mean_u_m_s == diag.roi_face_flux_m3_s["x_max"] == 0
    assert diag.roi_x_min_mean_u_m_s is None
    assert diag.roi_face_reference_errors["x_max"] == {
        "fluid_cells": 4,
        "velocity_vector_rmse_m_s": 5,
        "pressure_rmse_pa": 9,
    }
    assert diag.roi_face_reference_errors["x_min"] == {
        "fluid_cells": 0,
        "velocity_vector_rmse_m_s": None,
        "pressure_rmse_pa": None,
    }
    assert set(diag.roi_face_reference_errors) == set(diag.roi_face_flux_m3_s)
