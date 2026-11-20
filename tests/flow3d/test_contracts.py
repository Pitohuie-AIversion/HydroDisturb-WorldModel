from __future__ import annotations

import numpy as np
import pytest

from flow3d.contracts import (
    CYLINDER3D_STATE_SPEC,
    Flow3DBatch,
    Flow3DConditions,
    PhysicsContractError,
    SampleProvenance,
)
from flow3d.data.adapters import detect_temporal_rank_degeneracy
from flow3d.data.fixtures import make_cylinder_geometry_3d
from flow3d.data.schema import DatasetSchema3D
from flow3d.data.splits import group_split_by_provenance, temporal_block_split_with_purge


def test_cylinder3d_state_spec_and_schema_roundtrip():
    assert CYLINDER3D_STATE_SPEC.spatial_dim == 3
    assert CYLINDER3D_STATE_SPEC.field_channels == ("u", "v", "w", "p")
    assert CYLINDER3D_STATE_SPEC.geometry_channels == ("fluid_mask", "sdf_over_d")
    assert CYLINDER3D_STATE_SPEC.default_grid_size == (64, 96, 192)

    schema = DatasetSchema3D()
    restored = DatasetSchema3D.from_mapping(schema.to_dict())
    assert restored == schema
    with pytest.raises(PhysicsContractError):
        schema.validate_spatial_shape((96, 64, 192))


def test_reynolds_consistency_and_forbidden_future_condition_rejection():
    # Valid C04 condition: Re = 0.5 * 0.3 / 1.1889e-6 = 126167.05
    cond = Flow3DConditions(
        reynolds_number=126167.05,
        u0_m_s=0.5,
        diameter_m=0.3,
        viscosity_m2_s=1.1889e-6,
        dt_phys_s=0.06,
    )
    assert np.isclose(cond.resolved_dt_ctu, 0.1, atol=1e-6)

    # Mismatched Re (e.g. re180k solved at U0=0.5 m/s) -> must fail closed
    with pytest.raises(PhysicsContractError, match="Re consistency violation"):
        Flow3DConditions(
            reynolds_number=180000.0,
            u0_m_s=0.5,
            diameter_m=0.3,
            viscosity_m2_s=1.1889e-6,
            dt_phys_s=0.06,
        )

    # Forbidden future-leakage keys must be rejected
    with pytest.raises(PhysicsContractError, match="future-leakage"):
        Flow3DConditions(extra_metadata={"future_regime_label": "shedding_peak"})
    with pytest.raises(PhysicsContractError, match="strictly from history"):
        Flow3DConditions(history_regime_hint="future_vortex_label")


def test_flow3d_batch_shape_and_fluid_nan_validation():
    geom, _ = make_cylinder_geometry_3d((8, 12, 16))
    field = np.ones((2, 3, 4, 8, 12, 16), dtype=np.float32)
    geom_b = np.broadcast_to(geom[None, ...], (2, 2, 8, 12, 16)).copy()
    b_stats = np.ones((2, 3, 6), dtype=np.float32)

    batch = Flow3DBatch(field=field, geometry=geom_b, b=b_stats)
    assert batch.field.shape == (2, 3, 4, 8, 12, 16)

    # Wrong rank or wrong channel count must raise
    with pytest.raises(PhysicsContractError):
        Flow3DBatch(field=field[0], geometry=geom_b)

    # NaN inside fluid cell must raise
    bad_field = field.copy()
    fluid_zyx = np.argwhere(geom_b[0, 0] > 0.5)[0]
    bad_field[0, 0, 0, fluid_zyx[0], fluid_zyx[1], fluid_zyx[2]] = np.nan
    with pytest.raises(PhysicsContractError, match="NaN/Inf inside the valid fluid domain"):
        Flow3DBatch(field=bad_field, geometry=geom_b)


def test_group_split_prevents_ancestor_leakage_and_rejects_unverified_runs():
    provs = [
        SampleProvenance(
            case_id=f"case_{g}",
            geometry_id="cyl_d0p3",
            trajectory_id=f"traj_{g}_{t}",
            ancestor_id=f"anc_{g}",
            time_idx=t,
        )
        for g in range(5)
        for t in range(4)
    ]
    split = group_split_by_provenance(provs, group_by="ancestor_id", ratios=(0.6, 0.2, 0.2))
    assert split.leakage_verified
    assert not (set(split.train_groups) & set(split.val_groups))
    assert not (set(split.train_groups) & set(split.test_groups))
    assert not (set(split.val_groups) & set(split.test_groups))
    assert len(split.train_indices) + len(split.val_indices) + len(split.test_indices) == len(provs)

    # Unverified sample (e.g. c04_wm_25steps_golden) must be rejected before splitting
    bad_provs = list(provs)
    bad_provs[0] = SampleProvenance(
        case_id="c04_wm_25steps_golden",
        geometry_id="cyl_d0p3",
        trajectory_id="traj_fake",
        ancestor_id="anc_0",
        ground_truth_verified=False,
    )
    with pytest.raises(PhysicsContractError, match="quarantined"):
        group_split_by_provenance(bad_provs)

    # Temporal block split with purge gap for single trajectory
    t_split = temporal_block_split_with_purge(num_frames=50, purge_gap_frames=5)
    assert max(t_split.train_indices) + 5 < min(t_split.val_indices)
    assert max(t_split.val_indices) + 5 < min(t_split.test_indices)


def test_detect_temporal_rank_degeneracy_catches_synthetic_linear_interpolation():
    # Rank-2 linear interpolation between 2 frames across 25 steps
    f_start = np.random.RandomState(42).randn(4, 16, 16, 16).astype(np.float32)
    f_end = np.random.RandomState(43).randn(4, 16, 16, 16).astype(np.float32)
    alphas = np.linspace(0.0, 1.0, 25, dtype=np.float32)[:, None, None, None, None]
    synthetic_seq = (1.0 - alphas) * f_start[None, ...] + alphas * f_end[None, ...]
    is_degen, rank = detect_temporal_rank_degeneracy(synthetic_seq)
    assert is_degen and rank <= 2

    # Genuine multi-mode time evolution has rank > 2
    rng = np.random.RandomState(99)
    genuine_seq = rng.randn(12, 4, 16, 16, 16).astype(np.float32)
    is_degen_real, rank_real = detect_temporal_rank_degeneracy(genuine_seq)
    assert not is_degen_real and rank_real > 2
