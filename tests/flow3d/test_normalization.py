from __future__ import annotations

import numpy as np
import pytest

from flow3d.contracts import PhysicsContractError
from flow3d.data.fixtures import make_cylinder_geometry_3d
from flow3d.data.normalization import Flow3DNormalizer


def test_flow3d_normalizer_joint_velocity_variance_not_divided_by_three_and_roundtrip():
    geom, _ = make_cylinder_geometry_3d((8, 12, 16))
    fluid_mask = geom[0]
    valid = fluid_mask > 0.5

    rng = np.random.RandomState(7)
    field = rng.randn(2, 3, 4, 8, 12, 16).astype(np.float32)
    field[:, :, 0] += 0.5  # streamwise mean u
    field[:, :, 3] = field[:, :, 3] * 120.0 - 18.0  # pressure gauge offset & scale

    normalizer = Flow3DNormalizer(scale_floor_uvw=1e-6, scale_floor_p=1e-6)
    norm_field, b = normalizer.normalize(field, fluid_mask)
    assert b.shape == (2, 3, 6)
    assert b.dtype == np.float32

    # Joint kinetic energy 0.5 * mean(u_tilde^2 + v_tilde^2 + w_tilde^2) over valid fluid cells == 0.5
    f00 = norm_field[0, 0]
    joint_ke = 0.5 * float(np.mean(f00[0, valid] ** 2 + f00[1, valid] ** 2 + f00[2, valid] ** 2))
    assert np.isclose(joint_ke, 0.5, atol=1e-4)
    assert np.isclose(float(np.std(f00[3, valid], ddof=0)), 1.0, atol=1e-4)

    # Roundtrip reconstruction on valid fluid cells
    restored = normalizer.denormalize(norm_field, b, fluid_mask=fluid_mask)
    for b_idx in range(2):
        for t_idx in range(3):
            for c_idx in range(4):
                assert np.allclose(
                    restored[b_idx, t_idx, c_idx, valid],
                    field[b_idx, t_idx, c_idx, valid],
                    atol=1e-4,
                    rtol=1e-5,
                )


def test_solid_nans_never_poison_stats_and_fluid_nans_fail_closed():
    geom, _ = make_cylinder_geometry_3d((8, 12, 16))
    fluid_mask = geom[0]
    solid = fluid_mask <= 0.5
    assert np.any(solid)

    rng = np.random.RandomState(11)
    frame = rng.randn(4, 8, 12, 16).astype(np.float32)
    # Inject NaNs and Infs inside solid obstacle cells
    frame[:, solid] = np.nan

    normalizer = Flow3DNormalizer()
    norm_frame, b = normalizer.normalize(frame, fluid_mask)
    assert np.all(np.isfinite(b))
    assert np.all(np.isfinite(norm_frame))
    assert np.all(norm_frame[:, solid] == 0.0)

    # Inject NaN inside a valid fluid cell -> must fail closed immediately
    bad_frame = frame.copy()
    fluid_coord = np.argwhere(fluid_mask > 0.5)[0]
    bad_frame[0, fluid_coord[0], fluid_coord[1], fluid_coord[2]] = np.nan
    with pytest.raises(PhysicsContractError, match="NaN/Inf inside valid fluid domain"):
        normalizer.fit_frame(bad_frame, fluid_mask)


def test_zero_variance_and_empty_fluid_mask_handling():
    normalizer = Flow3DNormalizer(scale_floor_uvw=1e-5, scale_floor_p=2e-4)
    fluid_mask = np.ones((4, 6, 8), dtype=np.float32)
    const_frame = np.zeros((4, 4, 6, 8), dtype=np.float32)
    const_frame[0] = 0.5
    const_frame[3] = -25.0

    stats = normalizer.fit_frame(const_frame, fluid_mask)
    assert np.isclose(stats.sigma_uvw, 1e-5)
    assert np.isclose(stats.sigma_p, 2e-4)
    norm_frame, b = normalizer.normalize(const_frame, fluid_mask)
    restored = normalizer.denormalize(norm_frame, b, fluid_mask=fluid_mask)
    assert np.allclose(restored, const_frame, atol=1e-5)

    # Empty fluid mask -> fail closed
    empty_mask = np.zeros((4, 6, 8), dtype=np.float32)
    with pytest.raises(PhysicsContractError, match="zero valid fluid cells"):
        normalizer.fit_frame(const_frame, empty_mask)
