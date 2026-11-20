from __future__ import annotations

import numpy as np
import pytest

from flow3d.contracts import PhysicsContractError
from flow3d.data.fixtures import make_cylinder_geometry_3d
from flow3d.data.schema import convert_h5_6channel_to_field_and_geometry, transpose_yzx_to_zyx_fields
from flow3d.physics.boundary import evaluate_cylinder_boundary_conditions, first_fluid_shell_mask


def test_transpose_yzx_to_zyx_and_velocity_component_alignment():
    # Raw video_c04_shedding shape is (Y=96, Z=64, X=192) -> tested on compact (Y=12, Z=8, X=16)
    shape_yzx = (12, 8, 16)
    shape_zyx = (8, 12, 16)
    u_yzx = np.full(shape_yzx, 1.5, dtype=np.float32)
    w_cross_yzx = np.full(shape_yzx, -0.25, dtype=np.float32)  # cross-flow along Y
    v_span_yzx = np.full(shape_yzx, 0.05, dtype=np.float32)    # spanwise along Z
    p_yzx = np.full(shape_yzx, -10.0, dtype=np.float32)

    stacked_zyx = transpose_yzx_to_zyx_fields(u_yzx, w_cross_yzx, v_span_yzx, p_yzx, expected_zyx=shape_zyx)
    assert stacked_zyx.shape == (4, 8, 12, 16)
    assert np.allclose(stacked_zyx[0], 1.5)    # u (X)
    assert np.allclose(stacked_zyx[1], -0.25)  # v (Y = w_cross)
    assert np.allclose(stacked_zyx[2], 0.05)   # w (Z = v_span)
    assert np.allclose(stacked_zyx[3], -10.0)  # p

    # Wrong raw shape must fail closed
    with pytest.raises(PhysicsContractError):
        transpose_yzx_to_zyx_fields(
            np.zeros((8, 12, 16), dtype=np.float32),
            w_cross_yzx,
            v_span_yzx,
            p_yzx,
            expected_zyx=shape_zyx,
        )


def test_convert_h5_6channel_reorders_sdf_and_flips_solid_mask():
    # HDF5 channels: [u_star, v_cross_star, w_span_star, p_star, sdf_star, solid_mask]
    h5_raw = np.zeros((2, 6, 4, 6, 8), dtype=np.float32)
    h5_raw[:, 0] = 1.0
    h5_raw[:, 4] = 0.75  # sdf_star at index 4
    h5_raw[:, 5] = 1.0   # solid_mask = 1 (all solid) except interior fluid
    h5_raw[:, 5, :, 1:-1, 1:-1] = 0.0

    field, geometry = convert_h5_6channel_to_field_and_geometry(h5_raw)
    assert field.shape == (2, 4, 4, 6, 8)
    assert geometry.shape == (2, 2, 4, 6, 8)
    # Channel 0 of geometry is fluid_mask = 1 - solid_mask
    assert np.all(geometry[:, 0, :, 1:-1, 1:-1] == 1.0)
    assert np.all(geometry[:, 0, :, 0, :] == 0.0)
    # Channel 1 of geometry is sdf_over_d = sdf_star
    assert np.allclose(geometry[:, 1], 0.75)


def test_sdf_sign_convention_and_first_fluid_shell_boundary_diagnostics():
    geom, _ = make_cylinder_geometry_3d((8, 24, 32), spacing_zyx_m=(0.01875, 0.01875, 0.01875), cylinder_diameter_m=0.3)
    fluid_mask, sdf_over_d = geom[0], geom[1]
    fluid = fluid_mask > 0.5
    solid = ~fluid

    assert np.all(sdf_over_d[fluid] > 0.0)
    assert np.all(sdf_over_d[solid] <= 0.0)

    shell = first_fluid_shell_mask(fluid_mask)
    assert np.any(shell)
    assert not np.any(shell & solid)

    # Proof that zeroing solid interior does NOT imply no-slip on the first fluid shell
    field = np.zeros((4, 8, 24, 32), dtype=np.float32)
    field[0, fluid] = 0.5  # uniform slip flow right up to the first fluid shell, zero inside solid
    diag = evaluate_cylinder_boundary_conditions(field, fluid_mask, sdf_over_d)
    assert diag.solid_interior_max_abs == 0.0
    assert diag.first_shell_total_speed_rms > 0.4  # non-zero speed on first fluid shell detected!
