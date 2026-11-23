from __future__ import annotations

import numpy as np

from flow3d.data.fixtures import make_analytical_divergence_free_field_3d, make_cylinder_geometry_3d
from flow3d.physics.operators import curl_h, div_h, valid_div_of_curl_mask, verify_discrete_div_curl_identity


def test_analytical_divergence_free_field_and_compatible_div_curl_identity():
    shape_zyx = (12, 16, 20)
    spacing_zyx = (0.01875, 0.01875, 0.01875)
    field_4ch, a_vec = make_analytical_divergence_free_field_3d(shape_zyx, spacing_zyx, u0_m_s=0.5)

    # Verify discrete div_h(curl_h(A)) == 0 on double-eroded interior stencil
    report = verify_discrete_div_curl_identity(a_vec, spacing_zyx, fluid_mask=None)
    assert report["interior_valid_cells"] > 0
    assert report["interior_max_abs_div"] < 1e-5

    # Full velocity field u = curl_h(A) + (u0, 0, 0) is also discrete divergence-free on interior
    ones_mask = np.ones(shape_zyx, dtype=np.float32)
    interior = valid_div_of_curl_mask(ones_mask) > 0.5
    div_u = div_h(field_4ch[0], field_4ch[1], field_4ch[2], spacing_zyx, mask=None)
    assert float(np.max(np.abs(div_u[interior]))) < 1e-5


def test_div_curl_with_immersed_cylinder_holds_on_interior_and_exposes_boundary_truncation():
    shape_zyx = (12, 24, 32)
    spacing_zyx = (0.01875, 0.01875, 0.01875)
    geom, _ = make_cylinder_geometry_3d(shape_zyx, spacing_zyx, cylinder_diameter_m=0.3)
    fluid_mask = geom[0]
    _, a_vec = make_analytical_divergence_free_field_3d(shape_zyx, spacing_zyx)

    report = verify_discrete_div_curl_identity(a_vec, spacing_zyx, fluid_mask=fluid_mask)
    assert report["interior_valid_cells"] > 0
    # On the double-eroded valid interior mask, discrete div(curl(A)) is zero to float32 precision
    assert report["interior_max_abs_div"] < 1e-5
    # On the 1-cell truncation band adjacent to the obstacle/boundary, stencil truncation makes div(curl) != 0
    assert report["boundary_truncation_band_cells"] > 0
    assert report["boundary_truncation_band_max_abs_div"] > 1e-3


def test_curl_h_and_div_h_follow_right_handed_xyz_physical_axes_in_zyx_storage():
    # Grid (Z=5, Y=7, X=9), spacing (dz=0.4, dy=0.2, dx=0.1)
    dz, dy, dx = 0.4, 0.2, 0.1
    z = (np.arange(5, dtype=np.float32)[:, None, None]) * dz
    y = (np.arange(7, dtype=np.float32)[None, :, None]) * dy
    x = (np.arange(9, dtype=np.float32)[None, None, :]) * dx

    # Vector potential A = (0, 0, Az) with Az = 2*y - 3*x
    # curl(A) = (dAz/dY, -dAz/dX, 0) = (2, 3, 0)
    ax = np.zeros((5, 7, 9), dtype=np.float32)
    ay = np.zeros((5, 7, 9), dtype=np.float32)
    az = 2.0 * y - 3.0 * x + 0.0 * z
    a_vec = np.stack([ax, ay, az], axis=0)

    vel = curl_h(a_vec, (dz, dy, dx), mask=None)
    assert np.allclose(vel[0, 1:-1, 1:-1, 1:-1], 2.0, atol=1e-5)
    assert np.allclose(vel[1, 1:-1, 1:-1, 1:-1], 3.0, atol=1e-5)
    assert np.allclose(vel[2, 1:-1, 1:-1, 1:-1], 0.0, atol=1e-5)

    # With edge_order1=True and mask=None, even the 6 outer domain boundary faces are exact!
    vel_edge = curl_h(a_vec, (dz, dy, dx), mask=None, edge_order1=True)
    assert np.allclose(vel_edge[0], 2.0, atol=1e-5)
    assert np.allclose(vel_edge[1], 3.0, atol=1e-5)
    assert np.allclose(vel_edge[2], 0.0, atol=1e-5)
