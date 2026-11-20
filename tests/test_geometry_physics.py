import numpy as np

from stage1.geometry.masks import build_derivative_mask, sanitize_geometry
from stage1.physics.operators import centered_difference, divergence, vorticity_z


def test_geometry_sanitization_and_nonwrapping_derivative_mask():
    mask = np.ones((3, 4, 5), dtype=np.float32)
    mask[:, 2, 2] = 0
    sdf = np.full_like(mask, 100)
    clean_mask, clean_sdf = sanitize_geometry(mask, sdf)
    assert clean_mask[:, 2, 2].sum() == 0
    assert clean_sdf.max() == 1.0
    valid = build_derivative_mask(clean_mask, axis=-1)
    assert np.all(valid[..., 0] == 0)
    assert np.all(valid[..., -1] == 0)
    assert valid[:, 2, 2].sum() == 0


def test_centered_difference_uses_zero_boundary_and_mask():
    x = np.arange(27, dtype=np.float32).reshape(3, 3, 3)
    d = centered_difference(x, 1.0, axis=-1)
    assert np.all(d[..., 0] == 0) and np.all(d[..., -1] == 0)
    assert np.isclose(d[1, 1, 1], 1.0)


def test_divergence_of_constant_field_is_zero():
    shape = (4, 5, 6)
    mask = np.ones(shape, dtype=np.float32)
    u = np.ones(shape, dtype=np.float32)
    v = np.ones(shape, dtype=np.float32) * 2
    w = np.ones(shape, dtype=np.float32) * -3
    div = divergence(u, v, w, (1.0, 1.0, 1.0), mask)
    assert np.allclose(div, 0)


def test_divergence_and_vorticity_follow_zyx_axis_convention():
    # Shape is (Z=4, Y=5, X=6) with spacing (dz=0.5, dy=0.25, dx=0.125)
    z = np.arange(4, dtype=np.float32)[:, None, None] * 0.5
    y = np.arange(5, dtype=np.float32)[None, :, None] * 0.25
    x = np.arange(6, dtype=np.float32)[None, None, :] * 0.125
    mask = np.ones((4, 5, 6), dtype=np.float32)
    # u = 2*x + 3*y -> du/dX = 2, du/dY = 3
    # v = 5*x - 2*y -> dv/dX = 5, dv/dY = -2
    # w = 0*z       -> dw/dZ = 0
    # True divergence = du/dX + dv/dY + dw/dZ = 2 - 2 + 0 = 0
    # True vorticity_z = dv/dX - du/dY = 5 - 3 = 2
    u = 2.0 * x + 3.0 * y + 0.0 * z
    v = 5.0 * x - 2.0 * y + 0.0 * z
    w = np.zeros_like(u)
    spacing = (0.5, 0.25, 0.125)
    div = divergence(u, v, w, spacing, mask)
    vort = vorticity_z(u, v, spacing, mask)
    assert np.allclose(div[1:-1, 1:-1, 1:-1], 0.0, atol=1e-5)
    assert np.allclose(vort[1:-1, 1:-1, 1:-1], 2.0, atol=1e-5)
