import numpy as np

from stage1.geometry.masks import build_derivative_mask, sanitize_geometry
from stage1.physics.operators import centered_difference, divergence


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
