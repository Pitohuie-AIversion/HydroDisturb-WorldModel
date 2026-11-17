from __future__ import annotations

import numpy as np

from stage1.geometry.masks import build_derivative_mask


def centered_difference(field: np.ndarray, spacing: float, axis: int, valid_mask: np.ndarray | None = None) -> np.ndarray:
    if spacing <= 0:
        raise ValueError("spacing must be positive")
    x = np.asarray(field, dtype=np.float32)
    out = np.zeros_like(x)
    sl_mid = [slice(None)] * x.ndim
    sl_mid[axis] = slice(1, -1)
    sl_l = sl_mid.copy(); sl_l[axis] = slice(0, -2)
    sl_r = sl_mid.copy(); sl_r[axis] = slice(2, None)
    out[tuple(sl_mid)] = (x[tuple(sl_r)] - x[tuple(sl_l)]) / (2.0 * spacing)
    if valid_mask is not None:
        out *= np.asarray(valid_mask, dtype=np.float32)
    return out


def divergence(u: np.ndarray, v: np.ndarray, w: np.ndarray, spacing: tuple[float, float, float], mask: np.ndarray | None = None) -> np.ndarray:
    if np.shape(u) != np.shape(v) or np.shape(u) != np.shape(w):
        raise ValueError("u, v and w must have identical shapes")
    # The last three dimensions are always D/H/W; keep any leading batch
    # dimensions untouched.
    masks = [build_derivative_mask(mask, a) if mask is not None else None for a in (-3, -2, -1)]
    return (
        centered_difference(u, spacing[0], -3, masks[0])
        + centered_difference(v, spacing[1], -2, masks[1])
        + centered_difference(w, spacing[2], -1, masks[2])
    )


def vorticity_z(u: np.ndarray, v: np.ndarray, spacing: tuple[float, float, float], mask: np.ndarray | None = None) -> np.ndarray:
    """z component of curl for the last two spatial dimensions."""
    mask_x = build_derivative_mask(mask, -2) if mask is not None else None
    mask_y = build_derivative_mask(mask, -1) if mask is not None else None
    return centered_difference(v, spacing[1], -2, mask_x) - centered_difference(u, spacing[0], -1, mask_y)


def masked_mean(values: np.ndarray, mask: np.ndarray, eps: float = 1e-12) -> float:
    values = np.asarray(values, dtype=np.float64)
    mask = np.asarray(mask, dtype=np.float64)
    denom = float(mask.sum())
    if denom <= eps:
        raise ValueError("mask contains no valid cells")
    return float((values * mask).sum() / denom)
