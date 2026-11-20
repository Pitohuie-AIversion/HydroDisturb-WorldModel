from __future__ import annotations

import numpy as np

from flow3d.contracts import PhysicsContractError


def build_3d_derivative_mask(fluid_mask: np.ndarray, axis: int) -> np.ndarray:
    """Return mask of cells where central difference along `axis` uses only valid fluid neighbors (no wrap)."""
    m = (np.asarray(fluid_mask) > 0.5).astype(np.float32)
    out = np.zeros_like(m, dtype=np.float32)
    sl_mid = [slice(None)] * m.ndim
    sl_l = [slice(None)] * m.ndim
    sl_r = [slice(None)] * m.ndim
    sl_mid[axis] = slice(1, -1)
    sl_l[axis] = slice(0, -2)
    sl_r[axis] = slice(2, None)
    out[tuple(sl_mid)] = m[tuple(sl_mid)] * m[tuple(sl_l)] * m[tuple(sl_r)]
    return out


def build_full_3d_stencil_mask(fluid_mask: np.ndarray) -> np.ndarray:
    """Cells where 1st-order central differences are simultaneously valid along Z (-3), Y (-2), and X (-1)."""
    mz = build_3d_derivative_mask(fluid_mask, -3)
    my = build_3d_derivative_mask(fluid_mask, -2)
    mx = build_3d_derivative_mask(fluid_mask, -1)
    return (mz * my * mx).astype(np.float32)


def valid_div_of_curl_mask(fluid_mask: np.ndarray) -> np.ndarray:
    """Cells where composed 2nd-order `div_h(curl_h(A))` has a complete un-truncated fluid stencil.

    Because `curl_h` requires distance-1 fluid neighbors and `div_h` differentiates `curl_h` again,
    exact cancellation `div_h(curl_h(A)) == 0` holds ONLY on cells whose distance-1 neighbors are
    themselves valid for 1st-order derivatives (i.e. distance >= 2 from domain boundaries and obstacles).
    """
    first_valid = build_full_3d_stencil_mask(fluid_mask)
    return build_full_3d_stencil_mask(first_valid)


def centered_difference_3d(
    field: np.ndarray,
    spacing: float,
    axis: int,
    valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    if spacing <= 0 or not np.isfinite(spacing):
        raise PhysicsContractError(f"spacing must be a positive finite float, got {spacing}")
    x = np.asarray(field, dtype=np.float64)
    out = np.zeros_like(x, dtype=np.float64)
    sl_mid = [slice(None)] * x.ndim
    sl_l = [slice(None)] * x.ndim
    sl_r = [slice(None)] * x.ndim
    sl_mid[axis] = slice(1, -1)
    sl_l[axis] = slice(0, -2)
    sl_r[axis] = slice(2, None)
    # Use safe finite values before subtraction so solid NaNs outside valid_mask do not leak
    x_safe = np.where(np.isfinite(x), x, 0.0)
    out[tuple(sl_mid)] = (x_safe[tuple(sl_r)] - x_safe[tuple(sl_l)]) / (2.0 * float(spacing))
    if valid_mask is not None:
        out *= np.asarray(valid_mask, dtype=np.float64)
    return out.astype(np.float32)


def div_h(
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
    spacing_zyx: tuple[float, float, float],
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Compatible discrete divergence du/dX + dv/dY + dw/dZ on a (..., Z, Y, X) grid.

    - `u` is streamwise along X (axis -1, spacing_zyx[2] = dx)
    - `v` is cross-flow along Y (axis -2, spacing_zyx[1] = dy)
    - `w` is spanwise along Z (axis -3, spacing_zyx[0] = dz)
    """
    if np.shape(u) != np.shape(v) or np.shape(u) != np.shape(w):
        raise PhysicsContractError("u, v, and w must have identical shapes")
    if len(spacing_zyx) != 3 or any(h <= 0 for h in spacing_zyx):
        raise PhysicsContractError(f"spacing_zyx must be 3 positive floats (dz, dy, dx), got {spacing_zyx}")
    dz, dy, dx = spacing_zyx
    valid = build_full_3d_stencil_mask(mask) if mask is not None else None
    du_dx = centered_difference_3d(u, dx, -1, valid)
    dv_dy = centered_difference_3d(v, dy, -2, valid)
    dw_dz = centered_difference_3d(w, dz, -3, valid)
    return (du_dx + dv_dy + dw_dz).astype(np.float32)


def curl_h(
    a_vec: np.ndarray,
    spacing_zyx: tuple[float, float, float],
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Compatible discrete curl `u = curl_h(A)` for `A = [Ax, Ay, Az]` on a `(..., 3, Z, Y, X)` grid.

    Returns `[u, v, w]` where:
    - `u = (curl A)_x = d(Az)/dY - d(Ay)/dZ`
    - `v = (curl A)_y = d(Ax)/dZ - d(Az)/dX`
    - `w = (curl A)_z = d(Ay)/dX - d(Ax)/dY`
    """
    a = np.asarray(a_vec, dtype=np.float64)
    if a.ndim < 4 or a.shape[-4] != 3:
        raise PhysicsContractError(f"a_vec must have shape (..., 3, Z, Y, X), got {a.shape}")
    if len(spacing_zyx) != 3 or any(h <= 0 for h in spacing_zyx):
        raise PhysicsContractError(f"spacing_zyx must be 3 positive floats (dz, dy, dx), got {spacing_zyx}")
    dz, dy, dx = spacing_zyx
    ax = a[..., 0, :, :, :]
    ay = a[..., 1, :, :, :]
    az = a[..., 2, :, :, :]
    valid = build_full_3d_stencil_mask(mask) if mask is not None else None

    daz_dy = centered_difference_3d(az, dy, -2, valid)
    day_dz = centered_difference_3d(ay, dz, -3, valid)
    dax_dz = centered_difference_3d(ax, dz, -3, valid)
    daz_dx = centered_difference_3d(az, dx, -1, valid)
    day_dx = centered_difference_3d(ay, dx, -1, valid)
    dax_dy = centered_difference_3d(ax, dy, -2, valid)

    u = daz_dy - day_dz
    v = dax_dz - daz_dx
    w = day_dx - dax_dy
    return np.stack([u, v, w], axis=-4).astype(np.float32)


def verify_discrete_div_curl_identity(
    a_vec: np.ndarray,
    spacing_zyx: tuple[float, float, float],
    fluid_mask: np.ndarray | None = None,
) -> dict[str, float]:
    """Verify `div_h(curl_h(A)) == 0` on the double-eroded interior mask and report boundary truncation error."""
    vel = curl_h(a_vec, spacing_zyx, mask=fluid_mask)
    # Unmasked div of the masked curl vs double-eroded interior mask
    raw_div = div_h(vel[..., 0, :, :, :], vel[..., 1, :, :, :], vel[..., 2, :, :, :], spacing_zyx, mask=fluid_mask)
    if fluid_mask is None:
        ones = np.ones(a_vec.shape[-3:], dtype=np.float32)
        interior_mask = valid_div_of_curl_mask(ones) > 0.5
        single_mask = build_full_3d_stencil_mask(ones) > 0.5
    else:
        interior_mask = valid_div_of_curl_mask(fluid_mask) > 0.5
        single_mask = build_full_3d_stencil_mask(fluid_mask) > 0.5

    boundary_band = single_mask & (~interior_mask)
    interior_max = float(np.max(np.abs(raw_div[..., interior_mask]))) if np.any(interior_mask) else 0.0
    boundary_band_max = float(np.max(np.abs(raw_div[..., boundary_band]))) if np.any(boundary_band) else 0.0
    return {
        "interior_valid_cells": float(interior_mask.sum()),
        "interior_max_abs_div": interior_max,
        "boundary_truncation_band_cells": float(boundary_band.sum()),
        "boundary_truncation_band_max_abs_div": boundary_band_max,
    }
