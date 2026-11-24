"""Differentiable PyTorch 3D discrete differential and projection operators.

Strictly matches the axis convention, spacing order, non-wrapping slicing stencils,
joint 3-axis validity masks, and right-handed curl signs of `flow3d.physics.operators`:
- Spatial axes: (Z, Y, X) = (-3, -2, -1)
- Spacing: spacing_zyx = (dz, dy, dx)
- Velocity components: (u, v, w) along (X, Y, Z)
- All finite differences and projections run in FP32+ (half-precision autocast disabled).
"""

from __future__ import annotations

import math
from contextlib import nullcontext
from typing import Sequence

from flow3d.contracts import PhysicsContractError

try:  # Keep PyTorch optional so R1 `flow3d.data` remains importable without torch.
    import torch
except Exception:  # pragma: no cover - exercised only when torch is absent
    torch = None  # type: ignore[assignment]


def torch_available() -> bool:
    return torch is not None


def _require_torch() -> None:
    if torch is None:
        raise ImportError("flow3d.physics.torch_operators requires PyTorch. Install with `pip install -e .[torch]`.")


def _validate_spacing_zyx(spacing_zyx: Sequence[float]) -> tuple[float, float, float]:
    if len(spacing_zyx) != 3:
        raise PhysicsContractError(f"spacing_zyx must contain 3 positive finite floats (dz, dy, dx), got {spacing_zyx}")
    dz, dy, dx = (float(spacing_zyx[0]), float(spacing_zyx[1]), float(spacing_zyx[2]))
    if any((not math.isfinite(h)) or h <= 0.0 for h in (dz, dy, dx)):
        raise PhysicsContractError(f"spacing_zyx must contain 3 positive finite floats (dz, dy, dx), got {spacing_zyx}")
    return dz, dy, dx


def _fp32_context(tensor: "torch.Tensor"):
    dev_type = tensor.device.type
    if dev_type in ("cuda", "cpu"):
        return torch.amp.autocast(device_type=dev_type, enabled=False)
    return nullcontext()


def _work_dtype(tensor: "torch.Tensor") -> "torch.dtype":
    return torch.float64 if tensor.dtype == torch.float64 else torch.float32


def _broadcast_mask_to_spatial(mask: "torch.Tensor", target_shape: Sequence[int], target_device: "torch.device") -> "torch.Tensor":
    """Broadcast a fluid/stencil mask onto a scalar field of shape (*leading, Z, Y, X)."""
    m = mask.to(device=target_device)
    target_spatial = tuple(target_shape[-3:])
    if tuple(m.shape[-3:]) != target_spatial:
        raise PhysicsContractError(
            f"mask spatial shape {tuple(m.shape[-3:])} does not match field spatial shape {target_spatial}"
        )
    if m.ndim == len(target_shape):
        return m
    if m.ndim == 3:
        view_shape = (1,) * (len(target_shape) - 3) + target_spatial
        return m.reshape(view_shape)
    if m.ndim < len(target_shape):
        if m.ndim == len(target_shape) - 1 and m.shape[0] == target_shape[0]:
            return m.unsqueeze(1)
        if m.ndim == 4 and len(target_shape) == 5 and m.shape[0] == 1:
            return m.unsqueeze(1)
        raise PhysicsContractError(f"cannot broadcast mask shape {tuple(m.shape)} to scalar field shape {tuple(target_shape)}")
    if m.ndim == len(target_shape) + 1 and m.shape[-4] == 1:
        return m.squeeze(-4)
    raise PhysicsContractError(f"cannot broadcast mask shape {tuple(m.shape)} to scalar field shape {tuple(target_shape)}")


def torch_build_3d_derivative_mask(fluid_mask: "torch.Tensor", axis: int) -> "torch.Tensor":
    """Return mask of cells where central difference along `axis` uses only valid fluid neighbors (no wrap)."""
    _require_torch()
    if fluid_mask.ndim < 3:
        raise PhysicsContractError(f"fluid_mask must have at least 3 dimensions (..., Z, Y, X), got {tuple(fluid_mask.shape)}")
    norm_axis = axis if axis >= 0 else fluid_mask.ndim + axis
    if norm_axis < fluid_mask.ndim - 3 or norm_axis >= fluid_mask.ndim:
        raise PhysicsContractError(f"axis must be one of the last 3 spatial dimensions (-3, -2, -1), got {axis}")

    m = (fluid_mask > 0.5).to(dtype=torch.float32)
    size = m.shape[norm_axis]
    if size < 3:
        return torch.zeros_like(m, dtype=torch.float32)

    mid = m.narrow(norm_axis, 1, size - 2)
    left = m.narrow(norm_axis, 0, size - 2)
    right = m.narrow(norm_axis, 2, size - 2)
    inner = mid * left * right
    edge_shape = list(m.shape)
    edge_shape[norm_axis] = 1
    zeros = torch.zeros(edge_shape, dtype=torch.float32, device=m.device)
    return torch.cat([zeros, inner, zeros], dim=norm_axis)


def torch_build_full_3d_stencil_mask(fluid_mask: "torch.Tensor") -> "torch.Tensor":
    """Cells where 1st-order central differences are simultaneously valid along Z (-3), Y (-2), and X (-1)."""
    _require_torch()
    mz = torch_build_3d_derivative_mask(fluid_mask, -3)
    my = torch_build_3d_derivative_mask(fluid_mask, -2)
    mx = torch_build_3d_derivative_mask(fluid_mask, -1)
    return mz * my * mx


def torch_valid_div_of_curl_mask(fluid_mask: "torch.Tensor") -> "torch.Tensor":
    """Double-eroded mask where `div_h(curl_h(A))` has a complete un-truncated fluid stencil."""
    _require_torch()
    first_valid = torch_build_full_3d_stencil_mask(fluid_mask)
    return torch_build_full_3d_stencil_mask(first_valid)


def torch_centered_difference_3d(
    field: "torch.Tensor",
    spacing: float,
    axis: int,
    valid_mask: "torch.Tensor | None" = None,
    edge_order1: bool = False,
) -> "torch.Tensor":
    """Differentiable non-wrapping 2nd-order centered finite difference along `axis` in FP32+.

    When `edge_order1=True` and `valid_mask is None`, uses 1st-order one-sided differences
    at the outer boundary planes (index 0 and size-1) so curl/grad heads remain expressive on
    all 6 domain boundary faces while preserving exact interior `div_h(curl_h(A)) == 0`.
    """
    _require_torch()
    sp = float(spacing)
    if (not math.isfinite(sp)) or sp <= 0.0:
        raise PhysicsContractError(f"spacing must be a positive finite float, got {spacing}")
    if field.ndim < 3:
        raise PhysicsContractError(f"field must have at least 3 dimensions (..., Z, Y, X), got {tuple(field.shape)}")
    norm_axis = axis if axis >= 0 else field.ndim + axis
    if norm_axis < 0 or norm_axis >= field.ndim:
        raise PhysicsContractError(f"invalid axis {axis} for field with shape {tuple(field.shape)}")

    with _fp32_context(field):
        wdtype = _work_dtype(field)
        x = field.to(dtype=wdtype)
        size = x.shape[norm_axis]
        if size < 2 or (size < 3 and not (edge_order1 and valid_mask is None)):
            out = torch.zeros_like(x)
        else:
            if edge_order1 and valid_mask is None:
                left_edge = (x.narrow(norm_axis, 1, 1) - x.narrow(norm_axis, 0, 1)) / sp
                right_edge = (x.narrow(norm_axis, size - 1, 1) - x.narrow(norm_axis, size - 2, 1)) / sp
                if size == 2:
                    out = torch.cat([left_edge, right_edge], dim=norm_axis)
                else:
                    left = x.narrow(norm_axis, 0, size - 2)
                    right = x.narrow(norm_axis, 2, size - 2)
                    diff = (right - left) / (2.0 * sp)
                    out = torch.cat([left_edge, diff, right_edge], dim=norm_axis)
            else:
                left = x.narrow(norm_axis, 0, size - 2)
                right = x.narrow(norm_axis, 2, size - 2)
                diff = (right - left) / (2.0 * sp)
                edge_shape = list(x.shape)
                edge_shape[norm_axis] = 1
                zeros = torch.zeros(edge_shape, dtype=wdtype, device=x.device)
                out = torch.cat([zeros, diff, zeros], dim=norm_axis)
        if valid_mask is not None:
            vm = _broadcast_mask_to_spatial(valid_mask, out.shape, out.device)
            out = out.masked_fill(vm <= 0.5, 0)
        return out


def torch_div_h(
    u: "torch.Tensor",
    v: "torch.Tensor",
    w: "torch.Tensor",
    spacing_zyx: Sequence[float],
    mask: "torch.Tensor | None" = None,
) -> "torch.Tensor":
    """Compatible discrete divergence du/dX + dv/dY + dw/dZ on a (..., Z, Y, X) grid."""
    _require_torch()
    if u.shape != v.shape or u.shape != w.shape:
        raise PhysicsContractError(f"u, v, and w must have identical shapes, got {u.shape}, {v.shape}, {w.shape}")
    if u.ndim < 3:
        raise PhysicsContractError(f"u, v, w must have at least 3 spatial dimensions (..., Z, Y, X), got {u.shape}")
    dz, dy, dx = _validate_spacing_zyx(spacing_zyx)

    with _fp32_context(u):
        valid = torch_build_full_3d_stencil_mask(mask) if mask is not None else None
        du_dx = torch_centered_difference_3d(u, dx, -1, valid)
        dv_dy = torch_centered_difference_3d(v, dy, -2, valid)
        dw_dz = torch_centered_difference_3d(w, dz, -3, valid)
        return du_dx + dv_dy + dw_dz


def torch_curl_h(
    a_vec: "torch.Tensor",
    spacing_zyx: Sequence[float],
    mask: "torch.Tensor | None" = None,
    edge_order1: bool = False,
) -> "torch.Tensor":
    """Compatible discrete curl `u = curl_h(A)` for `A = [Ax, Ay, Az]` on `(..., 3, Z, Y, X)`."""
    _require_torch()
    if a_vec.ndim < 4 or a_vec.shape[-4] != 3:
        raise PhysicsContractError(f"a_vec must have shape (..., 3, Z, Y, X), got {tuple(a_vec.shape)}")
    dz, dy, dx = _validate_spacing_zyx(spacing_zyx)

    with _fp32_context(a_vec):
        ax = a_vec[..., 0, :, :, :]
        ay = a_vec[..., 1, :, :, :]
        az = a_vec[..., 2, :, :, :]
        valid = torch_build_full_3d_stencil_mask(mask) if mask is not None else None

        daz_dy = torch_centered_difference_3d(az, dy, -2, valid, edge_order1=edge_order1)
        day_dz = torch_centered_difference_3d(ay, dz, -3, valid, edge_order1=edge_order1)
        dax_dz = torch_centered_difference_3d(ax, dz, -3, valid, edge_order1=edge_order1)
        daz_dx = torch_centered_difference_3d(az, dx, -1, valid, edge_order1=edge_order1)
        day_dx = torch_centered_difference_3d(ay, dx, -1, valid, edge_order1=edge_order1)
        dax_dy = torch_centered_difference_3d(ax, dy, -2, valid, edge_order1=edge_order1)

        u = daz_dy - day_dz
        v = dax_dz - daz_dx
        w = day_dx - dax_dy
        return torch.stack([u, v, w], dim=-4)


def torch_grad_h(
    scalar: "torch.Tensor",
    spacing_zyx: Sequence[float],
    mask: "torch.Tensor | None" = None,
    edge_order1: bool = False,
) -> "torch.Tensor":
    """Discrete central gradient `[ds/dX, ds/dY, ds/dZ]` of shape `(..., 3, Z, Y, X)`."""
    _require_torch()
    if scalar.ndim < 3:
        raise PhysicsContractError(f"scalar must have at least 3 dimensions (..., Z, Y, X), got {tuple(scalar.shape)}")
    dz, dy, dx = _validate_spacing_zyx(spacing_zyx)

    with _fp32_context(scalar):
        valid = torch_build_full_3d_stencil_mask(mask) if mask is not None else None
        ds_dx = torch_centered_difference_3d(scalar, dx, -1, valid, edge_order1=edge_order1)
        ds_dy = torch_centered_difference_3d(scalar, dy, -2, valid, edge_order1=edge_order1)
        ds_dz = torch_centered_difference_3d(scalar, dz, -3, valid, edge_order1=edge_order1)
        return torch.stack([ds_dx, ds_dy, ds_dz], dim=-4)


def torch_laplacian_h(
    scalar: "torch.Tensor",
    spacing_zyx: Sequence[float],
    mask: "torch.Tensor | None" = None,
) -> "torch.Tensor":
    """Compact 7-point 2nd-order discrete Laplacian in FP32+ with non-wrapping slicing."""
    _require_torch()
    if scalar.ndim < 3:
        raise PhysicsContractError(f"scalar must have at least 3 dimensions (..., Z, Y, X), got {tuple(scalar.shape)}")
    dz, dy, dx = _validate_spacing_zyx(spacing_zyx)

    with _fp32_context(scalar):
        wdtype = _work_dtype(scalar)
        f = scalar.to(dtype=wdtype)

        def _d2(arr: "torch.Tensor", sp: float, axis: int) -> "torch.Tensor":
            norm_axis = axis if axis >= 0 else arr.ndim + axis
            size = arr.shape[norm_axis]
            if size < 3:
                return torch.zeros_like(arr)
            mid = arr.narrow(norm_axis, 1, size - 2)
            left = arr.narrow(norm_axis, 0, size - 2)
            right = arr.narrow(norm_axis, 2, size - 2)
            d2 = (right - 2.0 * mid + left) / (sp * sp)
            edge_shape = list(arr.shape)
            edge_shape[norm_axis] = 1
            zeros = torch.zeros(edge_shape, dtype=wdtype, device=arr.device)
            return torch.cat([zeros, d2, zeros], dim=norm_axis)

        lap = _d2(f, dz, -3) + _d2(f, dy, -2) + _d2(f, dx, -1)
        if mask is not None:
            valid = torch_build_full_3d_stencil_mask(mask)
            vm = _broadcast_mask_to_spatial(valid, lap.shape, lap.device)
            lap = lap.masked_fill(vm <= 0.5, 0)
        return lap


def _centered_difference_3d_adjoint(
    residual: "torch.Tensor",
    spacing: float,
    axis: int,
) -> "torch.Tensor":
    """Exact discrete transpose of `torch_centered_difference_3d` along `axis`."""
    norm_axis = axis if axis >= 0 else residual.ndim + axis
    size = residual.shape[norm_axis]
    if size < 3:
        return torch.zeros_like(residual)
    r_mid = residual.narrow(norm_axis, 1, size - 2) / (2.0 * float(spacing))
    pad2_shape = list(residual.shape)
    pad2_shape[norm_axis] = 2
    zeros2 = torch.zeros(pad2_shape, dtype=residual.dtype, device=residual.device)
    pos = torch.cat([zeros2, r_mid], dim=norm_axis)
    neg = torch.cat([r_mid, zeros2], dim=norm_axis)
    return pos - neg


def _div_h_adjoint(
    lam: "torch.Tensor",
    spacing_zyx: tuple[float, float, float],
    valid_stencil: "torch.Tensor | None",
    active_fluid: "torch.Tensor",
) -> "torch.Tensor":
    """Exact discrete adjoint `D_h^T(lam)` restricted to active fluid cells."""
    dz, dy, dx = spacing_zyx
    lam_v = lam if valid_stencil is None else lam * valid_stencil
    gx = _centered_difference_3d_adjoint(lam_v, dx, -1)
    gy = _centered_difference_3d_adjoint(lam_v, dy, -2)
    gz = _centered_difference_3d_adjoint(lam_v, dz, -3)
    return torch.stack([gx, gy, gz], dim=-4) * active_fluid.unsqueeze(-4)


def torch_helmholtz_project(
    vel: "torch.Tensor",
    spacing_zyx: Sequence[float],
    fluid_mask: "torch.Tensor | None" = None,
    num_iters: int = 20,
    omega: float = 0.8,
    method: str = "cg",
) -> tuple["torch.Tensor", "torch.Tensor"]:
    """Differentiable boundary-aware Helmholtz/Leray projection (NO periodic 3D FFT).

    Solves the compatible orthogonal projection `(D_h D_h^T) phi = D_h(vel)` on active
    fluid cells (`fluid_mask > 0.5`) using non-wrapping stencils, returning
    `(vel - D_h^T(phi), phi)` in FP32+.
    """
    _require_torch()
    if vel.ndim < 4 or vel.shape[-4] != 3:
        raise PhysicsContractError(f"vel must have shape (..., 3, Z, Y, X), got {tuple(vel.shape)}")
    if num_iters < 0:
        raise PhysicsContractError(f"num_iters must be >= 0, got {num_iters}")
    if method not in ("cg", "jacobi"):
        raise PhysicsContractError(f"method must be 'cg' or 'jacobi', got {method!r}")
    dz, dy, dx = _validate_spacing_zyx(spacing_zyx)

    with _fp32_context(vel):
        wdtype = _work_dtype(vel)
        v_work = vel.to(dtype=wdtype)
        scalar_shape = v_work.shape[:-4] + v_work.shape[-3:]

        if fluid_mask is not None:
            m_broad = _broadcast_mask_to_spatial(fluid_mask, scalar_shape, v_work.device)
            active = (m_broad > 0.5).to(dtype=wdtype).expand(scalar_shape)
            v_work = v_work.masked_fill(active.unsqueeze(-4) == 0, 0)
            valid_stencil = _broadcast_mask_to_spatial(
                torch_build_full_3d_stencil_mask(fluid_mask), scalar_shape, v_work.device
            ).to(dtype=wdtype)
        else:
            active = torch.ones(scalar_shape, dtype=wdtype, device=v_work.device)
            valid_stencil = None

        if not bool(torch.isfinite(v_work).all()):
            raise PhysicsContractError("non-finite projection velocity in the fluid domain")
        if num_iters == 0:
            phi = torch.zeros(scalar_shape, dtype=wdtype, device=v_work.device)
            return v_work, phi

        rhs = torch_div_h(
            v_work[..., 0, :, :, :], v_work[..., 1, :, :, :], v_work[..., 2, :, :, :], (dz, dy, dx), mask=fluid_mask
        )
        phi = torch.zeros_like(rhs)

        if method == "jacobi":
            diag_bound = 1.0 / (dz * dz) + 1.0 / (dy * dy) + 1.0 / (dx * dx)
            step = float(omega) / diag_bound
            for _ in range(num_iters):
                g_phi = _div_h_adjoint(phi, (dz, dy, dx), valid_stencil, active)
                ap = torch_div_h(
                    g_phi[..., 0, :, :, :], g_phi[..., 1, :, :, :], g_phi[..., 2, :, :, :], (dz, dy, dx), mask=fluid_mask
                )
                phi = phi + step * (rhs - ap)
        else:
            r = rhs
            p = r
            rz = (r * r).sum(dim=(-3, -2, -1), keepdim=True)
            for _ in range(num_iters):
                g_p = _div_h_adjoint(p, (dz, dy, dx), valid_stencil, active)
                ap = torch_div_h(
                    g_p[..., 0, :, :, :], g_p[..., 1, :, :, :], g_p[..., 2, :, :, :], (dz, dy, dx), mask=fluid_mask
                )
                p_ap = (p * ap).sum(dim=(-3, -2, -1), keepdim=True)
                safe_p_ap = torch.where(p_ap > 1e-30, p_ap, torch.ones_like(p_ap))
                alpha = torch.where(p_ap > 1e-30, rz / safe_p_ap, torch.zeros_like(rz))
                phi = phi + alpha * p
                r = r - alpha * ap
                rz_next = (r * r).sum(dim=(-3, -2, -1), keepdim=True)
                safe_rz = torch.where(rz > 1e-30, rz, torch.ones_like(rz))
                beta = torch.where(rz > 1e-30, rz_next / safe_rz, torch.zeros_like(rz_next))
                p = r + beta * p
                rz = rz_next

        corr = _div_h_adjoint(phi, (dz, dy, dx), valid_stencil, active)
        v_proj = v_work - corr
        if fluid_mask is not None:
            v_proj = torch.where(active.unsqueeze(-4) > 0.5, v_proj, torch.zeros_like(v_work))
        return v_proj, phi


__all__ = [
    "torch_available",
    "torch_build_3d_derivative_mask",
    "torch_build_full_3d_stencil_mask",
    "torch_centered_difference_3d",
    "torch_curl_h",
    "torch_div_h",
    "torch_grad_h",
    "torch_helmholtz_project",
    "torch_laplacian_h",
    "torch_valid_div_of_curl_mask",
]
