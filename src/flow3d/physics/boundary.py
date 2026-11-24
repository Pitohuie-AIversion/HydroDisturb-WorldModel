from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from flow3d.contracts import PhysicsContractError
from flow3d.physics.operators import centered_difference_3d


def first_fluid_shell_mask(fluid_mask: np.ndarray) -> np.ndarray:
    """Return boolean mask of valid fluid cells (fluid_mask > 0.5) adjacent to at least one solid cell.

    Zeroing out solid interior cells (fluid_mask == 0) is trivial masking and does NOT measure
    physical no-slip or no-penetration wall compliance. Physical wall diagnostics must inspect
    this first fluid shell adjacent to the cylinder obstacle.
    """
    m = np.asarray(fluid_mask)
    if m.ndim != 3:
        raise PhysicsContractError(f"fluid_mask must be 3D (Z,Y,X), got {m.shape}")
    fluid = m > 0.5
    solid = ~fluid
    if not np.any(solid) or not np.any(fluid):
        return np.zeros_like(fluid, dtype=bool)

    adj_solid = np.zeros_like(fluid, dtype=bool)
    # 6-connected neighborhood in (Z, Y, X)
    adj_solid[1:, :, :] |= solid[:-1, :, :]
    adj_solid[:-1, :, :] |= solid[1:, :, :]
    adj_solid[:, 1:, :] |= solid[:, :-1, :]
    adj_solid[:, :-1, :] |= solid[:, 1:, :]
    adj_solid[:, :, 1:] |= solid[:, :, :-1]
    adj_solid[:, :, :-1] |= solid[:, :, 1:]
    return fluid & adj_solid


@dataclass(frozen=True)
class BoundaryDiagnostics:
    """Near-cylinder velocity and signed transport across the sampled interior ROI."""

    first_shell_cell_count: int
    first_shell_normal_speed_rms: float
    first_shell_tangential_speed_rms: float
    first_shell_total_speed_rms: float
    solid_interior_max_abs: float
    roi_face_flux_m3_s: dict[str, float]
    roi_x_min_mean_u_m_s: float | None
    roi_x_max_mean_u_m_s: float | None
    roi_face_reference_errors: dict[str, dict] | None

    @property
    def roi_net_outward_flux_m3_s(self) -> float:
        return sum(self.roi_face_flux_m3_s.values())

    def to_dict(self) -> dict[str, Any]:
        return {
            "first_shell_cell_count": self.first_shell_cell_count,
            "first_shell_normal_speed_rms": self.first_shell_normal_speed_rms,
            "first_shell_tangential_speed_rms": self.first_shell_tangential_speed_rms,
            "first_shell_total_speed_rms": self.first_shell_total_speed_rms,
            "solid_interior_max_abs": self.solid_interior_max_abs,
            "roi_flux_method": "nearest_cell_center_velocity_times_face_area; outward_positive",
            "roi_x_min_mean_u_m_s": self.roi_x_min_mean_u_m_s,
            "roi_x_max_mean_u_m_s": self.roi_x_max_mean_u_m_s,
            "roi_face_reference_errors": self.roi_face_reference_errors,
            "roi_face_flux_m3_s": self.roi_face_flux_m3_s.copy(),
            "roi_net_outward_flux_m3_s": self.roi_net_outward_flux_m3_s,
        }


def evaluate_cylinder_boundary_conditions(
    field_4ch: np.ndarray,
    fluid_mask: np.ndarray,
    sdf_over_d: np.ndarray,
    spacing_zyx: tuple[float, float, float] = (0.01875, 0.01875, 0.01875),
    *,
    reference_4ch: np.ndarray | None = None,
) -> BoundaryDiagnostics:
    """Report six sampled ROI fluxes; optional SI CFD reference must use the identical grid and mask.

    Velocity-vector RMSE is sqrt(mean(du**2 + dv**2 + dw**2)) over the fluid cells on each face.
    A face fully occupied by solid has zero flux and undefined reference errors (None).
    """
    f = np.asarray(field_4ch, dtype=np.float64)
    if f.ndim != 4 or f.shape[0] != 4:
        raise PhysicsContractError(f"expected field_4ch shape (4,Z,Y,X), got {f.shape}")
    raw_mask = np.asarray(fluid_mask)
    if (
        not np.isfinite(raw_mask).all()
        or not np.isfinite(spacing_zyx).all()
        or len(spacing_zyx) != 3
        or min(spacing_zyx) <= 0
    ):
        raise PhysicsContractError("finite fluid mask and three positive SI spacings required")
    m = raw_mask > 0.5
    sdf = np.asarray(sdf_over_d, dtype=np.float64)
    if m.shape != f.shape[1:] or sdf.shape != f.shape[1:]:
        raise PhysicsContractError("fluid_mask and sdf_over_d must match field spatial shape (Z,Y,X)")

    if not np.isfinite(f[:3, m]).all() or not np.isfinite(sdf).all():
        raise PhysicsContractError("fluid velocity and geometry must be finite for boundary diagnostics")
    shell = first_fluid_shell_mask(m)
    dz, dy, dx = spacing_zyx
    # Wall outward normal n = grad(SDF) / |grad(SDF)| in (X, Y, Z) components
    dsdf_dx = centered_difference_3d(sdf, dx, -1)
    dsdf_dy = centered_difference_3d(sdf, dy, -2)
    dsdf_dz = centered_difference_3d(sdf, dz, -3)
    norm_g = np.sqrt(dsdf_dx**2 + dsdf_dy**2 + dsdf_dz**2) + 1e-12
    nx, ny, nz = dsdf_dx / norm_g, dsdf_dy / norm_g, dsdf_dz / norm_g

    u = np.where(np.isfinite(f[0]), f[0], 0.0)
    v = np.where(np.isfinite(f[1]), f[1], 0.0)
    w = np.where(np.isfinite(f[2]), f[2], 0.0)

    if np.any(shell):
        u_s, v_s, w_s = u[shell], v[shell], w[shell]
        un_s = u_s * nx[shell] + v_s * ny[shell] + w_s * nz[shell]
        u_tot_sq = u_s**2 + v_s**2 + w_s**2
        ut_sq = np.maximum(u_tot_sq - un_s**2, 0.0)
        un_rms = float(np.sqrt(np.mean(un_s**2)))
        ut_rms = float(np.sqrt(np.mean(ut_sq)))
        utot_rms = float(np.sqrt(np.mean(u_tot_sq)))
    else:
        un_rms = ut_rms = utot_rms = 0.0

    solid = ~m
    if np.any(solid):
        solid_vals = f[:3, solid]
        finite_solid = solid_vals[np.isfinite(solid_vals)]
        solid_max = float(np.max(np.abs(finite_solid))) if finite_solid.size > 0 else float("nan")
    else:
        solid_max = 0.0

    # Estimate normal velocities at the six cropped ROI faces from adjacent cell centres.
    # Solid portions of a face do not transport fluid. Opposite normals have opposite signs.
    fluxes = {}
    x_means = {}
    reference = None if reference_4ch is None else np.asarray(reference_4ch, dtype=np.float64)
    errors = None if reference is None else {}
    if reference is not None and (
        reference.shape != f.shape or not np.isfinite(reference[:, m]).all() or not np.isfinite(f[3, m]).all()
    ):
        raise PhysicsContractError(
            "ROI reference comparison requires matching SI frames finite on fluid cells"
        )
    for name, component, axis, area in (("x", u, 2, dy * dz), ("y", v, 1, dz * dx), ("z", w, 0, dx * dy)):
        for side, idx, sign in (("min", 0, -1), ("max", -1, 1)):
            key = f"{name}_{side}"
            valid = m.take(idx, axis=axis)
            plane = component.take(idx, axis=axis)[valid]
            fluxes[key] = float(sign * area * plane.sum())
            if name == "x":
                x_means[side] = float(plane.mean()) if plane.size else None
            if reference is not None:
                delta = f.take(idx, axis=axis + 1)[:, valid] - reference.take(idx, axis=axis + 1)[:, valid]
                errors[key] = {
                    "fluid_cells": plane.size,
                    "velocity_vector_rmse_m_s": float(np.sqrt(np.mean(np.sum(delta[:3] ** 2, axis=0))))
                    if plane.size
                    else None,
                    "pressure_rmse_pa": float(np.sqrt(np.mean(delta[3] ** 2))) if plane.size else None,
                }
    return BoundaryDiagnostics(
        first_shell_cell_count=int(shell.sum()),
        first_shell_normal_speed_rms=un_rms,
        first_shell_tangential_speed_rms=ut_rms,
        first_shell_total_speed_rms=utot_rms,
        solid_interior_max_abs=solid_max,
        roi_face_flux_m3_s=fluxes,
        roi_x_min_mean_u_m_s=x_means["min"],
        roi_x_max_mean_u_m_s=x_means["max"],
        roi_face_reference_errors=errors,
    )
