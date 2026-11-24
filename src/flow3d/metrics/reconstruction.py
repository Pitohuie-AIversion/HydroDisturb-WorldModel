"""G1 Stage 1 physical reconstruction metrics in SI units (m/s, Pa, s^-1).

Strictly separates velocity (m/s) and pressure (Pa), evaluates first-layer near-wall
error on `first_fluid_shell_mask`, computes interior discrete divergence on both
1-cell (`m1`) and 2-cell (`m2`) stencils, vorticity/enstrophy fidelity, and 3D
isotropic kinetic energy spectrum E(k) comparisons against ground-truth CFD.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from flow3d.contracts import PhysicsContractError
from flow3d.physics.boundary import evaluate_cylinder_boundary_conditions, first_fluid_shell_mask
from flow3d.physics.operators import (
    build_full_3d_stencil_mask,
    curl_h,
    div_h,
    valid_div_of_curl_mask,
)


def compute_3d_energy_spectrum(
    vel_3zyx: np.ndarray,
    fluid_mask: np.ndarray,
    spacing_zyx_m: Sequence[float],
    num_bins: int = 24,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute shell-integrated fluctuation kinetic energy from [3, Z, Y, X] velocities.

    Applies identical fluid-domain zeroing (fluctuation about fluid mean) and 3D Hann window
    to reduce spectral leakage. Returns bin centers in rad/m and energy per shell
    in m²/s². Divide shell energies by the wavenumber bin width for a density.
    """
    vel = np.asarray(vel_3zyx, dtype=np.float64)
    if vel.ndim != 4 or vel.shape[0] != 3:
        raise PhysicsContractError(f"vel_3zyx must have shape (3, Z, Y, X), got {vel.shape}")
    mask = np.asarray(fluid_mask) > 0.5
    nz, ny, nx = vel.shape[1:]
    dz, dy, dx = (float(spacing_zyx_m[0]), float(spacing_zyx_m[1]), float(spacing_zyx_m[2]))

    # Subtract fluid-domain mean velocity so spectrum reflects turbulent/wake fluctuations
    fluct = np.zeros_like(vel)
    for c in range(3):
        mean_c = float(vel[c][mask].mean()) if mask.any() else 0.0
        fluct[c] = np.where(mask, vel[c] - mean_c, 0.0)

    # 3D separable Hann window to suppress non-periodic boundary artifacts
    wz = np.hanning(nz)[:, None, None]
    wy = np.hanning(ny)[None, :, None]
    wx = np.hanning(nx)[None, None, :]
    win = wz * wy * wx
    win_norm = float(np.mean(win**2)) + 1e-12

    ek_3d = np.zeros((nz, ny, nx // 2 + 1), dtype=np.float64)
    n_tot = float(nz * ny * nx)
    for c in range(3):
        fk = np.fft.rfftn(fluct[c] * win)
        # Parseval-consistent energy density
        p_c = (np.abs(fk) ** 2) / (n_tot**2 * win_norm)
        # Account for Hermitian symmetry in rfftn along last axis
        weights = np.full(nx // 2 + 1, 2.0, dtype=np.float64)
        weights[0] = 1.0
        if nx % 2 == 0:
            weights[-1] = 1.0
        ek_3d += 0.5 * p_c * weights[None, None, :]

    kz = 2.0 * np.pi * np.fft.fftfreq(nz, d=dz)
    ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=dy)
    kx = 2.0 * np.pi * np.fft.rfftfreq(nx, d=dx)
    kmag = np.sqrt(kz[:, None, None] ** 2 + ky[None, :, None] ** 2 + kx[None, None, :] ** 2)

    k_max = float(min(np.pi / dz, np.pi / dy, np.pi / dx))
    k_min = float(min(2.0 * np.pi / (nz * dz), 2.0 * np.pi / (ny * dy), 2.0 * np.pi / (nx * dx)))
    edges = np.linspace(k_min * 0.5, k_max, int(num_bins) + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    e_bins = np.zeros(int(num_bins), dtype=np.float64)

    flat_k = kmag.ravel()
    flat_e = ek_3d.ravel()
    bin_idx = np.digitize(flat_k, edges) - 1
    for b in range(int(num_bins)):
        sel = bin_idx == b
        if np.any(sel):
            e_bins[b] = float(flat_e[sel].sum())

    return centers, e_bins


def evaluate_stage1_frame_si(
    pred_phys_4zyx: np.ndarray,
    gt_phys_4zyx: np.ndarray,
    geometry_2zyx: np.ndarray,
    spacing_zyx_m: Sequence[float],
    num_spectrum_bins: int = 24,
) -> dict[str, Any]:
    """Evaluate a single 3D reconstructed frame in SI units against ground truth."""
    pred = np.asarray(pred_phys_4zyx, dtype=np.float64)
    gt = np.asarray(gt_phys_4zyx, dtype=np.float64)
    geom = np.asarray(geometry_2zyx, dtype=np.float64)
    if pred.shape != gt.shape or pred.ndim != 4 or pred.shape[0] != 4:
        raise PhysicsContractError(f"pred and gt must match shape (4, Z, Y, X), got {pred.shape} vs {gt.shape}")
    if geom.ndim != 4 or geom.shape[0] != 2 or geom.shape[1:] != pred.shape[1:]:
        raise PhysicsContractError(f"geometry must have shape (2, Z, Y, X), got {geom.shape}")

    sp = (float(spacing_zyx_m[0]), float(spacing_zyx_m[1]), float(spacing_zyx_m[2]))
    fluid_mask = geom[0] > 0.5
    sdf_over_d = geom[1]
    solid_mask = ~fluid_mask
    if not np.any(fluid_mask):
        raise PhysicsContractError("fluid_mask has zero valid fluid cells")

    # 1. Valid-fluid Velocity (m/s) & Pressure (Pa) reconstruction metrics
    diff = pred - gt
    u_err = diff[0][fluid_mask]
    v_err = diff[1][fluid_mask]
    w_err = diff[2][fluid_mask]
    p_err = diff[3][fluid_mask]

    vel_err_sq = u_err**2 + v_err**2 + w_err**2
    vel_gt_sq = gt[0][fluid_mask] ** 2 + gt[1][fluid_mask] ** 2 + gt[2][fluid_mask] ** 2

    rmse_u = float(np.sqrt(np.mean(u_err**2)))
    rmse_v = float(np.sqrt(np.mean(v_err**2)))
    rmse_w = float(np.sqrt(np.mean(w_err**2)))
    rmse_vel = float(np.sqrt(np.mean(vel_err_sq)))
    mae_vel = float(np.mean(np.sqrt(vel_err_sq)))
    rel_l2_vel = float(np.sqrt(np.sum(vel_err_sq) / max(float(np.sum(vel_gt_sq)), 1e-12)))

    rmse_p = float(np.sqrt(np.mean(p_err**2)))
    mae_p = float(np.mean(np.abs(p_err)))
    rel_l2_p = float(np.sqrt(np.sum(p_err**2) / max(float(np.sum(gt[3][fluid_mask] ** 2)), 1e-12)))

    # Outer boundary faces (6 faces) velocity RMSE
    outer_face = np.zeros_like(fluid_mask, dtype=bool)
    outer_face[0, :, :] = True
    outer_face[-1, :, :] = True
    outer_face[:, 0, :] = True
    outer_face[:, -1, :] = True
    outer_face[:, :, 0] = True
    outer_face[:, :, -1] = True
    outer_fluid = outer_face & fluid_mask
    if np.any(outer_fluid):
        outer_vel_err_sq = (
            diff[0][outer_fluid] ** 2 + diff[1][outer_fluid] ** 2 + diff[2][outer_fluid] ** 2
        )
        outer_face_vel_rmse = float(np.sqrt(np.mean(outer_vel_err_sq)))
    else:
        outer_face_vel_rmse = 0.0

    # 2. First-layer near-wall error (first_fluid_shell_mask)
    shell = first_fluid_shell_mask(fluid_mask)
    if np.any(shell):
        sh_err_sq = diff[0][shell] ** 2 + diff[1][shell] ** 2 + diff[2][shell] ** 2
        sh_gt_sq = gt[0][shell] ** 2 + gt[1][shell] ** 2 + gt[2][shell] ** 2
        near_wall_vel_rmse = float(np.sqrt(np.mean(sh_err_sq)))
        near_wall_vel_rel_l2 = float(np.sqrt(np.sum(sh_err_sq) / max(float(np.sum(sh_gt_sq)), 1e-12)))
    else:
        near_wall_vel_rmse = 0.0
        near_wall_vel_rel_l2 = 0.0

    bc_pred = evaluate_cylinder_boundary_conditions(pred, geom[0], sdf_over_d, sp)
    bc_gt = evaluate_cylinder_boundary_conditions(gt, geom[0], sdf_over_d, sp)

    # Solid leakage check
    if np.any(solid_mask):
        solid_max = float(np.max(np.abs(pred[:3, solid_mask])))
    else:
        solid_max = 0.0

    # 3. Interior divergence on m1 (1-cell stencil) and m2 (2-cell div(curl) stencil)
    m1 = build_full_3d_stencil_mask(geom[0]) > 0.5
    m2 = valid_div_of_curl_mask(geom[0]) > 0.5
    div_pred = div_h(pred[0], pred[1], pred[2], sp, mask=geom[0]).astype(np.float64)
    div_gt = div_h(gt[0], gt[1], gt[2], sp, mask=geom[0]).astype(np.float64)

    div_rms_m1_pred = float(np.sqrt(np.mean(div_pred[m1] ** 2))) if np.any(m1) else 0.0
    div_rms_m1_gt = float(np.sqrt(np.mean(div_gt[m1] ** 2))) if np.any(m1) else 0.0
    div_rms_m2_pred = float(np.sqrt(np.mean(div_pred[m2] ** 2))) if np.any(m2) else 0.0
    div_rms_m2_gt = float(np.sqrt(np.mean(div_gt[m2] ** 2))) if np.any(m2) else 0.0

    # 4. Vorticity & Enstrophy on m1
    omega_pred = curl_h(pred[:3], sp, mask=geom[0]).astype(np.float64)
    omega_gt = curl_h(gt[:3], sp, mask=geom[0]).astype(np.float64)
    if np.any(m1):
        om_diff_sq = np.sum((omega_pred[:, m1] - omega_gt[:, m1]) ** 2, axis=0)
        om_gt_sq = np.sum(omega_gt[:, m1] ** 2, axis=0)
        om_pred_sq = np.sum(omega_pred[:, m1] ** 2, axis=0)
        vort_rmse = float(np.sqrt(np.mean(om_diff_sq)))
        vort_rel_l2 = float(np.sqrt(np.sum(om_diff_sq) / max(float(np.sum(om_gt_sq)), 1e-12)))
        enstrophy_pred = float(0.5 * np.mean(om_pred_sq))
        enstrophy_gt = float(0.5 * np.mean(om_gt_sq))
        enstrophy_rel_err = float(abs(enstrophy_pred - enstrophy_gt) / max(enstrophy_gt, 1e-12))
    else:
        vort_rmse = vort_rel_l2 = enstrophy_pred = enstrophy_gt = enstrophy_rel_err = 0.0

    # 5. 3D Kinetic Energy Spectrum E(k)
    k_bins, ek_pred = compute_3d_energy_spectrum(pred[:3], geom[0], sp, num_bins=num_spectrum_bins)
    _, ek_gt = compute_3d_energy_spectrum(gt[:3], geom[0], sp, num_bins=num_spectrum_bins)
    valid_k = ek_gt > 1e-14
    if np.any(valid_k):
        log_spec_rmse = float(
            np.sqrt(np.mean((np.log10(np.maximum(ek_pred[valid_k], 1e-16)) - np.log10(ek_gt[valid_k])) ** 2))
        )
    else:
        log_spec_rmse = 0.0

    return {
        "fluid_voxel_count": int(fluid_mask.sum()),
        "near_wall_voxel_count": int(shell.sum()),
        "div_rms_fluid_pred_s_inv": float(np.sqrt(np.mean(div_pred[fluid_mask] ** 2))),
        "rmse_u_ms": rmse_u,
        "rmse_v_ms": rmse_v,
        "rmse_w_ms": rmse_w,
        "rmse_vel_ms": rmse_vel,
        "mae_vel_ms": mae_vel,
        "rel_l2_vel": rel_l2_vel,
        "rmse_p_pa": rmse_p,
        "mae_p_pa": mae_p,
        "rel_l2_p": rel_l2_p,
        "outer_face_vel_rmse_ms": outer_face_vel_rmse,
        "near_wall_vel_rmse_ms": near_wall_vel_rmse,
        "near_wall_vel_rel_l2": near_wall_vel_rel_l2,
        "near_wall_un_rms_pred_ms": float(bc_pred.first_shell_normal_speed_rms),
        "near_wall_un_rms_gt_ms": float(bc_gt.first_shell_normal_speed_rms),
        "near_wall_ut_rms_pred_ms": float(bc_pred.first_shell_tangential_speed_rms),
        "near_wall_ut_rms_gt_ms": float(bc_gt.first_shell_tangential_speed_rms),
        "solid_interior_max_abs_ms": solid_max,
        "div_rms_m1_pred_s_inv": div_rms_m1_pred,
        "div_rms_m1_gt_s_inv": div_rms_m1_gt,
        "div_rms_m2_pred_s_inv": div_rms_m2_pred,
        "div_rms_m2_gt_s_inv": div_rms_m2_gt,
        "vorticity_rmse_s_inv": vort_rmse,
        "vorticity_rel_l2": vort_rel_l2,
        "enstrophy_pred_s_inv2": enstrophy_pred,
        "enstrophy_gt_s_inv2": enstrophy_gt,
        "enstrophy_rel_err": enstrophy_rel_err,
        "spectrum_log10_rmse": log_spec_rmse,
        "spectrum_k_rad_m": k_bins.tolist(),
        "spectrum_ek_pred": ek_pred.tolist(),
        "spectrum_ek_gt": ek_gt.tolist(),
    }
