from __future__ import annotations

import numpy as np

from flow3d.physics.operators import curl_h


def make_cylinder_geometry_3d(
    spatial_shape_zyx: tuple[int, int, int] = (16, 24, 32),
    spacing_zyx_m: tuple[float, float, float] = (0.01875, 0.01875, 0.01875),
    cylinder_diameter_m: float = 0.3,
    cylinder_center_yx_m: tuple[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a 3D circular cylinder `[fluid_mask, sdf_over_d]` on a `(Z, Y, X)` grid.

    Returns:
      - `geometry`: `(2, Z, Y, X)` float32 (`[fluid_mask, sdf_over_d]`)
      - `coords_zyx`: tuple of 1D coordinate arrays `(z, y, x)` in meters
    """
    nz, ny, nx = spatial_shape_zyx
    dz, dy, dx = spacing_zyx_m
    radius = 0.5 * cylinder_diameter_m
    z = (np.arange(nz, dtype=np.float32) + 0.5) * dz
    y = (np.arange(ny, dtype=np.float32) + 0.5) * dy
    x = (np.arange(nx, dtype=np.float32) + 0.5) * dx
    if cylinder_center_yx_m is None:
        cy = 0.5 * ny * dy
        cx = 0.25 * nx * dx
    else:
        cy, cx = cylinder_center_yx_m
    yy, xx = np.meshgrid(y, x, indexing="ij")
    dist_2d = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2) - radius
    sdf_3d = np.broadcast_to(dist_2d[None, :, :], (nz, ny, nx)).astype(np.float32)
    fluid_mask = (sdf_3d > 0.0).astype(np.float32)
    sdf_over_d = (sdf_3d / float(cylinder_diameter_m)).astype(np.float32)
    geometry = np.stack([fluid_mask, sdf_over_d], axis=0)
    return geometry, (z, y, x)  # type: ignore[return-value]


def make_analytical_divergence_free_field_3d(
    spatial_shape_zyx: tuple[int, int, int] = (16, 20, 24),
    spacing_zyx_m: tuple[float, float, float] = (0.01875, 0.01875, 0.01875),
    u0_m_s: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a smooth 3D vector potential A and velocity field u = curl_h(A) + (u0, 0, 0).

    Because `u - (u0,0,0) = curl_h(A)` is computed with the exact compatible discrete central-difference
    `curl_h`, its discrete divergence `div_h(u)` is identically zero (up to floating-point roundoff)
    on all interior valid stencil points!
    Returns `(field_4ch, vector_potential_3ch)` both of shape `(..., Z, Y, X)`.
    """
    nz, ny, nx = spatial_shape_zyx
    dz, dy, dx = spacing_zyx_m
    z = ((np.arange(nz, dtype=np.float64) + 0.5) * dz)[:, None, None]
    y = ((np.arange(ny, dtype=np.float64) + 0.5) * dy)[None, :, None]
    x = ((np.arange(nx, dtype=np.float64) + 0.5) * dx)[None, None, :]

    lz, ly, lx = nz * dz, ny * dy, nx * dx
    ax = np.broadcast_to(
        0.05 * np.sin(2.0 * np.pi * y / ly) * np.cos(2.0 * np.pi * z / lz) + 0.02 * np.cos(2.0 * np.pi * x / lx),
        spatial_shape_zyx,
    )
    ay = np.broadcast_to(0.04 * np.sin(2.0 * np.pi * z / lz) * np.cos(2.0 * np.pi * x / lx), spatial_shape_zyx)
    az = np.broadcast_to(0.06 * np.sin(2.0 * np.pi * x / lx) * np.cos(2.0 * np.pi * y / ly), spatial_shape_zyx)
    a_vec = np.stack([ax, ay, az], axis=0).astype(np.float32)

    vel = curl_h(a_vec, spacing_zyx_m, mask=None)
    vel[0] += float(u0_m_s)
    p = (-0.25 * (np.cos(4.0 * np.pi * x / lx) + np.cos(4.0 * np.pi * y / ly)) + 0.0 * z).astype(np.float32)
    field_4ch = np.concatenate([vel, p[None, ...]], axis=0).astype(np.float32)
    return field_4ch, a_vec
