from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from flow3d.contracts import PhysicsContractError
from stage1.config import _parse_simple_yaml


def load_yaml_or_json_config(path: str | Path) -> dict[str, Any]:
    """Load YAML or JSON config without requiring pyyaml when running in minimal environments."""
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if p.suffix.lower() == ".json":
        return dict(json.loads(text))
    return _parse_simple_yaml(text)


@dataclass(frozen=True)
class DatasetSchema3D:
    """Explicit coordinate, unit, and boundary schema for 3D cylinder wake datasets."""

    schema_version: str = "flow3d.schema.v1"
    tensor_layout: str = "B,T,C,Z,Y,X"
    spatial_shape_zyx: tuple[int, int, int] = (64, 96, 192)
    spacing_zyx_m: tuple[float, float, float] = (0.01875, 0.01875, 0.01875)
    cylinder_diameter_m: float = 0.3
    reference_density_kg_m3: float = 1025.8944
    reference_viscosity_m2_s: float = 1.1889e-6
    sdf_sign_convention: str = "positive_in_fluid_zero_on_wall_negative_in_solid"
    sdf_unit_convention: str = "dimensionless_SDF_over_D"
    mask_convention: str = "fluid_mask_1_fluid_0_solid"
    boundary_x_inlet: str = "dirichlet_inflow_U0"
    boundary_x_outlet: str = "pressure_outlet_zero_gauge"
    boundary_y_cross: str = "slip_farfield"
    boundary_z_span: str = "periodic"
    boundary_cylinder: str = "immersed_solid_noslip_impermeable"

    def validate_spatial_shape(self, shape_zyx: tuple[int, int, int]) -> None:
        if tuple(shape_zyx) != self.spatial_shape_zyx:
            raise PhysicsContractError(
                f"expected spatial shape (Z,Y,X)={self.spatial_shape_zyx}, got {tuple(shape_zyx)}"
            )

    def to_dict(self) -> dict[str, Any]:
        solver_boundaries = {
            "x_inlet": self.boundary_x_inlet,
            "x_outlet": self.boundary_x_outlet,
            "y_cross": self.boundary_y_cross,
            "z_span": self.boundary_z_span,
            "cylinder": self.boundary_cylinder,
        }
        return {
            "schema_version": self.schema_version,
            "tensor_layout": self.tensor_layout,
            "spatial_shape_zyx": list(self.spatial_shape_zyx),
            "spacing_zyx_m": list(self.spacing_zyx_m),
            "cylinder_diameter_m": self.cylinder_diameter_m,
            "reference_density_kg_m3": self.reference_density_kg_m3,
            "reference_viscosity_m2_s": self.reference_viscosity_m2_s,
            "sdf_sign_convention": self.sdf_sign_convention,
            "sdf_unit_convention": self.sdf_unit_convention,
            "mask_convention": self.mask_convention,
            "sampled_outer_faces": "interior_roi_cell_center_traces",
            "solver_boundaries": solver_boundaries,
            "boundaries": solver_boundaries.copy(),  # Legacy key for solver BCs.
        }

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "DatasetSchema3D":
        b = raw.get("solver_boundaries", raw.get("boundaries", {}))
        return cls(
            schema_version=str(raw.get("schema_version", "flow3d.schema.v1")),
            tensor_layout=str(raw.get("tensor_layout", "B,T,C,Z,Y,X")),
            spatial_shape_zyx=tuple(int(v) for v in raw.get("spatial_shape_zyx", (64, 96, 192))),
            spacing_zyx_m=tuple(float(v) for v in raw.get("spacing_zyx_m", (0.01875, 0.01875, 0.01875))),
            cylinder_diameter_m=float(raw.get("cylinder_diameter_m", 0.3)),
            reference_density_kg_m3=float(raw.get("reference_density_kg_m3", 1025.8944)),
            reference_viscosity_m2_s=float(raw.get("reference_viscosity_m2_s", 1.1889e-6)),
            sdf_sign_convention=str(raw.get("sdf_sign_convention", cls.sdf_sign_convention)),
            sdf_unit_convention=str(raw.get("sdf_unit_convention", cls.sdf_unit_convention)),
            mask_convention=str(raw.get("mask_convention", cls.mask_convention)),
            boundary_x_inlet=str(b.get("x_inlet", cls.boundary_x_inlet)),
            boundary_x_outlet=str(b.get("x_outlet", cls.boundary_x_outlet)),
            boundary_y_cross=str(b.get("y_cross", cls.boundary_y_cross)),
            boundary_z_span=str(b.get("z_span", cls.boundary_z_span)),
            boundary_cylinder=str(b.get("cylinder", cls.boundary_cylinder)),
        )


def transpose_yzx_to_zyx_fields(
    u_yzx: np.ndarray,
    w_cross_yzx: np.ndarray,
    v_span_yzx: np.ndarray,
    p_yzx: np.ndarray,
    expected_zyx: tuple[int, int, int] = (64, 96, 192),
) -> np.ndarray:
    """Convert raw COMSOL shedding arrays (Y_cross=96, Z_span=64, X_stream=192) to [4, Z=64, Y=96, X=192].

    In `video_c04_shedding`, raw arrays have shape (96, 64, 192) = (z_cross, y_span, x),
    where `u` is streamwise (X), `w_cross` is cross-flow (Y), and `v_span` is spanwise (Z).
    """
    expected_yzx = (expected_zyx[1], expected_zyx[0], expected_zyx[2])
    for name, arr in (("u", u_yzx), ("w_cross", w_cross_yzx), ("v_span", v_span_yzx), ("p", p_yzx)):
        if arr.shape != expected_yzx:
            raise PhysicsContractError(
                f"raw shedding array '{name}' expected YZX shape {expected_yzx}, got {arr.shape}"
            )
    u_zyx = np.transpose(u_yzx, (1, 0, 2))
    v_zyx = np.transpose(w_cross_yzx, (1, 0, 2))
    w_zyx = np.transpose(v_span_yzx, (1, 0, 2))
    p_zyx = np.transpose(p_yzx, (1, 0, 2))
    stacked = np.stack([u_zyx, v_zyx, w_zyx, p_zyx], axis=0).astype(np.float32)
    if stacked.shape[1:] != expected_zyx:
        raise PhysicsContractError(f"transposed field shape {stacked.shape[1:]} != {expected_zyx}")
    return stacked


def convert_h5_6channel_to_field_and_geometry(
    h5_frame_6ch: np.ndarray,
    sdf_clip: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert `03_world_model_dataset` 6-channel frame to (field[4,Z,Y,X], geometry[2,Z,Y,X]).

    HDF5 channel order in `03_world_model_dataset/manifest.json`:
      [0: u_star, 1: v_cross_star, 2: w_span_star, 3: p_star, 4: sdf_star, 5: solid_mask]
    Notice channel 4 is `sdf_star` and channel 5 is `solid_mask` (1=solid),
    which must be reordered and polarity-flipped to `geometry = [fluid_mask, sdf_over_d]`.
    """
    arr = np.asarray(h5_frame_6ch, dtype=np.float32)
    if arr.ndim not in (4, 5) or arr.shape[-4] != 6:
        raise PhysicsContractError(f"expected (..., 6, Z, Y, X) HDF5 tensor, got shape {arr.shape}")
    field = arr[..., :4, :, :, :].copy()
    sdf_star = arr[..., 4, :, :, :]
    solid_mask = arr[..., 5, :, :, :]
    if not np.isfinite(sdf_star).all() or not np.all((solid_mask == 0) | (solid_mask == 1)):
        raise PhysicsContractError("HDF5 geometry requires a finite SDF and binary solid_mask")
    if sdf_clip is not None:
        if not np.isfinite(sdf_clip) or sdf_clip <= 0:
            raise PhysicsContractError("sdf_clip must be positive")
        sdf_star = np.clip(sdf_star, -float(sdf_clip), float(sdf_clip))
    fluid_mask = (1.0 - (solid_mask > 0.5).astype(np.float32))
    geometry = np.stack([fluid_mask, sdf_star], axis=-4).astype(np.float32)
    return field, geometry


def denormalize_star_to_si(
    field_star_4ch: np.ndarray,
    u0_m_s: float,
    rho0_kg_m3: float = 1025.8944,
    p_inf_pa: float = 0.0,
) -> np.ndarray:
    """Convert dimensionless `[u*, v*, w*, p*]` from `03_world_model_dataset` to SI `[m/s, Pa]`.

    Matches upstream `HydroDisturb-Bench/scripts/tier1_tier2_acceptance_pipeline.py`:
      - `u, v, w = u* * U0, v* * U0, w* * U0` (m/s)
      - `p = p* * (rho0 * U0^2) + p_inf` (Pa; reference scale is `rho0 * U0^2`, NOT `0.5 * rho0 * U0^2`).
    """
    if u0_m_s <= 0 or rho0_kg_m3 <= 0:
        raise PhysicsContractError("u0_m_s and rho0_kg_m3 must be positive")
    f = np.asarray(field_star_4ch, dtype=np.float32)
    if f.ndim < 4 or f.shape[-4] != 4:
        raise PhysicsContractError(f"expected (..., 4, Z, Y, X) field_star_4ch, got {f.shape}")
    out = f.copy()
    out[..., :3, :, :, :] *= float(u0_m_s)
    p_ref = float(rho0_kg_m3) * (float(u0_m_s) ** 2)
    out[..., 3, :, :, :] = out[..., 3, :, :, :] * p_ref + float(p_inf_pa)
    return out


def normalize_si_to_star(
    field_si_4ch: np.ndarray,
    u0_m_s: float,
    rho0_kg_m3: float = 1025.8944,
    p_inf_pa: float = 0.0,
) -> np.ndarray:
    """Convert physical SI `[u, v, w, p]` (`m/s`, `Pa`) to dimensionless `[u*, v*, w*, p*]`."""
    if u0_m_s <= 0 or rho0_kg_m3 <= 0:
        raise PhysicsContractError("u0_m_s and rho0_kg_m3 must be positive")
    f = np.asarray(field_si_4ch, dtype=np.float32)
    if f.ndim < 4 or f.shape[-4] != 4:
        raise PhysicsContractError(f"expected (..., 4, Z, Y, X) field_si_4ch, got {f.shape}")
    out = f.copy()
    out[..., :3, :, :, :] /= float(u0_m_s)
    p_ref = float(rho0_kg_m3) * (float(u0_m_s) ** 2)
    out[..., 3, :, :, :] = (out[..., 3, :, :, :] - float(p_inf_pa)) / p_ref
    return out
