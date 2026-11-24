from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np


class PhysicsContractError(ValueError):
    """Raised when physical parameters, tensor layouts, or provenance violate G0 contracts."""


def condition_cadence(row: Mapping[str, Any], key: str, *, required: bool = False) -> float | None:
    """Resolve one forecast timestep from inference-known quantities in its own units."""
    if key not in ("dt_phys_s", "dt_ctu"):
        raise ValueError(f"unknown cadence field: {key}")
    known, valid = row.get("known_mask", {}), row.get("valid_mask", {})

    def available(name):
        return known.get(name, True) is True and valid.get(name, True) is True

    value = row.get(key) if available(key) else None
    if value is None and key == "dt_ctu" and available(key):
        factors = ("dt_phys_s", "u0_m_s", "diameter_m")
        if all(available(name) and row.get(name) is not None for name in factors):
            seconds, speed, diameter = (float(row[name]) for name in factors)
            if diameter <= 0 or not np.isfinite(diameter) or seconds <= 0:
                raise PhysicsContractError("invalid physical inputs for dt_ctu")
            value = seconds * abs(speed) / diameter
    if value is None:
        if required:
            raise PhysicsContractError(f"forecast requires a known, valid {key}")
        return None
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise PhysicsContractError(f"{key} must be finite and positive")
    return value


@dataclass(frozen=True)
class StateSpec3D:
    """State specification for 3D flow fields stored in (Z, Y, X) order."""

    name: str
    field_channels: tuple[str, ...] = ("u", "v", "w", "p")
    geometry_channels: tuple[str, ...] = ("fluid_mask", "sdf_over_d")
    spatial_dim: int = 3
    spatial_axes: tuple[str, str, str] = ("Z", "Y", "X")
    physical_axes: tuple[str, str, str] = ("z_span", "y_cross", "x_stream")
    default_grid_size: tuple[int, int, int] = (64, 96, 192)
    default_spacing_m: tuple[float, float, float] = (0.01875, 0.01875, 0.01875)
    cylinder_diameter_m: float = 0.3

    def __post_init__(self) -> None:
        if self.spatial_dim != 3:
            raise PhysicsContractError("StateSpec3D requires spatial_dim == 3")
        if len(self.field_channels) != 4:
            raise PhysicsContractError("StateSpec3D requires 4 field channels (u, v, w, p)")
        if len(self.geometry_channels) != 2:
            raise PhysicsContractError("StateSpec3D requires 2 geometry channels (fluid_mask, sdf_over_d)")
        if len(self.default_grid_size) != 3 or any(s <= 0 for s in self.default_grid_size):
            raise PhysicsContractError("default_grid_size must contain 3 positive integers (Z, Y, X)")
        if len(self.default_spacing_m) != 3 or any(h <= 0 for h in self.default_spacing_m):
            raise PhysicsContractError("default_spacing_m must contain 3 positive floats (dz, dy, dx)")
        if not np.isfinite(self.cylinder_diameter_m) or self.cylinder_diameter_m <= 0:
            raise PhysicsContractError("cylinder_diameter_m must be positive and finite")


CYLINDER3D_STATE_SPEC = StateSpec3D(name="cylinder_wake_3d_c04")


# Forbidden future-leakage keys that must never appear in inference conditions.
FORBIDDEN_FUTURE_CONDITION_KEYS = frozenset({
    "future_field",
    "future_z",
    "future_b",
    "future_stat",
    "future_regime_label",
    "future_vortex_phase",
    "unscheduled_future_disturbance",
    "target_field",
})


@dataclass(frozen=True)
class SampleProvenance:
    """Unified provenance record preventing train/val/test leakage and tracking CFD fidelity."""

    case_id: str
    geometry_id: str
    trajectory_id: str
    ancestor_id: str
    segment_id: str = "seg_0"
    time_idx: int = 0
    time_phys_s: float = 0.0
    time_ctu: float = 0.0
    ground_truth_verified: bool = False
    tier1_stability_status: str = "UNKNOWN"
    pressure_gauge: str = "unspecified_solver_pressure_reference"
    boundary_condition_id: str = "unspecified_solver_boundary_conditions"

    def __post_init__(self) -> None:
        for attr in ("case_id", "geometry_id", "trajectory_id", "ancestor_id", "segment_id"):
            val = getattr(self, attr)
            if not isinstance(val, str) or not val.strip():
                raise PhysicsContractError(f"provenance field '{attr}' must be a non-empty string")
        if not np.isfinite(self.time_phys_s) or not np.isfinite(self.time_ctu):
            raise PhysicsContractError("time_phys_s and time_ctu must be finite floats")
        if not isinstance(self.ground_truth_verified, (bool, np.bool_)):
            raise PhysicsContractError("ground_truth_verified must be a boolean")

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "geometry_id": self.geometry_id,
            "trajectory_id": self.trajectory_id,
            "ancestor_id": self.ancestor_id,
            "segment_id": self.segment_id,
            "time_idx": int(self.time_idx),
            "time_phys_s": float(self.time_phys_s),
            "time_ctu": float(self.time_ctu),
            "ground_truth_verified": bool(self.ground_truth_verified),
            "tier1_stability_status": self.tier1_stability_status,
            "pressure_gauge": self.pressure_gauge,
            "boundary_condition_id": self.boundary_condition_id,
        }

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "SampleProvenance":
        return cls(
            case_id=str(raw["case_id"]),
            geometry_id=str(raw["geometry_id"]),
            trajectory_id=str(raw["trajectory_id"]),
            ancestor_id=str(raw["ancestor_id"]),
            segment_id=str(raw.get("segment_id", "seg_0")),
            time_idx=int(raw.get("time_idx", 0)),
            time_phys_s=float(raw.get("time_phys_s", 0.0)),
            time_ctu=float(raw.get("time_ctu", 0.0)),
            ground_truth_verified=raw.get("ground_truth_verified", False),
            tier1_stability_status=str(raw.get("tier1_stability_status", "UNKNOWN")),
            pressure_gauge=str(raw.get("pressure_gauge", "unspecified_solver_pressure_reference")),
            boundary_condition_id=str(
                raw.get("boundary_condition_id", "unspecified_solver_boundary_conditions")
            ),
        )


@dataclass(frozen=True)
class Flow3DConditions:
    """Strictly typed inference-known physical and multimodal conditions."""

    reynolds_number: float | None = None
    u0_m_s: float | None = None
    diameter_m: float = 0.3
    viscosity_m2_s: float | None = 1.1889e-6
    dt_phys_s: float = 0.06
    dt_ctu: float | None = None
    action_schedule: tuple[float, ...] = ()
    text_prompt: str = ""
    history_regime_hint: str = "history_only"
    known_mask: Mapping[str, bool] = field(default_factory=dict)
    valid_mask: Mapping[str, bool] = field(default_factory=dict)
    re_consistency_rtol: float = 0.01
    dt_consistency_rtol: float = 0.05
    extra_metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # 1. Reject any forbidden future-leakage keys in extra_metadata or masks
        for container in (self.extra_metadata, self.known_mask, self.valid_mask):
            forbidden = FORBIDDEN_FUTURE_CONDITION_KEYS.intersection(container.keys())
            if forbidden:
                raise PhysicsContractError(f"future-leakage condition keys are forbidden: {sorted(forbidden)}")
        if "future" in self.history_regime_hint.lower():
            raise PhysicsContractError("e_regime / history_regime_hint must be inferred strictly from history")
        if self.diameter_m <= 0 or not np.isfinite(self.diameter_m):
            raise PhysicsContractError("diameter_m must be positive and finite")
        if self.dt_phys_s <= 0 or not np.isfinite(self.dt_phys_s):
            raise PhysicsContractError("dt_phys_s must be a positive finite float")

        # 2. Validate Re ≈ |U0| * D / nu when both are provided
        if (
            self.reynolds_number is not None
            and self.u0_m_s is not None
            and self.viscosity_m2_s is not None
        ):
            if self.reynolds_number <= 0 or self.viscosity_m2_s <= 0 or abs(self.u0_m_s) <= 0:
                raise PhysicsContractError("reynolds_number, |u0_m_s|, and viscosity_m2_s must be positive")
            re_computed = abs(self.u0_m_s) * self.diameter_m / self.viscosity_m2_s
            rel_err = abs(self.reynolds_number - re_computed) / self.reynolds_number
            if rel_err > self.re_consistency_rtol:
                raise PhysicsContractError(
                    f"Re consistency violation: declared Re={self.reynolds_number:.2f} vs "
                    f"|U0|*D/nu={re_computed:.2f} (rel_err={rel_err:.4f} > {self.re_consistency_rtol})"
                )

        # 3. Validate dt_ctu ≈ dt_phys_s * |U0| / D when both are provided
        if self.dt_ctu is not None:
            if self.dt_ctu <= 0 or not np.isfinite(self.dt_ctu):
                raise PhysicsContractError("dt_ctu must be a positive finite float")
            if self.u0_m_s is not None and abs(self.u0_m_s) > 0:
                dt_ctu_expected = self.dt_phys_s * abs(self.u0_m_s) / self.diameter_m
                dt_rel_err = abs(self.dt_ctu - dt_ctu_expected) / max(dt_ctu_expected, 1e-12)
                if dt_rel_err > self.dt_consistency_rtol:
                    raise PhysicsContractError(
                        f"dt_ctu consistency violation: declared dt_ctu={self.dt_ctu:.6f} vs "
                        f"dt_phys*|U0|/D={dt_ctu_expected:.6f} (rel_err={dt_rel_err:.4f} > {self.dt_consistency_rtol})"
                    )

    @property
    def resolved_dt_ctu(self) -> float | None:
        return condition_cadence(vars(self), "dt_ctu")

    def to_dict(self) -> dict[str, Any]:
        data = {
            "reynolds_number": self.reynolds_number,
            "u0_m_s": self.u0_m_s,
            "diameter_m": self.diameter_m,
            "viscosity_m2_s": self.viscosity_m2_s,
            "dt_phys_s": self.dt_phys_s,
            "dt_ctu": self.resolved_dt_ctu,
            "action_schedule": list(self.action_schedule),
            "text_prompt": self.text_prompt,
            "history_regime_hint": self.history_regime_hint,
            "known_mask": dict(self.known_mask),
            "valid_mask": dict(self.valid_mask),
        }
        if self.known_mask.get("action_schedule"):
            data["extra_metadata"] = {
                key: self.extra_metadata[key]
                for key in ("action_channel", "action_unit", "action_times_s", "action_known_at_time_s", "exogenous")
                if key in self.extra_metadata
            }
        return data


@dataclass(frozen=True)
class Flow3DBatch:
    """SI fields [B,T,4,Z,Y,X] in m/s and Pa, per-frame SI stats, and (dz,dy,dx) in metres.

    The geometry channels are dimensionless (fluid mask, SDF/D). Physical diagnostics
    use spacing directly; normalized-field physics heads use spacing_over_d().
    """

    field: np.ndarray
    geometry: np.ndarray
    b: np.ndarray | None = None
    spacing: tuple[float, float, float] = (0.01875, 0.01875, 0.01875)
    state_spec: StateSpec3D = CYLINDER3D_STATE_SPEC
    conditions: Sequence[Flow3DConditions] = ()
    provenance: Sequence[SampleProvenance] = ()

    def spacing_over_d(self, batch_index: int = 0) -> tuple[float, float, float]:
        """Return normalized-grid spacing for the example's cylinder diameter."""
        if not 0 <= batch_index < self.field.shape[0]:
            raise IndexError("batch_index out of range")
        diameter = self.conditions[batch_index].diameter_m if self.conditions else self.state_spec.cylinder_diameter_m
        dz, dy, dx = self.spacing
        return dz / diameter, dy / diameter, dx / diameter

    def __post_init__(self) -> None:
        f = np.asarray(self.field)
        g = np.asarray(self.geometry)
        if f.ndim != 6:
            raise PhysicsContractError(f"field must have 6 dimensions [B,T,4,Z,Y,X], got shape {f.shape}")
        bsz, t_len, c_f, nz, ny, nx = f.shape
        if c_f != len(self.state_spec.field_channels):
            raise PhysicsContractError(f"field channel axis (dim 2) must be 4, got {c_f}")
        if g.ndim != 5 or g.shape != (bsz, 2, nz, ny, nx):
            raise PhysicsContractError(
                f"geometry must have shape (B={bsz}, 2, Z={nz}, Y={ny}, X={nx}), got {g.shape}"
            )
        if len(self.spacing) != 3 or any(h <= 0 or not np.isfinite(h) for h in self.spacing):
            raise PhysicsContractError(f"spacing (dz, dy, dx) must be 3 positive finite floats, got {self.spacing}")
        if self.conditions and len(self.conditions) != bsz:
            raise PhysicsContractError("conditions must contain one entry per batch item")
        if not np.all(np.isfinite(g)):
            raise PhysicsContractError("geometry tensor [fluid_mask, sdf_over_d] contains non-finite values")
        fluid_mask = g[:, 0] > 0.5
        for b_idx in range(bsz):
            if int(fluid_mask[b_idx].sum()) == 0:
                raise PhysicsContractError(f"batch item {b_idx} has zero valid fluid cells")
            # Fluid cells must be strictly finite across all time steps and channels
            for frame in f[b_idx]:
                if not np.isfinite(frame[:, fluid_mask[b_idx]]).all():
                    raise PhysicsContractError(f"batch item {b_idx} contains NaN/Inf inside the valid fluid domain")
        if self.b is not None:
            b_arr = np.asarray(self.b)
            if b_arr.shape != (bsz, t_len, 6):
                raise PhysicsContractError(f"b must have shape (B={bsz}, T={t_len}, 6), got {b_arr.shape}")
            if b_arr.dtype != np.float32:
                raise PhysicsContractError(f"b must have dtype float32, got {b_arr.dtype}")
            if not np.all(np.isfinite(b_arr)):
                raise PhysicsContractError("b statistics contain non-finite values")
            if np.any(b_arr[..., 4] <= 0) or np.any(b_arr[..., 5] <= 0):
                raise PhysicsContractError("b scales (sigma_uvw, sigma_p) must be strictly positive")
