from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from flow3d.contracts import PhysicsContractError

SI_NORMALIZER_VERSION = "normalizer.flow3d.si.v2"


def si_roundtrip_diagnostics(
    reference_si: np.ndarray, restored_si: np.ndarray, fluid_mask: np.ndarray
) -> dict[str, Any]:
    """Check a single FP32 normalize/invert cycle in SI on valid fluid cells."""
    mask = np.asarray(fluid_mask)
    if mask.ndim != 3 or not np.isfinite(mask).all() or not (mask > 0.5).any():
        raise PhysicsContractError("SI roundtrip requires a nonempty finite fluid mask")
    valid = mask > 0.5
    reference = np.asarray(reference_si)
    restored = np.asarray(restored_si)
    if reference.shape != (4, *valid.shape) or restored.shape != reference.shape:
        raise PhysicsContractError("SI roundtrip expects matching (4,Z,Y,X) frames and spatial mask")

    rtol = 4 * float(np.finfo(np.float32).eps)
    channels = {}
    for index, name in enumerate(("u", "v", "w", "p")):
        target = reference[index, valid].astype(np.float64)
        actual = restored[index, valid].astype(np.float64)
        if not np.isfinite(target).all() or not np.isfinite(actual).all():
            raise PhysicsContractError(f"non-finite SI roundtrip value in fluid channel {name}")
        atol = 1e-5 if name == "p" else 1e-7
        # Centering and adding the frame mean back can dominate error at near-zero cells.
        tolerance = atol + rtol * np.maximum(np.abs(target), abs(target.mean()))
        error = np.abs(actual - target)
        channels[name] = {
            "unit": "Pa" if name == "p" else "m/s",
            "atol": atol,
            "max_abs_error": float(error.max()),
            "cells_outside_tolerance": int(np.count_nonzero(error > tolerance)),
        }
    return {
        "passed": all(c["cells_outside_tolerance"] == 0 for c in channels.values()),
        "rtol": rtol,
        "fluid_cell_count": int(valid.sum()),
        "channels": channels,
    }


@dataclass(frozen=True)
class Flow3DNormStats:
    """Per-frame 6D statistics b = [mu_u, mu_v, mu_w, mu_p, sigma_uvw, sigma_p] in FP32.

    Population variance (ddof=0) over valid fluid cells M = (fluid_mask > 0.5).
    `sigma_uvw = max(sqrt(Var_M(u) + Var_M(v) + Var_M(w)), scale_floor_uvw)` (NOT divided by 3).
    """

    mu_u: float
    mu_v: float
    mu_w: float
    mu_p: float
    sigma_uvw: float
    sigma_p: float
    scale_floor_uvw: float = 1e-6
    scale_floor_p: float = 1e-6
    version: str = SI_NORMALIZER_VERSION

    def as_array(self) -> np.ndarray:
        return np.asarray(
            [self.mu_u, self.mu_v, self.mu_w, self.mu_p, self.sigma_uvw, self.sigma_p],
            dtype=np.float32,
        )

    @classmethod
    def from_array(cls, b: np.ndarray, **kwargs: Any) -> Flow3DNormStats:
        arr = np.asarray(b, dtype=np.float32).reshape(-1)
        if arr.size != 6:
            raise PhysicsContractError(f"expected 6 normalization stats, got {arr.size}")
        if not np.all(np.isfinite(arr)):
            raise PhysicsContractError("normalization stats contain non-finite values")
        if arr[4] <= 0 or arr[5] <= 0:
            raise PhysicsContractError("sigma_uvw and sigma_p must be strictly positive")
        return cls(*(float(v) for v in arr), **kwargs)


class Flow3DNormalizer:
    """Per-frame valid-fluid normalizer for 3D fields [..., 4, Z, Y, X] with geometry [..., 2, Z, Y, X]."""

    def __init__(
        self,
        scale_floor_uvw: float = 1e-6,
        scale_floor_p: float = 1e-6,
        version: str = SI_NORMALIZER_VERSION,
    ) -> None:
        if scale_floor_uvw <= 0 or scale_floor_p <= 0:
            raise PhysicsContractError("scale_floor_uvw and scale_floor_p must be positive")
        self.scale_floor_uvw = float(scale_floor_uvw)
        self.scale_floor_p = float(scale_floor_p)
        self.version = version

    @staticmethod
    def _extract_valid_fluid_mask(fluid_mask: np.ndarray) -> np.ndarray:
        m = np.asarray(fluid_mask)
        if m.ndim != 3:
            raise PhysicsContractError(f"fluid_mask must have 3 spatial dimensions (Z,Y,X), got {m.shape}")
        if not np.all(np.isfinite(m)):
            raise PhysicsContractError("fluid_mask contains non-finite values")
        valid = m > 0.5
        if int(valid.sum()) == 0:
            raise PhysicsContractError("fluid_mask contains zero valid fluid cells (fail closed)")
        return valid

    def fit_frame(self, field_4ch: np.ndarray, fluid_mask: np.ndarray) -> Flow3DNormStats:
        f = np.asarray(field_4ch, dtype=np.float32)
        valid = self._extract_valid_fluid_mask(fluid_mask)
        if f.ndim != 4 or f.shape[0] != 4 or f.shape[1:] != valid.shape:
            raise PhysicsContractError(
                f"expected field shape (4, {valid.shape[0]}, {valid.shape[1]}, {valid.shape[2]}), got {f.shape}"
            )
        # Finite check inside valid fluid mask BEFORE any arithmetic; never use NaN * 0.
        fluid_channels: list[np.ndarray] = []
        for c_idx, c_name in enumerate(("u", "v", "w", "p")):
            fc = f[c_idx][valid].astype(np.float64)
            if not np.all(np.isfinite(fc)):
                raise PhysicsContractError(f"channel '{c_name}' contains NaN/Inf inside valid fluid domain")
            fluid_channels.append(fc)

        means = [float(fc.mean()) for fc in fluid_channels]
        # Population variance (ddof=0) summed across u, v, w (NOT divided by 3)
        var_uvw = sum(float(np.mean((fluid_channels[i] - means[i]) ** 2)) for i in range(3))
        var_p = float(np.mean((fluid_channels[3] - means[3]) ** 2))
        sigma_uvw = max(float(np.sqrt(var_uvw)), self.scale_floor_uvw)
        sigma_p = max(float(np.sqrt(var_p)), self.scale_floor_p)
        # Store as FP32 roundtrip-consistent stats
        b32 = np.asarray([*means, sigma_uvw, sigma_p], dtype=np.float32)
        b32[4] = max(float(b32[4]), self.scale_floor_uvw)
        b32[5] = max(float(b32[5]), self.scale_floor_p)
        return Flow3DNormStats.from_array(
            b32,
            scale_floor_uvw=self.scale_floor_uvw,
            scale_floor_p=self.scale_floor_p,
            version=self.version,
        )

    def normalize_frame(
        self,
        field_4ch: np.ndarray,
        fluid_mask: np.ndarray,
        stats: Flow3DNormStats | None = None,
        zero_solid: bool = True,
    ) -> tuple[np.ndarray, Flow3DNormStats]:
        f = np.asarray(field_4ch, dtype=np.float32)
        valid = self._extract_valid_fluid_mask(fluid_mask)
        stats = stats or self.fit_frame(f, valid)
        b = stats.as_array()
        mu_u, mu_v, mu_w, mu_p = float(b[0]), float(b[1]), float(b[2]), float(b[3])
        sig_uvw = max(float(b[4]), self.scale_floor_uvw)
        sig_p = max(float(b[5]), self.scale_floor_p)

        out = np.zeros_like(f, dtype=np.float32)
        for c_idx, (mu, sig) in enumerate(((mu_u, sig_uvw), (mu_v, sig_uvw), (mu_w, sig_uvw), (mu_p, sig_p))):
            ch = f[c_idx]
            if not np.all(np.isfinite(ch[valid])):
                raise PhysicsContractError(f"channel {c_idx} contains NaN/Inf inside valid fluid domain")
            if zero_solid:
                out[c_idx, valid] = (ch[valid] - mu) / sig
            else:
                safe_ch = np.where(np.isfinite(ch), ch, mu)
                out[c_idx] = (safe_ch - mu) / sig
        return out, stats

    def denormalize_frame(
        self,
        norm_field_4ch: np.ndarray,
        stats: Flow3DNormStats | np.ndarray,
        fluid_mask: np.ndarray | None = None,
        zero_solid: bool = True,
    ) -> np.ndarray:
        nf = np.asarray(norm_field_4ch, dtype=np.float32)
        if nf.ndim != 4 or nf.shape[0] != 4:
            raise PhysicsContractError(f"expected normalized frame (4,Z,Y,X), got {nf.shape}")
        if not np.all(np.isfinite(nf)):
            raise PhysicsContractError("normalized field contains non-finite values")
        st = stats if isinstance(stats, Flow3DNormStats) else Flow3DNormStats.from_array(stats)
        b = st.as_array()
        mu_u, mu_v, mu_w, mu_p = float(b[0]), float(b[1]), float(b[2]), float(b[3])
        sig_uvw = max(float(b[4]), self.scale_floor_uvw)
        sig_p = max(float(b[5]), self.scale_floor_p)

        out = np.empty_like(nf, dtype=np.float32)
        for c_idx, (mu, sig) in enumerate(((mu_u, sig_uvw), (mu_v, sig_uvw), (mu_w, sig_uvw), (mu_p, sig_p))):
            out[c_idx] = nf[c_idx] * sig + mu
        if fluid_mask is not None and zero_solid:
            valid = self._extract_valid_fluid_mask(fluid_mask)
            out[:, ~valid] = 0.0
        return out

    def normalize(
        self,
        field: np.ndarray,
        fluid_mask: np.ndarray,
        zero_solid: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Normalize [4,Z,Y,X], [T,4,Z,Y,X], or [B,T,4,Z,Y,X] and return (norm_field, b_fp32)."""
        f = np.asarray(field, dtype=np.float32)
        m = np.asarray(fluid_mask)
        if f.ndim == 4:
            nf, st = self.normalize_frame(f, m, zero_solid=zero_solid)
            return nf, st.as_array()
        if f.ndim == 5:
            # [T, 4, Z, Y, X]
            mask_3d = m[0] if m.ndim == 4 else m
            frames, stats = [], []
            for t in range(f.shape[0]):
                nf, st = self.normalize_frame(f[t], mask_3d, zero_solid=zero_solid)
                frames.append(nf)
                stats.append(st.as_array())
            return np.stack(frames, axis=0), np.stack(stats, axis=0).astype(np.float32)
        if f.ndim == 6:
            # [B, T, 4, Z, Y, X]
            bsz, t_len = f.shape[0], f.shape[1]
            out_f = np.empty_like(f, dtype=np.float32)
            out_b = np.empty((bsz, t_len, 6), dtype=np.float32)
            for b_idx in range(bsz):
                mb = m[b_idx, 0] if m.ndim == 5 else (m[b_idx] if m.ndim == 4 else m)
                for t_idx in range(t_len):
                    nf, st = self.normalize_frame(f[b_idx, t_idx], mb, zero_solid=zero_solid)
                    out_f[b_idx, t_idx] = nf
                    out_b[b_idx, t_idx] = st.as_array()
            return out_f, out_b
        raise PhysicsContractError(f"unsupported field rank {f.ndim}; expected 4, 5, or 6")

    def denormalize(
        self,
        norm_field: np.ndarray,
        b: np.ndarray,
        fluid_mask: np.ndarray | None = None,
        zero_solid: bool = True,
    ) -> np.ndarray:
        """Inverse transform matching `normalize` for rank 4, 5, or 6."""
        nf = np.asarray(norm_field, dtype=np.float32)
        b_arr = np.asarray(b, dtype=np.float32)
        if nf.ndim == 4:
            return self.denormalize_frame(nf, b_arr, fluid_mask=fluid_mask, zero_solid=zero_solid)
        if nf.ndim == 5:
            if b_arr.shape != (nf.shape[0], 6):
                raise PhysicsContractError(f"expected b shape ({nf.shape[0]}, 6), got {b_arr.shape}")
            m3 = None if fluid_mask is None else (fluid_mask[0] if fluid_mask.ndim == 4 else fluid_mask)
            return np.stack(
                [self.denormalize_frame(nf[t], b_arr[t], fluid_mask=m3, zero_solid=zero_solid) for t in range(nf.shape[0])],
                axis=0,
            )
        if nf.ndim == 6:
            bsz, t_len = nf.shape[0], nf.shape[1]
            if b_arr.shape != (bsz, t_len, 6):
                raise PhysicsContractError(f"expected b shape ({bsz}, {t_len}, 6), got {b_arr.shape}")
            out = np.empty_like(nf, dtype=np.float32)
            for b_idx in range(bsz):
                mb = None
                if fluid_mask is not None:
                    mb = fluid_mask[b_idx, 0] if fluid_mask.ndim == 5 else (fluid_mask[b_idx] if fluid_mask.ndim == 4 else fluid_mask)
                for t_idx in range(t_len):
                    out[b_idx, t_idx] = self.denormalize_frame(
                        nf[b_idx, t_idx], b_arr[b_idx, t_idx], fluid_mask=mb, zero_solid=zero_solid
                    )
            return out
        raise PhysicsContractError(f"unsupported normalized field rank {nf.ndim}")
