from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class GeometryContract:
    mask_threshold: float = 0.5
    sdf_clip: float = 8.0
    sdf_sign: str = "positive_fluid"
    version: str = "geometry.v1"


def sanitize_geometry(mask: np.ndarray, sdf: np.ndarray, contract: GeometryContract | None = None) -> tuple[np.ndarray, np.ndarray]:
    contract = contract or GeometryContract()
    mask = (np.asarray(mask, dtype=np.float32) >= contract.mask_threshold).astype(np.float32)
    sdf = np.asarray(sdf, dtype=np.float32)
    if sdf.shape != mask.shape:
        raise ValueError(f"mask/sdf shape mismatch: {mask.shape} vs {sdf.shape}")
    sdf = np.clip(sdf, -contract.sdf_clip, contract.sdf_clip) / contract.sdf_clip
    return mask, sdf


def build_derivative_mask(mask: np.ndarray, axis: int, contract: GeometryContract | None = None) -> np.ndarray:
    """Return a centered-difference validity mask for neighboring fluid cells."""
    contract = contract or GeometryContract()
    m = (np.asarray(mask) >= contract.mask_threshold).astype(np.float32)
    if axis < 0:
        axis += m.ndim
    if axis < 0 or axis >= m.ndim:
        raise ValueError("axis out of bounds")
    left = np.roll(m, 1, axis=axis)
    right = np.roll(m, -1, axis=axis)
    valid = m * left * right
    # Roll wraps at the boundary; explicitly invalidate both boundary planes.
    sl = [slice(None)] * m.ndim
    sl[axis] = 0
    valid[tuple(sl)] = 0
    sl[axis] = -1
    valid[tuple(sl)] = 0
    return valid.astype(np.float32)
