from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


FLOW_CHANNELS = ("u", "v", "w", "p", "mask", "sdf")


@dataclass(frozen=True)
class NormalizationStats:
    """Per-frame background statistics used by the Stage 1 contract.

    `velocity_scale` is the joint RMS of `(u-mean_u, v-mean_v, w-mean_w)` over
    valid fluid cells. The scale is not divided by three; this makes the
    normalized joint kinetic energy equal to 0.5 when no epsilon is active.
    """

    mean_u: float
    mean_v: float
    mean_w: float
    mean_p: float
    velocity_scale: float
    pressure_scale: float
    eps: float = 1e-6
    version: str = "normalizer.v1"

    def as_array(self) -> np.ndarray:
        return np.asarray(
            [self.mean_u, self.mean_v, self.mean_w, self.mean_p, self.velocity_scale, self.pressure_scale],
            dtype=np.float32,
        )

    @classmethod
    def from_array(cls, values: np.ndarray, **kwargs: Any) -> "NormalizationStats":
        values = np.asarray(values).reshape(-1)
        if values.size != 6:
            raise ValueError(f"expected six statistics, got {values.size}")
        return cls(*(float(v) for v in values), **kwargs)


class FlowNormalizer:
    """Normalize and restore `(u,v,w,p,mask,sdf)` frames with explicit masks."""

    def __init__(self, eps: float = 1e-6, sdf_clip: float = 8.0, version: str = "normalizer.v1"):
        if eps <= 0:
            raise ValueError("eps must be positive")
        self.eps = float(eps)
        self.sdf_clip = float(sdf_clip)
        self.version = version

    @staticmethod
    def _validate(frame: np.ndarray) -> np.ndarray:
        frame = np.asarray(frame, dtype=np.float32)
        if frame.ndim < 4 or frame.shape[-4] != 6:
            raise ValueError("frame must have channel dimension of six at axis -4")
        return frame

    def fit(self, frame: np.ndarray) -> NormalizationStats:
        frame = self._validate(frame)
        if frame.ndim != 4:
            raise ValueError("fit expects one frame with shape (6,D,H,W); use fit_batch for (B,6,D,H,W)")
        mask = np.take(frame, 4, axis=-4)
        if not np.all(np.isfinite(mask)):
            raise ValueError("fluid mask contains non-finite values")
        valid = mask > 0.5
        denom = float(valid.sum())
        if denom <= 0:
            raise ValueError("mask contains no fluid cells")
        vals = [np.take(frame, i, axis=-4).astype(np.float64) for i in range(4)]
        # Perform finite check on fluid cells BEFORE computing stats; never rely on NaN * 0.
        fluid_vals = []
        for idx, v in enumerate(vals):
            fv = v[valid]
            if not np.all(np.isfinite(fv)):
                raise ValueError(f"channel {FLOW_CHANNELS[idx]} contains non-finite values inside fluid mask")
            fluid_vals.append(fv)
        means = [float(fv.mean()) for fv in fluid_vals]
        vel_sq = sum((fluid_vals[i] - means[i]) ** 2 for i in range(3))
        velocity_scale = max(float(np.sqrt(vel_sq.mean())), self.eps)
        pressure_scale = max(float(np.sqrt(((fluid_vals[3] - means[3]) ** 2).mean())), self.eps)
        return NormalizationStats(*means, velocity_scale, pressure_scale, eps=self.eps, version=self.version)

    def fit_batch(self, frames: np.ndarray) -> list[NormalizationStats]:
        frames = self._validate(frames)
        if frames.ndim != 5:
            raise ValueError("fit_batch expects shape (B,6,D,H,W)")
        return [self.fit(frame) for frame in frames]

    def normalize(self, frame: np.ndarray, stats: NormalizationStats | None = None) -> tuple[np.ndarray, NormalizationStats]:
        frame = self._validate(frame)
        if frame.ndim != 4:
            raise ValueError("normalize expects one frame with shape (6,D,H,W); use normalize_batch for (B,6,D,H,W)")
        stats = stats or self.fit(frame)
        out = frame.copy()
        vel_scale = max(float(stats.velocity_scale), self.eps)
        p_scale = max(float(stats.pressure_scale), self.eps)
        valid = out[..., 4, :, :, :] > 0.5
        for i in range(4):
            ch = out[..., i, :, :, :]
            if not np.all(np.isfinite(ch[valid])):
                raise ValueError(f"channel {FLOW_CHANNELS[i]} contains non-finite values inside fluid mask")
            # Scrub non-finite values inside masked-out solid cells so NaNs never enter Conv3d.
            if not np.all(np.isfinite(ch)):
                ch = np.where(np.isfinite(ch), ch, 0.0)
                out[..., i, :, :, :] = ch
        for i, mean in enumerate((stats.mean_u, stats.mean_v, stats.mean_w)):
            out[..., i, :, :, :] = (out[..., i, :, :, :] - mean) / vel_scale
        out[..., 3, :, :, :] = (out[..., 3, :, :, :] - stats.mean_p) / p_scale
        out[..., 4, :, :, :] = valid.astype(np.float32)
        sdf = out[..., 5, :, :, :]
        if not np.all(np.isfinite(sdf)):
            raise ValueError("sdf channel contains non-finite values")
        out[..., 5, :, :, :] = np.clip(sdf, -self.sdf_clip, self.sdf_clip) / self.sdf_clip
        return out, stats

    def normalize_batch(self, frames: np.ndarray, stats: list[NormalizationStats] | None = None) -> tuple[np.ndarray, list[NormalizationStats]]:
        frames = self._validate(frames)
        if frames.ndim != 5:
            raise ValueError("normalize_batch expects shape (B,6,D,H,W)")
        stats = stats or self.fit_batch(frames)
        if len(stats) != frames.shape[0]:
            raise ValueError("one NormalizationStats object is required per frame")
        normalized = np.stack([self.normalize(frame, stat)[0] for frame, stat in zip(frames, stats)], axis=0)
        return normalized, stats

    def denormalize(self, normalized: np.ndarray, stats: NormalizationStats) -> np.ndarray:
        normalized = self._validate(normalized)
        if normalized.ndim != 4:
            raise ValueError("denormalize expects one frame; use denormalize_batch for (B,6,D,H,W)")
        out = normalized.copy()
        vel_scale = max(float(stats.velocity_scale), self.eps)
        p_scale = max(float(stats.pressure_scale), self.eps)
        for i, mean in enumerate((stats.mean_u, stats.mean_v, stats.mean_w)):
            out[..., i, :, :, :] = out[..., i, :, :, :] * vel_scale + mean
        out[..., 3, :, :, :] = out[..., 3, :, :, :] * p_scale + stats.mean_p
        out[..., 4, :, :, :] = (out[..., 4, :, :, :] > 0.5).astype(np.float32)
        out[..., 5, :, :, :] = out[..., 5, :, :, :] * self.sdf_clip
        return out

    def denormalize_batch(self, normalized: np.ndarray, stats: list[NormalizationStats]) -> np.ndarray:
        normalized = self._validate(normalized)
        if normalized.ndim != 5 or len(stats) != normalized.shape[0]:
            raise ValueError("denormalize_batch expects (B,6,D,H,W) and one stats object per frame")
        return np.stack([self.denormalize(frame, stat) for frame, stat in zip(normalized, stats)], axis=0)
