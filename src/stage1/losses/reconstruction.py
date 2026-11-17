from __future__ import annotations

from typing import Mapping
import numpy as np

from stage1.physics.operators import divergence, vorticity_z


def masked_mse(pred: np.ndarray, target: np.ndarray, mask: np.ndarray, eps: float = 1e-12) -> float:
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    mask = np.asarray(mask, dtype=np.float64)
    if pred.shape != target.shape:
        raise ValueError("pred and target must have identical shapes")
    if mask.ndim == pred.ndim - 1:
        mask = np.expand_dims(mask, axis=-4)
    if mask.shape != pred.shape and mask.shape[-3:] == pred.shape[-3:]:
        mask = np.broadcast_to(mask, pred.shape)
    denom = float(mask.sum())
    if denom <= eps:
        raise ValueError("mask contains no valid cells")
    return float((((pred - target) ** 2) * mask).sum() / denom)


def divergence_loss(pred_velocity: np.ndarray, spacing: tuple[float, float, float], mask: np.ndarray | None = None) -> float:
    if pred_velocity.shape[-4] != 3:
        raise ValueError("pred_velocity must have three channels at axis -4")
    div = divergence(pred_velocity[..., 0, :, :, :], pred_velocity[..., 1, :, :, :], pred_velocity[..., 2, :, :, :], spacing, mask)
    valid = mask if mask is not None else np.ones_like(div)
    return masked_mse(div, np.zeros_like(div), valid)


def vorticity_loss(pred_velocity: np.ndarray, target_velocity: np.ndarray, spacing: tuple[float, float, float], mask: np.ndarray | None = None) -> float:
    if pred_velocity.shape != target_velocity.shape or pred_velocity.shape[-4] != 3:
        raise ValueError("velocity tensors must match and have three channels")
    pred = vorticity_z(pred_velocity[..., 0, :, :, :], pred_velocity[..., 1, :, :, :], spacing, mask)
    target = vorticity_z(target_velocity[..., 0, :, :, :], target_velocity[..., 1, :, :, :], spacing, mask)
    return masked_mse(pred, target, mask if mask is not None else np.ones_like(pred))


def total_loss(pred: np.ndarray, target: np.ndarray, mask: np.ndarray, spacing: tuple[float, float, float], weights: Mapping[str, float] | None = None) -> dict[str, float]:
    weights = weights or {"reconstruction": 1.0, "divergence": 0.0, "vorticity": 0.0}
    rec = masked_mse(pred, target, np.expand_dims(mask, axis=-4))
    div = divergence_loss(pred[..., :3, :, :, :], spacing, mask)
    vort = vorticity_loss(pred[..., :3, :, :, :], target[..., :3, :, :, :], spacing, mask)
    total = weights.get("reconstruction", 1.0) * rec + weights.get("divergence", 0.0) * div + weights.get("vorticity", 0.0) * vort
    return {"total": float(total), "reconstruction": rec, "divergence": div, "vorticity": vort}
