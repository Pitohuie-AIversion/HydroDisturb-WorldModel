"""Deterministic residual forecasts on fixed-cadence Stage 1 latent segments."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import torch
from torch import Tensor, nn

from flow3d.contracts import condition_cadence

def _positive_int(value: int, name: str, minimum: int = 1) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _cadence(dt_ctu: float) -> float:
    value = float(dt_ctu)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("dt_ctu must be finite and positive")
    return value


def _sequence(z: Any, b: Any) -> tuple[Tensor, Tensor]:
    z = torch.as_tensor(z).detach().float()
    b = torch.as_tensor(b).detach().float()
    if z.ndim != 5 or min(z.shape[1:]) < 1 or b.shape != (len(z), 6):
        raise ValueError("expected z[N,C,Z,Y,X] and b[N,6]")
    if not torch.isfinite(z).all() or not torch.isfinite(b).all():
        raise ValueError("latent sequence contains non-finite z or b")
    return z, b


class ModeResForecaster(nn.Module):
    """Predict each channel's rate from its level and historical rates at the same voxel."""

    def __init__(self, channels: int, history_length: int = 4):
        super().__init__()
        self.channels = _positive_int(channels, "channels")
        self.history_length = _positive_int(history_length, "history_length", 2)
        self.level_weight = nn.Parameter(torch.zeros(channels, 1, 1, 1))
        self.lag_weights = nn.Parameter(torch.zeros(history_length - 1, channels, 1, 1, 1))
        self.bias = nn.Parameter(torch.zeros(channels, 1, 1, 1))

    def forward(self, history: Tensor, dt_ctu: Tensor) -> Tensor:
        """Advance batched history [B,L,C,Z,Y,X] by each segment's dt* [B]."""
        dt = dt_ctu.view(-1, 1, 1, 1, 1)
        recent = history[:, -1]
        # dt times a lagged rate is its increment on a fixed-cadence segment.
        lag_increment = (history.diff(dim=1) * self.lag_weights).sum(dim=1)
        return recent + dt * (self.bias + self.level_weight * recent) + lag_increment


def _rollout(model: ModeResForecaster, history: Tensor, dt: Tensor, horizon: int) -> Tensor:
    predictions = []
    for _ in range(horizon):
        following = model(history, dt)
        predictions.append(following)
        history = torch.cat((history[:, 1:], following[:, None]), dim=1)
    return torch.stack(predictions, dim=1)


@torch.no_grad()
def rollout_latents(
    model: ModeResForecaster, history_z: Any, history_b: Any, horizon: int, *, dt_ctu: float
) -> tuple[Tensor, Tensor]:
    """Free-run H latent frames from exactly L observed frames, carrying the last b forward."""
    _positive_int(horizon, "horizon")
    z, b = _sequence(history_z, history_b)
    if z.shape[:2] != (model.history_length, model.channels):
        raise ValueError(f"expected history length {model.history_length} and {model.channels} latent channels")
    z = z.to(model.bias.device)
    dt = z.new_tensor([_cadence(dt_ctu)])
    forecast = _rollout(model, z[None], dt, horizon)[0]
    if not torch.isfinite(forecast).all():
        raise ValueError("Mode-Res produced a non-finite forecast")
    return forecast, b[-1].to(z.device).expand(horizon, -1)


@torch.no_grad()
def delta_z_rollout(
    history_z: Any, history_b: Any, horizon: int, *, dt_ctu: float
) -> tuple[Tensor, Tensor]:
    """Extrapolate the last observed latent increment at the same dt*, carrying the last b."""
    _positive_int(horizon, "horizon")
    _cadence(dt_ctu)
    z, b = _sequence(history_z, history_b)
    if len(z) < 2:
        raise ValueError("Delta-z needs at least two observed frames")
    steps = torch.arange(1, horizon + 1, dtype=z.dtype, device=z.device).view(-1, 1, 1, 1, 1)
    forecast = z[-1] + steps * (z[-1] - z[-2])
    if not torch.isfinite(forecast).all():
        raise ValueError("Delta-z produced a non-finite forecast")
    return forecast, b[-1].to(z.device).expand(horizon, -1)


def persistence_rollout(history_si: Any, horizon: int) -> Tensor:
    """Repeat the last observed SI field directly, including its original solid-cell values."""
    _positive_int(horizon, "horizon")
    history = torch.as_tensor(history_si).detach()
    if history.ndim != 5 or history.shape[1] != 4 or len(history) == 0:
        raise ValueError("expected nonempty history_si[L,4,Z,Y,X]")
    return history[-1].expand(horizon, -1, -1, -1, -1)


def train_forecaster(
    train_sequences: Iterable[Mapping[str, Any]], *, history_length: int = 4, rollout_steps: int = 2,
    epochs: int = 10, learning_rate: float = 1e-3, batch_size: int = 4, seed: int = 0,
    device: str | torch.device = "cpu",
) -> tuple[ModeResForecaster, dict[str, Any]]:
    """Fit latent MSE over all free-running pushforward steps on within-segment train windows."""
    _positive_int(history_length, "history_length", 2)
    _positive_int(rollout_steps, "rollout_steps")
    _positive_int(epochs, "epochs")
    _positive_int(batch_size, "batch_size")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")
    width = history_length + rollout_steps
    segments: list[tuple[Tensor, float]] = []
    windows: list[tuple[int, int]] = []
    for row in train_sequences:
        if row.get("split", "train") != "train" or any(f["split"] != "train" for f in row.get("frames", ())):
            raise ValueError("train_forecaster accepts only train segments")
        z, _ = _sequence(row["z"] if "z" in row else row["z_clean"], row["b"])
        cadence = row.get("conditions", {})
        if "dt_ctu" in row:
            cadence = {**cadence, "dt_ctu": row["dt_ctu"]}
        dt = condition_cadence(cadence, "dt_ctu", required=True)
        if segments and z.shape[1:] != segments[0][0].shape[1:]:
            raise ValueError("train segments must share a latent grid and channel count")
        windows.extend((len(segments), start) for start in range(len(z) - width + 1))
        segments.append((z, dt))
    if not windows:
        raise ValueError(f"no train segment contains {width} consecutive frames")

    model = ModeResForecaster(segments[0][0].shape[1], history_length).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    generator = torch.Generator().manual_seed(seed)
    losses = []
    for _ in range(epochs):
        order = torch.randperm(len(windows), generator=generator).tolist()
        total_loss = 0.0
        for offset in range(0, len(order), batch_size):
            batch = [windows[i] for i in order[offset : offset + batch_size]]
            frames = torch.stack([segments[i][0][j : j + width].to(device) for i, j in batch])
            dt_batch = frames.new_tensor([segments[i][1] for i, _ in batch])
            forecast = _rollout(model, frames[:, :history_length], dt_batch, rollout_steps)
            loss = nn.functional.mse_loss(forecast, frames[:, history_length:])
            if not torch.isfinite(loss):
                raise ValueError("non-finite Mode-Res training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if not all(torch.isfinite(p.grad).all() for p in model.parameters()):
                raise ValueError("non-finite Mode-Res gradients")
            optimizer.step()
            total_loss += loss.item() * len(batch)
        losses.append(total_loss / len(windows))
    return model.eval(), {
        "model": "mode_res_channelwise", "history_length": history_length, "rollout_steps": rollout_steps,
        "train_segments": len(segments), "train_windows": len(windows), "epochs": epochs,
        "batch_size": batch_size, "learning_rate": learning_rate, "seed": seed, "epoch_latent_mse": losses,
        "b_rule": "persist_last_observed", "dt_unit": "CTU",
    }


_FORMAT = "flow3d.mode_res.channelwise.v1"


def save_forecaster(model: nn.Module, path: str | Path, *, metadata: dict | None = None) -> None:
    """Atomically write weights and JSON-compatible cache/training provenance."""
    path = Path(path)
    payload = {
        "format": _FORMAT, "channels": model.channels, "history_length": model.history_length,
        "state_dict": model.state_dict(), "metadata": json.loads(json.dumps(metadata or {}, allow_nan=False)),
    }
    if not isinstance(model, ModeResForecaster):
        from flow3d.training.dynamics import ConditionedForecaster

        if not isinstance(model, ConditionedForecaster):
            raise ValueError("unsupported forecaster type")
        payload.update(format="flow3d.mode_res.clean_query.v1", width=model.width, num_heads=model.num_heads)
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as temp:
        temp_path = Path(temp.name)
    try:
        torch.save(payload, temp_path)
        temp_path.replace(path)
    finally:
        temp_path.unlink(missing_ok=True)


def load_forecaster(
    path: str | Path, *, device: str | torch.device = "cpu"
) -> tuple[nn.Module, dict[str, Any]]:
    """Restore the saved dynamics and its run metadata."""
    payload = torch.load(path, weights_only=True, map_location="cpu")
    if payload.get("format") == _FORMAT:
        model = ModeResForecaster(payload["channels"], payload["history_length"])
    elif payload.get("format") == "flow3d.mode_res.clean_query.v1":
        from flow3d.training.dynamics import ConditionedForecaster

        model = ConditionedForecaster(payload["channels"], payload["history_length"], payload["width"], payload["num_heads"])
    else:
        raise ValueError("unsupported Mode-Res checkpoint format")
    model.load_state_dict(payload["state_dict"], strict=True)
    return model.to(device).eval(), payload["metadata"]
