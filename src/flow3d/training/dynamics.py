"""Fit and freely roll out the conditioned, deterministic Clean/Query model on frozen latents."""
from __future__ import annotations

from time import monotonic
from contextlib import nullcontext

import torch
from torch import nn
from torch.nn import functional as F

from flow3d.contracts import condition_cadence
from flow3d.models.conditions import ConditionEncoder
from flow3d.models.dynamics import CleanQueryModeRes
from flow3d.training.forecast import _sequence


class ConditionedForecaster(nn.Module):
    def __init__(self, channels, history_length=4, width=64, num_heads=4):
        super().__init__()
        if not 1 <= history_length <= 4:
            raise ValueError("Clean/Query supports a history of one to four frames")
        self.channels = channels
        self.history_length = history_length
        self.width = width
        self.num_heads = num_heads
        self.conditions = ConditionEncoder(dim=256)
        self.dynamics = CleanQueryModeRes(channels=channels, history_length=history_length,
                                          width=width, num_heads=num_heads, condition_dim=256)

    def forward(self, z, b, conditions, origin_times_s, horizon, *, mixed_precision=False):
        """Advance both z and b, keeping the issue time fixed for all scheduled actions."""
        if horizon < 1 or z.size(1) > self.history_length:
            raise ValueError("expected a positive horizon and a supported history length")
        dt = z.new_tensor([condition_cadence(row, "dt_ctu", required=True) for row in conditions])
        cadence_s = [condition_cadence(row, "dt_phys_s", required=True) for row in conditions]
        forecasts, statistics = [], []
        for lead in range(1, horizon + 1):
            times = [origin + lead * step for origin, step in zip(origin_times_s, cadence_s, strict=True)]
            with torch.autocast(device_type=z.device.type, dtype=torch.bfloat16, enabled=mixed_precision):
                tokens = self.conditions(conditions, z, b, origin_times_s=origin_times_s, target_times_s=times)
                next_z, next_b = self.dynamics(z, b, tokens, dt)
            next_z, next_b = next_z.float(), next_b.float()
            forecasts.append(next_z)
            statistics.append(next_b)
            z = torch.cat((z, next_z[:, None]), dim=1)[:, -self.history_length:]
            b = torch.cat((b, next_b[:, None]), dim=1)[:, -self.history_length:]
        return torch.stack(forecasts, dim=1), torch.stack(statistics, dim=1)


def train_conditioned(train_sequences, *, history_length, rollout_steps, epochs, learning_rate, batch_size,
                      seed, device, width=64):
    if min(history_length, rollout_steps, epochs, batch_size) < 1 or learning_rate <= 0:
        raise ValueError("positive history, rollout steps, epochs, batch size, and learning rate are required")
    device = torch.device(device)
    mixed_precision = device.type == "cuda" and torch.cuda.is_bf16_supported()
    generator = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    segments, windows = [], []
    length = history_length + rollout_steps
    for row in train_sequences:
        if row["split"] != "train" or any(frame["split"] != "train" for frame in row["frames"]):
            raise ValueError("conditioned dynamics accepts train segments only")
        z, b = _sequence(row["z_clean"], row["b"])
        if (b[:, 4:] <= 0).any() or len(row["time_physical_s"]) != len(z):
            raise ValueError("invalid training scales or time axis")
        windows.extend((len(segments), i) for i in range(len(z) - length + 1))
        segments.append((z, b, row["conditions"], row["time_physical_s"]))
    if not windows:
        raise ValueError("no contiguous train window is long enough")
    model = ConditionedForecaster(segments[0][0].size(1), history_length, width).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-3)
    effective_lengths = sorted({1, min(2, history_length), history_length})
    epoch_history = []
    sdpa_operators = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(epochs):
        started = monotonic()
        order = torch.randperm(len(windows), generator=generator).tolist()
        total_z = total_b = 0.
        for offset in range(0, len(order), batch_size):
            selected = [windows[i] for i in order[offset:offset + batch_size]]
            z = torch.stack([segments[s][0][i:i + length] for s, i in selected]).to(device)
            b = torch.stack([segments[s][1][i:i + length] for s, i in selected]).to(device)
            rows = [segments[s][2] for s, _ in selected]
            origins = [float(segments[s][3][i + history_length - 1]) for s, i in selected]
            active = effective_lengths[int(torch.randint(len(effective_lengths), (), generator=generator))]
            history = slice(history_length - active, history_length)
            context = z[:, history] + 0.01 * torch.randn_like(z[:, history])
            # Capture dispatch names from the first real training batch (CPU operator events
            # include ATen's concrete CUDA SDPA backend without recording GPU tensor data).
            trace = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) if epoch == 0 and offset == 0 else None
            with trace if trace is not None else nullcontext():
                pred, stats = model(context, b[:, history], rows, origins, rollout_steps, mixed_precision=mixed_precision)
            if trace is not None:
                sdpa_operators = sorted(event.key for event in trace.key_averages() if "_scaled_dot_product_" in event.key)
            # Put errors in SI means and scales on comparable units using only the last observation.
            sigma_u, sigma_p = b[:, history_length - 1, 4:6].clamp_min(1e-6).unbind(-1)
            scale = torch.stack((sigma_u, sigma_u, sigma_u, sigma_p, sigma_u, sigma_p), -1)
            loss_z = F.mse_loss(pred, z[:, history_length:])
            loss_b = ((stats - b[:, history_length:]) / scale[:, None]).square().mean()
            optimizer.zero_grad(set_to_none=True)
            (loss_z + loss_b).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            total_z += float(loss_z.detach()) * len(selected)
            total_b += float(loss_b.detach()) * len(selected)
        epoch_history.append({"epoch": epoch+1, "latent_mse": total_z/len(windows),
                              "scaled_b_mse": total_b/len(windows), "seconds": monotonic()-started})
        print(f"Conditioned epoch {epoch+1}/{epochs}: {epoch_history[-1]}", flush=True)
    return model.eval(), {
        "model": "clean_query_mode_res", "model_width": width, "condition_width": 256,
        "num_heads": model.num_heads, "history_length": history_length, "rollout_steps": rollout_steps,
        "epochs": epochs, "seed": seed, "batch_size": batch_size, "learning_rate": learning_rate,
        "train_segments": len(segments), "train_windows": len(windows), "history": epoch_history,
        "b_rule": "learned_history_only", "context_noise_std": 0.01, "effective_history_lengths": effective_lengths,
        "training_precision": "bf16_autocast" if mixed_precision else "float32",
        "first_batch_sdpa_operators": sdpa_operators,
        "peak_training_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
    }
