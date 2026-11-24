"""Five forecast condition tokens from declared inputs and the supplied history window."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import math
import re

import numpy as np
import torch
from torch import Tensor, nn

from flow3d.contracts import FORBIDDEN_FUTURE_CONDITION_KEYS, Flow3DConditions, condition_cadence


SLOTS = ("text", "physics", "step", "action", "regime")
_TEXT = re.compile(r"(?:3d\s+)?cylinder\s+wake(?:\s+re\s*=\s*([-+\d.eE]+))?", re.IGNORECASE)


def _available(row, key, default=True):
    return row["known_mask"].get(key, default) is True and row["valid_mask"].get(key, default) is True


def _numeric(row, key, transform=float):
    if not _available(row, key) or row.get(key) is None:
        return [0.0, 0.0]
    value = float(row[key])
    if not math.isfinite(value) or (key != "u0_m_s" and value <= 0):
        raise ValueError(f"invalid physical condition {key}")
    return [1.0, transform(value)]


def _symbol(label):
    # A fixed byte representation makes action identity independent of Python's hash seed.
    if not isinstance(label, str) or not label.strip():
        raise ValueError("declared actions require a channel and physical unit")
    return [byte / 127.5 - 1.0 for byte in hashlib.sha256(label.strip().encode()).digest()[:8]]


def _action(row, origin, target):
    missing = [0.0] * 18
    if not _available(row, "action_schedule", default=False):
        return missing
    metadata = row["extra_metadata"]
    if metadata.get("exogenous") is not True:
        return missing
    issued = float(metadata["action_known_at_time_s"])
    if not math.isfinite(issued):
        raise ValueError("nonfinite action publication time")
    if issued > origin:
        return missing
    times = np.asarray(metadata["action_times_s"], dtype=float)
    values = np.asarray(row["action_schedule"], dtype=float)
    if (times.ndim != 1 or not len(times) or values.shape != times.shape
            or not np.isfinite(times).all() or not np.isfinite(values).all() or np.any(np.diff(times) <= 0)):
        raise ValueError("invalid action schedule")
    # Schedules are sampled on the CFD time axis. Do not silently extrapolate a hold.
    i = int(np.argmin(np.abs(times - target)))
    tol = 2 * np.finfo(np.float32).eps * max(abs(target), 1.0)
    if abs(times[i] - target) > tol:
        return missing
    return [1.0, float(values[i]), *_symbol(metadata["action_channel"]), *_symbol(metadata["action_unit"])]


def _exogenous(row, origin, target):
    if isinstance(row, Flow3DConditions):
        row = row.to_dict()
    if not isinstance(row, Mapping) or not math.isfinite(origin) or not math.isfinite(target) or target < origin:
        raise ValueError("invalid forecast condition or timestamps")
    row = {"known_mask": {}, "valid_mask": {}, "extra_metadata": {}, **row}
    for sub in (row, row["known_mask"], row["valid_mask"], row["extra_metadata"]):
        if FORBIDDEN_FUTURE_CONDITION_KEYS.intersection(sub):
            raise ValueError("future truth is forbidden as a forecast condition")
    # The structured-text backend recognizes the G0 geometry and declared Re grammar.
    # Arbitrary prose or outcome labels cannot enter the model through this slot.
    text = [0.0, 0.0, 0.0]
    if _available(row, "text_prompt") and isinstance(row.get("text_prompt"), str):
        parsed = _TEXT.fullmatch(row["text_prompt"].strip())
        if parsed:
            text[0] = 1.0
            if parsed.group(1):
                re_text = float(parsed.group(1))
                if not math.isfinite(re_text) or re_text <= 0:
                    raise ValueError("invalid Reynolds number in structured text")
                text[1:] = [1.0, math.log10(re_text)]
    physics = (_numeric(row, "reynolds_number", math.log10) + _numeric(row, "u0_m_s")
               + _numeric(row, "diameter_m") + _numeric(row, "viscosity_m2_s", lambda nu: nu * 1e6))
    step = []
    for key in ("dt_phys_s", "dt_ctu"):
        cadence = condition_cadence(row, key)
        step.extend([float(cadence is not None), cadence or 0.0])
    return text, physics, step, _action(row, origin, target)


class ConditionEncoder(nn.Module):
    """Project structured text, physics, cadence, target-time action and past regime to [B,5,D].

    Numeric exogenous features include presence flags. Action lookup requires the original
    forecast issue time, even on later rollout steps. Regime uses only the passed history.
    """

    def __init__(self, dim: int = 256):
        super().__init__()
        if dim < 1:
            raise ValueError("condition embedding size must be positive")
        self.projections = nn.ModuleList(nn.Linear(width, dim) for width in (3, 8, 4, 18, 16))

    def forward(self, conditions: Sequence[Mapping | Flow3DConditions], history_z: Tensor, history_b: Tensor,
                *, origin_times_s: Sequence[float], target_times_s: Sequence[float]) -> Tensor:
        if history_z.ndim not in (4, 6) or not 1 <= history_z.shape[1] <= 4:
            raise ValueError("expected z[B,L,S,C] or z[B,L,C,Z,Y,X] with 1..4 history frames")
        batch, length = history_z.shape[:2]
        if (not batch or history_b.shape != (batch, length, 6) or len(conditions) != batch
                or len(origin_times_s) != batch or len(target_times_s) != batch):
            raise ValueError("condition batch dimensions must match the history")
        rows = [_exogenous(row, float(origin), float(target))
                for row, origin, target in zip(conditions, origin_times_s, target_times_s)]
        dtype = self.projections[0].weight.dtype
        external = [torch.tensor(slot, device=history_z.device, dtype=dtype) for slot in zip(*rows)]
        axes = tuple(range(2, history_z.ndim))
        z_mean = history_z.mean(dim=axes, dtype=torch.float32)
        z_rms = torch.linalg.vector_norm(history_z, dim=axes, dtype=torch.float32) / math.sqrt(math.prod(history_z.shape[2:]))
        # Compress different SI scales while preserving the sign of each mean.
        b = history_b.float()
        b = torch.sign(b) * torch.log1p(b.abs())
        summary = torch.cat((b, z_mean[..., None], z_rms[..., None]), dim=-1)
        regime = torch.cat((summary[:, -1], (summary[:, -1] - summary[:, 0]) / max(1, length - 1)), dim=-1)
        return torch.stack([layer(x) for layer, x in zip(self.projections, [*external, regime.to(dtype)])], dim=1)
