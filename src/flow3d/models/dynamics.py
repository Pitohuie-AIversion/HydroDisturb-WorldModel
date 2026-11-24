"""Deterministic Clean/Query dynamics in the frozen encoder's latent coordinates."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .spatial_transformer import SinusoidalPosEmb3D, SpatialTransformerBlock3D


class CleanQueryModeRes(nn.Module):
    """A single spatial/causal temporal Clean block and an unmasked next-frame Query.

    The encoder supplies z[B,L,C,Z,Y,X], SI background statistics b[B,L,6] and
    five known condition tokens [B,5,condition_dim]. dt_ctu[B] is the positive
    trajectory cadence in convective time units. Returns z[B,C,Z,Y,X], b[B,6].
    """

    def __init__(self, channels: int = 256, history_length: int = 4, width: int = 64,
                 num_heads: int = 4, condition_dim: int = 256,
                 scale_floor_uvw: float = 1e-6, scale_floor_p: float = 1e-6):
        super().__init__()
        if channels < 1 or history_length < 1 or condition_dim < 1:
            raise ValueError("positive channels and condition_dim and history_length >= 1 required")
        if width < 6 or num_heads < 1 or width % num_heads:
            raise ValueError("width must be >= 6 and divisible by positive num_heads")
        if not all(math.isfinite(f) and f > 0 for f in (scale_floor_uvw, scale_floor_p)):
            raise ValueError("velocity and pressure scale floors must be finite and positive")
        self.channels = channels
        self.history_length = history_length
        self.width = width
        self.num_heads = num_heads
        self.condition_dim = condition_dim
        self.register_buffer("scale_floors", torch.tensor([scale_floor_uvw, scale_floor_p], dtype=torch.float32))

        self.latent_in = nn.Linear(channels, width)
        self.spatial_pos = SinusoidalPosEmb3D(width)
        self.time_pos = nn.Parameter(torch.empty(history_length, width))
        self.spatial = SpatialTransformerBlock3D(width, num_heads, 2 * width)
        self.temporal_norm = nn.LayerNorm(width)
        self.temporal_qkv = nn.Linear(width, 3 * width)
        self.temporal_out = nn.Linear(width, width)

        self.condition_in = nn.Linear(condition_dim, width)
        self.slot_embedding = nn.Parameter(torch.empty(5, width))
        self.query = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width))
        self.memory_kv = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 2 * width))
        self.cross_out = nn.Linear(width, width)
        self.query_mlp = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 2 * width),
                                       nn.GELU(), nn.Linear(2 * width, width))
        self.latent_rate = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, channels))
        self.background_rate = nn.Sequential(nn.Linear(width + 6 * history_length, width),
                                             nn.GELU(), nn.Linear(width, 6))
        nn.init.normal_(self.time_pos, std=0.02)
        nn.init.normal_(self.slot_embedding, std=0.02)
        # Start with Delta-z and persisted b; optimize corrections to both in multi-step training.
        for head in (self.latent_rate, self.background_rate):
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)

    def _heads(self, tokens: Tensor) -> Tensor:
        batch, length, _ = tokens.shape
        return tokens.reshape(batch, length, self.num_heads, self.width // self.num_heads).transpose(1, 2)

    def clean_tokens(self, history_z: Tensor) -> Tensor:
        """Encode each spatial frame, then each voxel's ordered prefix, into [B,L,N,width]."""
        if (history_z.ndim != 6 or not 1 <= history_z.shape[1] <= self.history_length
                or history_z.shape[2] != self.channels):
            raise ValueError("expected history_z[B,L,channels,Z,Y,X] with 1 <= L <= history_length")
        batch, length, _, depth, height, width = history_z.shape
        voxels = depth * height * width
        tokens = self.latent_in(history_z.flatten(3).transpose(2, 3))
        positions = self.spatial_pos(depth, height, width, tokens.device, tokens.dtype)
        tokens = self.spatial((tokens + positions).reshape(batch * length, voxels, self.width))
        tokens = tokens.reshape(batch, length, voxels, self.width).transpose(1, 2)
        timeline = tokens.reshape(batch * voxels, length, self.width) + self.time_pos[-length:].to(tokens.dtype)
        q, k, v = self.temporal_qkv(self.temporal_norm(timeline)).chunk(3, dim=-1)
        attended = F.scaled_dot_product_attention(self._heads(q), self._heads(k), self._heads(v),
                                                  is_causal=True)
        timeline = timeline + self.temporal_out(attended.transpose(1, 2).reshape_as(timeline))
        return timeline.reshape(batch, voxels, length, self.width).transpose(1, 2)

    def _next_background(self, history_b: Tensor, clean_last: Tensor, dt: Tensor) -> Tensor:
        # Form dimensionless history in the coordinate system of the last observed frame.
        # The three velocity means share a single velocity scale. Pressure has its own scale.
        b = history_b.float()
        last = b[:, -1]
        floors = self.scale_floors.float()
        scales = torch.maximum(last[:, 4:], floors)
        units = torch.cat((scales[:, :1].expand(-1, 3), scales[:, 1:]), dim=-1)
        means = (b[:, :, :4] - last[:, None, :4]) / units[:, None]
        log_scales = torch.maximum(b[:, :, 4:], floors).log() - scales[:, None].log()
        relative = torch.cat((means, log_scales), dim=-1)
        # Expose only retained observations when history length dropout shortens the prefix.
        pad = self.history_length - relative.shape[1]
        relative = torch.cat((relative[:, :1].expand(-1, pad, -1), relative), dim=1).flatten(1)
        features = torch.cat((clean_last.mean(dim=1), relative), dim=-1)
        rate = self.background_rate(features).float()
        # Bound the learned change in dimensionless coordinates at each step.
        increment = (dt[:, None] * rate).tanh()
        next_means = last[:, :4] + units * increment[:, :4]
        # Anchor softplus at the latest observed SI scales, including at the floor.
        # Its inverse operates on a dimensionless ratio in (0, 1], even for large pressures.
        ratio = ((scales - floors) / scales).clamp_min(torch.finfo(b.dtype).eps)
        anchor = torch.expm1(ratio).log()
        next_scales = floors + scales * F.softplus(anchor + increment[:, 4:])
        return torch.cat((next_means, next_scales), dim=-1)

    def forward(self, history_z: Tensor, history_b: Tensor, conditions: Tensor,
                dt_ctu: Tensor) -> tuple[Tensor, Tensor]:
        if history_z.ndim != 6:
            raise ValueError("expected history_z[B,L,channels,Z,Y,X]")
        batch, length = history_z.shape[:2]
        if history_b.shape != (batch, length, 6):
            raise ValueError("expected history_b[B,L,6] matching history_z")
        if conditions.shape != (batch, 5, self.condition_dim):
            raise ValueError("expected five known condition tokens [B,5,condition_dim]")
        if dt_ctu.shape != (batch,) or not torch.all(torch.isfinite(dt_ctu) & (dt_ctu > 0)):
            raise ValueError("expected finite positive dt_ctu[B]")
        if not torch.isfinite(history_b).all() or not torch.all(history_b[..., 4:] > 0):
            raise ValueError("history background statistics must be finite with positive scales")

        clean = self.clean_tokens(history_z)
        clean_last = clean[:, -1]
        condition_tokens = self.condition_in(conditions) + self.slot_embedding.to(clean.dtype)
        memory = torch.cat((clean.flatten(1, 2), condition_tokens), dim=1)
        key, value = self.memory_kv(memory).chunk(2, dim=-1)
        attention = F.scaled_dot_product_attention(self._heads(self.query(clean_last)),
                                                  self._heads(key), self._heads(value), is_causal=False)
        query = clean_last + self.cross_out(attention.transpose(1, 2).reshape_as(clean_last))
        query = query + self.query_mlp(query)
        rate = self.latent_rate(query).transpose(1, 2).reshape_as(history_z[:, -1]).float()
        dt = dt_ctu.float()
        last = history_z[:, -1].float()
        # At fixed cadence dt, dt * the observed secant rate equals this last increment.
        previous = history_z[:, max(0, length - 2)].float()
        next_z = last + (last - previous) + dt.view(-1, 1, 1, 1, 1) * rate
        return next_z, self._next_background(history_b, clean_last, dt)
