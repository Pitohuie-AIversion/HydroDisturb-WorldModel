"""Mandatory 3D Spatial Transformer bottleneck (`Spatial Transformer x1`).

Operates strictly per-frame across flattened 3D spatial tokens `(Z_lat, Y_lat, X_lat)`
with PreLN, Multi-Head Self-Attention via `F.scaled_dot_product_attention` (SDPA),
dynamic 3D SinCos positional encoding, and FFN (`dim -> mlp_dim -> dim`).
- `depth >= 1` is MANDATORY (`depth=0` or `depth < 1` raises `ValueError`).
- ZERO temporal attention: frames in `[B, T, C, Dz, Hy, Wx]` are strictly independent.
"""

from __future__ import annotations

import math

try:  # Keep PyTorch optional for protocol-only environments.
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception:  # pragma: no cover - exercised only when torch is absent
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]


def torch_available() -> bool:
    return torch is not None


def _require_torch() -> None:
    if torch is None:
        raise ImportError("flow3d.models.spatial_transformer requires PyTorch. Install with `pip install -e .[torch]`.")


if torch is None:

    class SinusoidalPosEmb3D:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            _require_torch()

    class SpatialTransformerBlock3D:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            _require_torch()

    class SpatialTransformer3D:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            _require_torch()

else:

    def _axis_sincos(length: int, dim: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if dim <= 0:
            return torch.zeros((length, 0), device=device, dtype=dtype)
        half = dim // 2
        pos = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
        if half > 0:
            freq = torch.exp(
                torch.arange(half, device=device, dtype=torch.float32) * (-math.log(10000.0) / float(half))
            )
            angles = pos * freq.unsqueeze(0)
            emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        else:
            emb = torch.zeros((length, 0), device=device, dtype=torch.float32)
        if emb.shape[-1] < dim:
            pad = torch.zeros((length, dim - emb.shape[-1]), device=device, dtype=torch.float32)
            emb = torch.cat([emb, pad], dim=-1)
        return emb.to(dtype=dtype)

    class SinusoidalPosEmb3D(nn.Module):
        """Dynamic 3D sinusoidal positional encoding over `(Z_lat, Y_lat, X_lat)` tokens."""

        def __init__(self, dim: int = 256) -> None:
            super().__init__()
            if dim < 6:
                raise ValueError(f"SinusoidalPosEmb3D requires dim >= 6, got {dim}")
            self.dim = int(dim)
            d_z = (self.dim // 3 // 2) * 2
            d_y = (self.dim // 3 // 2) * 2
            d_x = self.dim - d_z - d_y
            self.d_z = d_z
            self.d_y = d_y
            self.d_x = d_x

        def forward(
            self,
            dz: int,
            hy: int,
            wx: int,
            device: torch.device,
            dtype: torch.dtype = torch.float32,
        ) -> torch.Tensor:
            ez = _axis_sincos(dz, self.d_z, device, dtype)[:, None, None, :].expand(dz, hy, wx, self.d_z)
            ey = _axis_sincos(hy, self.d_y, device, dtype)[None, :, None, :].expand(dz, hy, wx, self.d_y)
            ex = _axis_sincos(wx, self.d_x, device, dtype)[None, None, :, :].expand(dz, hy, wx, self.d_x)
            pos = torch.cat([ez, ey, ex], dim=-1)
            return pos.reshape(1, dz * hy * wx, self.dim)

    class SpatialTransformerBlock3D(nn.Module):
        """PreLN Transformer block with SDPA self-attention and GELU MLP."""

        def __init__(self, dim: int = 256, num_heads: int = 8, mlp_dim: int = 1024, dropout: float = 0.0) -> None:
            super().__init__()
            if dim % num_heads != 0:
                raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")
            self.dim = int(dim)
            self.num_heads = int(num_heads)
            self.head_dim = self.dim // self.num_heads
            self.dropout = float(dropout)

            self.norm1 = nn.LayerNorm(self.dim)
            self.qkv = nn.Linear(self.dim, 3 * self.dim)
            self.proj = nn.Linear(self.dim, self.dim)

            self.norm2 = nn.LayerNorm(self.dim)
            self.mlp = nn.Sequential(
                nn.Linear(self.dim, int(mlp_dim)),
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.Linear(int(mlp_dim), self.dim),
                nn.Dropout(self.dropout),
            )

        def forward(self, tokens: torch.Tensor) -> torch.Tensor:
            n, l, c = tokens.shape
            h = self.norm1(tokens)
            qkv = self.qkv(h).reshape(n, l, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            attn_out = F.scaled_dot_product_attention(
                q, k, v, dropout_p=self.dropout if self.training else 0.0, is_causal=False
            )
            attn_out = attn_out.transpose(1, 2).reshape(n, l, c)
            tokens = tokens + F.dropout(self.proj(attn_out), p=self.dropout, training=self.training)
            tokens = tokens + self.mlp(self.norm2(tokens))
            return tokens

    class SpatialTransformer3D(nn.Module):
        """Per-frame 3D Spatial Transformer (`depth >= 1` mandatory, zero temporal mixing)."""

        def __init__(
            self,
            dim: int = 256,
            depth: int = 1,
            num_heads: int = 8,
            mlp_dim: int = 1024,
            dropout: float = 0.0,
            pos_mode: str = "sincos3d",
            output_norm: str = "none",
        ) -> None:
            super().__init__()
            if int(depth) < 1:
                raise ValueError(f"SpatialTransformer3D requires depth >= 1 (depth=0 prohibited), got depth={depth}")
            if int(dim) % int(num_heads) != 0:
                raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")
            if pos_mode not in ("sincos3d",):
                raise ValueError(f"unsupported pos_mode {pos_mode!r}; expected 'sincos3d'")
            if output_norm not in ("none", "legacy_layer_norm"):
                raise ValueError(f"unknown transformer output_norm {output_norm!r}")

            self.dim = int(dim)
            self.depth = int(depth)
            self.num_heads = int(num_heads)
            self.mlp_dim = int(mlp_dim)
            self.pos_mode = str(pos_mode)

            self.pos_emb = SinusoidalPosEmb3D(self.dim)
            self.blocks = nn.ModuleList(
                [
                    SpatialTransformerBlock3D(
                        dim=self.dim,
                        num_heads=self.num_heads,
                        mlp_dim=self.mlp_dim,
                        dropout=dropout,
                    )
                    for _ in range(self.depth)
                ]
            )
            self.norm_out = nn.LayerNorm(self.dim) if output_norm == "legacy_layer_norm" else nn.Identity()

        def forward(self, z: torch.Tensor) -> torch.Tensor:
            if z.ndim not in (5, 6):
                raise ValueError(
                    f"SpatialTransformer3D expects 5D [N,C,Dz,Hy,Wx] or 6D [B,T,C,Dz,Hy,Wx], got {tuple(z.shape)}"
                )
            if not torch.isfinite(z).all():
                raise ValueError("SpatialTransformer3D received non-finite latent tensor")

            is_6d = z.ndim == 6
            if is_6d:
                b, t, c, dz, hy, wx = z.shape
                z_flat = z.reshape(b * t, c, dz, hy, wx)
            else:
                n, c, dz, hy, wx = z.shape
                z_flat = z

            if c != self.dim:
                raise ValueError(f"expected channel dim {self.dim}, got {c}")

            # Deterministic (Z, Y, X) token flatten order; strictly per-frame (B*T)
            tokens = z_flat.permute(0, 2, 3, 4, 1).reshape(z_flat.shape[0], dz * hy * wx, c)
            pos = self.pos_emb(dz, hy, wx, device=tokens.device, dtype=tokens.dtype)
            tokens = tokens + pos
            for blk in self.blocks:
                tokens = blk(tokens)
            tokens = self.norm_out(tokens)

            out_flat = tokens.reshape(z_flat.shape[0], dz, hy, wx, c).permute(0, 4, 1, 2, 3)
            if is_6d:
                return out_flat.reshape(b, t, c, dz, hy, wx)
            return out_flat


__all__ = [
    "SinusoidalPosEmb3D",
    "SpatialTransformer3D",
    "SpatialTransformerBlock3D",
    "torch_available",
]
