"""3D DC-AE Encoder, Independent Decoder, Geometry Injection, and Latent Normalization.

Implements V1.3 Route A Stage 1 backbone (`f8 / C=256`):
- `S2C3d` & `C2S3d`: exact 3D space-to-channel / channel-to-space (`factor=2`).
- `ResBlock3D`: GroupNorm -> SiLU -> Conv3d(3x3x3) -> GroupNorm -> SiLU -> Conv3d(3x3x3) + skip.
- `GeometryInject3D`: multi-scale 2-ch `[fluid_mask, sdf_over_d]` residual conditioning
  (`nearest` for binary mask, `trilinear` align_corners=False for SDF/D, optional tanh transform).
- `DownBlock3D` & `UpBlock3D`: DC-AE residual S2C / C2S blocks with parameter-free shortcuts.
- `DCAE3DEncoder`: accepts normalized 4-ch field + 2-ch geometry (or 6-ch), flattens B*T, outputs `[B*T, 256, Z//8, Y//8, X//8]`.
- `DCAE3DDecoder`: strictly independent; accepts ONLY `(z_clean, geometry)` with ZERO Encoder->Decoder skips.
- `LatentNorm3D`: configurable `group_norm` (default + dataset RMS scale), `dataset_rms`, `affine_free_rmsnorm`.
"""

from __future__ import annotations

import math
from typing import Sequence

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
        raise ImportError("flow3d.models.dc_ae3d requires PyTorch. Install with `pip install -e .[torch]`.")


if torch is None:

    class S2C3d:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            _require_torch()

    class C2S3d:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            _require_torch()

    class ResBlock3D:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            _require_torch()

    class GeometryInject3D:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            _require_torch()

    class DownBlock3D:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            _require_torch()

    class UpBlock3D:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            _require_torch()

    class LatentNorm3D:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            _require_torch()

    class DCAE3DEncoder:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            _require_torch()

    class DCAE3DDecoder:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            _require_torch()

else:

    def _resolve_gn_groups(channels: int, requested_groups: int) -> int:
        if requested_groups <= 0:
            raise ValueError(f"gn_groups must be positive, got {requested_groups}")
        if channels % requested_groups == 0:
            return requested_groups
        g = math.gcd(channels, requested_groups)
        return max(1, g)

    def _validate_finite(tensor: torch.Tensor, name: str) -> None:
        if not torch.isfinite(tensor).all():
            raise ValueError(f"{name} contains NaN or Inf values")

    def _prepare_geometry_for_bt(
        geometry: torch.Tensor,
        batch_size: int,
        t_len: int | None,
        spatial_zyx: tuple[int, int, int],
    ) -> torch.Tensor:
        """Normalize geometry shape to `[N, 2, Z, Y, X]` where `N = B*T` (or `B` if `t_len is None`)."""
        _validate_finite(geometry, "geometry")
        nz, ny, nx = spatial_zyx
        if t_len is None:
            if geometry.ndim != 5 or geometry.shape != (batch_size, 2, nz, ny, nx):
                raise ValueError(
                    f"expected geometry shape ({batch_size}, 2, {nz}, {ny}, {nx}), got {tuple(geometry.shape)}"
                )
            return geometry

        if geometry.ndim == 5:
            if geometry.shape != (batch_size, 2, nz, ny, nx):
                raise ValueError(
                    f"expected 5D geometry shape ({batch_size}, 2, {nz}, {ny}, {nx}), got {tuple(geometry.shape)}"
                )
            return geometry.unsqueeze(1).expand(batch_size, t_len, 2, nz, ny, nx).reshape(batch_size * t_len, 2, nz, ny, nx)
        if geometry.ndim == 6:
            if geometry.shape != (batch_size, t_len, 2, nz, ny, nx):
                raise ValueError(
                    f"expected 6D geometry shape ({batch_size}, {t_len}, 2, {nz}, {ny}, {nx}), got {tuple(geometry.shape)}"
                )
            return geometry.reshape(batch_size * t_len, 2, nz, ny, nx)
        raise ValueError(f"geometry must be 5D [B,2,Z,Y,X] or 6D [B,T,2,Z,Y,X], got shape {tuple(geometry.shape)}")

    class S2C3d(nn.Module):
        """3D Space-to-Channel: `[N, C, Z, Y, X] -> [N, C*f^3, Z//f, Y//f, X//f]`."""

        def __init__(self, factor: int = 2) -> None:
            super().__init__()
            if factor < 1:
                raise ValueError(f"factor must be >= 1, got {factor}")
            self.factor = int(factor)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if x.ndim != 5:
                raise ValueError(f"S2C3d expects 5D tensor [N,C,Z,Y,X], got {tuple(x.shape)}")
            n, c, z, y, x_dim = x.shape
            f = self.factor
            if z % f != 0 or y % f != 0 or x_dim % f != 0:
                raise ValueError(
                    f"spatial dims {(z, y, x_dim)} must be divisible by S2C3d factor={f}"
                )
            x = x.reshape(n, c, z // f, f, y // f, f, x_dim // f, f)
            x = x.permute(0, 1, 3, 5, 7, 2, 4, 6)
            return x.reshape(n, c * (f ** 3), z // f, y // f, x_dim // f)

    class C2S3d(nn.Module):
        """3D Channel-to-Space (exact inverse of S2C3d): `[N, C, Z, Y, X] -> [N, C//f^3, Z*f, Y*f, X*f]`."""

        def __init__(self, factor: int = 2) -> None:
            super().__init__()
            if factor < 1:
                raise ValueError(f"factor must be >= 1, got {factor}")
            self.factor = int(factor)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if x.ndim != 5:
                raise ValueError(f"C2S3d expects 5D tensor [N,C,Z,Y,X], got {tuple(x.shape)}")
            n, c, z, y, x_dim = x.shape
            f = self.factor
            f3 = f ** 3
            if c % f3 != 0:
                raise ValueError(f"channels {c} must be divisible by factor^3={f3}")
            c_out = c // f3
            x = x.reshape(n, c_out, f, f, f, z, y, x_dim)
            x = x.permute(0, 1, 5, 2, 6, 3, 7, 4)
            return x.reshape(n, c_out, z * f, y * f, x_dim * f)

    class ResBlock3D(nn.Module):
        """3D residual block: GroupNorm -> SiLU -> Conv3d -> GroupNorm -> SiLU -> Conv3d + skip."""

        def __init__(self, channels: int, gn_groups: int = 8) -> None:
            super().__init__()
            g = _resolve_gn_groups(channels, gn_groups)
            self.norm1 = nn.GroupNorm(g, channels)
            self.conv1 = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
            self.norm2 = nn.GroupNorm(g, channels)
            self.conv2 = nn.Conv3d(channels, channels, kernel_size=3, padding=1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            h = self.conv1(F.silu(self.norm1(x)))
            h = self.conv2(F.silu(self.norm2(h)))
            return x + h

    class GeometryInject3D(nn.Module):
        """Multi-scale 2-ch `[fluid_mask, sdf_over_d]` residual conditioning."""

        def __init__(self, channels: int, gn_groups: int = 8, sdf_tanh_alpha: float | None = 1.0) -> None:
            super().__init__()
            if sdf_tanh_alpha is not None and sdf_tanh_alpha <= 0.0:
                raise ValueError(f"sdf_tanh_alpha must be positive or None, got {sdf_tanh_alpha}")
            self.sdf_tanh_alpha = None if sdf_tanh_alpha is None else float(sdf_tanh_alpha)
            self.proj = nn.Conv3d(2, channels, kernel_size=1)
            self.norm = nn.GroupNorm(_resolve_gn_groups(channels, gn_groups), channels)

        def forward(self, x: torch.Tensor, geometry: torch.Tensor) -> torch.Tensor:
            if geometry.ndim != 5 or geometry.shape[1] != 2:
                raise ValueError(f"GeometryInject3D expects geometry [N, 2, Z, Y, X], got {tuple(geometry.shape)}")
            _validate_finite(geometry, "geometry")
            target_size = x.shape[-3:]
            mask_ch = geometry[:, 0:1]
            sdf_ch = geometry[:, 1:2]
            if tuple(geometry.shape[-3:]) != target_size:
                mask_ch = F.interpolate(mask_ch, size=target_size, mode="nearest")
                sdf_ch = F.interpolate(sdf_ch, size=target_size, mode="trilinear", align_corners=False)
            if self.sdf_tanh_alpha is not None:
                sdf_ch = torch.tanh(sdf_ch / self.sdf_tanh_alpha)
            geom_scaled = torch.cat([mask_ch, sdf_ch], dim=1)
            return x + self.norm(self.proj(geom_scaled))

    class DownBlock3D(nn.Module):
        """DC-AE 3D downsampling block: Conv3d -> S2C3d(2) + parameter-free S2C3d->GroupMean shortcut."""

        def __init__(self, in_ch: int, out_ch: int, gn_groups: int = 8, num_refine_blocks: int = 1) -> None:
            super().__init__()
            if out_ch % 8 != 0:
                raise ValueError(f"DownBlock3D requires out_ch % 8 == 0, got out_ch={out_ch}")
            if (in_ch * 8) % out_ch != 0:
                raise ValueError(f"DownBlock3D requires (in_ch * 8) % out_ch == 0, got in_ch={in_ch}, out_ch={out_ch}")
            if num_refine_blocks < 0:
                raise ValueError(f"num_refine_blocks must be >= 0, got {num_refine_blocks}")
            self.in_ch = int(in_ch)
            self.out_ch = int(out_ch)
            self.group_size = (in_ch * 8) // out_ch

            self.conv = nn.Conv3d(in_ch, out_ch // 8, kernel_size=3, padding=1)
            self.s2c = S2C3d(factor=2)
            self.refine = nn.Sequential(*[ResBlock3D(out_ch, gn_groups=gn_groups) for _ in range(num_refine_blocks)])

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            main = self.s2c(self.conv(x))
            sc = self.s2c(x)
            n, _, z, y, x_dim = sc.shape
            sc = sc.reshape(n, self.out_ch, self.group_size, z, y, x_dim).mean(dim=2)
            return self.refine(main + sc)

    class UpBlock3D(nn.Module):
        """DC-AE 3D upsampling block: Conv3d -> C2S3d(2) + parameter-free repeat_interleave->C2S3d shortcut."""

        def __init__(self, in_ch: int, out_ch: int, gn_groups: int = 8, num_refine_blocks: int = 1) -> None:
            super().__init__()
            if (out_ch * 8) % in_ch != 0:
                raise ValueError(f"UpBlock3D requires (out_ch * 8) % in_ch == 0, got in_ch={in_ch}, out_ch={out_ch}")
            if num_refine_blocks < 0:
                raise ValueError(f"num_refine_blocks must be >= 0, got {num_refine_blocks}")
            self.in_ch = int(in_ch)
            self.out_ch = int(out_ch)
            self.repeat_factor = (out_ch * 8) // in_ch

            self.conv = nn.Conv3d(in_ch, out_ch * 8, kernel_size=3, padding=1)
            self.c2s = C2S3d(factor=2)
            self.refine = nn.Sequential(*[ResBlock3D(out_ch, gn_groups=gn_groups) for _ in range(num_refine_blocks)])

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            main = self.c2s(self.conv(x))
            sc = self.c2s(x.repeat_interleave(self.repeat_factor, dim=1))
            return self.refine(main + sc)

    class LatentNorm3D(nn.Module):
        """Configurable bottleneck normalization for `z` `[N, C, Dz, Hy, Wx]` or `[B, T, C, Dz, Hy, Wx]`.

        Modes:
        - `group_norm` (default): `GroupNorm` + dataset-level global scalar RMS scaling (`dataset_rms_scale`)
        - `dataset_rms`: pure global scalar division `z / dataset_rms_scale`
        - `affine_free_rmsnorm`: explicit ablation switch (per-token channel RMSNorm without affine parameters)
        """

        VALID_MODES = ("group_norm", "dataset_rms", "affine_free_rmsnorm")

        def __init__(
            self,
            channels: int = 256,
            mode: str = "group_norm",
            num_groups: int = 32,
            dataset_rms_scale: float = 1.0,
            eps: float = 1e-6,
        ) -> None:
            super().__init__()
            if mode not in self.VALID_MODES:
                raise ValueError(f"unknown latent_norm mode {mode!r}; expected one of {self.VALID_MODES}")
            if not math.isfinite(dataset_rms_scale) or dataset_rms_scale <= 0.0:
                raise ValueError(f"dataset_rms_scale must be a positive finite float, got {dataset_rms_scale}")
            if eps <= 0.0:
                raise ValueError(f"eps must be positive, got {eps}")

            self.channels = int(channels)
            self.mode = str(mode)
            self.eps = float(eps)
            self.register_buffer("dataset_rms_scale", torch.tensor(float(dataset_rms_scale), dtype=torch.float32))

            if self.mode == "group_norm":
                g = _resolve_gn_groups(self.channels, int(num_groups))
                self.gn = nn.GroupNorm(g, self.channels, eps=self.eps)
            else:
                self.gn = None

        def forward(self, z: torch.Tensor) -> torch.Tensor:
            if z.ndim not in (5, 6):
                raise ValueError(f"LatentNorm3D expects 5D or 6D tensor, got {tuple(z.shape)}")
            c_axis = -4
            if z.shape[c_axis] != self.channels:
                raise ValueError(f"expected {self.channels} latent channels at dim {c_axis}, got {z.shape[c_axis]}")
            # Cache the same FP32 representation in every ablation under AMP.
            if z.dtype in (torch.float16, torch.bfloat16) and torch.is_autocast_enabled(z.device.type):
                z = z.float()

            if self.mode == "dataset_rms":
                return z / self.dataset_rms_scale

            if self.mode == "affine_free_rmsnorm":
                rms = torch.sqrt(z.pow(2).mean(dim=c_axis, keepdim=True) + self.eps)
                return z / rms

            # group_norm mode: GroupNorm over [N, C, Dz, Hy, Wx] + dataset-level global scalar RMS scaling
            assert self.gn is not None
            if z.ndim == 6:
                b, t = z.shape[:2]
                z_flat = z.reshape(b * t, self.channels, *z.shape[-3:])
                out = self.gn(z_flat).reshape(b, t, self.channels, *z.shape[-3:])
            else:
                out = self.gn(z)
            return out / self.dataset_rms_scale

    class DCAE3DEncoder(nn.Module):
        """3D DC-AE Encoder (`f8 / C=256`): `[B, (T,) 4|6, Z, Y, X]` -> `[B*T, 256, Z//8, Y//8, X//8]`."""

        def __init__(
            self,
            in_channels: int = 6,
            base_channels: int = 32,
            channel_mults: Sequence[int] = (1, 2, 4, 8),
            latent_channels: int = 256,
            downsample_factor: int = 8,
            gn_groups: int = 8,
            multiscale_geometry: bool = True,
            sdf_tanh_alpha: float | None = 1.0,
            num_refine_blocks: int = 1,
        ) -> None:
            super().__init__()
            if in_channels != 6:
                raise ValueError(f"DCAE3DEncoder expects 6 input channels ([u,v,w,p,mask,sdf]), got {in_channels}")
            if downsample_factor != 8:
                raise ValueError(f"DCAE3DEncoder requires downsample_factor=8 (3 S2C stages), got {downsample_factor}")
            if len(channel_mults) != 4:
                raise ValueError(f"channel_mults must have 4 entries for f8 (stem + 3 downstages), got {channel_mults}")

            self.in_channels = int(in_channels)
            self.base_channels = int(base_channels)
            self.channel_mults = tuple(int(m) for m in channel_mults)
            self.latent_channels = int(latent_channels)
            self.downsample_factor = int(downsample_factor)
            self.multiscale_geometry = bool(multiscale_geometry)

            c0, c1, c2, c3 = [self.base_channels * m for m in self.channel_mults]
            self.stem_conv = nn.Conv3d(in_channels, c0, kernel_size=3, padding=1)
            self.stem_res = ResBlock3D(c0, gn_groups=gn_groups)

            self.down1 = DownBlock3D(c0, c1, gn_groups=gn_groups, num_refine_blocks=num_refine_blocks)
            self.down2 = DownBlock3D(c1, c2, gn_groups=gn_groups, num_refine_blocks=num_refine_blocks)
            self.down3 = DownBlock3D(c2, c3, gn_groups=gn_groups, num_refine_blocks=num_refine_blocks)

            if self.multiscale_geometry:
                self.geom_stem = GeometryInject3D(c0, gn_groups=gn_groups, sdf_tanh_alpha=sdf_tanh_alpha)
                self.geom_down1 = GeometryInject3D(c1, gn_groups=gn_groups, sdf_tanh_alpha=sdf_tanh_alpha)
                self.geom_down2 = GeometryInject3D(c2, gn_groups=gn_groups, sdf_tanh_alpha=sdf_tanh_alpha)
                self.geom_down3 = GeometryInject3D(c3, gn_groups=gn_groups, sdf_tanh_alpha=sdf_tanh_alpha)
            else:
                self.geom_stem = self.geom_down1 = self.geom_down2 = self.geom_down3 = None

            self.to_latent = nn.Conv3d(c3, self.latent_channels, kernel_size=1)

        def forward(
            self,
            field_or_6ch: torch.Tensor,
            geometry: torch.Tensor | None = None,
        ) -> torch.Tensor:
            if field_or_6ch.ndim not in (5, 6):
                raise ValueError(
                    f"Encoder expects 5D [B,C,Z,Y,X] or 6D [B,T,C,Z,Y,X] input, got {tuple(field_or_6ch.shape)}"
                )
            _validate_finite(field_or_6ch, "encoder_input")

            if field_or_6ch.ndim == 6:
                b, t, c, nz, ny, nx = field_or_6ch.shape
                x_flat = field_or_6ch.reshape(b * t, c, nz, ny, nx)
                t_len: int | None = t
            else:
                b, c, nz, ny, nx = field_or_6ch.shape
                x_flat = field_or_6ch
                t_len = None

            if nz % self.downsample_factor != 0 or ny % self.downsample_factor != 0 or nx % self.downsample_factor != 0:
                raise ValueError(
                    f"spatial dims {(nz, ny, nx)} must be divisible by downsample_factor={self.downsample_factor}"
                )

            if c == 4:
                if geometry is None:
                    raise ValueError("geometry [B, (T,) 2, Z, Y, X] is required when input field has 4 channels")
                geom_flat = _prepare_geometry_for_bt(geometry, b, t_len, (nz, ny, nx))
                x_6ch = torch.cat([x_flat, geom_flat], dim=1)
            elif c == 6:
                if geometry is not None:
                    geom_flat = _prepare_geometry_for_bt(geometry, b, t_len, (nz, ny, nx))
                    x_6ch = torch.cat([x_flat[:, :4], geom_flat], dim=1)
                else:
                    geom_flat = x_flat[:, 4:6]
                    x_6ch = x_flat
            else:
                raise ValueError(f"expected 4 field channels or 6 combined channels, got {c}")

            h = self.stem_res(self.stem_conv(x_6ch))
            if self.geom_stem is not None:
                h = self.geom_stem(h, geom_flat)

            h = self.down1(h)
            if self.geom_down1 is not None:
                h = self.geom_down1(h, geom_flat)

            h = self.down2(h)
            if self.geom_down2 is not None:
                h = self.geom_down2(h, geom_flat)

            h = self.down3(h)
            if self.geom_down3 is not None:
                h = self.geom_down3(h, geom_flat)

            return self.to_latent(h)

    class DCAE3DDecoder(nn.Module):
        """Independent 3D DC-AE Decoder (`C2S x3`, strictly ZERO Encoder->Decoder cross-layer skips).

        Decodes solely from `(z_clean, geometry)` -> `[B*T, base_channels, Z, Y, X]`.
        """

        def __init__(
            self,
            latent_channels: int = 256,
            base_channels: int = 32,
            channel_mults: Sequence[int] = (1, 2, 4, 8),
            downsample_factor: int = 8,
            gn_groups: int = 8,
            multiscale_geometry: bool = True,
            sdf_tanh_alpha: float | None = 1.0,
            num_refine_blocks: int = 1,
            up3_num_refine_blocks: int | None = None,
        ) -> None:
            super().__init__()
            if downsample_factor != 8:
                raise ValueError(f"DCAE3DDecoder requires downsample_factor=8 (3 C2S stages), got {downsample_factor}")
            if len(channel_mults) != 4:
                raise ValueError(f"channel_mults must have 4 entries, got {channel_mults}")

            self.latent_channels = int(latent_channels)
            self.base_channels = int(base_channels)
            self.channel_mults = tuple(int(m) for m in channel_mults)
            self.downsample_factor = int(downsample_factor)
            self.multiscale_geometry = bool(multiscale_geometry)

            c0, c1, c2, c3 = [self.base_channels * m for m in self.channel_mults]
            up3_blocks = num_refine_blocks if up3_num_refine_blocks is None else int(up3_num_refine_blocks)

            self.from_latent = nn.Conv3d(self.latent_channels, c3, kernel_size=1)
            self.latent_res = ResBlock3D(c3, gn_groups=gn_groups)

            self.up1 = UpBlock3D(c3, c2, gn_groups=gn_groups, num_refine_blocks=num_refine_blocks)
            self.up2 = UpBlock3D(c2, c1, gn_groups=gn_groups, num_refine_blocks=num_refine_blocks)
            self.up3 = UpBlock3D(c1, c0, gn_groups=gn_groups, num_refine_blocks=up3_blocks)

            if self.multiscale_geometry:
                self.geom_latent = GeometryInject3D(c3, gn_groups=gn_groups, sdf_tanh_alpha=sdf_tanh_alpha)
                self.geom_up1 = GeometryInject3D(c2, gn_groups=gn_groups, sdf_tanh_alpha=sdf_tanh_alpha)
                self.geom_up2 = GeometryInject3D(c1, gn_groups=gn_groups, sdf_tanh_alpha=sdf_tanh_alpha)
                self.geom_up3 = GeometryInject3D(c0, gn_groups=gn_groups, sdf_tanh_alpha=sdf_tanh_alpha)
            else:
                self.geom_latent = self.geom_up1 = self.geom_up2 = self.geom_up3 = None

        def forward(self, z_clean: torch.Tensor, geometry: torch.Tensor) -> torch.Tensor:
            if geometry is None:
                raise ValueError("DCAE3DDecoder requires known 2-channel geometry [fluid_mask, sdf_over_d]")
            if z_clean.ndim not in (5, 6):
                raise ValueError(
                    f"Decoder expects 5D [N,C,Dz,Hy,Wx] or 6D [B,T,C,Dz,Hy,Wx] z_clean, got {tuple(z_clean.shape)}"
                )
            _validate_finite(z_clean, "z_clean")

            if z_clean.ndim == 6:
                b, t, c, dz, hy, wx = z_clean.shape
                z_flat = z_clean.reshape(b * t, c, dz, hy, wx)
                t_len: int | None = t
            else:
                b, c, dz, hy, wx = z_clean.shape
                z_flat = z_clean
                t_len = None

            if c != self.latent_channels:
                raise ValueError(f"expected latent_channels={self.latent_channels}, got {c}")

            expected_spatial = (dz * self.downsample_factor, hy * self.downsample_factor, wx * self.downsample_factor)
            geom_flat = _prepare_geometry_for_bt(geometry, b, t_len, expected_spatial)

            h = self.latent_res(self.from_latent(z_flat))
            if self.geom_latent is not None:
                h = self.geom_latent(h, geom_flat)

            h = self.up1(h)
            if self.geom_up1 is not None:
                h = self.geom_up1(h, geom_flat)

            h = self.up2(h)
            if self.geom_up2 is not None:
                h = self.geom_up2(h, geom_flat)

            h = self.up3(h)
            if self.geom_up3 is not None:
                h = self.geom_up3(h, geom_flat)

            return h


__all__ = [
    "C2S3d",
    "DCAE3DDecoder",
    "DCAE3DEncoder",
    "DownBlock3D",
    "GeometryInject3D",
    "LatentNorm3D",
    "ResBlock3D",
    "S2C3d",
    "UpBlock3D",
    "torch_available",
]
