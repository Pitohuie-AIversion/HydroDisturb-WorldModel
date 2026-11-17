from __future__ import annotations

from typing import Optional

try:  # PyTorch is optional for protocol-only development and CI.
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception:  # pragma: no cover - exercised in environments without torch
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]


def torch_available() -> bool:
    return torch is not None


def _require_torch() -> None:
    if torch is None:
        raise ImportError("Stage1DC_AE requires PyTorch. Install with `pip install -e .[torch]`.")


if torch is None:

    class Stage1Encoder:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            _require_torch()

    class Stage1Decoder:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            _require_torch()

    class Stage1DC_AE:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            _require_torch()

else:

    class GeometryInject(nn.Module):
        def __init__(self, hidden: int):
            super().__init__()
            self.proj = nn.Conv3d(2, hidden, kernel_size=1)
            self.norm = nn.GroupNorm(max(1, min(8, hidden // 4)), hidden)

        def forward(self, x: torch.Tensor, geometry: torch.Tensor) -> torch.Tensor:
            g = F.interpolate(geometry, size=x.shape[-3:], mode="trilinear", align_corners=False)
            return x + self.norm(self.proj(g))


    class Stage1Encoder(nn.Module):
        """Compact 3D DC-AE encoder with f8 spatial downsampling.

        The production architecture can replace the residual blocks without changing
        the latent or geometry contracts exposed by this module.
        """

        def __init__(self, in_channels: int = 6, latent_channels: int = 256, base_channels: int = 32, downsample_factor: int = 8):
            super().__init__()
            if downsample_factor != 8:
                raise ValueError("the current 3D encoder implementation supports downsample_factor=8")
            c1, c2, c3 = base_channels, base_channels * 2, base_channels * 4
            self.block1 = nn.Sequential(nn.Conv3d(in_channels, c1, 3, stride=2, padding=1), nn.GroupNorm(8, c1), nn.SiLU())
            self.block2 = nn.Sequential(nn.Conv3d(c1, c2, 3, stride=2, padding=1), nn.GroupNorm(8, c2), nn.SiLU())
            self.block3 = nn.Sequential(nn.Conv3d(c2, c3, 3, stride=2, padding=1), nn.GroupNorm(8, c3), nn.SiLU())
            self.geometry1 = GeometryInject(c1)
            self.geometry2 = GeometryInject(c2)
            self.geometry3 = GeometryInject(c3)
            self.to_latent = nn.Conv3d(c3, latent_channels, 3, padding=1)
            self.latent_norm = nn.GroupNorm(32 if latent_channels % 32 == 0 else 1, latent_channels)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if x.ndim != 5 or x.shape[1] != 6:
                raise ValueError("encoder expects (B,6,D,H,W)")
            geometry = x[:, 4:6]
            h = self.geometry1(self.block1(x), geometry)
            h = self.geometry2(self.block2(h), geometry)
            h = self.geometry3(self.block3(h), geometry)
            return self.latent_norm(self.to_latent(h))


    class Stage1Decoder(nn.Module):
        def __init__(self, latent_channels: int = 256, base_channels: int = 32, hard_mask_output: bool = True):
            super().__init__()
            c3, c2, c1 = base_channels * 4, base_channels * 2, base_channels
            self.hard_mask_output = hard_mask_output
            self.from_latent = nn.Sequential(nn.Conv3d(latent_channels, c3, 3, padding=1), nn.GroupNorm(8, c3), nn.SiLU())
            self.geometry3 = GeometryInject(c3)
            self.geometry2 = GeometryInject(c2)
            self.geometry1 = GeometryInject(c1)
            self.up1 = nn.Sequential(nn.Conv3d(c3, c2, 3, padding=1), nn.GroupNorm(8, c2), nn.SiLU())
            self.up2 = nn.Sequential(nn.Conv3d(c2, c1, 3, padding=1), nn.GroupNorm(8, c1), nn.SiLU())
            self.up3 = nn.Sequential(nn.Conv3d(c1, c1, 3, padding=1), nn.GroupNorm(8, c1), nn.SiLU())
            self.head = nn.Conv3d(c1, 6, 3, padding=1)

        def forward(self, z: torch.Tensor, geometry: Optional[torch.Tensor] = None) -> torch.Tensor:
            if z.ndim != 5:
                raise ValueError("decoder expects latent tensor (B,C,D,H,W)")
            h = self.from_latent(z)
            if geometry is not None:
                h = self.geometry3(h, geometry)
            h = self.up1(F.interpolate(h, scale_factor=2, mode="trilinear", align_corners=False))
            if geometry is not None:
                h = self.geometry2(h, geometry)
            h = self.up2(F.interpolate(h, scale_factor=2, mode="trilinear", align_corners=False))
            if geometry is not None:
                h = self.geometry1(h, geometry)
            h = self.up3(F.interpolate(h, scale_factor=2, mode="trilinear", align_corners=False))
            out = self.head(h)
            fields = out[:, :4]
            mask = torch.sigmoid(out[:, 4:5])
            sdf = torch.tanh(out[:, 5:6])
            if self.hard_mask_output:
                fields = fields * (mask >= 0.5).to(fields.dtype)
            return torch.cat((fields, mask, sdf), dim=1)


    class Stage1DC_AE(nn.Module):
        """Encoder/decoder pair returning `z_clean` and six-channel reconstruction."""

        def __init__(self, latent_channels: int = 256, base_channels: int = 32, hard_mask_output: bool = True):
            super().__init__()
            self.encoder = Stage1Encoder(latent_channels=latent_channels, base_channels=base_channels)
            self.decoder = Stage1Decoder(latent_channels=latent_channels, base_channels=base_channels, hard_mask_output=hard_mask_output)

        def encode(self, x: torch.Tensor) -> torch.Tensor:
            return self.encoder(x)

        def decode(self, z: torch.Tensor, x_or_geometry: Optional[torch.Tensor] = None) -> torch.Tensor:
            geometry = None if x_or_geometry is None else x_or_geometry[:, 4:6]
            return self.decoder(z, geometry)

        def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
            z = self.encode(x)
            return {"z_clean": z, "reconstruction": self.decode(z, x)}
