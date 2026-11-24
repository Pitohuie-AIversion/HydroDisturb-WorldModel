"""Optional clean-latent physics probe (`LatentPhysicsProbe`).

Reads ONLY clean latent `z_clean` (`[B, (T,) C, Dz, Hy, Wx]`) and never feeds back
into the decoder. Pools spatial tokens via `concat(MeanPool, MaxPool)` -> MLP predicting
dimensionless enstrophy and normalized log radial energy spectrum.
"""

from __future__ import annotations

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
        raise ImportError("flow3d.models.probe requires PyTorch. Install with `pip install -e .[torch]`.")


if torch is None:

    class LatentPhysicsProbe:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            _require_torch()

else:

    class LatentPhysicsProbe(nn.Module):
        """Predicts dimensionless enstrophy and log radial energy spectrum from `z_clean`."""

        def __init__(
            self,
            latent_channels: int = 256,
            hidden_dim: int = 256,
            spectrum_bins: int = 16,
        ) -> None:
            super().__init__()
            if latent_channels <= 0 or hidden_dim <= 0 or spectrum_bins <= 0:
                raise ValueError("latent_channels, hidden_dim, and spectrum_bins must be positive")
            self.latent_channels = int(latent_channels)
            self.hidden_dim = int(hidden_dim)
            self.spectrum_bins = int(spectrum_bins)

            in_dim = 2 * self.latent_channels
            self.trunk = nn.Sequential(
                nn.Linear(in_dim, self.hidden_dim),
                nn.SiLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.SiLU(),
            )
            self.enstrophy_head = nn.Linear(self.hidden_dim, 1)
            self.spectrum_head = nn.Linear(self.hidden_dim, self.spectrum_bins)

        def forward(self, z_clean: torch.Tensor) -> dict[str, torch.Tensor]:
            if z_clean.ndim not in (5, 6):
                raise ValueError(
                    f"LatentPhysicsProbe expects 5D [N,C,Dz,Hy,Wx] or 6D [B,T,C,Dz,Hy,Wx], got {tuple(z_clean.shape)}"
                )
            if z_clean.shape[-4] != self.latent_channels:
                raise ValueError(
                    f"expected latent_channels={self.latent_channels} at dim -4, got {z_clean.shape[-4]}"
                )
            if not torch.isfinite(z_clean).all():
                raise ValueError("LatentPhysicsProbe received non-finite z_clean")

            mean_pool = z_clean.mean(dim=(-3, -2, -1))
            max_pool = z_clean.amax(dim=(-3, -2, -1))
            pooled = torch.cat([mean_pool, max_pool], dim=-1)
            feat = self.trunk(pooled)
            enstrophy_tilde = F.softplus(self.enstrophy_head(feat))
            log_spectrum_tilde = self.spectrum_head(feat)
            return {
                "enstrophy_tilde": enstrophy_tilde,
                "log_spectrum_tilde": log_spectrum_tilde,
            }


__all__ = ["LatentPhysicsProbe", "torch_available"]
