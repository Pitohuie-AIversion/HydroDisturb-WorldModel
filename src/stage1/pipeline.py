from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import LatentContract, Stage1Config
from .data.normalization import FlowNormalizer, NormalizationStats
from .models.dcae import Stage1DC_AE, torch_available


@dataclass
class Stage1Output:
    z_clean: Any
    reconstruction: Any
    stats: NormalizationStats
    contract: LatentContract


class Stage1Autoencoder:
    """High-level Stage 1 entry point used by training and offline export."""

    def __init__(self, config: Stage1Config | None = None, model: Any | None = None):
        self.config = config or Stage1Config()
        self.normalizer = FlowNormalizer(sdf_clip=self.config.grid.sdf_clip, version=self.config.normalizer_version)
        if model is None:
            if not torch_available():
                raise ImportError("Stage1Autoencoder model path requires PyTorch; install `pip install -e .[torch]`.")
            model = Stage1DC_AE(
                latent_channels=self.config.model.latent_channels,
                base_channels=self.config.model.base_channels,
                hard_mask_output=self.config.model.hard_mask_output,
            )
        self.model = model

    def latent_contract(self, latent_shape: tuple[int, int, int, int]) -> LatentContract:
        return LatentContract(
            version=self.config.latent_contract_version,
            shape=latent_shape,
            normalizer_version=self.config.normalizer_version,
        )

    def forward(self, frame: Any, stats: NormalizationStats | None = None) -> Stage1Output:
        """Normalize a NumPy frame, then run deterministic `z_clean` inference."""
        normalized, stats = self.normalizer.normalize(frame, stats)
        if not torch_available():
            raise ImportError("forward requires PyTorch; install `pip install -e .[torch]`.")
        import torch

        tensor = torch.from_numpy(normalized).float()
        if tensor.ndim == 4:
            tensor = tensor.unsqueeze(0)
        self.model.eval()
        with torch.no_grad():
            result = self.model(tensor)
        z = result["z_clean"]
        contract = self.latent_contract(tuple(int(v) for v in z.shape))
        return Stage1Output(z_clean=z, reconstruction=result["reconstruction"], stats=stats, contract=contract)
