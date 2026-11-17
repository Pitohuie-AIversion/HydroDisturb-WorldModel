from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import Stage1Config
from .models.dcae import Stage1DC_AE, torch_available


@dataclass
class TrainStepResult:
    loss: float
    reconstruction: float
    divergence: float
    vorticity: float


if torch_available():
    import torch
    import torch.nn.functional as F
    from .physics.torch_operators import torch_divergence
else:  # pragma: no cover - only used when torch is unavailable
    torch = None  # type: ignore[assignment]


class Stage1Trainer:
    """Minimal reconstruction-first trainer with explicit auxiliary-loss switches."""

    def __init__(self, model: Any | None = None, config: Stage1Config | None = None, optimizer: Any | None = None):
        if not torch_available():
            raise ImportError("Stage1Trainer requires PyTorch; install `pip install -e .[torch]`.")
        self.config = config or Stage1Config()
        self.model = model or Stage1DC_AE(
            latent_channels=self.config.model.latent_channels,
            base_channels=self.config.model.base_channels,
            hard_mask_output=self.config.model.hard_mask_output,
        )
        self.optimizer = optimizer or torch.optim.AdamW(self.model.parameters(), lr=2e-4, betas=(0.9, 0.95), weight_decay=1e-4)
        self.scheduler = None
        self.scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    @staticmethod
    def _masked_mse(pred, target, mask):
        mask = mask.expand_as(pred)
        return ((pred - target).square() * mask).sum() / mask.sum().clamp_min(1.0)

    def train_step(self, batch: Any) -> TrainStepResult:
        if batch.ndim != 5 or batch.shape[1] != 6:
            raise ValueError("batch must have shape (B,6,D,H,W)")
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=batch.device.type, enabled=self.scaler.is_enabled()):
            output = self.model(batch)
            pred = output["reconstruction"]
            mask = batch[:, 4:5].detach().ge(0.5).to(pred.dtype)
            rec = self._masked_mse(pred[:, :4], batch[:, :4], mask)
            div = torch_divergence(pred[:, :3], batch[:, 4:5])
            div_loss = div.square().mean()
            # z-vorticity is intentionally an opt-in placeholder until the exact
            # grid-axis convention is chosen for the training dataset.
            vort_loss = pred.new_zeros(())
            loss = (
                self.config.loss.reconstruction_weight * rec
                + self.config.loss.divergence_weight * div_loss
                + self.config.loss.vorticity_weight * vort_loss
            )
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        return TrainStepResult(float(loss.detach()), float(rec.detach()), float(div_loss.detach()), float(vort_loss.detach()))
