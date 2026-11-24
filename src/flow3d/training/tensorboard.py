"""TensorBoard logging for Stage 1 optimizer updates and epoch records."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import AbstractContextManager
from datetime import datetime, timezone
from pathlib import Path
from tempfile import mkdtemp
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .trainer import Flow3DTrainMetrics


_EPOCH_TAGS = {
    "train_loss": "train/loss",
    "train_loss_vel": "train/velocity_loss",
    "train_loss_press": "train/pressure_loss",
    "train_loss_aux": "train/auxiliary_loss",
    "val_loss": "validation/loss",
    "val_loss_vel": "validation/velocity_loss",
    "val_loss_press": "validation/pressure_loss",
    "rmse_uvw_ms": "validation/velocity_rmse_m_s",
    "rmse_p_pa": "validation/pressure_rmse_pa",
    "div_rms_phys": "validation/divergence_rms_s_inv",
    "near_wall_vel_rmse_ms": "validation/near_wall_velocity_rmse_m_s",
    "solid_max_speed": "validation/solid_max_abs_velocity_component_m_s",
    "global_step": "optimization/updates",
    "learning_rate": "optimization/learning_rate",
    "next_learning_rate": "optimization/next_learning_rate",
    "epoch_time_s": "timing/epoch_seconds",
}


class Stage1TensorBoard(AbstractContextManager):
    """A separate event directory for each invocation, including checkpoint branches."""

    def __init__(self, parent_dir: str | Path, head_name: str) -> None:
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ModuleNotFoundError as exc:
            if exc.name != "tensorboard":
                raise
            raise ImportError(
                "Stage 1 TensorBoard logging requires tensorboard. "
                "Install it with 'uv sync --locked --extra all' or pass --no-tensorboard."
            ) from exc

        parent = Path(parent_dir).resolve()
        parent.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.log_dir = Path(mkdtemp(dir=parent, prefix=f"{head_name}-{timestamp}-"))
        self.writer = SummaryWriter(log_dir=str(self.log_dir), flush_secs=10)

    def log_step(self, step: int, metrics: Flow3DTrainMetrics) -> None:
        self.writer.add_scalar("step/loss", metrics.total_loss, step)
        self.writer.add_scalar("step/grad_norm", metrics.grad_norm, step)
        self.writer.add_scalar("step/learning_rate", metrics.learning_rate, step)

    def log_epoch(self, record: Mapping[str, Any]) -> None:
        epoch = record["epoch"]
        for key, tag in _EPOCH_TAGS.items():
            if key in record:
                self.writer.add_scalar(tag, record[key], epoch)
        self.writer.flush()

    def __exit__(self, *exc_info) -> None:
        self.writer.close()
