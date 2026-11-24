"""Stage 1 3D DC-AE Trainer with multi-GPU elastic support, physics loss weighting, and atomic checkpoints."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import dataclass, field
import json
import logging
import math
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np

from flow3d.contracts import CYLINDER3D_STATE_SPEC
from flow3d.models import (
    Flow3DStage1Autoencoder,
    configure_elastic_hardware,
    resolve_elastic_runtime_plan,
)
from flow3d.physics.boundary import first_fluid_shell_mask
from flow3d.physics.torch_operators import torch_div_h
from stage1.io.checkpoint import CheckpointManager, CheckpointStateError

try:
    import torch
    from torch.utils.data import DataLoader
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    DataLoader = None  # type: ignore[assignment]

logger = logging.getLogger("flow3d.training")


def _require_torch() -> Any:
    if torch is None:
        raise ImportError("PyTorch is required for Flow3DStage1Trainer")
    return torch


@dataclass
class Flow3DTrainMetrics:
    """Detailed loss & optimization metrics for a single training step or epoch average."""

    total_loss: float
    loss_vel: float
    loss_press: float
    loss_aux: float
    grad_norm: float
    learning_rate: float
    step_time_ms: float
    div_rms_fluid: float = 0.0


@dataclass
class Flow3DEvalMetrics:
    """Comprehensive physical validation metrics in SI units and normalized space."""

    val_loss: float
    val_loss_vel: float
    val_loss_press: float
    rmse_u_ms: float
    rmse_v_ms: float
    rmse_w_ms: float
    rmse_uvw_ms: float
    rmse_p_pa: float
    div_rms_phys_s_inv: float
    solid_interior_max_abs: float
    eval_time_ms: float
    extra: dict[str, float] = field(default_factory=dict)
    near_wall_vel_rmse_ms: float = 0.0


def _resolve_amp_dtype(amp_dtype_str: str) -> Any:
    t = _require_torch()
    s = str(amp_dtype_str).strip().lower()
    if s == "bf16":
        return t.bfloat16
    if s == "fp16":
        return t.float16
    if s == "fp32":
        return None
    raise ValueError(f"unsupported amp_dtype: {amp_dtype_str!r}; expected 'bf16', 'fp16', or 'fp32'")


class Flow3DStage1Trainer:
    """Stage 1 autoencoder trainer for FlowWorld-3D.

    - Supports elastic n-GPU scaling (1..N CUDA devices) or CPU execution.
    - Full physics loss weighting: velocity MSE, pressure MSE, soft divergence penalty (Head-A),
      Coulomb gauge penalty & harmonic Laplacian penalty (Head-B).
    - Periodic physical metric validation in SI (RMSE in m/s & Pa, divergence RMS, boundary diagnostics).
    - Atomic, resumable checkpointing via CheckpointManager.
    """

    def __init__(
        self,
        model: Flow3DStage1Autoencoder,
        config: Mapping[str, Any] | None = None,
        optimizer: Any | None = None,
        scheduler: Any | None = None,
        checkpoint_manager: CheckpointManager | None = None,
        device: str | Any = "auto",
        num_gpus: int | str = "auto",
        amp: bool = True,
        amp_dtype: str = "bf16",
        clip_grad_norm: float = 1.0,
        save_dir: str | Path = "checkpoints/stage1",
    ) -> None:
        _require_torch()
        self.config = dict(config or {})
        self.clip_grad_norm = float(clip_grad_norm)

        # 1. Hardware & Elastic Runtime Configuration
        runtime_cfg = self.config.get("runtime", {})
        self.gather_trainer_outputs_only = bool(runtime_cfg.get("gather_trainer_outputs_only", False))
        req_gpus = num_gpus if num_gpus != "auto" else runtime_cfg.get("num_gpus", "auto")
        self.plan = resolve_elastic_runtime_plan(
            self.config,
            num_gpus=req_gpus,
        )
        self.hw_config = configure_elastic_hardware(self.plan)

        if str(device).lower() == "auto":
            self.device = torch.device(self.plan.primary_device)
        else:
            self.device = torch.device(device)

        self.num_active_gpus = self.plan.active_gpus
        self.device_ids = self.plan.device_ids

        # 2. Model setup
        self.model = model.to(self.device)

        # 3. Precision (AMP)
        self.use_amp = bool(amp and self.device.type == "cuda")
        self.amp_dtype_str = str(amp_dtype or runtime_cfg.get("amp_dtype", "bf16"))
        self.amp_torch_dtype = _resolve_amp_dtype(self.amp_dtype_str)
        if self.use_amp and self.amp_torch_dtype == torch.float16:
            self.scaler = torch.amp.GradScaler("cuda")
        else:
            self.scaler = None

        # 4. Loss Weights
        loss_cfg = self.config.get("loss", {})
        self.w_recon = float(loss_cfg.get("reconstruction_weight", 1.0))
        self.w_press = float(loss_cfg.get("pressure_weight", 1.0))
        self.w_div = float(loss_cfg.get("divergence_weight", 0.0))
        self.w_coulomb = float(loss_cfg.get("coulomb_gauge_weight", 0.01))
        self.w_harm = float(loss_cfg.get("harmonic_laplacian_weight", 0.01))

        # 5. Optimizer & Scheduler
        train_cfg = self.config.get("training", {})
        lr = float(train_cfg.get("learning_rate", 2e-4))
        weight_decay = float(train_cfg.get("weight_decay", 1e-4))
        betas = tuple(float(b) for b in train_cfg.get("betas", (0.9, 0.95)))

        self.optimizer = optimizer or torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
        )
        self.scheduler = scheduler
        self.adaptive = train_cfg.get("scheduler") == "warmup_plateau"
        self.base_lrs = [float(group["lr"]) for group in self.optimizer.param_groups]
        self.warmup_start_lr = float(train_cfg.get("warmup_start_lr", 1e-5))
        self.warmup_epochs = int(train_cfg.get("warmup_epochs", 8))
        self.warmup_steps: int | None = None
        self.min_epochs = int(train_cfg.get("min_epochs", 60))
        self.stop_patience = int(train_cfg.get("early_stop_patience", 30))
        self.stop_threshold = float(train_cfg.get("early_stop_threshold", 0.005))
        if self.adaptive:
            patience = int(train_cfg.get("plateau_patience", 12))
            if self.warmup_epochs < 0 or self.warmup_start_lr <= 0 or self.min_epochs < 0:
                raise ValueError("invalid warmup or minimum epoch settings")
            if self.stop_patience <= patience or patience < 0 or not 0 <= self.stop_threshold < 1:
                raise ValueError("early stopping requires a longer patience than the LR plateau")
            if scheduler is not None:
                raise ValueError("warmup_plateau constructs its own ReduceLROnPlateau")
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="min", factor=float(train_cfg.get("plateau_factor", 0.5)),
                patience=patience, threshold=float(train_cfg.get("plateau_threshold", 1e-3)),
                threshold_mode="rel", min_lr=float(train_cfg.get("min_lr", 5e-6)),
            )
        self.ema_decay = float(train_cfg.get("ema_decay", 0.995))
        self.ema_model = deepcopy(self.model).requires_grad_(False).eval() if train_cfg.get("ema", False) else None
        if self.ema_model is not None and not 0 <= self.ema_decay < 1:
            raise ValueError("ema_decay must be in [0, 1)")

        # 6. Checkpoint Manager
        self.save_dir = Path(save_dir or train_cfg.get("save_dir", "checkpoints/stage1"))
        self.checkpoint_manager = checkpoint_manager or CheckpointManager(
            save_dir=self.save_dir,
            model_name="dcae3d_stage1",
        )

        self.global_step = 0
        self.current_epoch = 0
        self.best_val_loss = float("inf")
        self.history: list[dict[str, Any]] = []
        self.progress_loss = float("inf")
        self.last_progress_epoch = 0
        self.stop_reason: str | None = None
        self.best_pointer: dict[str, Any] | None = None

    def _forward_model(
        self,
        norm_field: torch.Tensor,
        geometry: torch.Tensor,
        b_stats: torch.Tensor | None,
        spacing_star: Sequence[float],
        model: Any | None = None,
    ) -> dict[str, Any]:
        """Forward through single model or elastic multi-GPU."""
        bsz = int(norm_field.shape[0])
        model = self.model if model is None else model
        if self.num_active_gpus >= 2 and bsz >= 2:
            gather_options = {}
            if self.gather_trainer_outputs_only:
                gather_options["gather_for"] = "train" if model.training else "validation"
            return model.forward_elastic_multi_gpu(
                norm_field,
                geometry,
                b=b_stats,
                spacing_zyx=spacing_star,
                num_gpus=self.num_active_gpus,
                device_ids=self.device_ids,
                **gather_options,
            )
        return model(
            norm_field,
            geometry,
            b=b_stats,
            spacing_zyx=spacing_star,
        )

    def train_step(self, batch: dict[str, Any]) -> Flow3DTrainMetrics:
        """Executes one training step with forward, loss computation, backward & optimization."""
        t_start = time.perf_counter()
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        norm_field = batch["norm_field"].to(self.device, non_blocking=True)
        geometry = batch["geometry"].to(self.device, non_blocking=True)
        b_stats = batch["b"].to(self.device, non_blocking=True) if "b" in batch else None
        default_spacing_star = tuple(
            float(v) / CYLINDER3D_STATE_SPEC.cylinder_diameter_m
            for v in CYLINDER3D_STATE_SPEC.default_spacing_m
        )
        spacing_star = batch.get("spacing_star", default_spacing_star)

        amp_active = bool(self.use_amp and self.amp_torch_dtype is not None)
        amp_ctx = (
            torch.amp.autocast(device_type="cuda", dtype=self.amp_torch_dtype)
            if amp_active
            else nullcontext()
        )

        with amp_ctx:
            out = self._forward_model(norm_field, geometry, b_stats, spacing_star)
            pred_norm = out["norm_fields"]
            fluid_mask = geometry[:, 0:1] > 0.5
            mask_count = fluid_mask.sum().clamp_min(1)

            # 1. Masked Reconstruction Losses (in normalized space)
            # Velocity: channels 0, 1, 2
            diff_vel_sq = (pred_norm[:, :3] - norm_field[:, :3]).pow(2)
            loss_vel = (diff_vel_sq * fluid_mask).sum() / mask_count

            # Pressure: channel 3
            diff_p_sq = (pred_norm[:, 3:4] - norm_field[:, 3:4]).pow(2)
            loss_press = (diff_p_sq * fluid_mask).sum() / mask_count

            # 2. Auxiliary Head Losses
            head_aux = out.get("head_aux", {})
            loss_aux = torch.zeros((), dtype=loss_vel.dtype, device=loss_vel.device)

            div_loss = head_aux.get("div_loss")
            if div_loss is not None and self.w_div > 0:
                loss_aux = loss_aux + self.w_div * div_loss

            coulomb_loss = head_aux.get("coulomb_gauge_loss")
            if coulomb_loss is not None and self.w_coulomb > 0:
                loss_aux = loss_aux + self.w_coulomb * coulomb_loss

            harm_loss = head_aux.get("harmonic_laplacian_loss")
            if harm_loss is not None and self.w_harm > 0:
                loss_aux = loss_aux + self.w_harm * harm_loss

            total_loss = self.w_recon * loss_vel + self.w_press * loss_press + loss_aux

        # 3. Backward Pass & Optimizer Step
        updated = True
        if self.scaler is not None:
            self.scaler.scale(total_loss).backward()
            self.scaler.unscale_(self.optimizer)
            grad_norm = float(torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm).item())
            old_scale = self.scaler.get_scale()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            updated = self.scaler.get_scale() >= old_scale
        else:
            total_loss.backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm).item())
            self.optimizer.step()

        if updated:
            self.global_step += 1
            if self.ema_model is not None:
                decay = min(self.ema_decay, 1.0 - 1.0 / (self.global_step + 10))
                with torch.no_grad():
                    for avg, online in zip(self.ema_model.parameters(), self.model.parameters()):
                        avg.lerp_(online, 1.0 - decay)
                    for avg, online in zip(self.ema_model.buffers(), self.model.buffers()):
                        avg.copy_(online)
        step_ms = (time.perf_counter() - t_start) * 1000.0

        current_lr = float(self.optimizer.param_groups[0]["lr"])

        return Flow3DTrainMetrics(
            total_loss=float(total_loss.item()),
            loss_vel=float(loss_vel.item()),
            loss_press=float(loss_press.item()),
            loss_aux=float(loss_aux.item()),
            grad_norm=grad_norm,
            learning_rate=current_lr,
            step_time_ms=step_ms,
        )

    def evaluate(self, val_loader: DataLoader) -> Flow3DEvalMetrics:
        """Evaluates model performance across validation set in physical and normalized units."""
        t_start = time.perf_counter()
        eval_model = self.model if self.ema_model is None else self.ema_model
        eval_model.eval()

        norm_sq = torch.zeros(4, dtype=torch.float64, device=self.device)
        phys_sq = torch.zeros_like(norm_sq)
        fluid_count = torch.zeros((), dtype=torch.long, device=self.device)
        phys_count = torch.zeros_like(fluid_count)
        shell_count = torch.zeros_like(fluid_count)
        shell_sq = norm_sq.new_zeros(())
        div_sq = norm_sq.new_zeros(())
        solid_max = norm_sq.new_zeros(())
        default_spacing = CYLINDER3D_STATE_SPEC.default_spacing_m
        default_star = tuple(h / CYLINDER3D_STATE_SPEC.cylinder_diameter_m for h in default_spacing)
        amp = self.use_amp and self.amp_torch_dtype is not None
        with torch.no_grad():
            for batch in val_loader:
                norm_field = batch["norm_field"].to(self.device, non_blocking=True)
                geometry = batch["geometry"].to(self.device, non_blocking=True)
                b_stats = batch["b"].to(self.device, non_blocking=True) if "b" in batch else None
                with torch.amp.autocast("cuda", dtype=self.amp_torch_dtype, enabled=amp):
                    out = self._forward_model(
                        norm_field, geometry, b_stats, batch.get("spacing_star", default_star), eval_model
                    )
                fluid = geometry[:, :1] > 0.5
                count = fluid.sum()
                fluid_count += count
                dims = (0, 2, 3, 4)
                error_sq = (out["norm_fields"].float() - norm_field).square()
                norm_sq += (error_sq * fluid).sum(dims, dtype=torch.float64)
                if out["phys_fields"] is None:
                    continue

                phys = out["phys_fields"].float()
                target = batch["phys_field"].to(self.device, non_blocking=True)
                error_sq = (phys - target).square()
                phys_sq += (error_sq * fluid).sum(dims, dtype=torch.float64)
                phys_count += count
                div = torch_div_h(phys[:, 0], phys[:, 1], phys[:, 2],
                                  batch.get("spacing_m", default_spacing), mask=fluid[:, 0])
                div_sq += (div.square() * fluid[:, 0]).sum(dtype=torch.float64)
                solid_max = torch.maximum(solid_max, phys[:, :3].abs().masked_fill(fluid, 0).amax())

                masks = batch["geometry"][:, 0].detach().cpu().numpy()
                shell = torch.from_numpy(np.stack([first_fluid_shell_mask(m) for m in masks])).to(self.device)
                shell_sq += (error_sq[:, :3].sum(dim=1) * shell).sum(dtype=torch.float64)
                shell_count += shell.sum()

        if fluid_count.item() == 0:
            raise ValueError("validation dataset contains no fluid cells")
        mse = (norm_sq / fluid_count).tolist()
        vel_loss, p_loss = sum(mse[:3]), mse[3]
        phys_mse = (phys_sq / phys_count.clamp_min(1)).tolist()
        return Flow3DEvalMetrics(
            val_loss=self.w_recon * vel_loss + self.w_press * p_loss,
            val_loss_vel=vel_loss, val_loss_press=p_loss,
            rmse_u_ms=math.sqrt(phys_mse[0]), rmse_v_ms=math.sqrt(phys_mse[1]),
            rmse_w_ms=math.sqrt(phys_mse[2]), rmse_uvw_ms=math.sqrt(sum(phys_mse[:3])),
            rmse_p_pa=math.sqrt(phys_mse[3]),
            div_rms_phys_s_inv=math.sqrt((div_sq / phys_count.clamp_min(1)).item()),
            near_wall_vel_rmse_ms=math.sqrt((shell_sq / shell_count.clamp_min(1)).item()),
            solid_interior_max_abs=solid_max.item(),
            eval_time_ms=(time.perf_counter() - t_start) * 1000.0,
        )

    def _validate_resume(self, payload: Mapping[str, Any]) -> None:
        saved_cfg = payload.get("config", {})
        saved = saved_cfg.get("training", {})
        current = self.config.get("training", {})
        if payload.get("model_name") != self.checkpoint_manager.model_name:
            raise CheckpointStateError("checkpoint model name differs from this run")
        if (saved.get("scheduler") == "warmup_plateau") != self.adaptive:
            raise CheckpointStateError("checkpoint scheduler differs from requested schedule")
        if (payload.get("ema_state_dict") is not None) != (self.ema_model is not None):
            raise CheckpointStateError("checkpoint EMA mode differs from requested evaluation weights")
        for key in ("head", "loss", "grid", "normalizer", "encoder", "decoder", "spatial_transformer", "latent_norm"):
            if saved_cfg.get(key) != self.config.get(key):
                raise CheckpointStateError(f"checkpoint config differs for {key}")
        for key in (
            "split_sha256", "split_strategy", "purge_gap_frames", "train_ratio", "val_ratio", "test_ratio",
            "seed", "batch_size", "amp", "amp_dtype", "learning_rate", "weight_decay", "betas", "clip_grad_norm",
            "warmup_epochs", "warmup_start_lr", "plateau_factor", "plateau_patience", "plateau_threshold",
            "min_lr", "ema_decay", "min_epochs", "early_stop_patience", "early_stop_threshold",
        ):
            if key in saved and saved[key] != current.get(key):
                raise CheckpointStateError(f"checkpoint training config differs for {key}")
        if self.adaptive and "flow3d_trainer" not in payload.get("extra_state", {}):
            raise CheckpointStateError("checkpoint lacks adaptive trainer state")

    def resume(self, checkpoint: str | Path, train_loader: DataLoader) -> None:
        """Restore training at an epoch boundary. The requested epoch budget may be extended."""
        payload = self.checkpoint_manager.load_checkpoint(
            checkpoint, self.model, self.optimizer, self.scheduler, self.scaler,
            validate_payload=self._validate_resume, map_location="cpu",
        )
        if self.ema_model is not None:
            self.ema_model.load_state_dict(payload["ema_state_dict"])
        self.current_epoch = int(payload["epoch"])
        self.global_step = int(payload["global_step"])
        src = Path(payload["loaded_path"])
        state = payload.get("extra_state", {}).get("flow3d_trainer")
        if state is None:
            # Older cosine runs saved their history at completion in a separate summary.
            summary = src.parent / "training_summary.json"
            records = json.loads(summary.read_text())["history"] if summary.is_file() else []
            self.history = [r for r in records if r["epoch"] <= self.current_epoch]
            if not self.history or self.history[-1]["epoch"] < self.current_epoch:
                self.history.append(dict(payload["metrics"]))
            self.best_val_loss = min((r["val_loss"] for r in self.history if "val_loss" in r), default=math.inf)
            pointer_path = src.parent / self.checkpoint_manager.best_pointer_path.name
            previous_best = json.loads(pointer_path.read_text()) if pointer_path.is_file() else None
            if previous_best:
                origin = CheckpointManager(src.parent, model_name=self.checkpoint_manager.model_name)
                best = origin.verify_checkpoint("best")
                if best["epoch"] <= self.current_epoch:
                    self.best_val_loss = min(self.best_val_loss, best["metrics"]["val_loss"])
                else:
                    previous_best = None
            is_best = payload["metrics"].get("val_loss") == self.best_val_loss
            logger.warning("Resuming a legacy checkpoint with the available epoch history")
        else:
            self.history = list(state["history"])
            self.best_val_loss = state["best_val_loss"] if state["best_val_loss"] is not None else math.inf
            self.progress_loss = state["progress_loss"] if state["progress_loss"] is not None else math.inf
            self.last_progress_epoch = state["last_progress_epoch"]
            self.warmup_steps = state["warmup_steps"]
            self.base_lrs = state["base_lrs"]
            self.stop_reason = state["stop_reason"]
            previous_best, is_best = state["previous_best"], state["is_best"]
            for name, owner in (("dataloader", train_loader), ("sampler", getattr(train_loader, "sampler", None))):
                saved_rng = payload.get(f"{name}_state")
                generator = getattr(owner, "generator", None)
                if (saved_rng is None) != (generator is None):
                    raise CheckpointStateError(f"{name} RNG setup differs from checkpoint")
                if generator is not None:
                    generator.set_state(saved_rng.cpu())

        if is_best or previous_best is not None:
            path = src if is_best else src.parent / previous_best["checkpoint"]
            digest = payload["loaded_sha256"] if is_best else previous_best["sha256"]
            self.checkpoint_manager.restore_best_pointer(path, digest, payload["contract_sha256"])
            self.best_pointer = json.loads(self.checkpoint_manager.best_pointer_path.read_text())

    def _save_epoch(self, train_loader: DataLoader, record: dict[str, Any], is_best: bool) -> None:
        loader_rng = getattr(train_loader, "generator", None)
        sampler_rng = getattr(getattr(train_loader, "sampler", None), "generator", None)
        state = {
            "history": self.history, "best_val_loss": self.best_val_loss if math.isfinite(self.best_val_loss) else None,
            "progress_loss": self.progress_loss if math.isfinite(self.progress_loss) else None,
            "last_progress_epoch": self.last_progress_epoch, "warmup_steps": self.warmup_steps,
            "base_lrs": self.base_lrs, "stop_reason": self.stop_reason, "is_best": is_best,
            "previous_best": None if is_best else self.best_pointer,
        }
        self.checkpoint_manager.save_checkpoint(
            epoch=self.current_epoch, global_step=self.global_step, model=self.model,
            optimizer=self.optimizer, scheduler=self.scheduler, scaler=self.scaler,
            config=self.config, metrics=record, is_best=is_best, extra_state={"flow3d_trainer": state},
            dataloader_state=loader_rng.get_state() if loader_rng is not None else None,
            sampler_state=sampler_rng.get_state() if sampler_rng is not None else None,
            ema_state_dict=self.ema_model.state_dict() if self.ema_model is not None else None,
        )
        if is_best:
            self.best_pointer = json.loads(self.checkpoint_manager.best_pointer_path.read_text())

    def _prepare_warmup_step(self) -> None:
        if not self.warmup_steps or self.global_step >= self.warmup_steps:
            return
        fraction = self.global_step / (self.warmup_steps - 1) if self.warmup_steps > 1 else 1.0
        for group, base in zip(self.optimizer.param_groups, self.base_lrs):
            start = self.warmup_start_lr * (base / self.base_lrs[0])
            group["lr"] = base if fraction == 1.0 else start + fraction * (base - start)

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        epochs: int = 50,
        eval_interval: int = 5,
        save_interval: int = 10,
        log_callback: Any | None = None,
        step_callback: Callable[[int, Flow3DTrainMetrics], None] | None = None,
    ) -> list[dict[str, Any]]:
        """Executes full training epochs, evaluation, and atomic checkpointing."""
        if epochs < 1 or eval_interval < 1 or save_interval < 1:
            raise ValueError("epoch budget and evaluation/checkpoint intervals must be positive")
        if self.current_epoch > epochs:
            raise ValueError("epoch budget is smaller than the restored epoch")
        if self.stop_reason == "CONVERGED" or self.current_epoch == epochs:
            return self.history
        if self.adaptive:
            if val_loader is None or eval_interval != 1:
                raise ValueError("warmup_plateau needs a validation metric every epoch")
            steps = self.warmup_epochs * len(train_loader)
            if self.warmup_steps is not None and self.warmup_steps != steps:
                raise ValueError("training steps per epoch changed since checkpoint")
            self.warmup_steps = steps
        self.stop_reason = None

        logger.info(
            f"Starting Stage 1 training for {epochs} epochs on device={self.device} "
            f"(active GPUs={self.num_active_gpus}, AMP={self.use_amp})"
        )

        for ep in range(self.current_epoch + 1, epochs + 1):
            self.current_epoch = ep
            past_warmup = self.global_step >= (self.warmup_steps or 0)
            ep_loss_sum = 0.0
            ep_vel_sum = 0.0
            ep_p_sum = 0.0
            ep_aux_sum = 0.0
            ep_steps = 0
            t_epoch_start = time.perf_counter()

            for batch in train_loader:
                self._prepare_warmup_step()
                previous_step = self.global_step
                m = self.train_step(batch)
                if step_callback is not None and self.global_step > previous_step:
                    step_callback(self.global_step, m)
                ep_loss_sum += m.total_loss
                ep_vel_sum += m.loss_vel
                ep_p_sum += m.loss_press
                ep_aux_sum += m.loss_aux
                ep_steps += 1

            if ep_steps == 0:
                raise ValueError("training dataset is empty")
            last_lr = float(self.optimizer.param_groups[0]["lr"])
            is_plateau = isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)
            if self.scheduler is not None and not is_plateau:
                self.scheduler.step()

            n_s = ep_steps
            ep_time_s = time.perf_counter() - t_epoch_start
            ep_record: dict[str, Any] = {
                "epoch": ep,
                "global_step": self.global_step,
                "train_loss": ep_loss_sum / n_s,
                "train_loss_vel": ep_vel_sum / n_s,
                "train_loss_press": ep_p_sum / n_s,
                "train_loss_aux": ep_aux_sum / n_s,
                "learning_rate": last_lr,
                "epoch_time_s": ep_time_s,
            }

            # Periodic Evaluation
            if val_loader is not None and (ep % eval_interval == 0 or ep == epochs):
                eval_m = self.evaluate(val_loader)
                if not math.isfinite(eval_m.val_loss):
                    raise FloatingPointError(f"non-finite validation loss at epoch {ep}")
                ep_record.update({
                    "val_loss": eval_m.val_loss,
                    "val_loss_vel": eval_m.val_loss_vel,
                    "val_loss_press": eval_m.val_loss_press,
                    "rmse_uvw_ms": eval_m.rmse_uvw_ms,
                    "rmse_p_pa": eval_m.rmse_p_pa,
                    "div_rms_phys": eval_m.div_rms_phys_s_inv,
                    "solid_max_speed": eval_m.solid_interior_max_abs,
                    "near_wall_vel_rmse_ms": eval_m.near_wall_vel_rmse_ms,
                })
                is_best = bool(eval_m.val_loss < self.best_val_loss)
                if is_best:
                    self.best_val_loss = eval_m.val_loss
                if is_plateau and past_warmup:
                    self.scheduler.step(eval_m.val_loss)
                if self.adaptive:
                    if eval_m.val_loss < self.progress_loss and eval_m.val_loss <= self.progress_loss * (1 - self.stop_threshold):
                        self.progress_loss = eval_m.val_loss
                        self.last_progress_epoch = ep
                    stale = ep - self.last_progress_epoch
                    ep_record["epochs_without_significant_improvement"] = stale
                    just_reduced = self.optimizer.param_groups[0]["lr"] < last_lr
                    if ep >= max(self.min_epochs, self.warmup_epochs) and stale >= self.stop_patience and not just_reduced:
                        self.stop_reason = "CONVERGED"
            else:
                is_best = False

            self._prepare_warmup_step()
            ep_record["next_learning_rate"] = float(self.optimizer.param_groups[0]["lr"])
            if ep == epochs and self.stop_reason is None:
                self.stop_reason = "NOT_CONVERGED" if self.adaptive else "MAX_EPOCHS"
            if self.stop_reason:
                ep_record["stop_reason"] = self.stop_reason
            self.history.append(ep_record)
            if ep % save_interval == 0 or self.stop_reason or is_best:
                self._save_epoch(train_loader, ep_record, is_best)
            if log_callback is not None:
                log_callback(ep_record)
            if self.stop_reason:
                if self.stop_reason == "NOT_CONVERGED":
                    logger.warning("NOT_CONVERGED: epoch budget exhausted before the plateau criterion was met")
                break

        return self.history
