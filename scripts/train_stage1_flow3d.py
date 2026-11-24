#!/usr/bin/env python3
"""Production Stage 1 Training Entrypoint for FlowWorld-3D (R3).

Trains the 3D DC-AE Autoencoder on certified multi-Re CFD trajectories
using the available compute resources, physics losses, and atomic checkpoints.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import datetime as _dt
from functools import partial
import json
import logging
from pathlib import Path
import random
import sys
import time
from typing import Any

import numpy as np

# Ensure src is on path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from flow3d.data.dataset import Flow3DH5Dataset, collate_flow3d_stage1, stage1_split_identity
from flow3d.data.schema import load_yaml_or_json_config
from flow3d.models import (
    Flow3DStage1Autoencoder,
    configure_elastic_hardware,
    resolve_elastic_runtime_plan,
    torch_available,
)
from flow3d.training.trainer import Flow3DStage1Trainer
from flow3d.training.tensorboard import Stage1TensorBoard
from flow3d.training.artifact import build_stage1_contract, configure_stage1_resume
from stage1.io.checkpoint import CheckpointManager, sha256_json

if torch_available():
    import torch
    from torch.utils.data import DataLoader, RandomSampler
else:
    torch = None
    DataLoader = None


def setup_logger(log_level: str = "INFO") -> logging.Logger:
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=getattr(logging, log_level.upper(), logging.INFO),
    )
    return logging.getLogger("flow3d.train_stage1")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FlowWorld-3D Stage 1 3D DC-AE Production Training with Elastic Multi-GPU"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/flow3d/stage1.yaml",
        help="Path to stage 1 YAML configuration file",
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="/root/autodl-tmp/datasets/world_model",
        help="Path to directory containing certified HDF5 dataset files",
    )
    parser.add_argument("--eligible-files", nargs="+", help="Train on these certified HDF5 filenames only")
    parser.add_argument("--ema", action=argparse.BooleanOptionalAction, default=None,
                        help="Use EMA weights for validation and G1")
    parser.add_argument("--tensorboard", action=argparse.BooleanOptionalAction, default=None,
                        help="Write TensorBoard scalars (default: training.tensorboard in config)")
    parser.add_argument("--tensorboard-dir", type=Path,
                        help="Enable TensorBoard and put the new run under this directory (default: SAVE_DIR/tensorboard)")
    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="Directory to save checkpoints (defaults to config value or 'checkpoints/stage1')",
    )
    parser.add_argument(
        "--head-type",
        type=str,
        default=None,
        choices=["head_a", "head_b"],
        help="Override physics readout head type ('head_a' direct or 'head_b' regularized curl)",
    )
    parser.add_argument(
        "--latent-norm",
        type=str,
        default=None,
        choices=["group_norm", "dataset_rms", "affine_free_rmsnorm"],
        help="Override latent normalization mode",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Total training epochs (overrides config if provided)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Global batch size (defaults to elastic plan: 1 per active GPU)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate for AdamW optimizer",
    )
    parser.add_argument(
        "--num-gpus",
        type=str,
        default=None,
        help="Override runtime.num_gpus ('auto' uses all visible GPUs, 0 uses CPU)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="DataLoader worker processes (defaults to elastic plan recommendation)",
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable automatic mixed precision (AMP)",
    )
    parser.add_argument(
        "--amp-dtype",
        type=str,
        default="bf16",
        choices=["bf16", "fp16", "fp32"],
        help="AMP precision dtype (default bf16)",
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=None,
        help="Epoch interval between evaluations",
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=None,
        help="Epoch interval between saving checkpoints",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint .pt file to resume training from",
    )
    parser.add_argument(
        "--split-strategy",
        type=str,
        default=None,
        choices=["temporal_block", "group_by_trajectory", "group_by_ancestor"],
        help="Train/val/test split strategy",
    )
    parser.add_argument(
        "--divergence-weight",
        type=float,
        default=None,
        help="Override loss.divergence_weight (Head-A FP32 soft divergence penalty)",
    )
    parser.add_argument(
        "--coulomb-gauge-weight",
        type=float,
        default=None,
        help="Override loss.coulomb_gauge_weight (Head-B Coulomb gauge regularization)",
    )
    parser.add_argument(
        "--harmonic-laplacian-weight",
        type=float,
        default=None,
        help="Override loss.harmonic_laplacian_weight (Head-B harmonic potential regularization)",
    )
    parser.add_argument(
        "--use-potential-branch",
        action="store_true",
        help="Enable Head-B scalar potential branch grad_h(phi) with harmonic Laplacian constraint",
    )
    parser.add_argument(
        "--no-edge-order1",
        action="store_true",
        help="Disable 1st-order boundary differences in Head-B curl/grad",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(line_buffering=True)
    logger = setup_logger("INFO")

    if not torch_available():
        logger.error("PyTorch is required for training; please install PyTorch with CUDA.")
        sys.exit(1)

    set_seed(args.seed)

    # 1. Load Configuration
    cfg_path = Path(args.config)
    if not cfg_path.is_file():
        cfg_path = REPO_ROOT / args.config
    if not cfg_path.is_file():
        logger.error(f"Configuration file not found: {args.config} (resolved: {cfg_path})")
        sys.exit(1)
    cfg = load_yaml_or_json_config(cfg_path)

    # Overrides from command line
    if args.head_type:
        cfg.setdefault("head", {})["type"] = args.head_type
    if args.use_potential_branch:
        cfg.setdefault("head", {})["use_potential_branch"] = True
    if args.no_edge_order1:
        cfg.setdefault("head", {})["edge_order1"] = False
    if args.latent_norm:
        cfg.setdefault("latent_norm", {})["type"] = args.latent_norm
    if args.divergence_weight is not None:
        cfg.setdefault("loss", {})["divergence_weight"] = float(args.divergence_weight)
    if args.coulomb_gauge_weight is not None:
        cfg.setdefault("loss", {})["coulomb_gauge_weight"] = float(args.coulomb_gauge_weight)
    if args.harmonic_laplacian_weight is not None:
        cfg.setdefault("loss", {})["harmonic_laplacian_weight"] = float(args.harmonic_laplacian_weight)

    train_cfg = cfg.setdefault("training", {})
    for key, value in (
        ("max_epochs", args.epochs), ("learning_rate", args.lr),
        ("eval_interval", args.eval_interval), ("save_interval", args.save_interval),
        ("split_strategy", args.split_strategy), ("eligible_files", args.eligible_files),
        ("ema", args.ema),
    ):
        if value is not None:
            train_cfg[key] = value
    epochs = int(train_cfg.get("max_epochs", train_cfg.get("epochs", 50)))
    learning_rate = float(train_cfg.get("learning_rate", 2e-4))
    eval_interval = int(train_cfg.get("eval_interval", 5))
    save_interval = int(train_cfg.get("save_interval", 10))
    schedule = train_cfg.get("scheduler", "cosine")
    if schedule not in {"warmup_plateau", "cosine"}:
        raise ValueError(f"Unknown training scheduler: {schedule}")
    if min(epochs, eval_interval, save_interval, learning_rate) <= 0:
        raise ValueError("Epoch budget, intervals, and learning rate must be positive")
    if schedule == "warmup_plateau" and eval_interval != 1:
        raise ValueError("warmup_plateau requires --eval-interval 1 so patience counts epochs")
    save_dir_raw = args.save_dir or train_cfg.get("save_dir", "checkpoints/stage1")
    save_dir = Path(save_dir_raw) if Path(save_dir_raw).is_absolute() else (REPO_ROOT / save_dir_raw)
    save_dir.mkdir(parents=True, exist_ok=True)

    use_amp = not args.no_amp
    amp_dtype = args.amp_dtype

    # 2. Elastic Hardware & Multi-GPU Runtime Plan
    plan = resolve_elastic_runtime_plan(cfg, num_gpus=args.num_gpus)
    configure_elastic_hardware(plan)
    batch_size = args.batch_size if args.batch_size is not None else plan.global_batch_size
    num_workers = args.num_workers
    if num_workers is None:
        num_workers = cfg.get("runtime", {}).get("num_workers", min(8, plan.total_dataloader_workers))

    logger.info("=" * 70)
    logger.info("FlowWorld-3D Stage 1 (3D DC-AE) Training Pipeline Initialized")
    logger.info("=" * 70)
    logger.info(f"Hardware Plan: Strategy={plan.strategy}, Active GPUs={plan.active_gpus}/{plan.detected_gpus}")
    logger.info(f"Devices: {plan.devices}, Primary Device: {plan.primary_device}")
    gpu_inventory = []
    for device_id in plan.device_ids:
        props = torch.cuda.get_device_properties(device_id)
        gpu_inventory.append({"device": f"cuda:{device_id}", "name": props.name, "total_bytes": props.total_memory})
        logger.info("cuda:%d %s, %.2f GiB", device_id, props.name, props.total_memory / (1024 ** 3))
    logger.info("CPU budget: %d threads, RAM headroom at startup: %.2f GiB",
                plan.detected_cpus, plan.detected_ram_gib)
    logger.info(
        f"CPU threads: {plan.cpu_threads_total}, workers per DataLoader: {num_workers}, global batch: {batch_size}"
    )
    logger.info(
        f"Model Head: {cfg.get('head', {}).get('type')} "
        f"(potential={cfg.get('head', {}).get('use_potential_branch', False)}, "
        f"edge_order1={cfg.get('head', {}).get('edge_order1', True)}), "
        f"Latent Norm: {cfg.get('latent_norm', {}).get('type')}"
    )
    logger.info(
        f"Loss Weights: recon={cfg.get('loss', {}).get('reconstruction_weight', 1.0)}, "
        f"press={cfg.get('loss', {}).get('pressure_weight', 1.0)}, "
        f"div={cfg.get('loss', {}).get('divergence_weight', 0.0)}, "
        f"coulomb={cfg.get('loss', {}).get('coulomb_gauge_weight', 0.01)}, "
        f"harm={cfg.get('loss', {}).get('harmonic_laplacian_weight', 0.01)}"
    )
    logger.info(f"Epochs: {epochs}, Learning Rate: {learning_rate}, AMP: {use_amp} ({amp_dtype})")
    logger.info(f"Save Directory: {save_dir.resolve()}")

    # 3. Datasets & DataLoaders
    train_cfg.update(seed=args.seed, batch_size=batch_size, amp=use_amp, amp_dtype=amp_dtype)

    logger.info(f"Loading certified CFD dataset from: {args.dataset_dir}")
    t_data0 = time.perf_counter()

    train_ds, val_ds, test_ds = Flow3DH5Dataset.create_splits(
        dataset_dir=args.dataset_dir,
        eligible_files=train_cfg.get("eligible_files"),
        split_strategy=train_cfg.get("split_strategy", "temporal_block"),
        purge_gap_frames=int(train_cfg.get("purge_gap_frames", 4)),
        train_ratio=float(train_cfg.get("train_ratio", 0.7)),
        val_ratio=float(train_cfg.get("val_ratio", 0.15)),
        test_ratio=float(train_cfg.get("test_ratio", 0.15)),
    )
    split_identity = stage1_split_identity(train_ds, val_ds, test_ds)
    train_cfg["split_sha256"] = sha256_json(split_identity)

    data_load_ms = round((time.perf_counter() - t_data0) * 1000.0, 2)
    logger.info(f"Dataset preloaded into RAM in {data_load_ms} ms: Train={len(train_ds)} frames, Val={len(val_ds)} frames")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=RandomSampler(train_ds, generator=torch.Generator().manual_seed(args.seed)),
        generator=torch.Generator().manual_seed(args.seed + 1),
        num_workers=num_workers,
        persistent_workers=bool(num_workers and plan.persistent_workers),
        pin_memory=plan.pin_memory and (plan.active_gpus > 0),
        collate_fn=partial(
            collate_flow3d_stage1,
            include_phys_field=cfg.get("runtime", {}).get("train_include_phys_field", False),
        ),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        generator=torch.Generator().manual_seed(args.seed + 2),
        num_workers=num_workers,
        persistent_workers=bool(num_workers and plan.persistent_workers),
        pin_memory=plan.pin_memory and (plan.active_gpus > 0),
        collate_fn=collate_flow3d_stage1,
        drop_last=False,
    )

    # 4. Model Construction
    primary_device = torch.device(plan.primary_device)
    model = Flow3DStage1Autoencoder.from_config(cfg).to(primary_device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model Built: Total Parameters={total_params:,}, Trainable={trainable_params:,}")

    # 5. Optimizer & Scheduler
    weight_decay = float(train_cfg.get("weight_decay", 1e-4))
    betas = tuple(float(b) for b in train_cfg.get("betas", (0.9, 0.95)))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=betas,
        weight_decay=weight_decay,
    )

    scheduler = None
    if schedule == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=float(train_cfg.get("min_lr", 1e-5)),
        )

    # 6. Checkpoint Manager & Resuming
    ckpt_mgr = CheckpointManager(
        save_dir=save_dir,
        model_name=f"dcae3d_{cfg.get('head', {}).get('type', 'stage1')}",
        contract=build_stage1_contract(cfg, split_identity, Path(args.dataset_dir)),
    )

    # 7. Trainer Initialization
    trainer = Flow3DStage1Trainer(
        model=model,
        config=cfg,
        optimizer=optimizer,
        scheduler=scheduler,
        checkpoint_manager=ckpt_mgr,
        device=primary_device,
        num_gpus=plan.active_gpus,
        amp=use_amp,
        amp_dtype=amp_dtype,
        clip_grad_norm=float(train_cfg.get("clip_grad_norm", 1.0)),
        save_dir=save_dir,
    )
    if args.resume:
        resume_path = configure_stage1_resume(ckpt_mgr, args.resume, train_cfg)
        trainer.resume(resume_path, train_loader)
        logger.info(f"Resumed at epoch={trainer.current_epoch}, step={trainer.global_step}")

    # 8. Training Execution Loop
    logger.info("-" * 70)
    logger.info("Starting Training Fit Loop...")
    t_train0 = time.perf_counter()

    tb_enabled = args.tensorboard
    if tb_enabled is None:
        tb_enabled = args.tensorboard_dir is not None or train_cfg.get("tensorboard", False)
    tb = None
    if tb_enabled:
        parent = args.tensorboard_dir or save_dir / "tensorboard"
        tb = Stage1TensorBoard(parent, cfg.get("head", {}).get("type", "stage1"))
        logger.info("TensorBoard run: %s", tb.log_dir)

    def log_epoch(record: dict[str, Any]) -> None:
        ep = record["epoch"]
        t_loss = record["train_loss"]
        v_loss = record["train_loss_vel"]
        p_loss = record["train_loss_press"]
        a_loss = record.get("train_loss_aux", 0.0)
        lr_now = record["learning_rate"]
        time_s = record["epoch_time_s"]

        msg = f"Epoch [{ep:03d}/{epochs:03d}] Loss: {t_loss:.5f} (Vel: {v_loss:.5f}, P: {p_loss:.5f}, Aux: {a_loss:.5f}) | LR: {lr_now:.2e} | Time: {time_s:.2f}s"
        if "val_loss" in record:
            val_l = record["val_loss"]
            uvw_rmse = record["rmse_uvw_ms"]
            p_rmse = record["rmse_p_pa"]
            div_rms = record["div_rms_phys"]
            near_wall = record["near_wall_vel_rmse_ms"]
            msg += (f" | Val Loss: {val_l:.6g} [UVW-RMSE: {uvw_rmse:.4g} m/s, P-RMSE: {p_rmse:.4g} Pa,"
                    f" Div-RMS: {div_rms:.3g} s^-1, near-wall RMSE: {near_wall:.4g} m/s]")
        logger.info(msg)
        if tb is not None:
            tb.log_epoch(record)

    with tb if tb is not None else nullcontext():
        if tb is not None:
            for record in trainer.history:
                tb.log_epoch(record)
        history = trainer.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs,
            eval_interval=eval_interval,
            save_interval=save_interval,
            log_callback=log_epoch,
            step_callback=tb.log_step if tb is not None else None,
        )

    total_train_sec = round(time.perf_counter() - t_train0, 2)
    logger.info("=" * 70)
    logger.info(f"Stage 1 Training Complete in {total_train_sec} seconds ({total_train_sec / 60:.2f} minutes)!")
    logger.info(f"Best Validation Loss: {trainer.best_val_loss:.6f}")
    log_status = logger.warning if trainer.stop_reason == "NOT_CONVERGED" else logger.info
    log_status(f"{trainer.stop_reason} at epoch {trainer.current_epoch}/{epochs}")
    logger.info(f"Checkpoints and best model pointer saved to: {save_dir.resolve()}")

    # 9. Save structured training log JSON
    summary_path = save_dir / "training_summary.json"
    summary_data = {
        "timestamp_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "total_train_seconds": total_train_sec,
        "epochs": trainer.current_epoch,
        "max_epochs": epochs,
        "stop_reason": trainer.stop_reason,
        "global_steps": trainer.global_step,
        "best_val_loss": trainer.best_val_loss,
        "active_gpus": plan.active_gpus,
        "devices": plan.devices,
        "detected_cpu_threads": plan.detected_cpus,
        "available_ram_gib_at_start": plan.detected_ram_gib,
        "active_gpu_hardware": gpu_inventory,
        "batch_size": batch_size,
        "cpu_threads": plan.cpu_threads_total,
        "num_workers_per_loader": num_workers,
        "config": cfg,
        "split_identity": split_identity if train_cfg.get("split_sha256") else None,
        "split_identity_status": "verified" if train_cfg.get("split_sha256") else "unverified_legacy",
        "tensorboard_log_dir": str(tb.log_dir) if tb is not None else None,
        "history": history,
    }
    summary_path.write_text(json.dumps(summary_data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    logger.info(f"Training summary saved: {summary_path}")


if __name__ == "__main__":
    main()
