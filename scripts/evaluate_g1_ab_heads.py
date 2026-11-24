#!/usr/bin/env python3
"""G1 Held-Out Test Evaluation & Head-A vs Head-B Comparison Suite for FlowWorld-3D.

Loads best checkpoints for Head-A and Head-B, evaluates all held-out test frames in SI units
(velocity m/s, pressure Pa, near-wall first fluid shell, interior divergence m1/m2, vorticity,
enstrophy, 3D kinetic energy spectrum E(k)), saves JSON report, and generates 3 comparison PNGs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

import torch

from flow3d.data.dataset import Flow3DH5Dataset, stage1_split_identity
from flow3d.metrics.reconstruction import evaluate_stage1_frame_si
from flow3d.models import Flow3DStage1Autoencoder
from flow3d.physics.operators import curl_h, div_h
from stage1.io.checkpoint import CheckpointManager, sha256_json

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8.5,
    "figure.titlesize": 13.5,
})


def _load_best_model(
    ckpt_dir: Path, device: torch.device, expected_head: str
) -> tuple[Flow3DStage1Autoencoder, dict[str, Any], dict[str, Any]]:
    summary_path = ckpt_dir / "training_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"missing training_summary.json in {ckpt_dir}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    cfg = summary["config"]
    head_type = cfg.get("head", {}).get("type", "head_a")
    if head_type != expected_head:
        raise ValueError(f"{ckpt_dir}: expected {expected_head}, found {head_type}")
    model_name = f"dcae3d_{head_type}"

    best_ptr = ckpt_dir / f"{model_name}_best.json"
    if not best_ptr.is_file():
        raise FileNotFoundError(f"missing best checkpoint pointer: {best_ptr}")
    ptr_data = json.loads(best_ptr.read_text(encoding="utf-8"))
    ckpt_path = ckpt_dir / ptr_data["checkpoint"]

    model = Flow3DStage1Autoencoder.from_config(cfg).to(device)
    mgr = CheckpointManager(save_dir=ckpt_dir, model_name=model_name)
    payload = mgr.load_checkpoint(
        checkpoint="best",
        model=model,
        optimizer=None,
        scheduler=None,
        map_location="cpu",
        strict=True,
        restore_rng=False,
        use_ema=True,
    )
    identity = summary.get("split_identity")
    saved_cfg = payload["config"]
    model_sections = ("grid", "encoder", "spatial_transformer", "decoder", "head", "latent_norm", "normalizer", "probe")
    if any(cfg.get(key) != saved_cfg.get(key) for key in model_sections):
        raise ValueError(f"{ckpt_dir}: model configuration differs from the checkpoint")
    saved_training = saved_cfg.get("training", {})
    saved_digest = saved_training.get("split_sha256")
    has_ema = payload.get("ema_state_dict") is not None
    if bool(saved_training.get("ema", False)) != has_ema:
        raise ValueError(f"{ckpt_dir}: validation weights are missing or inconsistent with the checkpoint")
    if saved_digest is not None or identity is not None:
        if identity is None or sha256_json(identity) != saved_digest:
            raise ValueError(f"{ckpt_dir}: held-out split identity differs from checkpoint")
    val_loss = payload["metrics"]["val_loss"]
    if val_loss != summary["best_val_loss"] or payload["model_name"] != model_name:
        raise ValueError(f"{ckpt_dir}: best checkpoint does not match the training summary")
    model.eval()
    meta = {
        "checkpoint": str(ckpt_path),
        "epoch": payload.get("epoch"),
        "global_step": payload.get("global_step"),
        "sha256": ptr_data.get("sha256"),
        "val_loss": val_loss,
        "weights": "ema" if has_ema else "raw",
        "head_type": head_type,
        "use_potential_branch": bool(cfg.get("head", {}).get("use_potential_branch", False)),
        "edge_order1": bool(cfg.get("head", {}).get("edge_order1", True)),
        "loss_cfg": saved_cfg.get("loss", {}),
    }
    return model, summary, meta


def _held_out_split(
    dataset_dir: str | Path, sum_a: dict[str, Any], sum_b: dict[str, Any]
) -> tuple[Flow3DH5Dataset, dict[str, Any]]:
    expected = sum_a.get("split_identity")
    if expected != sum_b.get("split_identity"):
        raise ValueError("Head-A and Head-B were trained on different or unrecorded Stage 1 splits")
    # Older summaries did not record input hashes or frame identities. Tag
    # such reports so a reconstruction of the old split cannot claim verification.
    legacy_defaults = dict(split_strategy="temporal_block", purge_gap_frames=4,
                           train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
    settings = {
        key: sum_a["config"]["training"].get(key, default) for key, default in legacy_defaults.items()
    }
    for key, default in legacy_defaults.items():
        if sum_b["config"]["training"].get(key, default) != settings[key]:
            raise ValueError(f"inconsistent Stage 1 split setting: {key}")
    if expected is None:
        print("[G1 Eval] Legacy summaries lack file hashes and frame IDs. Held-out membership is unverified.")
    splits = Flow3DH5Dataset.create_splits(
        dataset_dir=dataset_dir,
        eligible_files=list(expected["files"]) if expected is not None else None,
        **settings,
    )
    actual = stage1_split_identity(*splits)
    if expected is not None and actual != expected:
        raise ValueError("current HDF5 contents or train/val/test frame IDs differ from the training splits")
    if not len(splits[2]):
        raise ValueError("empty held-out test split")
    return splits[2], {"status": "verified" if expected is not None else "unverified_legacy",
                       "sha256": sha256_json(actual), **actual}


def _aggregate_metrics(records: list[dict[str, Any]]) -> dict[str, float]:
    scalar_keys = [
        k for k, v in records[0].items()
        if isinstance(v, (int, float)) and k not in (
            "step_index", "u0_m_s", "reynolds_number", "fluid_voxel_count", "near_wall_voxel_count",
        )
    ]
    out: dict[str, float] = {}
    for k in scalar_keys:
        vals = [float(r[k]) for r in records]
        out[k] = float(np.mean(vals))
    return out


def _pooled_physical_metrics(records: list[dict[str, Any]]) -> dict[str, float | None]:
    """Pool squared errors over voxels, using the same masks as training validation."""
    result: dict[str, float | None] = {}
    for key, support in (
        ("rmse_vel_ms", "fluid_voxel_count"), ("rmse_p_pa", "fluid_voxel_count"),
        ("div_rms_fluid_pred_s_inv", "fluid_voxel_count"), ("near_wall_vel_rmse_ms", "near_wall_voxel_count"),
    ):
        count = sum(r[support] for r in records)
        result[key] = float(np.sqrt(sum(r[key] ** 2 * r[support] for r in records) / count)) if count else None
    return result


def _select_head(
    frames_a: list[dict[str, Any]], frames_b: list[dict[str, Any]],
    loss_a: dict[str, Any], loss_b: dict[str, Any],
) -> dict[str, Any]:
    """Score both validation-selected checkpoints on the same test voxels with common weights."""
    def weights(loss: dict[str, Any]) -> tuple[float, float]:
        return float(loss.get("reconstruction_weight", 1.0)), float(loss.get("pressure_weight", 1.0))

    weights_a, weights_b = weights(loss_a), weights(loss_b)
    same_weights = weights_a == weights_b
    w_vel, w_p = weights_a if same_weights else (1.0, 1.0)
    scores = {}
    for head, frames in (("head_a", frames_a), ("head_b", frames_b)):
        count = sum(r["fluid_voxel_count"] for r in frames)
        total_vel = sum(r["normalized_reconstruction_sse"]["uvw"] for r in frames)
        total_p = sum(r["normalized_reconstruction_sse"]["p"] for r in frames)
        scores[head] = (w_vel * total_vel + w_p * total_p) / count
        if not np.isfinite(scores[head]):
            raise ValueError(f"{head} produced a nonfinite normalized reconstruction score")
    winner = "tie" if scores["head_a"] == scores["head_b"] else min(scores, key=scores.get)
    return {
        "criterion": "pooled_normalized_fluid_reconstruction_loss",
        "definition": "Sum of w_uvw * (du² + dv² + dw²) + w_p * dp² over all test fluid voxels, "
                      "divided by the total number of test fluid voxels. Errors use normalized fields.",
        "direction": "minimize",
        "weights": {"w_uvw": w_vel, "w_p": w_p},
        "weight_source": "shared_validation_weights" if same_weights else "common_unit_weights",
        "scores": scores,
        "winner": winner,
    }


def _evaluate_model_on_dataset(
    model: Flow3DStage1Autoencoder,
    dataset: Flow3DH5Dataset,
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    """Run FP32 evaluation on all frames of `dataset` and cache representative test frames for plotting."""
    per_frame: list[dict[str, Any]] = []
    cached_preds: dict[str, np.ndarray] = {}

    with torch.no_grad():
        for idx in range(len(dataset)):
            item = dataset[idx]
            norm_f = item["norm_field"].unsqueeze(0).to(device)
            geom = item["geometry"].unsqueeze(0).to(device)
            b_stats = item["b"].unsqueeze(0).to(device)
            sp_star = item["spacing_star"]
            sp_m = item["spacing_m"]

            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            out = model(norm_f, geom, b=b_stats, spacing_zyx=sp_star)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            infer_ms = (time.perf_counter() - t0) * 1000.0

            pred_phys = out["phys_fields"][0].detach().cpu().numpy().astype(np.float64)
            gt_phys = item["phys_field"].numpy().astype(np.float64)
            geom_np = item["geometry"].numpy().astype(np.float64)

            m = evaluate_stage1_frame_si(pred_phys, gt_phys, geom_np, sp_m)
            fluid = geom[0, 0] > 0.5
            norm_error = out["norm_fields"][0, :, fluid].double() - norm_f[0, :, fluid].double()
            sse = norm_error.square().sum(dim=1)
            m["normalized_reconstruction_sse"] = {"uvw": sse[:3].sum().item(), "p": sse[3].item()}
            m["case_id"] = str(item["case_id"])
            m["reynolds_number"] = float(item["reynolds_number"])
            m["u0_m_s"] = float(item["u0_m_s"])
            m["step_index"] = int(item["step_index"])
            m["infer_ms"] = float(infer_ms)
            per_frame.append(m)

            # Cache last frame of each case_id for slice visualization
            cached_preds[str(item["case_id"])] = pred_phys.astype(np.float32)

    return per_frame, cached_preds


def _plot_fig1_training_curves(
    sum_a: dict[str, Any],
    sum_b: dict[str, Any],
    out_path: Path,
) -> None:
    ha = sum_a["history"]
    hb = sum_b["history"]

    ep_a = [r["epoch"] for r in ha]
    ep_b = [r["epoch"] for r in hb]

    fig, axes = plt.subplots(2, 3, figsize=(18.5, 9.8), dpi=180, constrained_layout=True)

    # (0,0) Train Total Loss
    ax = axes[0, 0]
    ax.plot(ep_a, [r["train_loss"] for r in ha], label="Head-A", color="#1f77b4", lw=2.0)
    ax.plot(ep_b, [r["train_loss"] for r in hb], label="Head-B", color="#d62728", lw=2.0)
    ax.set_yscale("log")
    ax.set_title("Training Total Loss (log)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Total Loss")
    ax.grid(True, which="both", alpha=0.28)
    ax.legend()

    # (0,1) Train Velocity vs Pressure MSE
    ax = axes[0, 1]
    ax.plot(ep_a, [r["train_loss_vel"] for r in ha], label="Head-A Vel MSE", color="#1f77b4", lw=1.9)
    ax.plot(ep_a, [r["train_loss_press"] for r in ha], label="Head-A Press MSE", color="#1f77b4", ls="--", lw=1.5)
    ax.plot(ep_b, [r["train_loss_vel"] for r in hb], label="Head-B Vel MSE", color="#d62728", lw=1.9)
    ax.plot(ep_b, [r["train_loss_press"] for r in hb], label="Head-B Press MSE", color="#d62728", ls="--", lw=1.5)
    ax.set_yscale("log")
    ax.set_title("Training Normalized Vel & Press MSE (log)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Masked MSE")
    ax.grid(True, which="both", alpha=0.28)
    ax.legend(fontsize=8)

    # (0,2) Validation Loss
    va_ep = [r["epoch"] for r in ha if "val_loss" in r]
    vb_ep = [r["epoch"] for r in hb if "val_loss" in r]
    ax = axes[0, 2]
    ax.plot(va_ep, [r["val_loss"] for r in ha if "val_loss" in r], "o-", label="Head-A Val Loss", color="#1f77b4", lw=2.0, ms=4)
    ax.plot(vb_ep, [r["val_loss"] for r in hb if "val_loss" in r], "s-", label="Head-B Val Loss", color="#d62728", lw=2.0, ms=4)
    ax.set_yscale("log")
    ax.set_title("Validation Reconstruction Loss (log)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Val Loss (Vel + Press)")
    ax.grid(True, which="both", alpha=0.28)
    ax.legend()

    # (1,0) Validation UVW-RMSE (m/s)
    ax = axes[1, 0]
    ax.plot(va_ep, [r["rmse_uvw_ms"] for r in ha if "val_loss" in r], "o-", label="Head-A", color="#1f77b4", lw=2.0, ms=4)
    ax.plot(vb_ep, [r["rmse_uvw_ms"] for r in hb if "val_loss" in r], "s-", label="Head-B", color="#d62728", lw=2.0, ms=4)
    ax.set_title("Validation Velocity Vector RMSE [m/s]")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("RMSE (m/s)")
    ax.grid(True, alpha=0.28)
    ax.legend()

    # (1,1) Validation Pressure RMSE (Pa)
    ax = axes[1, 1]
    ax.plot(va_ep, [r["rmse_p_pa"] for r in ha if "val_loss" in r], "o-", label="Head-A", color="#1f77b4", lw=2.0, ms=4)
    ax.plot(vb_ep, [r["rmse_p_pa"] for r in hb if "val_loss" in r], "s-", label="Head-B", color="#d62728", lw=2.0, ms=4)
    ax.set_title("Validation Pressure RMSE [Pa]")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("RMSE (Pa)")
    ax.grid(True, alpha=0.28)
    ax.legend()

    # (1,2) Validation Physical Divergence RMS (s^-1)
    ax = axes[1, 2]
    ax.plot(va_ep, [r["div_rms_phys"] for r in ha if "val_loss" in r], "o-", label="Head-A", color="#1f77b4", lw=2.0, ms=4)
    ax.plot(vb_ep, [r["div_rms_phys"] for r in hb if "val_loss" in r], "s-", label="Head-B", color="#d62728", lw=2.0, ms=4)
    ax.set_yscale("log")
    ax.set_title("Validation Physical Divergence RMS [s^-1] (log)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Div RMS (s^-1)")
    ax.grid(True, which="both", alpha=0.28)
    ax.legend()

    fig.suptitle(f"FlowWorld-3D Stage 1 training and validation — "
                 f"Head-A ({ep_a[-1]} epochs) and Head-B ({ep_b[-1]} epochs)", fontweight="bold")
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_fig2_reconstruction_slices(
    test_ds: Flow3DH5Dataset,
    preds_a: dict[str, np.ndarray],
    preds_b: dict[str, np.ndarray],
    out_path: Path,
    case_id: str = "re1m_wm_50steps",
) -> None:
    # Find last test frame for chosen case_id
    target_item = None
    for i in range(len(test_ds)):
        it = test_ds[i]
        if str(it["case_id"]) == case_id:
            target_item = it
    if target_item is None:
        target_item = test_ds[-1]
        case_id = str(target_item["case_id"])

    gt = target_item["phys_field"].numpy()
    geom = target_item["geometry"].numpy()
    fluid = geom[0] > 0.5
    sp_m = target_item["spacing_m"]
    pa = preds_a[case_id]
    pb = preds_b[case_id]

    z_mid = gt.shape[1] // 2
    mask_2d = fluid[z_mid]

    # Compute vorticity magnitude and divergence
    om_gt = np.sqrt(np.sum(curl_h(gt[:3], sp_m, mask=geom[0]).astype(np.float64) ** 2, axis=0))[z_mid]
    om_a = np.sqrt(np.sum(curl_h(pa[:3], sp_m, mask=geom[0]).astype(np.float64) ** 2, axis=0))[z_mid]
    om_b = np.sqrt(np.sum(curl_h(pb[:3], sp_m, mask=geom[0]).astype(np.float64) ** 2, axis=0))[z_mid]

    dv_gt = div_h(gt[0], gt[1], gt[2], sp_m, mask=geom[0])[z_mid]
    dv_a = div_h(pa[0], pa[1], pa[2], sp_m, mask=geom[0])[z_mid]
    dv_b = div_h(pb[0], pb[1], pb[2], sp_m, mask=geom[0])[z_mid]

    def _m(arr2d: np.ndarray) -> np.ma.MaskedArray:
        return np.ma.masked_where(~mask_2d, arr2d)

    rows = [
        ("Streamwise u [m/s]", gt[0, z_mid], pa[0, z_mid], pb[0, z_mid], "RdBu_r", False),
        ("Pressure p [Pa]", gt[3, z_mid], pa[3, z_mid], pb[3, z_mid], "coolwarm", False),
        ("Vorticity |w| [s^-1]", om_gt, om_a, om_b, "inferno", True),
        ("Divergence div(u) [s^-1]", dv_gt, dv_a, dv_b, "PuOr", False),
    ]

    fig, axes = plt.subplots(4, 5, figsize=(23.5, 13.5), dpi=180, constrained_layout=True)
    col_titles = ["Ground Truth (CFD)", "Head-A Prediction", "Head-B Prediction", "|Head-A - GT| Error", "|Head-B - GT| Error"]

    for r, (row_lbl, f_gt, f_a, f_b, cmap_name, non_neg) in enumerate(rows):
        cmap = plt.get_cmap(cmap_name).copy()
        cmap.set_bad("#2b2b2b")
        cmap_err = plt.get_cmap("magma").copy()
        cmap_err.set_bad("#2b2b2b")

        vals_gt = f_gt[mask_2d]
        if non_neg:
            vmin, vmax = 0.0, float(np.percentile(vals_gt, 99.2))
        elif r == 3:
            vlim = max(float(np.percentile(np.abs(vals_gt), 99.0)) * 2.5, 0.5)
            vmin, vmax = -vlim, vlim
        else:
            vmin = float(np.percentile(vals_gt, 1.0))
            vmax = float(np.percentile(vals_gt, 99.0))

        err_a = np.abs(f_a - f_gt)
        err_b = np.abs(f_b - f_gt)
        emax = max(float(np.percentile(np.concatenate([err_a[mask_2d], err_b[mask_2d]]), 99.0)), 1e-6)

        panels = [_m(f_gt), _m(f_a), _m(f_b), _m(err_a), _m(err_b)]
        for c, arr in enumerate(panels):
            ax = axes[r, c]
            if c < 3:
                im = ax.imshow(arr, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
            else:
                im = ax.imshow(arr, origin="lower", cmap=cmap_err, vmin=0.0, vmax=emax, aspect="auto")
            if r == 0:
                ax.set_title(col_titles[c], fontweight="bold")
            if c == 0:
                ax.set_ylabel(row_lbl, fontweight="bold")
            ax.set_xticks([])
            ax.set_yticks([])
            if c == 2:
                fig.colorbar(im, ax=axes[r, :3], location="right", shrink=0.88, pad=0.015, aspect=18)
            if c == 4:
                fig.colorbar(im, ax=axes[r, 3:], location="right", shrink=0.88, pad=0.015, aspect=18)

    fig.suptitle(
        f"FlowWorld-3D G1 Held-Out Test Frame Reconstruction Slices (Mid-Span z={z_mid}, Case={case_id}, Step={target_item['step_index']})\n"
        "Direct velocity (Head-A) and potential-based velocity (Head-B). Dark gray marks solid cells.",
        fontweight="bold",
    )
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_fig3_physics_and_spectrum(
    frames_a: list[dict[str, Any]],
    frames_b: list[dict[str, Any]],
    out_path: Path,
) -> None:
    case_re = {r["case_id"]: r["reynolds_number"] for r in frames_a}
    cases = sorted(case_re, key=lambda cid: (case_re[cid], cid))
    short_lbl = [f"Re={case_re[cid]:g}" for cid in cases]

    def _by_case(frames: list[dict[str, Any]], key: str) -> list[float]:
        res = []
        for cid in cases:
            sub = [r[key] for r in frames if r["case_id"] == cid]
            res.append(float(np.mean(sub)))
        return res

    n_rows = 1 + (len(cases) + 2) // 3
    fig, axes = plt.subplots(n_rows, 3, figsize=(19.0, 5.1 * n_rows), dpi=180, constrained_layout=True)
    x = np.arange(len(cases))
    w = 0.28

    # (0,0) Velocity & Pressure Rel-L2 (%)
    ax = axes[0, 0]
    ax.bar(x - w, [v * 100 for v in _by_case(frames_a, "rel_l2_vel")], width=w, label="Head-A Vel Rel-L2", color="#1f77b4")
    ax.bar(x, [v * 100 for v in _by_case(frames_b, "rel_l2_vel")], width=w, label="Head-B Vel Rel-L2", color="#d62728")
    ax.bar(x + w, [v * 100 for v in _by_case(frames_a, "rel_l2_p")], width=w * 0.5, label="Head-A P Rel-L2", color="#aec7e8")
    ax.bar(x + 1.5 * w, [v * 100 for v in _by_case(frames_b, "rel_l2_p")], width=w * 0.5, label="Head-B P Rel-L2", color="#ff9896")
    ax.set_xticks(x + 0.15)
    ax.set_xticklabels(short_lbl)
    ax.set_ylabel("Relative L2 Error [%]")
    ax.set_title("Held-Out Test Velocity & Pressure Rel-L2 (%)")
    ax.grid(True, axis="y", alpha=0.28)
    ax.legend(fontsize=8)

    # (0,1) First-Layer Near-Wall Velocity RMSE (m/s) & Outer Face RMSE
    ax = axes[0, 1]
    ax.bar(x - w, _by_case(frames_a, "near_wall_vel_rmse_ms"), width=w, label="Head-A Near-Wall", color="#1f77b4")
    ax.bar(x, _by_case(frames_b, "near_wall_vel_rmse_ms"), width=w, label="Head-B Near-Wall", color="#d62728")
    ax.bar(x + w, _by_case(frames_a, "outer_face_vel_rmse_ms"), width=w * 0.5, label="Head-A Outer Face", color="#aec7e8")
    ax.bar(x + 1.5 * w, _by_case(frames_b, "outer_face_vel_rmse_ms"), width=w * 0.5, label="Head-B Outer Face", color="#ff9896")
    ax.set_xticks(x + 0.15)
    ax.set_xticklabels(short_lbl)
    ax.set_ylabel("Velocity RMSE [m/s]")
    ax.set_title("Near-Wall 1st-Shell & Outer-Face Velocity RMSE")
    ax.grid(True, axis="y", alpha=0.28)
    ax.legend(fontsize=8)

    # (0,2) Interior Divergence RMS on m2 (div(curl) stencil) vs GT
    ax = axes[0, 2]
    ax.bar(x - w, _by_case(frames_a, "div_rms_m2_gt_s_inv"), width=w, label="Ground Truth CFD", color="#2ca02c")
    ax.bar(x, _by_case(frames_a, "div_rms_m2_pred_s_inv"), width=w, label="Head-A", color="#1f77b4")
    ax.bar(x + w, _by_case(frames_b, "div_rms_m2_pred_s_inv"), width=w, label="Head-B", color="#d62728")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(short_lbl)
    ax.set_ylabel("Interior Div RMS m2 [s^-1] (log)")
    ax.set_title("Interior Discrete Divergence RMS (m2 Stencil)")
    ax.grid(True, axis="y", which="both", alpha=0.28)
    ax.legend(fontsize=8)

    spectrum_axes = axes[1:, :].ravel()
    for ax in spectrum_axes[len(cases):]:
        ax.set_visible(False)
    for ax, cid, lbl in zip(spectrum_axes, cases, short_lbl):
        sub_a = [r for r in frames_a if r["case_id"] == cid]
        sub_b = [r for r in frames_b if r["case_id"] == cid]
        if not sub_a:
            continue
        k = np.asarray(sub_a[0]["spectrum_k_rad_m"])
        ek_gt = np.mean([r["spectrum_ek_gt"] for r in sub_a], axis=0)
        ek_a = np.mean([r["spectrum_ek_pred"] for r in sub_a], axis=0)
        ek_b = np.mean([r["spectrum_ek_pred"] for r in sub_b], axis=0)

        ax.loglog(k, np.maximum(ek_gt, 1e-16), "k-", lw=2.2, label="Ground Truth CFD")
        ax.loglog(k, np.maximum(ek_a, 1e-16), "o--", color="#1f77b4", lw=1.7, ms=3.5, label=f"Head-A (logRMSE={np.mean([r['spectrum_log10_rmse'] for r in sub_a]):.3f})")
        ax.loglog(k, np.maximum(ek_b, 1e-16), "s:", color="#d62728", lw=1.7, ms=3.5, label=f"Head-B (logRMSE={np.mean([r['spectrum_log10_rmse'] for r in sub_b]):.3f})")
        ax.set_title(f"3D Kinetic Energy Spectrum E(k) — {lbl}")
        ax.set_xlabel("Wavenumber k [rad/m]")
        ax.set_ylabel("Shell-integrated kinetic energy [m²/s²]")
        ax.grid(True, which="both", alpha=0.28)
        ax.legend(fontsize=8)

    fig.suptitle(
        f"FlowWorld-3D G1 physical reconstruction on {len(frames_a)} held-out frames "
        f"across {len(cases)} trajectories",
        fontweight="bold",
    )
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="G1 Head-A vs Head-B held-out test evaluation")
    parser.add_argument("--head-a-dir", type=str, required=True)
    parser.add_argument("--head-b-dir", type=str, required=True)
    parser.add_argument("--dataset-dir", type=str, default="/root/autodl-tmp/datasets/world_model")
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(f"[G1 Eval] Loading Head-A from {args.head_a_dir} ...")
    model_a, sum_a, meta_a = _load_best_model(Path(args.head_a_dir), device, "head_a")
    print(f"[G1 Eval] Loading Head-B from {args.head_b_dir} ...")
    model_b, sum_b, meta_b = _load_best_model(Path(args.head_b_dir), device, "head_b")

    print(f"[G1 Eval] Loading strictly held-out TEST split from {args.dataset_dir} ...")
    test_ds, split_record = _held_out_split(args.dataset_dir, sum_a, sum_b)
    print(f"[G1 Eval] Held-out Test frames: {len(test_ds)}")

    print("[G1 Eval] Evaluating Head-A on Test split (FP32 SI)...")
    frames_a, preds_a = _evaluate_model_on_dataset(model_a, test_ds, device)
    print("[G1 Eval] Evaluating Head-B on Test split (FP32 SI)...")
    frames_b, preds_b = _evaluate_model_on_dataset(model_b, test_ds, device)

    def _summarize(frames: list[dict[str, Any]]) -> dict[str, Any]:
        cases = sorted({r["case_id"] for r in frames})
        per_case = {cid: _aggregate_metrics([r for r in frames if r["case_id"] == cid]) for cid in cases}
        return {
            "overall": _aggregate_metrics(frames),
            "pooled_physical": _pooled_physical_metrics(frames),
            "per_case": per_case,
        }

    report = {
        "test_frame_count": len(test_ds),
        "split_identity": split_record,
        "selection": _select_head(frames_a, frames_b, meta_a["loss_cfg"], meta_b["loss_cfg"]),
        "aggregation": "overall and per_case are means over frames. pooled_physical uses all relevant voxels.",
        "head_a": {
            "meta": meta_a,
            "summary": _summarize(frames_a),
            "frames": frames_a,
        },
        "head_b": {
            "meta": meta_b,
            "summary": _summarize(frames_b),
            "frames": frames_b,
        },
    }
    report_path = out_dir / "g1_ab_evaluation_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(f"[G1 Eval] Saved JSON report: {report_path}")

    print("[G1 Eval] Generating 3 diagnostic comparison figures...")
    _plot_fig1_training_curves(sum_a, sum_b, out_dir / "g1_fig1_training_curves_ab.png")
    _plot_fig2_reconstruction_slices(test_ds, preds_a, preds_b, out_dir / "g1_fig2_reconstruction_slices_ab.png")
    _plot_fig3_physics_and_spectrum(frames_a, frames_b, out_dir / "g1_fig3_physics_and_spectrum_ab.png")
    print(f"[G1 Eval] Done! Outputs saved in {out_dir}")


if __name__ == "__main__":
    main()
