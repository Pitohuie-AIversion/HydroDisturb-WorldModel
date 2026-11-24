"""Real CUDA/CPU full-size Stage 1 smoke & elastic n-GPU VRAM/latency benchmark for FlowWorld-3D (R2)."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import copy
from dataclasses import replace
import json
import math
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from flow3d.data.fixtures import make_analytical_divergence_free_field_3d, make_cylinder_geometry_3d
from flow3d.data.normalization import Flow3DNormalizer
from flow3d.data.schema import load_yaml_or_json_config
from flow3d.models import (
    Flow3DStage1Autoencoder,
    VectorPotentialCurlHead3D,
    configure_elastic_hardware,
    resolve_elastic_runtime_plan,
    torch_available,
)
from flow3d.physics.torch_operators import torch_div_h, torch_valid_div_of_curl_mask


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Flow3D Stage 1 full-size GPU/CPU forward+backward smoke with elastic n-GPU scaling"
    )
    parser.add_argument("--config", type=str, default="configs/flow3d/stage1.yaml")
    parser.add_argument("--device", type=str, default="auto", help="'auto', 'cuda', 'cuda:0', or 'cpu'")
    parser.add_argument(
        "--num-gpus",
        type=str,
        default="auto",
        help="Elastic GPU count n ('auto' to use all available GPUs, or any natural number n >= 1, or 0 for CPU)",
    )
    parser.add_argument("--shape", type=int, nargs=5, default=[1, 1, 64, 96, 192], metavar=("B", "T", "Z", "Y", "X"))
    parser.add_argument(
        "--latent-norm", type=str, default=None, choices=["group_norm", "dataset_rms", "affine_free_rmsnorm"]
    )
    parser.add_argument("--amp", action="store_true", help="Enable FP16/BF16 autocast during forward (operators stay FP32+)")
    parser.add_argument("--skip-helmholtz-proj", action="store_true", help="Skip Head-A Helmholtz projection smoke")
    parser.add_argument(
        "--skip-elastic-batch",
        action="store_true",
        help="Skip Phase 2 elastic multi-GPU data-parallel forward+backward benchmark",
    )
    parser.add_argument(
        "--elastic-batch-size",
        type=int,
        default=None,
        help="Global batch size B for Phase 2 elastic multi-GPU smoke (defaults to plan.global_batch_size)",
    )
    return parser.parse_args()


def _sync_if_cuda(device) -> None:
    import torch

    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _resolve_amp_torch_dtype(amp_dtype: str):
    import torch

    norm = str(amp_dtype).strip().lower()
    if norm == "bf16":
        return torch.bfloat16
    if norm == "fp16":
        return torch.float16
    if norm == "fp32":
        return None
    raise ValueError(f"unsupported amp_dtype {amp_dtype!r}; expected 'bf16', 'fp16', or 'fp32'")


def _grid_spacings(grid: dict[str, Any]) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Return model spacing in diameters and physical spacing in meters."""
    diameter = float(grid["cylinder_diameter_m"])
    if not np.isfinite(diameter) or diameter <= 0:
        raise ValueError("cylinder_diameter_m must be positive and finite")
    spacing_m = grid.get("spacing_zyx_m")
    if spacing_m is None:
        spacing_m = np.asarray(grid.get("spacing_zyx", (0.01875 / diameter,) * 3)) * diameter
    spacing_m = np.asarray(spacing_m, dtype=float)
    if spacing_m.shape != (3,) or not np.isfinite(spacing_m).all() or not (spacing_m > 0).all():
        raise ValueError("grid spacing must contain three positive finite values")
    spacing_star = spacing_m / diameter
    if "spacing_zyx" in grid:
        supplied = np.asarray(grid["spacing_zyx"], dtype=float)
        if supplied.shape != (3,) or not np.allclose(supplied, spacing_star, rtol=1e-6, atol=0):
            raise ValueError("dimensionless spacing_zyx must equal spacing_zyx_m / cylinder_diameter_m")
    return tuple(spacing_star.tolist()), tuple(spacing_m.tolist())


def _smoke_loss(out):
    loss = out["phys_fields"].square().mean()
    for name in ("div_loss", "coulomb_gauge_loss", "harmonic_laplacian_loss"):
        if name in out["head_aux"]:
            loss = loss + 0.01 * out["head_aux"][name]
    probe = out.get("probe")
    if probe is not None:
        # Exercise both probe readouts during the synthetic backward benchmark.
        probe_loss = probe["enstrophy_tilde"].float().mean() + probe["log_spectrum_tilde"].float().square().mean()
        loss = loss + 0.01 * probe_loss
    return loss


def _check_smoke_backward(model, loss) -> dict[str, Any]:
    import torch

    if not bool(torch.isfinite(loss.detach())):
        raise RuntimeError("non-finite smoke loss")
    trainable = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("no trainable parameters in the smoke benchmark")
    missing = [name for name, p in trainable if p.grad is None]
    if missing:
        raise RuntimeError(f"missing gradients: {', '.join(missing)}")
    maxima = {name: p.grad.detach().abs().amax().item() for name, p in trainable}
    bad = [name for name, maximum in maxima.items() if not math.isfinite(maximum)]
    if bad:
        raise RuntimeError(f"non-finite gradients: {', '.join(bad)}")
    allowed_zero = set()
    if isinstance(getattr(model, "head", None), VectorPotentialCurlHead3D):
        # A spatially constant shift vanishes under curl(A), div(A), and grad(phi).
        # No other parameter is exempt, even within those convolutions.
        allowed_zero = {"head.a_proj.bias", "head.phi_proj.bias"}.intersection(maxima)
    zero = {name for name, maximum in maxima.items() if maximum == 0}
    unexpected = zero - allowed_zero
    if unexpected:
        raise RuntimeError(f"zero gradient tensors: {', '.join(sorted(unexpected))}")
    return {
        "all_grads_finite": True,
        "checked_gradient_tensors": len(trainable),
        "allowed_zero_gradient_tensors": sorted(allowed_zero),
        "zero_gradient_tensors": sorted(zero),
    }


def _compute_physics_diagnostics(phys, geom, spacing_m: tuple[float, float, float]) -> dict[str, float]:
    import torch

    with torch.no_grad():
        bsz, t_len = phys.shape[:2]
        fluid_mask = (geom[:, 0] > 0.5).unsqueeze(1).expand(bsz, t_len, *geom.shape[-3:])
        solid_mask = ~fluid_mask
        div_phys = torch_div_h(phys[:, :, 0], phys[:, :, 1], phys[:, :, 2], spacing_m, mask=geom[:, 0])
        interior_mask = (torch_valid_div_of_curl_mask(geom[:, 0]) > 0.5).unsqueeze(1).expand(
            bsz, t_len, *geom.shape[-3:]
        )
        div_rms_fluid = float(div_phys[fluid_mask].pow(2).mean().sqrt().item()) if bool(fluid_mask.any()) else 0.0
        div_interior_max = float(div_phys[interior_mask].abs().max().item()) if bool(interior_mask.any()) else 0.0
        vel = phys[:, :, :3]
        solid_vel_mask = solid_mask.unsqueeze(2).expand_as(vel)
        solid_max = float(vel[solid_vel_mask].abs().max().item()) if bool(solid_vel_mask.any()) else 0.0
    return {
        "div_phys_rms_fluid": div_rms_fluid,
        "div_phys_interior_max_abs": div_interior_max,
        "solid_interior_max_abs": solid_max,
    }


def _run_head_smoke(
    label: str,
    cfg: dict[str, Any],
    x_norm_host,
    geom_host,
    b_stats_host,
    spacing_star: tuple[float, float, float],
    spacing_m: tuple[float, float, float],
    device_str: str,
    use_amp: bool,
    amp_dtype: str,
    non_blocking: bool,
    dev_lock: threading.Lock,
) -> dict[str, Any]:
    import torch

    device = torch.device(device_str)
    amp_torch_dtype = _resolve_amp_torch_dtype(amp_dtype)
    amp_active = bool(use_amp and device.type == "cuda" and amp_torch_dtype is not None)
    with dev_lock:
        if device.type == "cuda":
            torch.cuda.set_device(device)
        x_norm = x_norm_host.to(device, non_blocking=non_blocking)
        geom = geom_host.to(device, non_blocking=non_blocking)
        b_stats = b_stats_host.to(device, non_blocking=non_blocking)

        model = Flow3DStage1Autoencoder.from_config(cfg).to(device)
        model.train()

        # 1 warmup step to initialize cuDNN/CUDA kernels on this device before timed measurement
        model.zero_grad(set_to_none=True)
        _sync_if_cuda(device)
        w0 = time.perf_counter()
        amp_ctx = (
            torch.amp.autocast(device_type="cuda", dtype=amp_torch_dtype) if amp_active else nullcontext()
        )
        with amp_ctx:
            w_out = model(
                x_norm,
                geom,
                b=b_stats,
                spacing_zyx=spacing_star,
                run_probe=bool(cfg.get("probe", {}).get("enabled", False)),
            )
        w_loss = _smoke_loss(w_out)
        w_loss.backward()
        _sync_if_cuda(device)
        warmup_ms = round((time.perf_counter() - w0) * 1000.0, 3)
        del w_out, w_loss
        model.zero_grad(set_to_none=True)

        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)

        _sync_if_cuda(device)
        t0 = time.perf_counter()
        amp_ctx = (
            torch.amp.autocast(device_type="cuda", dtype=amp_torch_dtype) if amp_active else nullcontext()
        )
        with amp_ctx:
            out = model(
                x_norm,
                geom,
                b=b_stats,
                spacing_zyx=spacing_star,
                run_probe=bool(cfg.get("probe", {}).get("enabled", False)),
            )
        _sync_if_cuda(device)
        t1 = time.perf_counter()

        loss = _smoke_loss(out)

        t2 = time.perf_counter()
        loss.backward()
        _sync_if_cuda(device)
        t3 = time.perf_counter()

        gradient_checks = _check_smoke_backward(model, loss)

        diag = _compute_physics_diagnostics(out["phys_fields"].detach(), geom, spacing_m)

        stats: dict[str, Any] = {
            "mode": label,
            "device": str(device),
            "head_type": cfg["head"]["type"],
            "latent_norm_type": cfg["latent_norm"]["type"],
            "amp_dtype": (amp_dtype if amp_active else "fp32"),
            "amp_bf16": bool(amp_active and amp_torch_dtype == torch.bfloat16),
            "helmholtz_projection": bool(cfg["head"].get("helmholtz_projection", False)),
            "param_count": int(sum(p.numel() for p in model.parameters())),
            "latent_shape": list(out["z_clean"].shape),
            "norm_fields_shape": list(out["norm_fields"].shape),
            "phys_fields_shape": list(out["phys_fields"].shape),
            "warmup_total_ms": warmup_ms,
            "forward_ms": round((t1 - t0) * 1000.0, 3),
            "backward_ms": round((t3 - t2) * 1000.0, 3),
            "loss": float(loss.detach().item()),
            **gradient_checks,
            **diag,
        }
        for name in ("div_loss", "coulomb_gauge_loss", "harmonic_laplacian_loss"):
            if name in out["head_aux"]:
                stats[name] = float(out["head_aux"][name].detach().item())

        if device.type == "cuda":
            alloc_bytes = torch.cuda.max_memory_allocated(device)
            res_bytes = torch.cuda.max_memory_reserved(device)
            stats["max_memory_allocated_mib"] = round(alloc_bytes / (1024.0 ** 2), 2)
            stats["max_memory_allocated_gib"] = round(alloc_bytes / (1024.0 ** 3), 4)
            stats["max_memory_reserved_mib"] = round(res_bytes / (1024.0 ** 2), 2)
            stats["max_memory_reserved_gib"] = round(res_bytes / (1024.0 ** 3), 4)

        del model, out, loss, x_norm, geom, b_stats
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return stats


def _run_elastic_multi_gpu_batch_smoke(
    cfg_base: dict[str, Any],
    field_np: np.ndarray,
    geom_np: np.ndarray,
    spacing_star: tuple[float, float, float],
    spacing_m: tuple[float, float, float],
    plan,
    t_len: int,
    nz: int,
    ny: int,
    nx: int,
    elastic_batch_size: int | None = None,
    use_amp: bool = False,
) -> dict[str, Any]:
    """Run an elastic B=n (or custom --elastic-batch-size B) forward+backward pass across elastic GPUs."""
    import torch

    n_gpus = int(plan.active_gpus)
    global_b = max(1, int(plan.global_batch_size)) if elastic_batch_size is None else int(elastic_batch_size)
    if global_b < 1:
        raise ValueError(f"--elastic-batch-size must be >= 1, got {global_b}")
    fields_bt = np.broadcast_to(field_np[None, None, ...], (global_b, t_len, 4, nz, ny, nx)).copy()
    geom_b = np.broadcast_to(geom_np[None, ...], (global_b, 2, nz, ny, nx)).copy()
    normalizer = Flow3DNormalizer()
    norm_np, b_np = normalizer.normalize(fields_bt, geom_np[0])

    x_norm = torch.from_numpy(norm_np)
    geom = torch.from_numpy(geom_b)
    b_stats = torch.from_numpy(b_np)
    if plan.pin_memory:
        x_norm = x_norm.pin_memory()
        geom = geom.pin_memory()
        b_stats = b_stats.pin_memory()

    cfg_elastic = copy.deepcopy(cfg_base)
    cfg_elastic["head"]["type"] = "head_b"
    primary_dev = torch.device(plan.primary_device)
    model = Flow3DStage1Autoencoder.from_config(cfg_elastic).to(primary_dev)
    model.train()

    amp_torch_dtype = _resolve_amp_torch_dtype(plan.amp_dtype)
    amp_active = bool(use_amp and n_gpus >= 1 and amp_torch_dtype is not None)

    # 1 warmup step across active elastic devices before timed measurement & peak VRAM tracking
    model.zero_grad(set_to_none=True)
    for dev_id in plan.device_ids:
        torch.cuda.synchronize(dev_id)
    w0 = time.perf_counter()
    amp_ctx = torch.amp.autocast(device_type="cuda", dtype=amp_torch_dtype) if amp_active else nullcontext()
    with amp_ctx:
        w_out = model.forward_elastic_multi_gpu(
            x_norm,
            geom,
            b=b_stats,
            spacing_zyx=spacing_star,
            run_probe=True,
            device_ids=plan.device_ids,
            non_blocking=plan.non_blocking,
        )
    w_loss = _smoke_loss(w_out)
    w_loss.backward()
    for dev_id in plan.device_ids:
        torch.cuda.synchronize(dev_id)
    warmup_ms = round((time.perf_counter() - w0) * 1000.0, 3)
    del w_out, w_loss
    model.zero_grad(set_to_none=True)

    for dev_id in plan.device_ids:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(dev_id)
        torch.cuda.synchronize(dev_id)

    t0 = time.perf_counter()
    amp_ctx = torch.amp.autocast(device_type="cuda", dtype=amp_torch_dtype) if amp_active else nullcontext()
    with amp_ctx:
        out = model.forward_elastic_multi_gpu(
            x_norm,
            geom,
            b=b_stats,
            spacing_zyx=spacing_star,
            run_probe=True,
            device_ids=plan.device_ids,
            non_blocking=plan.non_blocking,
        )
    for dev_id in plan.device_ids:
        torch.cuda.synchronize(dev_id)
    t1 = time.perf_counter()

    loss = _smoke_loss(out)
    t2 = time.perf_counter()
    loss.backward()
    for dev_id in plan.device_ids:
        torch.cuda.synchronize(dev_id)
    t3 = time.perf_counter()

    gradient_checks = _check_smoke_backward(model, loss)

    phys_det = out["phys_fields"].detach()
    diag = _compute_physics_diagnostics(phys_det, geom.to(phys_det.device), spacing_m)

    per_gpu_vram: list[dict[str, Any]] = []
    for dev_id in plan.device_ids:
        alloc_b = torch.cuda.max_memory_allocated(dev_id)
        res_b = torch.cuda.max_memory_reserved(dev_id)
        per_gpu_vram.append(
            {
                "device": f"cuda:{dev_id}",
                "max_memory_allocated_mib": round(alloc_b / (1024.0 ** 2), 2),
                "max_memory_allocated_gib": round(alloc_b / (1024.0 ** 3), 4),
                "max_memory_reserved_mib": round(res_b / (1024.0 ** 2), 2),
                "max_memory_reserved_gib": round(res_b / (1024.0 ** 3), 4),
            }
        )

    fwd_ms = round((t1 - t0) * 1000.0, 3)
    bwd_ms = round((t3 - t2) * 1000.0, 3)
    total_s = max(1e-6, (t1 - t0) + (t3 - t2))

    summary = {
        "mode": f"elastic_n_gpu_data_parallel_B{global_b}_across_{n_gpus}_gpus",
        "plan_active_gpus": n_gpus,
        "active_gpus": int(out["elastic_runtime"]["active_gpus"]),
        "used_devices": out["elastic_runtime"]["used_devices"],
        "shard_batch_sizes": out["elastic_runtime"]["shard_batch_sizes"],
        "global_batch_shape_btzyx": [global_b, t_len, nz, ny, nx],
        "latent_shape": list(out["z_clean"].shape),
        "phys_fields_shape": list(out["phys_fields"].shape),
        "amp_dtype": (plan.amp_dtype if amp_active else "fp32"),
        "warmup_total_ms": warmup_ms,
        "forward_ms": fwd_ms,
        "backward_ms": bwd_ms,
        "aggregate_volumes_per_sec": round(float(global_b * t_len) / total_s, 3),
        **gradient_checks,
        "coulomb_gauge_loss": float(out["head_aux"]["coulomb_gauge_loss"].detach().item()),
        **diag,
        "per_gpu_memory": per_gpu_vram,
    }
    del model, out, loss, x_norm, geom, b_stats
    if n_gpus > 0:
        torch.cuda.empty_cache()
    return summary


def main() -> None:
    args = _parse_args()
    if not torch_available():
        raise SystemExit("PyTorch is required for the Stage 1 smoke benchmark. Install .[torch].")

    import torch

    bsz, t_len, nz, ny, nx = (int(v) for v in args.shape)
    cfg_base = load_yaml_or_json_config(Path(args.config))
    if args.latent_norm is not None:
        cfg_base["latent_norm"]["type"] = args.latent_norm
    cfg_base["probe"]["enabled"] = True

    explicit_dev_idx: int | None = None
    if args.device == "cpu":
        req_gpus: int | str = 0
    elif args.device.startswith("cuda:"):
        explicit_dev_idx = int(args.device.split(":", 1)[1])
        avail = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if explicit_dev_idx < 0 or explicit_dev_idx >= avail:
            raise ValueError(f"requested device {args.device!r} out of range for {avail} CUDA device(s)")
        req_gpus = 1
    elif args.device in ("auto", "cuda"):
        req_gpus = args.num_gpus
    else:
        raise ValueError(f"unsupported device {args.device!r}; expected 'auto', 'cpu', 'cuda', or 'cuda:<id>'")

    plan = resolve_elastic_runtime_plan(cfg_base, num_gpus=req_gpus, strict_gpu_count=True)
    if args.device == "cuda" and plan.active_gpus == 0:
        raise ValueError("--device cuda requires at least one active CUDA GPU")
    if args.amp and (plan.active_gpus == 0 or plan.amp_dtype == "fp32"):
        raise ValueError("--amp requires CUDA and runtime.amp_dtype set to bf16 or fp16")
    if explicit_dev_idx is not None and plan.active_gpus == 1:
        dev_str = f"cuda:{explicit_dev_idx}"
        plan = replace(plan, device_ids=(explicit_dev_idx,), devices=(dev_str,), primary_device=dev_str)
    hw_applied = configure_elastic_hardware(plan)

    spacing_star, spacing_m = _grid_spacings(cfg_base["grid"])
    diameter = float(cfg_base["grid"]["cylinder_diameter_m"])

    geom_np, _ = make_cylinder_geometry_3d((nz, ny, nx), spacing_m, cylinder_diameter_m=diameter)
    field_np, _ = make_analytical_divergence_free_field_3d((nz, ny, nx), spacing_m, u0_m_s=0.5)
    fluid_mask_np = geom_np[0] > 0.5
    field_np[:, ~fluid_mask_np] = 0.0

    fields_bt = np.broadcast_to(field_np[None, None, ...], (bsz, t_len, 4, nz, ny, nx)).copy()
    geom_b = np.broadcast_to(geom_np[None, ...], (bsz, 2, nz, ny, nx)).copy()

    normalizer = Flow3DNormalizer()
    norm_np, b_np = normalizer.normalize(fields_bt, geom_np[0])

    x_norm_host = torch.from_numpy(norm_np)
    geom_host = torch.from_numpy(geom_b)
    b_stats_host = torch.from_numpy(b_np)
    if plan.pin_memory:
        x_norm_host = x_norm_host.pin_memory()
        geom_host = geom_host.pin_memory()
        b_stats_host = b_stats_host.pin_memory()

    # Build the Stage 1 verification & ablation task suite and map across n elastic GPUs
    task_specs: list[tuple[str, dict[str, Any], bool, str]] = []

    # 1. Head-A (Direct 4ch, GroupNorm)
    cfg_a = copy.deepcopy(cfg_base)
    cfg_a["head"]["type"] = "head_a"
    cfg_a["head"]["helmholtz_projection"] = False
    task_specs.append(("head_a", cfg_a, args.amp, plan.amp_dtype))

    # 2. Head-B (Vector Potential A* -> curl_h(mask=None) + Coulomb gauge, GroupNorm)
    cfg_b = copy.deepcopy(cfg_base)
    cfg_b["head"]["type"] = "head_b"
    task_specs.append(("head_b", cfg_b, args.amp, plan.amp_dtype))

    # 3. Head-A with differentiable Helmholtz projection (FP32)
    if not args.skip_helmholtz_proj:
        cfg_a_proj = copy.deepcopy(cfg_base)
        cfg_a_proj["head"]["type"] = "head_a"
        cfg_a_proj["head"]["helmholtz_projection"] = True
        task_specs.append(("head_a_helmholtz_proj", cfg_a_proj, args.amp, plan.amp_dtype))

    # 4. When multiple GPUs are active, utilize remaining elastic GPUs concurrently
    if plan.active_gpus >= 2:
        cfg_b_rmsnorm = copy.deepcopy(cfg_base)
        cfg_b_rmsnorm["head"]["type"] = "head_b"
        cfg_b_rmsnorm["latent_norm"]["type"] = "affine_free_rmsnorm"
        task_specs.append(("head_b_affine_free_rmsnorm", cfg_b_rmsnorm, args.amp, plan.amp_dtype))

        cfg_b_bf16 = copy.deepcopy(cfg_base)
        cfg_b_bf16["head"]["type"] = "head_b"
        cfg_b_bf16["latent_norm"]["type"] = "dataset_rms"
        bf16_or_plan_amp = "bf16" if plan.amp_dtype == "fp32" else plan.amp_dtype
        task_specs.append(("head_b_dataset_rms_bf16", cfg_b_bf16, True, bf16_or_plan_amp))

    device_locks = {dev_str: threading.Lock() for dev_str in plan.devices}
    t_pool_start = time.perf_counter()
    max_workers = max(1, len(plan.devices))
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = []
        for idx, (label, cfg_item, use_amp_flag, amp_dtype_item) in enumerate(task_specs):
            target_dev = plan.task_device_for_index(idx)
            lock = device_locks.setdefault(target_dev, threading.Lock())
            futures.append(
                pool.submit(
                    _run_head_smoke,
                    label,
                    cfg_item,
                    x_norm_host,
                    geom_host,
                    b_stats_host,
                    spacing_star,
                    spacing_m,
                    target_dev,
                    use_amp_flag,
                    amp_dtype_item,
                    plan.non_blocking,
                    lock,
                )
            )
        results = [fut.result() for fut in futures]
    pool_wall_ms = round((time.perf_counter() - t_pool_start) * 1000.0, 3)

    elastic_batch_result: dict[str, Any] | None = None
    if not args.skip_elastic_batch and plan.active_gpus >= 1:
        elastic_batch_result = _run_elastic_multi_gpu_batch_smoke(
            cfg_base,
            field_np,
            geom_np,
            spacing_star,
            spacing_m,
            plan,
            t_len,
            nz,
            ny,
            nx,
            elastic_batch_size=args.elastic_batch_size,
            use_amp=args.amp,
        )

    device_info: dict[str, Any] = {
        "primary_device": plan.primary_device,
        "active_gpus": plan.active_gpus,
        "devices": list(plan.devices),
        "torch_version": torch.__version__,
        "amp_default": bool(args.amp),
        "amp_dtype": plan.amp_dtype,
        "hardware_config_applied": hw_applied,
        "parallel_pool_wall_ms": pool_wall_ms,
    }
    if plan.active_gpus > 0 and torch.cuda.is_available():
        prop_idx = plan.device_ids[0] if plan.device_ids else 0
        props = torch.cuda.get_device_properties(prop_idx)
        device_info["gpu_name"] = props.name
        device_info["gpu_total_memory_gib"] = round(props.total_memory / (1024.0 ** 3), 2)

    print(
        json.dumps(
            {
                "device_info": device_info,
                "elastic_runtime_plan": plan.to_dict(),
                "shape_btzyx": [bsz, t_len, nz, ny, nx],
                "spacing_zyx_star": list(spacing_star),
                "spacing_zyx_m": list(spacing_m),
                "runs": results,
                "elastic_multi_gpu_batch_smoke": elastic_batch_result,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
