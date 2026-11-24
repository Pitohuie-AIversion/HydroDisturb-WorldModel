"""Stage 1 Physics Heads (Head-A & Head-B), PhysicalReadout3D, and Composite Autoencoder.

- `DirectFieldHead3D` (Head-A): linear Conv3d predicting unbounded `[u~, v~, w~, p~]`
  with optional soft divergence residual and differentiable boundary-aware Helmholtz projection.
- `VectorPotentialCurlHead3D` (Head-B): predicts `A* [3ch]` + `p~ [1ch]` (+ optional harmonic
  mean-flow separation branch) -> compatible `torch_curl_h(A*)` + Coulomb gauge regularization.
- `PhysicalReadout3D`: FP32+ inverse of `Flow3DNormalizer` using paired `b [B, (T,) 6]`
  and known `fluid_mask` via `torch.where(fluid_mask > 0.5, phys, 0.0)`.
- `Flow3DStage1Autoencoder` (alias `Flow3DStage1Model`): full Stage 1 composition with `from_config`.
"""

from __future__ import annotations

import os
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Literal, Mapping, Sequence

from flow3d.contracts import CYLINDER3D_STATE_SPEC, PhysicsContractError
from flow3d.models.dc_ae3d import DCAE3DDecoder, DCAE3DEncoder, LatentNorm3D
from flow3d.models.probe import LatentPhysicsProbe
from flow3d.models.spatial_transformer import SpatialTransformer3D
from flow3d.physics.torch_operators import (
    torch_build_full_3d_stencil_mask,
    torch_curl_h,
    torch_div_h,
    torch_grad_h,
    torch_helmholtz_project,
    torch_valid_div_of_curl_mask,
)

try:  # Keep PyTorch optional for protocol-only environments.
    import torch
    import torch.nn as nn
except Exception:  # pragma: no cover - exercised only when torch is absent
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]


def torch_available() -> bool:
    return torch is not None


def _trainer_shard_output(output: Mapping[str, Any], phase: Literal["train", "validation"]) -> dict[str, Any]:
    """Select references before gathering, keeping the original tensors and autograd graph."""
    if phase == "validation":
        return {"norm_fields": output["norm_fields"], "phys_fields": output["phys_fields"]}
    if phase != "train":
        raise ValueError(f"unknown trainer phase {phase!r}")
    aux = output["head_aux"]
    losses = ("div_loss", "coulomb_gauge_loss", "harmonic_laplacian_loss")
    return {"norm_fields": output["norm_fields"], "head_aux": {key: aux[key] for key in losses if key in aux}}


def _require_torch() -> None:
    if torch is None:
        raise ImportError("flow3d.models.heads requires PyTorch. Install with `pip install -e .[torch]`.")


_PROC = Path("/proc")


def _read_kernel_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeError):
        return ""


def _cgroup_directories(controller: str) -> Iterator[tuple[Path, bool]]:
    """Yield visible cgroup ancestors for the process on unified and v1 mounts."""
    groups = {}
    for line in _read_kernel_file(_PROC / "self/cgroup").splitlines():
        fields = line.split(":", 2)
        if len(fields) != 3:
            continue
        hierarchy, controllers, group = fields
        if hierarchy == "0" or controller in controllers.split(","):
            groups[hierarchy == "0"] = Path(group)
    if not groups:
        return
    for line in _read_kernel_file(_PROC / "self/mountinfo").splitlines():
        mount_info, _, fs_info = line.partition(" - ")
        mount_fields, fs_fields = mount_info.split(), fs_info.split()
        if len(mount_fields) < 5 or len(fs_fields) < 3:
            continue
        unified = fs_fields[0] == "cgroup2"
        if not unified and not (fs_fields[0] == "cgroup" and controller in fs_fields[2].split(",")):
            continue
        group = groups.get(unified)
        if group is None or not group.is_absolute() or ".." in group.parts:
            continue
        # mountinfo encodes literal spaces in path fields as backslash-040.
        root, mount = (Path(field.replace(r"\040", " ")) for field in mount_fields[3:5])
        # Private cgroup namespaces may report membership relative to the mount root.
        relative = group.relative_to(root) if group.is_relative_to(root) else group.relative_to("/")
        directory = mount / relative
        if not directory.is_dir():
            continue
        while True:
            yield directory, unified
            if directory == mount:
                break
            directory = directory.parent


def _detect_cpu_count() -> int:
    try:
        count = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        count = os.cpu_count() or 1
    for directory, unified in _cgroup_directories("cpu"):
        try:
            if unified:
                quota, period = map(int, _read_kernel_file(directory / "cpu.max").split())
            else:
                quota = int(_read_kernel_file(directory / "cpu.cfs_quota_us"))
                period = int(_read_kernel_file(directory / "cpu.cfs_period_us"))
        except ValueError:  # Includes unlimited ("max") or unavailable quota files.
            continue
        if quota > 0 and period > 0:
            count = min(count, (quota + period - 1) // period)
    return max(1, count)


def _detect_host_ram_gib() -> float:
    """Available RAM, bounded by the remaining hard limit at each cgroup ancestor."""
    bounds = []
    meminfo = {}
    for line in _read_kernel_file(_PROC / "meminfo").splitlines():
        fields = line.split()
        if len(fields) >= 2 and fields[0] in {"MemAvailable:", "MemTotal:"} and fields[1].isdigit():
            meminfo[fields[0]] = int(fields[1]) * 1024
    for key in ("MemAvailable:", "MemTotal:"):
        if key in meminfo:
            bounds.append(meminfo[key])
            break
    for directory, unified in _cgroup_directories("memory"):
        limit_name, usage_name = ("memory.max", "memory.current") if unified else (
            "memory.limit_in_bytes", "memory.usage_in_bytes",
        )
        limit = _read_kernel_file(directory / limit_name)
        if limit.isdigit():
            usage = _read_kernel_file(directory / usage_name)
            bounds.append(max(0, int(limit) - (int(usage) if usage.isdigit() else 0)))
    return round(min(bounds) / (1024 ** 3), 2) if bounds else 0.0


@dataclass(frozen=True)
class ElasticRuntimePlan:
    """Elastic n-GPU & full server resource (CPU / GPU / RAM) execution plan.

    Designed for elastic cloud environments where the available GPU count `n`
    is a variable natural number (`n in {1, 2, 3, ...}`, or `0` on CPU-only hosts).
    """

    requested_num_gpus: int | str
    active_gpus: int
    detected_gpus: int
    detected_cpus: int
    detected_ram_gib: float
    device_ids: tuple[int, ...]
    devices: tuple[str, ...]
    primary_device: str
    strategy: str
    per_gpu_batch_size: int
    global_batch_size: int
    scale_batch_linearly_with_n: bool
    maximize_all_resources: bool
    cpu_threads_total: int
    threads_per_gpu_worker: int
    dataloader_workers_per_gpu: int
    total_dataloader_workers: int
    pin_memory: bool
    non_blocking: bool
    persistent_workers: bool
    prefetch_factor: int
    allow_tf32: bool
    cudnn_benchmark: bool
    amp_dtype: str

    def task_device_for_index(self, task_idx: int) -> str:
        """Assign task `task_idx` across the `n` elastic devices via round-robin `task_idx % n`."""
        if int(task_idx) < 0:
            raise ValueError(f"task_idx must be non-negative, got {task_idx}")
        return self.devices[int(task_idx) % len(self.devices)]

    def to_dict(self) -> dict[str, Any]:
        return {
            "requested_num_gpus": self.requested_num_gpus,
            "active_gpus": self.active_gpus,
            "detected_gpus": self.detected_gpus,
            "detected_cpus": self.detected_cpus,
            "detected_ram_gib": self.detected_ram_gib,
            "device_ids": list(self.device_ids),
            "devices": list(self.devices),
            "primary_device": self.primary_device,
            "strategy": self.strategy,
            "per_gpu_batch_size": self.per_gpu_batch_size,
            "global_batch_size": self.global_batch_size,
            "scale_batch_linearly_with_n": self.scale_batch_linearly_with_n,
            "maximize_all_resources": self.maximize_all_resources,
            "cpu_threads_total": self.cpu_threads_total,
            "threads_per_gpu_worker": self.threads_per_gpu_worker,
            "dataloader_workers_per_gpu": self.dataloader_workers_per_gpu,
            "total_dataloader_workers": self.total_dataloader_workers,
            "pin_memory": self.pin_memory,
            "non_blocking": self.non_blocking,
            "persistent_workers": self.persistent_workers,
            "prefetch_factor": self.prefetch_factor,
            "allow_tf32": self.allow_tf32,
            "cudnn_benchmark": self.cudnn_benchmark,
            "amp_dtype": self.amp_dtype,
        }


def resolve_elastic_runtime_plan(
    cfg_or_runtime: Mapping[str, Any] | None = None,
    *,
    num_gpus: int | str | None = None,
    detected_gpus: int | None = None,
    detected_cpus: int | None = None,
    detected_ram_gib: float | None = None,
    strict_gpu_count: bool = False,
) -> ElasticRuntimePlan:
    """Resolve an elastic n-GPU and full-machine CPU/RAM utilization plan without hardcoded card counts."""
    raw_map = dict(cfg_or_runtime) if cfg_or_runtime is not None else {}
    rt = dict(raw_map.get("runtime", raw_map)) if isinstance(raw_map.get("runtime", raw_map), Mapping) else {}

    avail_gpus = (
        int(detected_gpus)
        if detected_gpus is not None
        else (int(torch.cuda.device_count()) if (torch is not None and torch.cuda.is_available()) else 0)
    )
    if avail_gpus < 0:
        raise ValueError(f"detected_gpus must be >= 0, got {avail_gpus}")

    avail_cpus = int(detected_cpus) if detected_cpus is not None else _detect_cpu_count()
    if avail_cpus < 1:
        raise ValueError(f"detected_cpus must be >= 1, got {avail_cpus}")

    avail_ram = float(detected_ram_gib) if detected_ram_gib is not None else _detect_host_ram_gib()

    req_raw = rt.get("num_gpus", "auto") if num_gpus is None else num_gpus
    if isinstance(req_raw, bool):
        raise ValueError("num_gpus must be 'auto' or an integer >= 0, not bool")

    if req_raw is None or (isinstance(req_raw, str) and req_raw.strip().lower() == "auto"):
        requested_num_gpus: int | str = "auto"
        active_n = avail_gpus
    else:
        try:
            req_int = int(req_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"num_gpus must be 'auto' or an integer >= 0, got {req_raw!r}") from exc
        if req_int < 0:
            raise ValueError(f"num_gpus must be >= 0 (natural number n for GPUs, or 0 for CPU), got {req_int}")
        requested_num_gpus = req_int
        if req_int > avail_gpus:
            if strict_gpu_count:
                raise ValueError(f"requested num_gpus={req_int} exceeds available GPUs ({avail_gpus})")
            active_n = avail_gpus
        else:
            active_n = req_int

    per_gpu_bs = int(rt.get("per_gpu_batch_size", 1))
    if per_gpu_bs < 1:
        raise ValueError(f"per_gpu_batch_size must be >= 1, got {per_gpu_bs}")

    scale_linear = bool(rt.get("scale_batch_linearly_with_n", True))
    effective_slots = max(1, active_n)
    global_bs = per_gpu_bs * effective_slots if scale_linear else per_gpu_bs

    raw_strategy = str(rt.get("strategy", "auto")).strip().lower()
    valid_strategies = ("auto", "cpu", "single_gpu", "data_parallel", "ddp", "worker_pool")
    if raw_strategy not in valid_strategies:
        raise ValueError(f"unknown runtime strategy {raw_strategy!r}; expected one of {valid_strategies}")
    if active_n == 0:
        strategy = "cpu"
    elif active_n == 1:
        strategy = "single_gpu" if raw_strategy in ("auto", "data_parallel", "ddp", "cpu") else raw_strategy
    else:
        strategy = "data_parallel" if raw_strategy in ("auto", "single_gpu", "cpu") else raw_strategy

    maximize_all = bool(rt.get("maximize_all_resources", True))
    raw_cpu_threads = rt.get("cpu_threads", "auto")
    if raw_cpu_threads is None or (isinstance(raw_cpu_threads, str) and raw_cpu_threads.strip().lower() == "auto"):
        cpu_threads_total = avail_cpus
    else:
        cpu_threads_total = max(1, min(avail_cpus, int(raw_cpu_threads)))

    threads_per_gpu_worker = max(1, cpu_threads_total // effective_slots)

    raw_dl_workers = rt.get("dataloader_workers_per_gpu", "auto")
    if raw_dl_workers is None or (isinstance(raw_dl_workers, str) and raw_dl_workers.strip().lower() == "auto"):
        dl_workers_per_gpu = max(1, min(16, threads_per_gpu_worker))
    else:
        dl_workers_per_gpu = max(0, int(raw_dl_workers))
    total_dl_workers = dl_workers_per_gpu * effective_slots

    device_ids = tuple(range(active_n))
    devices = tuple(f"cuda:{i}" for i in device_ids) if active_n > 0 else ("cpu",)
    primary_device = devices[0]

    pin_memory = bool(rt.get("pin_memory", True)) and active_n > 0
    non_blocking = bool(rt.get("non_blocking", True)) and active_n > 0
    persistent_workers = bool(rt.get("persistent_workers", True)) and dl_workers_per_gpu > 0
    prefetch_factor = max(1, int(rt.get("prefetch_factor", 4)))
    allow_tf32 = bool(rt.get("allow_tf32", False))
    cudnn_benchmark = bool(rt.get("cudnn_benchmark", False))
    amp_dtype = str(rt.get("amp_dtype", "bf16")).strip().lower()
    if amp_dtype not in ("fp32", "bf16", "fp16"):
        raise ValueError(f"amp_dtype must be one of ('fp32', 'bf16', 'fp16'), got {amp_dtype!r}")

    return ElasticRuntimePlan(
        requested_num_gpus=requested_num_gpus,
        active_gpus=active_n,
        detected_gpus=avail_gpus,
        detected_cpus=avail_cpus,
        detected_ram_gib=avail_ram,
        device_ids=device_ids,
        devices=devices,
        primary_device=primary_device,
        strategy=strategy,
        per_gpu_batch_size=per_gpu_bs,
        global_batch_size=global_bs,
        scale_batch_linearly_with_n=scale_linear,
        maximize_all_resources=maximize_all,
        cpu_threads_total=cpu_threads_total,
        threads_per_gpu_worker=threads_per_gpu_worker,
        dataloader_workers_per_gpu=dl_workers_per_gpu,
        total_dataloader_workers=total_dl_workers,
        pin_memory=pin_memory,
        non_blocking=non_blocking,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        allow_tf32=allow_tf32,
        cudnn_benchmark=cudnn_benchmark,
        amp_dtype=amp_dtype,
    )


def configure_elastic_hardware(plan: ElasticRuntimePlan) -> dict[str, Any]:
    """Apply CPU thread budget and CUDA TF32/cuDNN acceleration flags from `ElasticRuntimePlan`."""
    _require_torch()
    torch.set_num_threads(int(plan.cpu_threads_total))
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = bool(plan.allow_tf32)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = bool(plan.allow_tf32)
        torch.backends.cudnn.benchmark = bool(plan.cudnn_benchmark)
    return {
        "torch_num_threads": int(torch.get_num_threads()),
        "allow_tf32": bool(plan.allow_tf32),
        "cudnn_benchmark": bool(plan.cudnn_benchmark),
        "active_gpus": int(plan.active_gpus),
        "devices": list(plan.devices),
    }


if torch is None:

    class DirectFieldHead3D:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            _require_torch()

    class VectorPotentialCurlHead3D:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            _require_torch()

    class PhysicalReadout3D:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            _require_torch()

    class Flow3DStage1Autoencoder:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            _require_torch()

    Flow3DStage1Model = Flow3DStage1Autoencoder

else:

    def _fp32_context(tensor: torch.Tensor):
        dev_type = tensor.device.type
        if dev_type in ("cuda", "cpu"):
            return torch.amp.autocast(device_type=dev_type, enabled=False)
        return nullcontext()

    def _require_finite_prediction(tensor: torch.Tensor, name: str) -> None:
        if not bool(torch.isfinite(tensor).all()):
            raise PhysicsContractError(f"non-finite {name} prediction")

    def _flatten_feat(feat: torch.Tensor) -> tuple[torch.Tensor, bool, tuple[int, ...]]:
        if feat.ndim == 6:
            b, t, c, nz, ny, nx = feat.shape
            return feat.reshape(b * t, c, nz, ny, nx), True, (b, t)
        if feat.ndim == 5:
            b, c, nz, ny, nx = feat.shape
            return feat, False, (b,)
        raise ValueError(f"head expects 5D [N,C,Z,Y,X] or 6D [B,T,C,Z,Y,X] features, got {tuple(feat.shape)}")

    def _interior_mean_square(
        residual: torch.Tensor,
        fluid_mask: torch.Tensor | None,
        *,
        composed_derivative: bool = False,
    ) -> torch.Tensor:
        """Average over complete fluid stencils per frame, then average frames equally."""
        if fluid_mask is None:
            fluid_mask = residual.new_ones(residual.shape[-3:])
        if composed_derivative:
            valid = torch_valid_div_of_curl_mask(fluid_mask)
        else:
            valid = torch_build_full_3d_stencil_mask(fluid_mask)
        dims = (-3, -2, -1)
        counts = valid.sum(dim=dims)
        if bool((counts == 0).any()):
            raise PhysicsContractError("physics loss requires a complete interior fluid stencil in every frame")
        squared_sum = residual.masked_fill(valid == 0, 0).square().sum(dim=dims)
        return (squared_sum / counts).mean()

    def _align_fluid_mask_for_flat(
        fluid_mask: torch.Tensor | None,
        is_6d: bool,
        lead_dims: tuple[int, ...],
        spatial_zyx: tuple[int, int, int],
        device: torch.device,
    ) -> torch.Tensor | None:
        if fluid_mask is None:
            return None
        m = fluid_mask.to(device=device)
        if not torch.isfinite(m).all():
            raise ValueError("fluid_mask contains non-finite values")
        nz, ny, nx = spatial_zyx
        if not is_6d:
            b = lead_dims[0]
            if m.ndim == 4 and m.shape == (b, nz, ny, nx):
                return m
            if m.ndim == 5 and m.shape == (b, 1, nz, ny, nx):
                return m.squeeze(1)
            if m.ndim == 3 and m.shape == (nz, ny, nx):
                return m.unsqueeze(0).expand(b, nz, ny, nx)
            raise ValueError(f"cannot align fluid_mask shape {tuple(m.shape)} with batch {b} and grid {(nz, ny, nx)}")

        b, t = lead_dims
        if m.ndim == 5 and m.shape == (b, 1, nz, ny, nx):
            return m.expand(b, t, nz, ny, nx).reshape(b * t, nz, ny, nx)
        if m.ndim == 6 and m.shape == (b, t, 1, nz, ny, nx):
            return m.reshape(b * t, nz, ny, nx)
        if m.ndim == 4 and m.shape == (b, nz, ny, nx):
            return m.unsqueeze(1).expand(b, t, nz, ny, nx).reshape(b * t, nz, ny, nx)
        if m.ndim == 3 and m.shape == (nz, ny, nx):
            return m.unsqueeze(0).expand(b * t, nz, ny, nx)
        raise ValueError(
            f"cannot align fluid_mask shape {tuple(m.shape)} with batch {(b, t)} and grid {(nz, ny, nx)}"
        )

    class DirectFieldHead3D(nn.Module):
        """Head-A: linear Conv3d predicting `[u~, v~, w~, p~]` + optional Helmholtz projection."""

        def __init__(
            self,
            in_channels: int = 32,
            helmholtz_projection: bool = False,
            helmholtz_iters: int = 20,
            helmholtz_omega: float = 0.8,
            helmholtz_method: str = "cg",
        ) -> None:
            super().__init__()
            self.in_channels = int(in_channels)
            self.helmholtz_projection = bool(helmholtz_projection)
            self.helmholtz_iters = int(helmholtz_iters)
            self.helmholtz_omega = float(helmholtz_omega)
            self.helmholtz_method = str(helmholtz_method)
            self.proj = nn.Conv3d(self.in_channels, 4, kernel_size=3, padding=1)

        def forward(
            self,
            feat: torch.Tensor,
            fluid_mask: torch.Tensor | None = None,
            spacing_zyx: Sequence[float] = CYLINDER3D_STATE_SPEC.default_spacing_m,
        ) -> dict[str, Any]:
            flat, is_6d, lead = _flatten_feat(feat)
            raw = self.proj(flat)
            _require_finite_prediction(raw, "direct field")
            mask_flat = _align_fluid_mask_for_flat(fluid_mask, is_6d, lead, raw.shape[-3:], raw.device)

            phi: torch.Tensor | None = None
            if self.helmholtz_projection:
                vel_proj, phi = torch_helmholtz_project(
                    raw[:, :3],
                    spacing_zyx,
                    fluid_mask=mask_flat,
                    num_iters=self.helmholtz_iters,
                    omega=self.helmholtz_omega,
                    method=self.helmholtz_method,
                )
                norm_flat = torch.cat([vel_proj, raw[:, 3:4].to(dtype=vel_proj.dtype)], dim=1)
                _require_finite_prediction(norm_flat, "projected field")
            else:
                norm_flat = raw

            div_res = torch_div_h(
                norm_flat[:, 0], norm_flat[:, 1], norm_flat[:, 2], spacing_zyx, mask=mask_flat
            )
            div_loss = _interior_mean_square(div_res, mask_flat)

            if is_6d:
                b, t = lead
                nz, ny, nx = norm_flat.shape[-3:]
                norm_out = norm_flat.reshape(b, t, 4, nz, ny, nx)
                div_out = div_res.reshape(b, t, nz, ny, nx)
                phi_out = None if phi is None else phi.reshape(b, t, nz, ny, nx)
            else:
                norm_out = norm_flat
                div_out = div_res
                phi_out = phi

            return {
                "norm_fields": norm_out,
                "div_residual": div_out,
                "div_loss": div_loss,
                "phi": phi_out,
            }

    class VectorPotentialCurlHead3D(nn.Module):
        """Head-B: regularized vector potential `A*` + `p~` -> compatible `curl_h(A*)` (+ optional `grad_h(phi)`) + gauge."""

        def __init__(
            self,
            in_channels: int = 32,
            separate_mean_potential: bool = True,
            use_potential_branch: bool = False,
            hard_mask_output: bool = True,
            edge_order1: bool = True,
        ) -> None:
            super().__init__()
            self.in_channels = int(in_channels)
            self.separate_mean_potential = bool(separate_mean_potential)
            self.use_potential_branch = bool(use_potential_branch)
            self.hard_mask_output = bool(hard_mask_output)
            self.edge_order1 = bool(edge_order1)
            self.a_proj = nn.Conv3d(self.in_channels, 3, kernel_size=3, padding=1)
            self.p_proj = nn.Conv3d(self.in_channels, 1, kernel_size=3, padding=1)
            self.phi_proj = nn.Conv3d(self.in_channels, 1, kernel_size=3, padding=1) if self.use_potential_branch else None
            self.mean_head = nn.Linear(self.in_channels, 3) if self.separate_mean_potential else None

        def forward(
            self,
            feat: torch.Tensor,
            fluid_mask: torch.Tensor | None = None,
            spacing_zyx: Sequence[float] = CYLINDER3D_STATE_SPEC.default_spacing_m,
        ) -> dict[str, Any]:
            flat, is_6d, lead = _flatten_feat(feat)
            a_star = self.a_proj(flat)
            p_tilde = self.p_proj(flat)
            mask_flat = _align_fluid_mask_for_flat(fluid_mask, is_6d, lead, a_star.shape[-3:], a_star.device)

            # Never truncate derivatives with fluid_mask inside torch_curl_h / torch_grad_h
            _require_finite_prediction(a_star, "vector potential")
            u_curl = torch_curl_h(a_star, spacing_zyx, mask=None, edge_order1=self.edge_order1)
            u_tilde = u_curl

            if self.phi_proj is not None:
                phi_scalar = self.phi_proj(flat)[:, 0]
                _require_finite_prediction(phi_scalar, "scalar potential")
                u_pot = torch_grad_h(phi_scalar, spacing_zyx, mask=None, edge_order1=self.edge_order1)
                u_tilde = u_tilde + u_pot
                # Two centered derivatives span two cells. Use the actual velocity
                # contribution so this loss has the same discrete nullspace as div(grad(phi)).
                div_pot = torch_div_h(u_pot[:, 0], u_pot[:, 1], u_pot[:, 2], spacing_zyx)
                harmonic_laplacian_loss = _interior_mean_square(div_pot, mask_flat, composed_derivative=True)
            else:
                phi_scalar = None
                harmonic_laplacian_loss = torch.zeros((), dtype=u_curl.dtype, device=u_curl.device)

            if self.mean_head is not None:
                pooled = flat.mean(dim=(-3, -2, -1))
                mean_flow = self.mean_head(pooled).to(dtype=u_curl.dtype).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                u_tilde = u_tilde + mean_flow
            else:
                mean_flow = None

            if self.hard_mask_output and mask_flat is not None:
                m_bool = (mask_flat > 0.5).unsqueeze(1)
                u_tilde = torch.where(m_bool, u_tilde, torch.zeros_like(u_tilde))

            norm_flat = torch.cat([u_tilde, p_tilde.to(dtype=u_tilde.dtype)], dim=1)
            _require_finite_prediction(norm_flat, "curl head field")
            coulomb_div_a = torch_div_h(
                a_star[:, 0], a_star[:, 1], a_star[:, 2], spacing_zyx, mask=mask_flat
            )
            coulomb_gauge_loss = _interior_mean_square(coulomb_div_a, mask_flat)
            div_res = torch_div_h(
                norm_flat[:, 0], norm_flat[:, 1], norm_flat[:, 2], spacing_zyx, mask=mask_flat
            )

            if is_6d:
                b, t = lead
                nz, ny, nx = norm_flat.shape[-3:]
                norm_out = norm_flat.reshape(b, t, 4, nz, ny, nx)
                a_out = a_star.reshape(b, t, 3, nz, ny, nx)
                coulomb_out = coulomb_div_a.reshape(b, t, nz, ny, nx)
                div_out = div_res.reshape(b, t, nz, ny, nx)
                mean_out = None if mean_flow is None else mean_flow.reshape(b, t, 3)
                phi_out = None if phi_scalar is None else phi_scalar.reshape(b, t, nz, ny, nx)
            else:
                norm_out = norm_flat
                a_out = a_star
                coulomb_out = coulomb_div_a
                div_out = div_res
                mean_out = None if mean_flow is None else mean_flow.reshape(lead[0], 3)
                phi_out = phi_scalar

            return {
                "norm_fields": norm_out,
                "vector_potential": a_out,
                "coulomb_div_a": coulomb_out,
                "coulomb_gauge_loss": coulomb_gauge_loss,
                "potential_phi": phi_out,
                "harmonic_laplacian_loss": harmonic_laplacian_loss,
                "div_residual": div_out,
                "mean_flow": mean_out,
            }

    class PhysicalReadout3D(nn.Module):
        """FP32+ dimensional recovery from normalized `[u~,v~,w~,p~]`, `b [B,(T,)6]`, and known `fluid_mask`."""

        def __init__(
            self,
            scale_floor_uvw: float = 1e-6,
            scale_floor_p: float = 1e-6,
            hard_mask_output: bool = True,
        ) -> None:
            super().__init__()
            if scale_floor_uvw <= 0.0 or scale_floor_p <= 0.0:
                raise ValueError("scale_floor_uvw and scale_floor_p must be positive")
            self.scale_floor_uvw = float(scale_floor_uvw)
            self.scale_floor_p = float(scale_floor_p)
            self.hard_mask_output = bool(hard_mask_output)

        def forward(
            self,
            norm_fields_4ch: torch.Tensor,
            b: torch.Tensor,
            fluid_mask: torch.Tensor,
        ) -> torch.Tensor:
            if norm_fields_4ch.ndim not in (5, 6) or norm_fields_4ch.shape[-4] != 4:
                raise ValueError(
                    f"PhysicalReadout3D expects [B,(T,)4,Z,Y,X] normalized fields, got {tuple(norm_fields_4ch.shape)}"
                )
            if not torch.isfinite(b).all():
                raise ValueError("b statistics contain NaN or Inf values")
            if not torch.isfinite(fluid_mask).all():
                raise ValueError("fluid_mask contains NaN or Inf values")
            if b.device != norm_fields_4ch.device or fluid_mask.device != norm_fields_4ch.device:
                raise ValueError("norm_fields_4ch, b, and fluid_mask must reside on the same device")

            is_6d = norm_fields_4ch.ndim == 6
            nz, ny, nx = norm_fields_4ch.shape[-3:]
            if is_6d:
                bsz, t_len = norm_fields_4ch.shape[:2]
                if b.shape != (bsz, t_len, 6):
                    raise ValueError(f"expected b shape ({bsz}, {t_len}, 6), got {tuple(b.shape)}")
                if fluid_mask.ndim == 5 and fluid_mask.shape == (bsz, 1, nz, ny, nx):
                    mask_aligned = fluid_mask.unsqueeze(1).expand(bsz, t_len, 1, nz, ny, nx)
                elif fluid_mask.ndim == 6 and fluid_mask.shape == (bsz, t_len, 1, nz, ny, nx):
                    mask_aligned = fluid_mask
                elif fluid_mask.ndim == 4 and fluid_mask.shape == (bsz, nz, ny, nx):
                    mask_aligned = fluid_mask.unsqueeze(1).unsqueeze(2).expand(bsz, t_len, 1, nz, ny, nx)
                else:
                    raise ValueError(
                        f"expected fluid_mask [B,1,Z,Y,X] or [B,T,1,Z,Y,X], got {tuple(fluid_mask.shape)}"
                    )
            else:
                bsz = norm_fields_4ch.shape[0]
                if b.shape != (bsz, 6):
                    raise ValueError(f"expected b shape ({bsz}, 6), got {tuple(b.shape)}")
                if fluid_mask.ndim == 5 and fluid_mask.shape == (bsz, 1, nz, ny, nx):
                    mask_aligned = fluid_mask
                elif fluid_mask.ndim == 4 and fluid_mask.shape == (bsz, nz, ny, nx):
                    mask_aligned = fluid_mask.unsqueeze(1)
                else:
                    raise ValueError(
                        f"expected fluid_mask [B,1,Z,Y,X] or [B,Z,Y,X], got {tuple(fluid_mask.shape)}"
                    )

            mask_bool = mask_aligned > 0.5
            if not bool(mask_bool.any()):
                raise ValueError("fluid_mask contains zero valid fluid cells")
            # Check finiteness strictly inside valid fluid domain BEFORE arithmetic (never NaN * 0)
            fluid_vals = norm_fields_4ch.masked_select(mask_bool.expand_as(norm_fields_4ch))
            if not torch.isfinite(fluid_vals).all():
                raise ValueError("normalized predictions contain NaN or Inf inside the valid fluid domain")

            if bool((b[..., 4] <= 0.0).any()) or bool((b[..., 5] <= 0.0).any()):
                raise ValueError("b scales (sigma_uvw, sigma_p) must be strictly positive")

            with _fp32_context(norm_fields_4ch):
                wdtype = torch.float64 if norm_fields_4ch.dtype == torch.float64 else torch.float32
                nf = norm_fields_4ch.to(dtype=wdtype)
                b_work = b.to(dtype=wdtype)

                mu = b_work[..., :4]
                sig_uvw = b_work[..., 4:5]
                sig_p = b_work[..., 5:6]
                sig = torch.cat([sig_uvw, sig_uvw, sig_uvw, sig_p], dim=-1)

                mu = mu.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                sig = sig.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

                # Scrub any NaN/Inf strictly outside mask_bool before linear affine transform
                if self.hard_mask_output:
                    nf_safe = torch.where(mask_bool, nf, torch.zeros_like(nf))
                    phys = nf_safe * sig + mu
                    return torch.where(mask_bool, phys, torch.zeros_like(phys))
                nf_finite = torch.where(torch.isfinite(nf), nf, torch.zeros_like(nf))
                return nf_finite * sig + mu

    class _ReplicaAutocastWrapper(nn.Module):
        """Preserve caller CUDA autocast dtype inside `torch.nn.parallel.parallel_apply` worker threads."""

        def __init__(self, module: nn.Module, amp_dtype: torch.dtype) -> None:
            super().__init__()
            self.module = module
            self.amp_dtype = amp_dtype

        def forward(self, *args: Any, **kwargs: Any) -> Any:
            with torch.amp.autocast(device_type="cuda", dtype=self.amp_dtype, enabled=True):
                return self.module(*args, **kwargs)

    class Flow3DStage1Autoencoder(nn.Module):
        """Composite Stage 1 pipeline: Encoder -> SpatialTransformer3D -> LatentNorm3D -> Decoder -> Head -> Readout."""

        def __init__(
            self,
            encoder: DCAE3DEncoder,
            spatial_transformer: SpatialTransformer3D,
            latent_norm: LatentNorm3D,
            decoder: DCAE3DDecoder,
            head: DirectFieldHead3D | VectorPotentialCurlHead3D,
            readout: PhysicalReadout3D,
            probe: LatentPhysicsProbe | None = None,
            default_spacing_zyx: Sequence[float] = CYLINDER3D_STATE_SPEC.default_spacing_m,
            hard_mask_output: bool | None = None,
        ) -> None:
            super().__init__()
            self.encoder = encoder
            self.spatial_transformer = spatial_transformer
            self.latent_norm = latent_norm
            self.decoder = decoder
            self.head = head
            self.readout = readout
            self.probe = probe
            self.default_spacing_zyx = tuple(float(v) for v in default_spacing_zyx)
            self.hard_mask_output = (
                bool(readout.hard_mask_output) if hard_mask_output is None else bool(hard_mask_output)
            )

        @classmethod
        def from_config(cls, cfg: Mapping[str, Any]) -> "Flow3DStage1Autoencoder":
            grid = cfg.get("grid", {})
            enc_cfg = cfg.get("encoder", {})
            st_cfg = cfg.get("spatial_transformer", {})
            ln_cfg = cfg.get("latent_norm", {})
            dec_cfg = cfg.get("decoder", {})
            head_cfg = cfg.get("head", {})
            probe_cfg = cfg.get("probe", {})
            norm_cfg = cfg.get("normalizer", {})

            # Fail-closed architectural invariant checks
            if bool(st_cfg.get("temporal_attention", False)):
                raise ValueError("Stage 1 SpatialTransformer3D forbids temporal_attention=True")
            if not bool(st_cfg.get("pre_ln", True)):
                raise ValueError("Stage 1 SpatialTransformer3D requires pre_ln=True")
            if not bool(st_cfg.get("use_sdpa", True)):
                raise ValueError("Stage 1 SpatialTransformer3D requires use_sdpa=True")
            if not bool(enc_cfg.get("use_s2c", True)):
                raise ValueError("Stage 1 encoder requires use_s2c=True")
            if not bool(dec_cfg.get("no_encoder_skip", True)):
                raise ValueError("Stage 1 decoder requires no_encoder_skip=True (zero encoder-decoder skips)")
            if not bool(dec_cfg.get("use_c2s", True)):
                raise ValueError("Stage 1 decoder requires use_c2s=True")
            if int(dec_cfg.get("geometry_channels", 2)) != 2:
                raise ValueError(
                    f"Stage 1 decoder requires geometry_channels == 2, got {dec_cfg.get('geometry_channels')!r}"
                )

            spacing_zyx = tuple(
                float(v)
                for v in grid.get("spacing_zyx", grid.get("spacing_zyx_m", CYLINDER3D_STATE_SPEC.default_spacing_m))
            )
            base_channels = int(enc_cfg.get("base_channels", 32))
            channel_mults = tuple(int(v) for v in enc_cfg.get("channel_mults", (1, 2, 4, 8)))
            latent_channels = int(enc_cfg.get("latent_channels", 256))
            downsample_factor = int(enc_cfg.get("downsample_factor", 8))
            gn_groups = int(enc_cfg.get("gn_groups", 8))
            hard_mask_output = bool(dec_cfg.get("hard_mask_output", True))

            encoder = DCAE3DEncoder(
                in_channels=int(enc_cfg.get("in_channels", 6)),
                base_channels=base_channels,
                channel_mults=channel_mults,
                latent_channels=latent_channels,
                downsample_factor=downsample_factor,
                gn_groups=gn_groups,
                multiscale_geometry=bool(enc_cfg.get("multiscale_geometry", True)),
                sdf_tanh_alpha=enc_cfg.get("sdf_tanh_alpha", 1.0),
                num_refine_blocks=int(enc_cfg.get("num_refine_blocks", 1)),
            )

            spatial_transformer = SpatialTransformer3D(
                dim=int(st_cfg.get("embed_dim", latent_channels)),
                depth=int(st_cfg.get("depth", 1)),
                num_heads=int(st_cfg.get("num_heads", 8)),
                mlp_dim=int(st_cfg.get("mlp_dim", 1024)),
                dropout=float(st_cfg.get("dropout", 0.0)),
                pos_mode=str(st_cfg.get("pos_encoding", "sincos3d")),
                # Archived v1 configs predate this key and use a learned terminal LayerNorm.
                output_norm=st_cfg.get(
                    "output_norm",
                    "legacy_layer_norm" if cfg.get("latent_contract_version") == "latent.flow3d.v1" else "none",
                ),
            )

            latent_norm = LatentNorm3D(
                channels=latent_channels,
                mode=str(ln_cfg.get("type", "group_norm")),
                num_groups=int(ln_cfg.get("num_groups", 32)),
                dataset_rms_scale=float(ln_cfg.get("dataset_rms_scale", 1.0)),
                eps=float(ln_cfg.get("eps", 1e-6)),
            )

            decoder = DCAE3DDecoder(
                latent_channels=latent_channels,
                base_channels=int(dec_cfg.get("base_channels", base_channels)),
                channel_mults=tuple(int(v) for v in dec_cfg.get("channel_mults", channel_mults)),
                downsample_factor=downsample_factor,
                gn_groups=int(dec_cfg.get("gn_groups", gn_groups)),
                multiscale_geometry=bool(dec_cfg.get("multiscale_geometry", True)),
                sdf_tanh_alpha=dec_cfg.get("sdf_tanh_alpha", enc_cfg.get("sdf_tanh_alpha", 1.0)),
                num_refine_blocks=int(dec_cfg.get("num_refine_blocks", 1)),
                up3_num_refine_blocks=dec_cfg.get("up3_num_refine_blocks", None),
            )

            head_type = str(head_cfg.get("type", "head_a")).lower()
            if head_type in ("head_a", "direct"):
                head: DirectFieldHead3D | VectorPotentialCurlHead3D = DirectFieldHead3D(
                    in_channels=decoder.base_channels,
                    helmholtz_projection=bool(head_cfg.get("helmholtz_projection", False)),
                    helmholtz_iters=int(head_cfg.get("helmholtz_iters", 20)),
                    helmholtz_omega=float(head_cfg.get("helmholtz_omega", 0.8)),
                    helmholtz_method=str(head_cfg.get("helmholtz_method", "cg")),
                )
            elif head_type in ("head_b", "vector_potential", "curl"):
                head = VectorPotentialCurlHead3D(
                    in_channels=decoder.base_channels,
                    separate_mean_potential=bool(head_cfg.get("separate_mean_potential", True)),
                    use_potential_branch=bool(head_cfg.get("use_potential_branch", False)),
                    hard_mask_output=hard_mask_output,
                    edge_order1=bool(head_cfg.get("edge_order1", True)),
                )
            else:
                raise ValueError(f"unknown head type {head_type!r}; expected 'head_a' or 'head_b'")

            readout = PhysicalReadout3D(
                scale_floor_uvw=float(norm_cfg.get("scale_floor_uvw", 1e-6)),
                scale_floor_p=float(norm_cfg.get("scale_floor_p", 1e-6)),
                hard_mask_output=hard_mask_output,
            )

            probe = (
                LatentPhysicsProbe(
                    latent_channels=latent_channels,
                    hidden_dim=int(probe_cfg.get("hidden_dim", 256)),
                    spectrum_bins=int(probe_cfg.get("spectrum_bins", 16)),
                )
                if bool(probe_cfg.get("enabled", False))
                else None
            )

            return cls(
                encoder=encoder,
                spatial_transformer=spatial_transformer,
                latent_norm=latent_norm,
                decoder=decoder,
                head=head,
                readout=readout,
                probe=probe,
                default_spacing_zyx=spacing_zyx,
                hard_mask_output=hard_mask_output,
            )

        def encode(self, field: torch.Tensor, geometry: torch.Tensor | None = None) -> torch.Tensor:
            is_6d = field.ndim == 6
            b = field.shape[0]
            t = field.shape[1] if is_6d else 1

            z_enc = self.encoder(field, geometry)
            if is_6d:
                z_seq = z_enc.reshape(b, t, *z_enc.shape[1:])
                z_st = self.spatial_transformer(z_seq)
                return self.latent_norm(z_st)
            z_st = self.spatial_transformer(z_enc)
            return self.latent_norm(z_st)

        def decode(
            self,
            z_clean: torch.Tensor,
            geometry: torch.Tensor,
            b: torch.Tensor | None = None,
            spacing_zyx: Sequence[float] | None = None,
        ) -> dict[str, Any]:
            spacing = self.default_spacing_zyx if spacing_zyx is None else spacing_zyx
            is_6d = z_clean.ndim == 6
            dec_feat_flat = self.decoder(z_clean, geometry)
            if is_6d:
                bsz, t_len = z_clean.shape[:2]
                dec_feat = dec_feat_flat.reshape(bsz, t_len, *dec_feat_flat.shape[1:])
            else:
                dec_feat = dec_feat_flat

            fluid_mask = geometry[..., 0:1, :, :, :]
            head_out = self.head(dec_feat, fluid_mask=fluid_mask, spacing_zyx=spacing)
            norm_fields = head_out["norm_fields"]
            phys_fields = None if b is None else self.readout(norm_fields, b, fluid_mask)
            return {
                "norm_fields": norm_fields,
                "phys_fields": phys_fields,
                "head_aux": head_out,
            }

        def forward(
            self,
            field: torch.Tensor,
            geometry: torch.Tensor,
            b: torch.Tensor | None = None,
            spacing_zyx: Sequence[float] | None = None,
            run_probe: bool = False,
        ) -> dict[str, Any]:
            z_clean = self.encode(field, geometry)
            dec_out = self.decode(z_clean, geometry, b=b, spacing_zyx=spacing_zyx)
            probe_out = self.probe(z_clean) if (self.probe is not None and run_probe) else None
            return {
                "z_clean": z_clean,
                "norm_fields": dec_out["norm_fields"],
                "phys_fields": dec_out["phys_fields"],
                "head_aux": dec_out["head_aux"],
                "probe": probe_out,
            }

        @staticmethod
        def resolve_runtime_plan(
            cfg_or_runtime: Mapping[str, Any] | None = None,
            *,
            num_gpus: int | str | None = None,
            detected_gpus: int | None = None,
            detected_cpus: int | None = None,
            strict_gpu_count: bool = False,
        ) -> ElasticRuntimePlan:
            return resolve_elastic_runtime_plan(
                cfg_or_runtime,
                num_gpus=num_gpus,
                detected_gpus=detected_gpus,
                detected_cpus=detected_cpus,
                strict_gpu_count=strict_gpu_count,
            )

        def forward_elastic_multi_gpu(
            self,
            field: torch.Tensor,
            geometry: torch.Tensor,
            b: torch.Tensor | None = None,
            spacing_zyx: Sequence[float] | None = None,
            run_probe: bool = False,
            *,
            num_gpus: int | str | None = "auto",
            device_ids: Sequence[int] | None = None,
            non_blocking: bool = True,
            gather_for: Literal["train", "validation"] | None = None,
        ) -> dict[str, Any]:
            """Execute Stage 1 forward elastically across `n` GPUs (`n in N`, variable natural number).

            - Dynamically discovers `n` GPUs (or uses explicit `device_ids` / `num_gpus`).
            - When `B >= 2` and `n >= 2` CUDA GPUs are active, shards batch `B` across
              `m = min(n, B)` GPUs via `replicate` + `parallel_apply` + differentiable `gather`,
              preserving exact full-batch autograd backward into primary model parameters.
            - Falls back cleanly to single-GPU (`n=1` or `B=1`) or CPU (`n=0`).
            - With multiple shards, gather_for may restrict transfers to trainer-consumed outputs.
            """
            if field.ndim not in (5, 6):
                raise ValueError(
                    f"field must be 5D [B,4|6,Z,Y,X] or 6D [B,T,4|6,Z,Y,X], got {tuple(field.shape)}"
                )
            bsz = int(field.shape[0])
            if bsz < 1:
                raise ValueError(f"batch size B must be >= 1, got {bsz}")
            if int(geometry.shape[0]) != bsz:
                raise ValueError(
                    f"geometry batch size ({geometry.shape[0]}) must match field batch size ({bsz})"
                )
            if b is not None and int(b.shape[0]) != bsz:
                raise ValueError(f"b statistics batch size ({b.shape[0]}) must match field batch size ({bsz})")

            if device_ids is not None:
                raw_ids = list(device_ids)
                if len(raw_ids) == 0:
                    raise ValueError("device_ids must be non-empty when explicitly provided")
                if any(isinstance(i, bool) for i in raw_ids):
                    raise ValueError(f"device_ids must contain integer indices, not bool: {raw_ids}")
                cand_ids = [int(i) for i in raw_ids]
                if any(i < 0 for i in cand_ids):
                    raise ValueError(f"device_ids must be non-negative, got {cand_ids}")
                if len(set(cand_ids)) != len(cand_ids):
                    raise ValueError(f"device_ids must contain unique indices, got {cand_ids}")
                avail_gpus = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
                if avail_gpus == 0:
                    raise ValueError("explicit device_ids provided, but CUDA is not available")
                if any(i >= avail_gpus for i in cand_ids):
                    raise ValueError(
                        f"device_ids contains out-of-range index for available CUDA device count ({avail_gpus}): {cand_ids}"
                    )
                cuda_ok = True
            else:
                plan = resolve_elastic_runtime_plan({"num_gpus": num_gpus})
                cand_ids = list(plan.device_ids)
                cuda_ok = torch.cuda.is_available() and plan.active_gpus > 0

            if not cuda_ok:
                out = self.forward(field, geometry, b=b, spacing_zyx=spacing_zyx, run_probe=run_probe)
                out["elastic_runtime"] = {
                    "active_gpus": 0,
                    "used_devices": [str(field.device)],
                    "shard_batch_sizes": [bsz],
                }
                return out

            active_ids = cand_ids[: max(1, min(len(cand_ids), bsz))]
            primary_dev = torch.device(f"cuda:{active_ids[0]}")
            param_dev = next(self.parameters()).device
            if param_dev != primary_dev:
                self.to(primary_dev)

            if len(active_ids) == 1:
                f_dev = field.to(primary_dev, non_blocking=non_blocking)
                g_dev = geometry.to(primary_dev, non_blocking=non_blocking)
                b_dev = None if b is None else b.to(primary_dev, non_blocking=non_blocking)
                out = self.forward(f_dev, g_dev, b=b_dev, spacing_zyx=spacing_zyx, run_probe=run_probe)
                out["elastic_runtime"] = {
                    "active_gpus": 1,
                    "used_devices": [str(primary_dev)],
                    "shard_batch_sizes": [bsz],
                }
                return out

            m = len(active_ids)
            f_splits = torch.tensor_split(field, m, dim=0)
            g_splits = torch.tensor_split(geometry, m, dim=0)
            b_splits = [None] * m if b is None else list(torch.tensor_split(b, m, dim=0))

            shard_sizes = [int(s.shape[0]) for s in f_splits]
            shard_weights = [float(sz) / float(bsz) for sz in shard_sizes]

            inputs = [
                (
                    f_s.to(f"cuda:{dev_id}", non_blocking=non_blocking),
                    g_s.to(f"cuda:{dev_id}", non_blocking=non_blocking),
                )
                for f_s, g_s, dev_id in zip(f_splits, g_splits, active_ids)
            ]
            kwargs_tup = [
                {
                    "b": (None if b_s is None else b_s.to(f"cuda:{dev_id}", non_blocking=non_blocking)),
                    "spacing_zyx": spacing_zyx,
                    "run_probe": run_probe,
                }
                for b_s, dev_id in zip(b_splits, active_ids)
            ]

            caller_autocast_enabled = torch.is_autocast_enabled("cuda")
            caller_autocast_dtype = torch.get_autocast_dtype("cuda") if caller_autocast_enabled else None

            replicas: Sequence[nn.Module] = torch.nn.parallel.replicate(
                self, active_ids, detach=not torch.is_grad_enabled()
            )
            if caller_autocast_enabled and caller_autocast_dtype is not None:
                replicas = [_ReplicaAutocastWrapper(rep, caller_autocast_dtype) for rep in replicas]
            shard_outputs = torch.nn.parallel.parallel_apply(replicas, inputs, kwargs_tup, active_ids)
            if gather_for is not None:
                shard_outputs = [_trainer_shard_output(out, gather_for) for out in shard_outputs]

            def _gather_tree(items: Sequence[Any]) -> Any:
                first = items[0]
                if first is None:
                    return None
                if isinstance(first, Mapping):
                    return {k: _gather_tree([it[k] for it in items]) for k in first}
                if isinstance(first, torch.Tensor):
                    if first.ndim == 0:
                        gathered_scalars = torch.nn.parallel.gather(
                            [t.unsqueeze(0) for t in items],
                            target_device=active_ids[0],
                            dim=0,
                        )
                        weights_t = torch.tensor(
                            shard_weights,
                            dtype=gathered_scalars.dtype,
                            device=gathered_scalars.device,
                        )
                        return (gathered_scalars * weights_t).sum()
                    return torch.nn.parallel.gather(list(items), target_device=active_ids[0], dim=0)
                return first

            gathered: dict[str, Any] = _gather_tree(shard_outputs)
            gathered["elastic_runtime"] = {
                "active_gpus": m,
                "used_devices": [f"cuda:{i}" for i in active_ids],
                "shard_batch_sizes": shard_sizes,
            }
            return gathered

    Flow3DStage1Model = Flow3DStage1Autoencoder


__all__ = [
    "DirectFieldHead3D",
    "ElasticRuntimePlan",
    "Flow3DStage1Autoencoder",
    "Flow3DStage1Model",
    "PhysicalReadout3D",
    "VectorPotentialCurlHead3D",
    "configure_elastic_hardware",
    "resolve_elastic_runtime_plan",
    "torch_available",
]
