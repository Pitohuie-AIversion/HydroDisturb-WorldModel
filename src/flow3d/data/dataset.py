"""PyTorch Dataset & DataLoader adapters for FlowWorld-3D certified multi-Re HDF5 datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from flow3d.contracts import Flow3DBatch, PhysicsContractError, SampleProvenance
from flow3d.data.adapters import inspect_world_model_dataset_manifest, load_world_model_h5_trajectory
from flow3d.data.normalization import Flow3DNormalizer, Flow3DNormStats
from flow3d.data.splits import SplitAssignment, temporal_block_split_with_purge, group_split_by_provenance
from stage1.io.manifest import sha256_file

try:
    import torch
    from torch.utils.data import DataLoader, Dataset
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    Dataset = object  # type: ignore[misc,assignment]
    DataLoader = None  # type: ignore[assignment]


def _require_torch() -> Any:
    if torch is None:
        raise ImportError("PyTorch is required for Flow3DH5Dataset and Stage 1 training")
    return torch


@dataclass(frozen=True)
class Flow3DFrameSample:
    """Individual frame data sample for Stage 1 autoencoding."""

    norm_field: np.ndarray  # [4, Z, Y, X] float32
    phys_field: np.ndarray  # [4, Z, Y, X] float32 (m/s and Pa)
    geometry: np.ndarray    # [2, Z, Y, X] float32 (fluid_mask, SDF/D)
    b: np.ndarray           # [6] float32 (mu_u, mu_v, mu_w, mu_p, sigma_uvw, sigma_p)
    spacing_m: tuple[float, float, float]
    spacing_star: tuple[float, float, float]
    case_id: str
    reynolds_number: float
    u0_m_s: float
    cylinder_diameter_m: float
    step_index: int
    time_physical_s: float
    time_nondim_ctu: float
    trajectory_index: int
    frame_index: int


# In-memory shared cache for loaded trajectory batches to prevent redundant disk reads
_TRAJECTORY_CACHE: dict[str, tuple[Flow3DBatch, dict[str, Any]]] = {}


class Flow3DH5Dataset(Dataset):
    """Certified multi-Re HDF5 dataset for FlowWorld-3D Stage 1 training & evaluation.

    - Loads exclusively verified ELIGIBLE trajectories (refuses quarantined pseudo-data).
    - Caches normalized & physical fields in RAM by default (~2.9 GB for 150 frames, 0.6% of 475 GB RAM)
      for zero-I/O overhead, deterministic multi-worker batching, and elimination of h5py thread contention.
    - Provides leak-free split filtering (temporal block with purge gaps or trajectory provenance).
    """

    def __init__(
        self,
        dataset_dir: str | Path = "/root/autodl-tmp/datasets/world_model",
        eligible_files: Sequence[str] | None = None,
        split: str = "all",
        split_strategy: str = "temporal_block",
        purge_gap_frames: int = 4,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        preload_in_memory: bool = True,
        normalizer: Flow3DNormalizer | None = None,
        as_torch_tensors: bool = True,
        _shared_samples: list[Flow3DFrameSample] | None = None,
        _shared_provenances: list[SampleProvenance] | None = None,
        trajectory_cache: dict[str, tuple[Flow3DBatch, dict[str, Any]]] | None = None,
    ) -> None:
        _require_torch()
        self.dataset_dir = Path(dataset_dir)
        if not self.dataset_dir.is_dir():
            raise FileNotFoundError(f"dataset directory does not exist: {self.dataset_dir}")

        self.normalizer = normalizer or Flow3DNormalizer()
        self.split = str(split).lower()
        if self.split not in ("all", "train", "val", "test"):
            raise ValueError(f"split must be 'all', 'train', 'val', or 'test', got {split!r}")
        self.split_strategy = str(split_strategy).lower()
        if self.split_strategy not in {"temporal_block", "group_by_trajectory", "group_by_ancestor"}:
            raise ValueError(f"unknown Stage 1 split_strategy: {split_strategy}")
        self.as_torch_tensors = bool(as_torch_tensors)
        self.preload_in_memory = bool(preload_in_memory)

        if _shared_samples is not None and _shared_provenances is not None:
            self._samples = _shared_samples
            self._provenances = _shared_provenances
            self.file_names = list(eligible_files) if eligible_files is not None else []
        else:
            # 1. Discover eligible files
            if eligible_files is not None:
                self.file_names = [Path(f).name for f in eligible_files]
            else:
                manifest_info = inspect_world_model_dataset_manifest(self.dataset_dir)
                self.file_names = [t["file_name"] for t in manifest_info["eligible_trajectories"]]
                if not self.file_names:
                    raise PhysicsContractError("no eligible trajectories found in dataset directory")

            # 2. Load trajectories & build frames (with RAM caching)
            self._samples: list[Flow3DFrameSample] = []
            self._provenances: list[SampleProvenance] = []

            cache = _TRAJECTORY_CACHE if trajectory_cache is None else trajectory_cache
            for traj_idx, fn in enumerate(sorted(self.file_names)):
                file_path = self.dataset_dir / fn
                cache_key = str(file_path.resolve())
                if cache_key not in cache:
                    cache[cache_key] = load_world_model_h5_trajectory(
                        file_path,
                        normalize=True,
                        check_rank_degeneracy=True,
                    )
                batch, details = cache[cache_key]
                # batch.field is [1, T, 4, Z, Y, X], batch.geometry is [1, 2, Z, Y, X]
                f_si = np.asarray(batch.field[0], dtype=np.float32)      # [T, 4, Z, Y, X]
                geom = np.asarray(batch.geometry[0], dtype=np.float32)   # [2, Z, Y, X]
                b_arr = np.asarray(batch.b[0], dtype=np.float32)        # [T, 6]
                sp_m = tuple(float(v) for v in batch.spacing)
                sp_star = tuple(float(v) for v in batch.spacing_over_d(0))
                cond = batch.conditions[0]
                t_len = f_si.shape[0]

                fluid_mask = geom[0] > 0.5
                for t in range(t_len):
                    prov = batch.provenance[t]
                    frame_si = f_si[t]
                    # Normalized field [4, Z, Y, X]
                    mu_u, mu_v, mu_w, mu_p = float(b_arr[t, 0]), float(b_arr[t, 1]), float(b_arr[t, 2]), float(b_arr[t, 3])
                    s_uvw, s_p = float(b_arr[t, 4]), float(b_arr[t, 5])
                    norm_frame = np.zeros_like(frame_si)
                    norm_frame[0] = np.where(fluid_mask, (frame_si[0] - mu_u) / s_uvw, 0.0)
                    norm_frame[1] = np.where(fluid_mask, (frame_si[1] - mu_v) / s_uvw, 0.0)
                    norm_frame[2] = np.where(fluid_mask, (frame_si[2] - mu_w) / s_uvw, 0.0)
                    norm_frame[3] = np.where(fluid_mask, (frame_si[3] - mu_p) / s_p, 0.0)

                    sample = Flow3DFrameSample(
                        norm_field=norm_frame,
                        phys_field=frame_si,
                        geometry=geom,
                        b=b_arr[t],
                        spacing_m=sp_m,
                        spacing_star=sp_star,
                        case_id=prov.case_id,
                        reynolds_number=float(cond.reynolds_number) if cond.reynolds_number is not None else 0.0,
                        u0_m_s=float(cond.u0_m_s) if cond.u0_m_s is not None else 0.0,
                        cylinder_diameter_m=cond.diameter_m,
                        step_index=prov.time_idx,
                        time_physical_s=prov.time_phys_s,
                        time_nondim_ctu=prov.time_ctu,
                        trajectory_index=traj_idx,
                        frame_index=t,
                    )
                    self._samples.append(sample)
                    self._provenances.append(prov)

        # 3. Determine split indices
        self.active_indices = self._compute_split_indices(
            split=self.split,
            split_strategy=self.split_strategy,
            purge_gap_frames=purge_gap_frames,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
        )

    @classmethod
    def create_splits(
        cls,
        dataset_dir: str | Path = "/root/autodl-tmp/datasets/world_model",
        eligible_files: Sequence[str] | None = None,
        split_strategy: str = "temporal_block",
        purge_gap_frames: int = 4,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        normalizer: Flow3DNormalizer | None = None,
        as_torch_tensors: bool = True,
        trajectory_cache: dict[str, tuple[Flow3DBatch, dict[str, Any]]] | None = None,
    ) -> tuple["Flow3DH5Dataset", "Flow3DH5Dataset", "Flow3DH5Dataset"]:
        """Instantiate train, val, test datasets sharing single preloaded memory buffer."""
        master = cls(
            dataset_dir=dataset_dir,
            eligible_files=eligible_files,
            split="all",
            split_strategy=split_strategy,
            purge_gap_frames=purge_gap_frames,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            normalizer=normalizer,
            as_torch_tensors=as_torch_tensors,
            trajectory_cache=trajectory_cache,
        )
        train_ds = cls(
            dataset_dir=dataset_dir,
            eligible_files=master.file_names,
            split="train",
            split_strategy=split_strategy,
            purge_gap_frames=purge_gap_frames,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            normalizer=normalizer,
            as_torch_tensors=as_torch_tensors,
            _shared_samples=master._samples,
            _shared_provenances=master._provenances,
        )
        val_ds = cls(
            dataset_dir=dataset_dir,
            eligible_files=master.file_names,
            split="val",
            split_strategy=split_strategy,
            purge_gap_frames=purge_gap_frames,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            normalizer=normalizer,
            as_torch_tensors=as_torch_tensors,
            _shared_samples=master._samples,
            _shared_provenances=master._provenances,
        )
        test_ds = cls(
            dataset_dir=dataset_dir,
            eligible_files=master.file_names,
            split="test",
            split_strategy=split_strategy,
            purge_gap_frames=purge_gap_frames,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            normalizer=normalizer,
            as_torch_tensors=as_torch_tensors,
            _shared_samples=master._samples,
            _shared_provenances=master._provenances,
        )
        return train_ds, val_ds, test_ds

    def _compute_split_indices(
        self,
        split: str,
        split_strategy: str,
        purge_gap_frames: int,
        train_ratio: float,
        val_ratio: float,
        test_ratio: float,
    ) -> list[int]:
        if split == "all":
            return list(range(len(self._samples)))

        if split_strategy in {"group_by_trajectory", "group_by_ancestor"}:
            ratios = (train_ratio, val_ratio, test_ratio)
            assignment = group_split_by_provenance(
                self._provenances,
                group_by="ancestor_id" if split_strategy == "group_by_ancestor" else "trajectory_id",
                ratios=ratios,
            )
            if split == "train":
                return list(assignment.train_indices)
            elif split == "val":
                return list(assignment.val_indices)
            else:
                return list(assignment.test_indices)
        else:
            # Per-trajectory contiguous temporal block with purge gaps
            traj_groups: dict[int, list[int]] = {}
            for g_idx, s in enumerate(self._samples):
                traj_groups.setdefault(s.trajectory_index, []).append(g_idx)

            owners = {}
            for indices in traj_groups.values():
                provenance = self._provenances[indices[0]]
                previous = owners.setdefault(provenance.ancestor_id, provenance.trajectory_id)
                if previous != provenance.trajectory_id:
                    raise PhysicsContractError("shared ancestor across trajectories: use split_strategy='group_by_ancestor' to hold branches together")
            selected_indices: list[int] = []
            for t_idx, global_indices in traj_groups.items():
                sub_provs = [self._provenances[i] for i in global_indices]
                assign = temporal_block_split_with_purge(
                    sub_provs,
                    history_frames=1,
                    forecast_frames=0,
                    purge_gap_frames=purge_gap_frames,
                    ratios=(train_ratio, val_ratio, test_ratio),
                )
                if split == "train":
                    sub = assign.train_indices
                elif split == "val":
                    sub = assign.val_indices
                else:
                    sub = assign.test_indices
                selected_indices.extend(global_indices[i] for i in sub)
            return sorted(selected_indices)

    def __len__(self) -> int:
        return len(self.active_indices)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        global_idx = self.active_indices[idx]
        sample = self._samples[global_idx]

        if not self.as_torch_tensors:
            return {
                "norm_field": sample.norm_field,
                "phys_field": sample.phys_field,
                "geometry": sample.geometry,
                "b": sample.b,
                "spacing_m": sample.spacing_m,
                "spacing_star": sample.spacing_star,
                "case_id": sample.case_id,
                "reynolds_number": sample.reynolds_number,
                "u0_m_s": sample.u0_m_s,
                "time_physical_s": sample.time_physical_s,
                "time_nondim_ctu": sample.time_nondim_ctu,
                "step_index": sample.step_index,
            }

        t_norm = torch.from_numpy(sample.norm_field)
        t_phys = torch.from_numpy(sample.phys_field)
        t_geom = torch.from_numpy(sample.geometry)
        t_b = torch.from_numpy(sample.b)

        return {
            "norm_field": t_norm,
            "phys_field": t_phys,
            "geometry": t_geom,
            "b": t_b,
            "spacing_m": sample.spacing_m,
            "spacing_star": sample.spacing_star,
            "case_id": sample.case_id,
            "reynolds_number": sample.reynolds_number,
            "u0_m_s": sample.u0_m_s,
            "time_physical_s": sample.time_physical_s,
            "time_nondim_ctu": sample.time_nondim_ctu,
            "step_index": sample.step_index,
        }


def stage1_split_identity(
    train_ds: Flow3DH5Dataset, val_ds: Flow3DH5Dataset, test_ds: Flow3DH5Dataset
) -> dict[str, Any]:
    """Identify exact inputs and ordered frames in each split without fetching field tensors."""
    files = sorted(train_ds.file_names)
    if len(files) != len(set(files)):
        raise PhysicsContractError("duplicate Stage 1 trajectory file")
    splits: dict[str, list[list[Any]]] = {}
    seen: set[tuple[str, int]] = set()
    for name, ds in (("train", train_ds), ("val", val_ds), ("test", test_ds)):
        if ds.split != name or ds.dataset_dir != train_ds.dataset_dir or sorted(ds.file_names) != files:
            raise PhysicsContractError(f"{name} has different Stage 1 source files or split membership")
        frames = []
        for idx in ds.active_indices:
            sample = ds._samples[idx]
            key = (files[sample.trajectory_index], int(sample.step_index))
            if key in seen:
                raise PhysicsContractError(f"duplicate Stage 1 frame across train/val/test: {key}")
            seen.add(key)
            frames.append([key[0], sample.case_id, key[1]])
        splits[name] = frames
    return {"files": {name: sha256_file(train_ds.dataset_dir / name) for name in files}, "splits": splits}


def collate_flow3d_stage1(
    batch_list: Sequence[dict[str, Any]], *, include_phys_field: bool = True,
) -> dict[str, Any]:
    """Collate frames into [B, C, Z, Y, X] tensors, optionally omitting the evaluation target."""
    _require_torch()
    batch: dict[str, Any] = {"norm_field": torch.stack([item["norm_field"] for item in batch_list], dim=0)}
    if include_phys_field:
        batch["phys_field"] = torch.stack([item["phys_field"] for item in batch_list], dim=0)
    batch["geometry"] = torch.stack([item["geometry"] for item in batch_list], dim=0)
    batch["b"] = torch.stack([item["b"] for item in batch_list], dim=0)

    # Spacings should be uniform across the verified dataset
    batch["spacing_m"] = batch_list[0]["spacing_m"]
    batch["spacing_star"] = batch_list[0]["spacing_star"]

    batch["meta"] = [
        {
            "case_id": item["case_id"],
            "reynolds_number": item["reynolds_number"],
            "u0_m_s": item["u0_m_s"],
            "time_physical_s": item["time_physical_s"],
            "time_nondim_ctu": item["time_nondim_ctu"],
            "step_index": item["step_index"],
        }
        for item in batch_list
    ]
    return batch


def create_stage1_dataloaders(
    dataset_dir: str | Path = "/root/autodl-tmp/datasets/world_model",
    batch_size: int = 1,
    num_workers: int = 4,
    pin_memory: bool = True,
    split_strategy: str = "temporal_block",
    purge_gap_frames: int = 4,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Creates train, validation, and test PyTorch DataLoaders for Stage 1."""
    _require_torch()
    train_ds = Flow3DH5Dataset(
        dataset_dir=dataset_dir,
        split="train",
        split_strategy=split_strategy,
        purge_gap_frames=purge_gap_frames,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )
    val_ds = Flow3DH5Dataset(
        dataset_dir=dataset_dir,
        split="val",
        split_strategy=split_strategy,
        purge_gap_frames=purge_gap_frames,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )
    test_ds = Flow3DH5Dataset(
        dataset_dir=dataset_dir,
        split="test",
        split_strategy=split_strategy,
        purge_gap_frames=purge_gap_frames,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_flow3d_stage1,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_flow3d_stage1,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_flow3d_stage1,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader
