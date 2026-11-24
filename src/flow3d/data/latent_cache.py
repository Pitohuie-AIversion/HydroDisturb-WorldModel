"""Export and read verified Stage 1 latents in contiguous train, validation and test segments."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from flow3d.data.adapters import _admit_manifest_row, _audit_files, _source_key, load_world_model_h5_trajectory
from flow3d.data.dataset import Flow3DH5Dataset, stage1_split_identity
from flow3d.data.interventions import validated_intervention
from flow3d.models import Flow3DStage1Autoencoder
from flow3d.training.artifact import prepare_frozen_checkpoint
from stage1.io.checkpoint import (
    CheckpointContractError, CheckpointIntegrityError, CheckpointManager, sha256_file, sha256_json,
)


_SIDECARS = ("manifest.json", "multi_re_tier1_certification_summary.json", "extraction_and_rampool_assessment.json")
_SPLITS = ("train", "val", "test")


def _sidecar_hashes(directory):
    return {name: sha256_file(directory / name) for name in _SIDECARS}


def _model_from_payload(payload, device):
    model = Flow3DStage1Autoencoder.from_config(payload["config"])
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return model.to(device).eval().requires_grad_(False)


def _contiguous_groups(dataset):
    group = []
    previous_key = None
    previous_frame = -1
    for index in dataset.active_indices:
        sample = dataset._samples[index]
        provenance = dataset._provenances[index]
        key = (sample.trajectory_index, provenance.ancestor_id, provenance.trajectory_id, provenance.segment_id)
        if group and (key != previous_key or sample.frame_index != previous_frame + 1):
            yield group
            group = []
        group.append(index)
        previous_key, previous_frame = key, sample.frame_index
    if group:
        yield group


def _export_segment(dataset, indices, batch, file_name, path, model, device, latent_shape, *, split=None):
    import torch

    split = dataset.split if split is None else split
    frames = []
    latents = np.empty((len(indices), *latent_shape), dtype=np.float32)
    stats = np.empty((len(indices), 6), dtype=np.float32)
    geometry = torch.from_numpy(batch.geometry).to(device)
    with torch.inference_mode():
        for pos, index in enumerate(indices):
            sample = dataset._samples[index]
            field = torch.from_numpy(sample.norm_field).unsqueeze(0).to(device)
            z = model.encode(field, geometry).squeeze(0).float().cpu().numpy()
            if tuple(z.shape) != tuple(latent_shape) or not np.isfinite(z).all():
                raise CheckpointContractError("encoder produced invalid z_clean shape or values")
            latents[pos] = z
            stats[pos] = sample.b
            frames.append(dict(dataset._provenances[index].to_dict(), frame_index=sample.frame_index, split=split))
    np.savez(path, z_clean=latents, b=stats, geometry=batch.geometry[0])
    return {
        "shard": path.name, "file_name": file_name, "split": split,
        "trajectory_id": frames[0]["trajectory_id"], "ancestor_id": frames[0]["ancestor_id"],
        "spacing_m": list(batch.spacing), "spacing_star": list(batch.spacing_over_d(0)),
        "conditions": batch.conditions[0].to_dict(), "frames": frames,
    }


def export_latent_cache(checkpoint_dir: Path, dataset_dir: Path, output_dir: Path, device="cpu") -> Path:
    """Publish all assigned Stage 1 frames atomically into a new or empty directory."""
    checkpoint_dir, dataset_dir, output_dir = (Path(p).resolve() for p in (checkpoint_dir, dataset_dir, output_dir))
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise FileExistsError(f"cache output directory must be new or empty: {output_dir}")
    summary = json.loads((checkpoint_dir / "training_summary.json").read_text())
    expected = summary.get("split_identity")
    if not isinstance(expected, dict) or not expected.get("files"):
        raise CheckpointContractError("cache export requires recorded Stage 1 split identity")
    for name, digest in expected["files"].items():
        if Path(name).name != name or sha256_file(dataset_dir / name) != digest:
            raise CheckpointIntegrityError(f"trajectory differs from training: {name}")
    audit = _sidecar_hashes(dataset_dir)
    cfg = summary["config"]["training"]
    split_keys = ("split_strategy", "purge_gap_frames", "train_ratio", "val_ratio", "test_ratio")
    settings = {key: cfg[key] for key in split_keys if key in cfg}
    # A private cache reads the currently hashed files, even after earlier in-process Stage 1 runs.
    batches = {}
    splits = Flow3DH5Dataset.create_splits(
        dataset_dir, eligible_files=list(expected["files"]), as_torch_tensors=False,
        trajectory_cache=batches, **settings,
    )
    identity = stage1_split_identity(*splits)
    if identity != expected:
        raise CheckpointContractError("recomputed Stage 1 frame membership differs from training")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".latent-export-", dir=output_dir.parent) as tmp:
        staging = Path(tmp)
        manager, checkpoint, contract, _, _ = prepare_frozen_checkpoint(
            checkpoint_dir, staging, identity, dataset_dir=dataset_dir,
        )
        payload = manager.verify_checkpoint(checkpoint, expected_contract=contract)
        model = _model_from_payload(payload, device)
        del payload
        segments = []
        for dataset in splits:
            names = sorted(dataset.file_names)
            for indices in _contiguous_groups(dataset):
                file_name = names[dataset._samples[indices[0]].trajectory_index]
                batch, _ = batches[str((dataset_dir / file_name).resolve())]
                path = staging / f"segment-{len(segments):04d}.npz"
                segment = _export_segment(dataset, indices, batch, file_name, path, model, device, contract["latent_shape"])
                segments.append(segment)
        if _sidecar_hashes(dataset_dir) != audit or audit["manifest.json"] != contract["dataset_manifest_sha256"]:
            raise CheckpointIntegrityError("source admission metadata changed during cache export")
        index = {
            "format": "flow3d.stage1.latents.v2", "checkpoint_sha256": sha256_file(checkpoint),
            "contract_sha256": sha256_json(contract), "split_identity": identity, "admission_sha256": audit,
            "segments": segments, "shards": [row["shard"] for row in segments],
        }
        index_path = staging / "index.json"
        index_path.write_text(json.dumps(index, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        manifest = staging / "cache_manifest.json"
        manager.create_frozen_manifest(checkpoint, contract=contract, output_path=manifest, cache_index=index_path.name)
        staging.rename(output_dir)
    return output_dir / "cache_manifest.json"


_DOWNSTREAM_FORMAT = "flow3d.downstream.latents.v1"


def _planned_trajectories(plan):
    """Validate whole-trajectory assignments, including the reserved test ancestors."""
    rows = {row["file_name"]: row for row in plan["trajectories"]}
    if len(rows) != len(plan["trajectories"]) or set(plan["splits"]) != set(_SPLITS):
        raise CheckpointContractError("duplicate trajectories or invalid downstream split")
    ordered = []
    owners = {}
    payload_splits = {}
    seen = set()
    original = set(plan["stage1_ancestor_ids"])
    for split in _SPLITS:
        if not plan["splits"][split]:
            raise CheckpointContractError(f"empty downstream split: {split}")
        for name in plan["splits"][split]:
            row = rows.get(name)
            if not row or Path(name).name != name or name in seen or row["split"] != split:
                raise CheckpointContractError("downstream trajectory assignment is inconsistent")
            seen.add(name)
            ancestor = row["ancestor_id"]
            if not ancestor or ancestor in original or owners.setdefault(ancestor, split) != split:
                raise CheckpointContractError(f"downstream ancestor is reused across splits or by Stage 1: {ancestor}")
            digest = row["source_sha256"]
            if (not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest)
                    or payload_splits.setdefault(digest, split) != split):
                raise CheckpointContractError(f"invalid or reused downstream CFD hash: {name}")
            steps = row["step_indices"]
            if (row["frame_indices"] != list(range(row["num_steps"])) or not steps
                    or len(steps) != row["num_steps"] or any(b <= a for a, b in zip(steps, steps[1:]))):
                raise CheckpointContractError(f"invalid planned frame order for {name}")
            ordered.append(row)
    if seen != set(rows):
        raise CheckpointContractError("downstream manifest has unassigned trajectories")
    return ordered


def _audited_sources(directory, file_names):
    manifest, tier, audit = _audit_files(directory)
    sources = {}
    for row in manifest["trajectories"]:
        name = row["file_name"]
        if name in file_names:
            ancestor, _ = _admit_manifest_row(row, manifest, tier, audit)
            sources[name] = (ancestor, _source_key(row["source_model"]))
    if set(sources) != set(file_names):
        raise CheckpointIntegrityError("trajectory is missing from the source admission manifest")
    return sources


def _check_planned_segment(segment, row):
    frames = segment["frames"]
    if (segment["file_name"] != row["file_name"] or segment["split"] != row["split"]
            or segment["trajectory_id"] != row["trajectory_id"] or segment["ancestor_id"] != row["ancestor_id"]
            or [f["frame_index"] for f in frames] != row["frame_indices"]
            or [f["time_idx"] for f in frames] != row["step_indices"]
            or not all(np.isclose(row[key], segment["conditions"][key], rtol=1e-4, atol=1e-7)
                       for key in ("dt_ctu", "dt_phys_s"))):
        raise CheckpointIntegrityError(f"cached trajectory differs from downstream manifest: {row['file_name']}")


def _encode_downstream_file(directory, row, shard, model, device, latent_shape):
    # Use a private one-trajectory buffer; the default Dataset cache retains every source in RAM.
    name = row["file_name"]
    source = directory / name
    if sha256_file(source) != row["source_sha256"]:
        raise CheckpointIntegrityError(f"downstream CFD source hash mismatch: {name}")
    batches = {}
    ds = Flow3DH5Dataset(directory, eligible_files=[name], split="all",
                         as_torch_tensors=False, trajectory_cache=batches)
    batch, _ = batches[str(source.resolve())]
    segment = _export_segment(ds, ds.active_indices, batch, name, shard, model, device,
                              latent_shape, split=row["split"])
    _check_planned_segment(segment, row)
    if sha256_file(source) != row["source_sha256"]:
        raise CheckpointIntegrityError(f"downstream CFD source changed during export: {name}")
    return segment


def export_downstream_cache(checkpoint_dir, stage1_data_dir, dataset_dir, split_manifest, output_dir,
                            device="cpu", *, expected_checkpoint_sha256=None):
    """Freeze a Stage 1 best and encode only train/validation from a separate G0 manifest."""
    source, original_dir, data, plan_path, out = (
        Path(p).resolve() for p in (checkpoint_dir, stage1_data_dir, dataset_dir, split_manifest, output_dir)
    )
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise FileExistsError(f"cache output directory must be new or empty: {out}")
    plan_bytes = plan_path.read_bytes()
    plan = json.loads(plan_bytes)
    g0_path = plan_path.with_name("g0_audit.json")
    if plan.get("schema_version") != "flow3d.g2.split.v1" or sha256_file(g0_path) != plan["g0_audit_sha256"]:
        raise CheckpointIntegrityError("downstream G0 audit or split version differs from registration")
    planned = _planned_trajectories(plan)
    g0_sources = {Path(r["path"]).name: r["ancestor_id"] for r in json.loads(g0_path.read_bytes())["eligible_trajectories"]}
    if any(g0_sources.get(row["file_name"]) != row["ancestor_id"] for row in planned):
        raise CheckpointIntegrityError("a downstream source is absent from the registered G0 eligible ancestors")
    audit = _sidecar_hashes(data)
    if audit != plan["admission_sha256"]:
        raise CheckpointIntegrityError("downstream G0 admission records differ from the split manifest")
    stage1 = json.loads((source / "training_summary.json").read_text())["split_identity"]
    if plan["stage1_split_identity_sha256"] != sha256_json(stage1) or plan["stage1_files"] != stage1["files"]:
        raise CheckpointIntegrityError("downstream registration is for a different Stage 1 training population")
    stage1_audit = _sidecar_hashes(original_dir)
    originals = _audited_sources(original_dir, stage1["files"])
    if set(plan["stage1_ancestor_ids"]) != {v[0] for v in originals.values()}:
        raise CheckpointIntegrityError("reserved Stage 1 ancestry differs from original training data")
    original_solvers = {v[1] for v in originals.values()}
    sources = _audited_sources(data, {r["file_name"] for r in planned})
    solver_splits = {}
    for row in planned:
        ancestor, solver = sources[row["file_name"]]
        if (ancestor != row["ancestor_id"] or solver in original_solvers or row["source_sha256"] in stage1["files"].values()
                or solver_splits.setdefault(solver, row["split"]) != row["split"]):
            raise CheckpointIntegrityError(f"downstream solver lineage differs from split: {row['file_name']}")
    out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".downstream-export-", dir=out.parent) as tmp:
        staging = Path(tmp)
        manager, checkpoint, contract, _, digest = prepare_frozen_checkpoint(
            source, staging, stage1, dataset_dir=original_dir)
        if expected_checkpoint_sha256 and digest != expected_checkpoint_sha256:
            raise CheckpointIntegrityError("Stage 1 best pointer differs from the selected checkpoint")
        payload = manager.verify_checkpoint(checkpoint, expected_contract=contract)
        model = _model_from_payload(payload, device)
        del payload
        split_copy = staging / "downstream_split.json"
        split_copy.write_bytes(plan_bytes)
        segments = []
        for row in planned:
            if row["split"] == "test":
                continue
            print(f"Encoding {row['split']} {row['file_name']} on {device}", flush=True)
            shard = staging / f"segment-{len(segments):04d}.npz"
            segments.append(_encode_downstream_file(data, row, shard, model, device, contract["latent_shape"]))
        if (_sidecar_hashes(data) != audit or _sidecar_hashes(original_dir) != stage1_audit
                or contract["dataset_manifest_sha256"] != stage1_audit["manifest.json"]):
            raise CheckpointIntegrityError("source admission records changed during encoding")
        identity = {"files": {r["file_name"]: r["source_sha256"] for r in planned},
                    "splits": {key: [] for key in _SPLITS}}
        for row in segments:
            identity["splits"][row["split"]].extend([row["file_name"], f["case_id"], f["time_idx"]]
                                                     for f in row["frames"])
        index = {
            "format": _DOWNSTREAM_FORMAT, "checkpoint_sha256": sha256_file(checkpoint),
            "contract_sha256": sha256_json(contract), "stage1_split_identity": stage1,
            "stage1_admission_sha256": stage1_audit,
            "downstream_split_sha256": sha256_file(split_copy), "split_identity": identity,
            "admission_sha256": audit, "segments": segments, "shards": [r["shard"] for r in segments],
        }
        (staging / "index.json").write_text(json.dumps(index, indent=2, allow_nan=False) + "\n")
        manager.create_frozen_manifest(checkpoint, contract=contract,
                                       output_path=staging / "cache_manifest.json", cache_index="index.json")
        staging.rename(out)
    return out / "cache_manifest.json"


class LatentCache:
    """A verified frozen model and time-ordered sequences, loaded one shard at a time."""

    def __init__(self, verified, index, contract, split_strategy, source_dir=None):
        self.checkpoint_path = Path(verified["checkpoint_path"])
        self.checkpoint_sha256 = verified["checkpoint_sha256"]
        self.contract = contract
        self.weight_source = contract["weight_source"]
        self.stage1_split_strategy = split_strategy
        self.split_strategy = split_strategy
        self.split_identity = index["split_identity"]
        self.segments = index["segments"]
        self._root = Path(verified["cache_index_path"]).parent
        self._hashes = verified["shards_sha256"]
        self._audit = index["admission_sha256"]
        self._source_dir = Path(source_dir).resolve() if source_dir is not None else None
        self.downstream_split_sha256 = None
        if index["format"] == _DOWNSTREAM_FORMAT:
            split_file = self._root / "downstream_split.json"
            self.downstream_split_sha256 = sha256_file(split_file)
            if self.downstream_split_sha256 != index["downstream_split_sha256"]:
                raise CheckpointIntegrityError("downstream split manifest was changed")
            plan = json.loads(split_file.read_bytes())
            planned = _planned_trajectories(plan)
            cached = [row for row in planned if row["split"] != "test"]
            if (sha256_json(index["stage1_split_identity"]) != contract["split_sha256"]
                    or index["stage1_admission_sha256"]["manifest.json"] != contract["dataset_manifest_sha256"]
                    or self._audit != plan["admission_sha256"] or len(cached) != len(self.segments)
                    or self.split_identity["files"] != {r["file_name"]: r["source_sha256"] for r in planned}):
                raise CheckpointIntegrityError("downstream cache differs from its frozen sources")
            for segment, row in zip(self.segments, cached):
                _check_planned_segment(segment, row)
            self.split_strategy = "group_by_ancestor"
        elif (index["format"] != "flow3d.stage1.latents.v2"
              or sha256_json(self.split_identity) != contract["split_sha256"]
              or self._audit["manifest.json"] != contract["dataset_manifest_sha256"]):
            raise CheckpointIntegrityError("latent cache index differs from frozen Stage 1 contract")
        if [row["shard"] for row in self.segments] != index["shards"]:
            raise CheckpointIntegrityError("latent cache segments differ from the published shards")
        membership = {key: [] for key in _SPLITS}
        for row in self.segments:
            frames = row["frames"]
            for i, frame in enumerate(frames):
                if (
                    not frame["ground_truth_verified"] or frame["split"] != row["split"]
                    or frame["trajectory_id"] != row["trajectory_id"] or frame["ancestor_id"] != row["ancestor_id"]
                    or (i and frame["frame_index"] != frames[i - 1]["frame_index"] + 1)
                ):
                    raise CheckpointIntegrityError("latent cache frame provenance or temporal continuity invalid")
                membership[row["split"]].append([row["file_name"], frame["case_id"], frame["time_idx"]])
        if membership != self.split_identity["splits"]:
            raise CheckpointIntegrityError("cached frames differ from the ordered training split identity")

    def load_model(self, device="cpu"):
        manager = CheckpointManager(self.checkpoint_path.parent)
        payload = manager.verify_checkpoint(self.checkpoint_path, self.checkpoint_sha256, self.contract)
        return _model_from_payload(payload, device)

    def _sequence_conditions(self, row):
        """Recover an omitted legacy action marker only from the original, hash-bound G0 input."""
        cond = row["conditions"]
        meta = cond.get("extra_metadata", {})
        if (not cond.get("action_schedule") or "exogenous" in meta
                or cond.get("known_mask", {}).get("action_schedule") is not True
                or cond.get("valid_mask", {}).get("action_schedule") is not True):
            return cond
        if self._source_dir is None:
            raise CheckpointContractError("legacy action provenance requires --dataset-dir with the original G0 sources")
        name = row["file_name"]
        path = self._source_dir / name
        if (_sidecar_hashes(self._source_dir) != self._audit
                or sha256_file(path) != self.split_identity["files"][name]):
            raise CheckpointIntegrityError(f"legacy action source differs from frozen cache: {name}")
        manifest, tier, audit = _audit_files(self._source_dir)
        sources = [item for item in manifest["trajectories"] if item["file_name"] == name]
        if len(sources) != 1 or _admit_manifest_row(sources[0], manifest, tier, audit)[0] != row["ancestor_id"]:
            raise CheckpointIntegrityError(f"legacy action lineage differs from cache: {name}")
        import h5py

        with h5py.File(path, "r") as source:
            times = source["time_physical_s"][:]
        spec = validated_intervention(sources[0], times)
        if (spec is None or cond["action_schedule"] != spec["values"]
                or meta.get("action_times_s") != times.tolist() or meta.get("action_unit") != spec["unit"]
                or meta.get("action_channel") != spec["channel"]
                or meta.get("action_known_at_time_s") != spec["known_at_time_s"]):
            raise CheckpointIntegrityError(f"legacy cached action differs from G0 intervention: {name}")
        return {**cond, "extra_metadata": {**meta, "exogenous": True}}

    def sequences(self, split=None):
        """Yield float32 latents and stats with a time axis, plus static geometry and provenance."""
        if split is not None and split not in _SPLITS:
            raise ValueError(f"unknown cache split: {split}")
        for row in self.segments:
            if split is not None and row["split"] != split:
                continue
            conditions = self._sequence_conditions(row)
            path = self._root / row["shard"]
            if sha256_file(path) != self._hashes[row["shard"]]:
                raise CheckpointIntegrityError(f"changed latent shard: {path}")
            with np.load(path, allow_pickle=False) as archive:
                arrays = {key: archive[key] for key in ("z_clean", "b", "geometry")}
            frames = row["frames"]
            shapes = {
                "z_clean": (len(frames), *self.contract["latent_shape"]), "b": (len(frames), 6),
                "geometry": (2, *self.contract["spatial_shape_zyx"]),
            }
            for key, arr in arrays.items():
                if arr.shape != shapes[key] or arr.dtype != np.float32 or not np.isfinite(arr).all():
                    raise CheckpointIntegrityError(f"invalid cached {key} in {path}")
            yield dict(
                row, **arrays, conditions=conditions, step_indices=[f["time_idx"] for f in frames],
                time_physical_s=[f["time_phys_s"] for f in frames],
                time_nondim_ctu=[f["time_ctu"] for f in frames],
            )

    def targets_si(self, sequence, dataset_dir, frame_indices=None):
        """Return [T,4,Z,Y,X] original SI observations. Optional indices are relative to the cached segment."""
        directory = Path(dataset_dir)
        matches = [row for row in self.segments if row["shard"] == sequence["shard"]]
        if len(matches) != 1:
            raise CheckpointContractError("unknown latent segment")
        row = matches[0]
        source = directory / row["file_name"]
        expected = self.split_identity["files"][source.name]
        if _sidecar_hashes(directory) != self._audit or sha256_file(source) != expected:
            raise CheckpointIntegrityError(f"source data or admission records differ from cache: {source}")
        batch, _ = load_world_model_h5_trajectory(source, normalize=False)
        frames = row["frames"]
        selected = frames if frame_indices is None else [frames[i] for i in frame_indices]
        positions = []
        for frame in selected:
            pos = frame["frame_index"]
            provenance = batch.provenance[pos].to_dict()
            if any(frame.get(key) != value for key, value in provenance.items()):
                raise CheckpointIntegrityError("CFD target identity differs from cached frame")
            positions.append(pos)
        return batch.field[0, np.asarray(positions, dtype=np.intp)]


def load_latent_cache(manifest_path: Path, *, expected_contract=None, source_dir=None) -> LatentCache:
    """Verify the v2 frozen checkpoint, index and every shard before returning a cache."""
    path = Path(manifest_path).resolve()
    manager = CheckpointManager(path.parent)
    verified = manager.verify_frozen_manifest(path, expected_contract=expected_contract)
    payload = manager.verify_checkpoint(verified["checkpoint_path"], verified["checkpoint_sha256"])
    index = json.loads(Path(verified["cache_index_path"]).read_text(encoding="utf-8"))
    strategy = payload["config"]["training"].get("split_strategy", "temporal_block")
    return LatentCache(verified, index, payload["contract"], strategy, source_dir=source_dir)
