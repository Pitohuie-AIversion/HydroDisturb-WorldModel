"""Bind Stage 1 training splits and validation weights to a frozen encoder/decoder."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from flow3d.data.normalization import Flow3DNormalizer, SI_NORMALIZER_VERSION
from flow3d.models import Flow3DStage1Autoencoder
from stage1.io.checkpoint import (
    CheckpointContractError, CheckpointIntegrityError, CheckpointManager,
    sha256_bytes, sha256_file, sha256_json,
)


def build_stage1_contract(
    config: dict, split_identity: dict, dataset_dir: Path
) -> dict[str, Any]:
    """Use the admitted HDF5 grid and ordered split, including non-default grids."""
    import h5py

    dataset_dir = Path(dataset_dir)
    manifest = (dataset_dir / "manifest.json").read_bytes()
    trajectories = json.loads(manifest)["trajectories"]
    files = split_identity["files"]
    shapes = set()
    for name in files:
        if Path(name).name != name:
            raise CheckpointContractError(f"invalid Stage 1 trajectory filename: {name}")
        rows = [row for row in trajectories if row["file_name"] == name]
        with h5py.File(dataset_dir / name, "r") as source:
            shape = source["flow_fields"].shape
        if len(rows) != 1 or list(shape) != rows[0].get("tensor_shape") or len(shape) != 5 or shape[1] != 6:
            raise CheckpointContractError(f"{name}: HDF5 grid differs from the dataset manifest")
        shapes.add(tuple(shape[-3:]))
    if len(shapes) != 1:
        raise CheckpointContractError("Stage 1 cache requires one common spatial grid")
    spatial_shape = shapes.pop()
    encoder = config.get("encoder", {})
    reduction = int(encoder.get("downsample_factor", 8))
    channels = int(encoder.get("latent_channels", 256))
    if reduction != 8 or channels <= 0 or any(n <= 0 or n % reduction for n in spatial_shape):
        raise CheckpointContractError("grid cannot be encoded by the configured three 2x downsampling stages")
    normalizer = Flow3DNormalizer()
    if config.get("normalizer_version", SI_NORMALIZER_VERSION) != normalizer.version:
        raise CheckpointContractError("unsupported Stage 1 normalizer_version for SI export")
    model_sections = (
        "schema_version", "latent_contract_version", "grid", "encoder", "spatial_transformer",
        "decoder", "head", "latent_norm", "normalizer", "probe",
    )
    return {
        "schema_version": 1,
        "latent_layout": "B,T,C,Z,Y,X",
        "latent_shape": [channels, *(n // reduction for n in spatial_shape)],
        "spatial_shape_zyx": list(spatial_shape),
        "normalizer_version": normalizer.version,
        "normalization": {"scale_floor_uvw": normalizer.scale_floor_uvw, "scale_floor_p": normalizer.scale_floor_p},
        "latent_contract_version": config.get("latent_contract_version", "latent.flow3d.v2"),
        "model_config_sha256": sha256_json({key: config.get(key) for key in model_sections}),
        "dataset_manifest_sha256": sha256_bytes(manifest),
        "split_sha256": sha256_json(split_identity),
        "weight_source": "ema" if config.get("training", {}).get("ema", False) else "raw",
    }


def prepare_frozen_checkpoint(
    checkpoint_dir: Path, output_dir: Path, split_identity: dict, *, dataset_dir: Path
) -> tuple[CheckpointManager, Path, dict, str, str]:
    """Export the validated best as a standalone single-weight checkpoint.

    Pass the identity recomputed by stage1_split_identity on the export dataset.
    A legacy run must prove its split using the digest embedded during training.
    Its manifest bytes are first bound at export, which is recorded explicitly.
    """
    source_dir = Path(checkpoint_dir).resolve()
    destination = Path(output_dir).resolve()
    dataset_dir = Path(dataset_dir)
    if destination == source_dir:
        raise CheckpointContractError("frozen weights require a separate output directory")
    pointers = list(source_dir.glob("*_best.json"))
    if len(pointers) != 1:
        raise CheckpointContractError(f"{source_dir}: expected exactly one Stage 1 best pointer")
    pointer = json.loads(pointers[0].read_text(encoding="utf-8"))
    model_name = pointers[0].name.removesuffix("_best.json")
    source = source_dir / pointer["checkpoint"]
    digest = pointer["checkpoint_sha256"]
    if (
        not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest)
        or source.resolve().parent != source_dir or source.name != pointer["checkpoint"]
        or pointer.get("pointer_version") != 1 or pointer.get("model_name") != model_name
        or pointer.get("sha256") != digest or pointer.get("checkpoint_id") != digest[:16]
    ):
        raise CheckpointIntegrityError("inconsistent Stage 1 best pointer")
    original = CheckpointManager(source_dir, model_name=model_name)
    payload = original.verify_checkpoint(source, expected_sha256=digest)
    config = payload["config"]
    saved = payload["contract"]
    if (
        payload["model_name"] != model_name or payload["config_sha256"] != sha256_json(config)
        or not isinstance(saved, dict) or payload["contract_sha256"] != sha256_json(saved)
        or payload["contract_sha256"] != pointer.get("contract_sha256")
    ):
        raise CheckpointIntegrityError("checkpoint configuration or contract differs from the original best pointer")
    summary = json.loads((source_dir / "training_summary.json").read_text(encoding="utf-8"))
    recorded_split = config.get("training", {}).get("split_sha256")
    if summary.get("split_identity") != split_identity or recorded_split != sha256_json(split_identity):
        raise CheckpointContractError("training split identity differs from the actual export split or is missing")
    if "val_loss" not in payload["metrics"] or payload["metrics"]["val_loss"] != summary.get("best_val_loss"):
        raise CheckpointContractError("best checkpoint validation loss differs from the training summary")
    if not saved and config.get("normalizer_version") != SI_NORMALIZER_VERSION:
        raise CheckpointContractError("legacy checkpoint lacks a verified SI normalizer version")
    contract = build_stage1_contract(config, split_identity, dataset_dir)
    if saved and saved != contract:
        raise CheckpointContractError("training checkpoint contract differs from the export data or configuration")
    for filename, expected in split_identity["files"].items():
        if sha256_file(dataset_dir / filename) != expected:
            raise CheckpointIntegrityError(f"trajectory content differs from training: {filename}")
    weight_source = contract["weight_source"]
    if (payload.get("ema_state_dict") is not None) != (weight_source == "ema"):
        raise CheckpointContractError("checkpoint validation weights disagree with the training EMA configuration")
    contract.update(
        original_checkpoint_sha256=digest,
        dataset_manifest_binding="training" if saved else "legacy_export",
    )
    import torch

    with torch.random.fork_rng(devices=[]):
        model = Flow3DStage1Autoencoder.from_config(config)
    state = payload["ema_state_dict"] if weight_source == "ema" else payload["model_state_dict"]
    model.load_state_dict(state, strict=True)
    frozen = CheckpointManager(destination, model_name=f"{model_name}_frozen", contract=contract)
    path = frozen.save_checkpoint(
        epoch=payload["epoch"], global_step=payload["global_step"], model=model, optimizer=None,
        config=config, metrics=payload["metrics"], include_rng=False,
    )
    return frozen, path, contract, weight_source, digest


def configure_stage1_resume(
    manager: CheckpointManager, checkpoint: str | Path, training_config: dict,
) -> Path:
    """Continue an older training run without retroactively certifying its history."""
    path, digest = manager._resolve_path(checkpoint)
    reader = CheckpointManager(manager.save_dir, model_name=manager.model_name)
    payload = reader.verify_checkpoint(path, expected_sha256=digest)
    stored = payload["contract"]
    if sha256_json(stored) != payload["contract_sha256"]:
        raise CheckpointIntegrityError("resume checkpoint contract hash differs from its contents")
    if not stored:
        trained_split = payload["config"].get("training", {}).get("split_sha256")
        if trained_split is None:
            training_config.pop("split_sha256", None)
        elif trained_split != training_config["split_sha256"]:
            raise CheckpointContractError("cannot resume the verified legacy run on a different split")
        manager.contract = {}
    return path.resolve()
