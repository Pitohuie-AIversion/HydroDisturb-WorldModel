"""Exercise an actual frozen model and certified HDF5 fixture without training."""

import json
import shutil

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from tests.flow3d.test_h5_integrity import _branch_pair, trajectory

from flow3d.data.dataset import Flow3DH5Dataset, stage1_split_identity
from flow3d.data.latent_cache import export_downstream_cache, export_latent_cache, load_latent_cache
from flow3d.data.normalization import SI_NORMALIZER_VERSION
from flow3d.models import Flow3DStage1Autoencoder
from stage1.io.checkpoint import CheckpointContractError, CheckpointIntegrityError, CheckpointManager, sha256_file, sha256_json


def test_cache_reproduces_heldout_frames_and_weights_and_detects_changed_bytes(trajectory, tmp_path, monkeypatch):
    from flow3d.data.dataset import _TRAJECTORY_CACHE

    path, raw = trajectory
    splits = Flow3DH5Dataset.create_splits(path.parent, eligible_files=[path.name], purge_gap_frames=1)
    identity = stage1_split_identity(*splits)
    cfg = {
        "encoder": {"latent_channels": 8, "base_channels": 4, "gn_groups": 1},
        "latent_norm": {"num_groups": 1},
        "spatial_transformer": {"num_heads": 2, "mlp_dim": 16},
        "normalizer_version": SI_NORMALIZER_VERSION,
        "head": {"type": "head_a"},
        "training": {"purge_gap_frames": 1, "split_sha256": sha256_json(identity)},
    }
    model = Flow3DStage1Autoencoder.from_config(cfg).eval()
    source = tmp_path / "run"
    manager = CheckpointManager(source, model_name="dcae3d_head_a")
    checkpoint = manager.save_checkpoint(
        1, 1, model, None, config=cfg, is_best=True, include_rng=False, metrics={"val_loss": 0.1},
    )
    summary = {"config": cfg, "split_identity": identity, "best_val_loss": 0.1}
    (source / "training_summary.json").write_text(json.dumps(summary))
    cache_dir = tmp_path / "cache"
    # An earlier training process can hold data under the same path. Export must read its own audited snapshot.
    monkeypatch.setitem(_TRAJECTORY_CACHE, str(path.resolve()), (None, {}))
    manifest = export_latent_cache(source, path.parent, cache_dir, "cpu")
    cache = load_latent_cache(manifest)
    assert manifest == cache_dir / "cache_manifest.json"
    assert cache.checkpoint_path.is_absolute() and cache.weight_source == "raw"
    assert cache.split_strategy == "temporal_block"
    assert cache.contract["original_checkpoint_sha256"] == sha256_file(checkpoint)
    assert cache.split_identity == identity
    frozen = cache.load_model()
    assert not frozen.training and all(not p.requires_grad for p in frozen.parameters())

    for split, dataset in zip(("train", "val", "test"), splits):
        sequence, = cache.sequences(split)
        frames = sequence["frames"]
        assert sequence["step_indices"] == [row[2] for row in identity["splits"][split]]
        assert sequence["conditions"]["reynolds_number"] == 100
        assert sequence["conditions"]["dt_phys_s"] == pytest.approx(0.04)
        assert sequence["ancestor_id"] == dataset._provenances[dataset.active_indices[0]].ancestor_id
        assert [f["time_phys_s"] for f in frames] == sequence["time_physical_s"]
        targets = cache.targets_si(sequence, path.parent)
        np.testing.assert_allclose(targets, torch.stack([item["phys_field"] for item in dataset]).numpy())
        for i, item in enumerate(dataset):
            with torch.inference_mode():
                expected = model.encode(item["norm_field"][None], item["geometry"][None])
            np.testing.assert_allclose(sequence["z_clean"][i], expected.numpy()[0])
            np.testing.assert_array_equal(sequence["b"][i], item["b"].numpy())
            assert targets[i, 3, 1, 1, 1] == pytest.approx(raw[frames[i]["frame_index"], 3, 1, 1, 1] * 40 + 5)

    with pytest.raises(FileExistsError):
        export_latent_cache(source, path.parent, cache_dir)
    with path.open("ab") as stream:
        stream.write(b"different CFD source")
    with pytest.raises(CheckpointIntegrityError):
        cache.targets_si(sequence, path.parent)
    with pytest.raises(CheckpointIntegrityError):
        export_latent_cache(source, path.parent, tmp_path / "changed-cache")
    assert not (tmp_path / "changed-cache").exists()
    shard = cache_dir / sequence["shard"]
    with shard.open("ab") as stream:
        stream.write(b"different latents")
    with pytest.raises(CheckpointIntegrityError):
        load_latent_cache(manifest)


def test_downstream_export_seals_new_split_without_opening_test_fields(trajectory, tmp_path):
    import h5py

    path, _ = trajectory
    datasets = Flow3DH5Dataset.create_splits(path.parent, eligible_files=[path.name], purge_gap_frames=1)
    identity = stage1_split_identity(*datasets)
    cfg = {
        "encoder": {"latent_channels": 8, "base_channels": 4, "gn_groups": 1},
        "latent_norm": {"num_groups": 1}, "spatial_transformer": {"num_heads": 2, "mlp_dim": 16},
        "head": {"type": "head_a"}, "normalizer_version": SI_NORMALIZER_VERSION,
        "training": {"purge_gap_frames": 1, "split_sha256": sha256_json(identity)},
    }
    source = tmp_path / "run"
    manager = CheckpointManager(source, model_name="head_a")
    checkpoint = manager.save_checkpoint(1, 1, Flow3DStage1Autoencoder.from_config(cfg), None,
                                         config=cfg, is_best=True, include_rng=False, metrics={"val_loss": 0.1})
    summary = {"config": cfg, "split_identity": identity, "best_val_loss": 0.1}
    (source / "training_summary.json").write_text(json.dumps(summary))
    downstream = tmp_path / "downstream"
    downstream.mkdir()
    original = json.loads((path.parent / "manifest.json").read_text())["trajectories"][0]
    sidecars = ["manifest.json", "multi_re_tier1_certification_summary.json", "extraction_and_rampool_assessment.json"]
    for name in sidecars[1:]:
        shutil.copyfile(path.parent / name, downstream / name)
    names = {split: [f"{split}.h5"] for split in ("train", "val", "test")}
    rows, planned = [], []
    for split, (name,) in names.items():
        solver = f"e:/solver/run/{split}.mph"
        rows.append(dict(original, file_name=name, case_id=f"case_{split}", source_model=solver))
        h5 = downstream / name
        shutil.copyfile(path, h5)
        with h5py.File(h5, "r+") as handle:
            handle.attrs["source_model"] = solver
        planned.append(dict(file_name=name, trajectory_id=split, ancestor_id=solver, split=split,
                            source_sha256=sha256_file(h5), num_steps=12, frame_indices=list(range(12)),
                            step_indices=list(range(110, 122)), dt_phys_s=0.04, dt_ctu=0.04 * 2 / 0.3))
    (downstream / "manifest.json").write_text(json.dumps({"trajectories": rows}))
    g0 = downstream / "g0_audit.json"
    g0.write_text(json.dumps({"eligible_trajectories": [
        {"path": str(downstream / row["file_name"]), "ancestor_id": row["ancestor_id"]} for row in planned
    ]}))
    plan = dict(schema_version="flow3d.g2.split.v1", g0_audit_sha256=sha256_file(g0), splits=names,
                stage1_split_identity_sha256=sha256_json(identity), stage1_files=identity["files"],
                trajectories=planned, stage1_ancestor_ids=["e:/solver/run/a.mph"],
                admission_sha256={name: sha256_file(downstream / name) for name in sidecars})
    split_file = downstream / "split.json"
    # A reserved origin is already registered. Its CFD payload is intentionally unavailable to the exporter.
    (downstream / "test.h5").unlink()
    output = tmp_path / "export"
    leaked = dict(plan, stage1_ancestor_ids=[*plan["stage1_ancestor_ids"], planned[1]["ancestor_id"]])
    split_file.write_text(json.dumps(leaked))
    with pytest.raises(CheckpointContractError, match="ancestor is reused"):
        export_downstream_cache(source, path.parent, downstream, split_file, output)
    assert not output.exists()
    split_file.write_text(json.dumps(plan))
    manifest = export_downstream_cache(source, path.parent, downstream, split_file, output,
                                       expected_checkpoint_sha256=sha256_file(checkpoint))
    cache = load_latent_cache(manifest)
    assert cache.downstream_split_sha256 == sha256_file(split_file)
    assert cache.contract["split_sha256"] == sha256_json(identity)
    assert cache.split_strategy == "group_by_ancestor" and not list(cache.sequences("test"))
    val, = cache.sequences("val")
    assert val["step_indices"] == list(range(110, 122)) and val["ancestor_id"] == planned[1]["ancestor_id"]
    assert cache.targets_si(val, downstream).shape == (12, 4, 8, 8, 8)
    (output / "downstream_split.json").write_text("{}")
    with pytest.raises(CheckpointIntegrityError, match="split manifest was changed"):
        load_latent_cache(manifest)


def test_legacy_action_cache_requires_exact_original_g0_provenance(trajectory, tmp_path):
    path, _ = trajectory
    pulse, _ = _branch_pair(path)
    splits = Flow3DH5Dataset.create_splits(path.parent, eligible_files=[pulse.name], purge_gap_frames=1)
    identity = stage1_split_identity(*splits)
    cfg = {"encoder": {"latent_channels": 8, "base_channels": 4, "gn_groups": 1},
           "latent_norm": {"num_groups": 1}, "spatial_transformer": {"num_heads": 2, "mlp_dim": 16},
           "normalizer_version": SI_NORMALIZER_VERSION, "head": {"type": "head_a"},
           "training": {"split_sha256": sha256_json(identity), "purge_gap_frames": 1}}
    run = tmp_path / "source"
    manager = CheckpointManager(run, model_name="head_a")
    manager.save_checkpoint(1, 1, Flow3DStage1Autoencoder.from_config(cfg), None, config=cfg,
                            is_best=True, metrics={"val_loss": 0.1}, include_rng=False)
    (run / "training_summary.json").write_text(json.dumps({"config": cfg, "split_identity": identity,
                                                          "best_val_loss": 0.1}))
    cache_dir = tmp_path / "cache"
    manifest = export_latent_cache(run, path.parent, cache_dir)
    current = load_latent_cache(manifest)
    expected = next(current.sequences("train"))["conditions"]
    assert expected["extra_metadata"]["exogenous"] is True

    # Publish the exact historical form of an admitted cache and bind its index/shards normally.
    index_file = cache_dir / "index.json"
    index = json.loads(index_file.read_text())
    for row in index["segments"]:
        del row["conditions"]["extra_metadata"]["exogenous"]

    def publish_legacy():
        index_file.write_text(json.dumps(index))
        CheckpointManager(cache_dir, contract=current.contract).create_frozen_manifest(
            current.checkpoint_path, contract=current.contract, cache_index="index.json", output_path=manifest)

    publish_legacy()
    with pytest.raises(CheckpointContractError, match="original G0"):
        next(load_latent_cache(manifest).sequences("train"))
    registered = sha256_file(manifest)
    recovered = next(load_latent_cache(manifest, source_dir=path.parent).sequences("train"))["conditions"]
    assert recovered == expected and sha256_file(manifest) == registered
    index["segments"][0]["conditions"]["action_schedule"][-1] += 1.
    publish_legacy()
    with pytest.raises(CheckpointIntegrityError, match="differs from G0"):
        next(load_latent_cache(manifest, source_dir=path.parent).sequences("train"))
