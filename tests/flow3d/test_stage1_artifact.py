import json

import pytest

h5py = pytest.importorskip("h5py")
torch = pytest.importorskip("torch")

from flow3d.models import Flow3DStage1Autoencoder
from flow3d.training.artifact import build_stage1_contract, configure_stage1_resume, prepare_frozen_checkpoint
from stage1.io.checkpoint import CheckpointContractError, CheckpointIntegrityError, CheckpointManager, sha256_file, sha256_json


def source_run(root, *, legacy=False, ema=False, record_split=True):
    data = root / "dataset"
    data.mkdir()
    filename = "trajectory.h5"
    shape = (5, 6, 8, 16, 24)
    with h5py.File(data / filename, "w") as h5:
        h5.create_dataset("flow_fields", shape=shape, dtype="f")
    (data / "manifest.json").write_text(json.dumps({"trajectories": [{"file_name": filename, "tensor_shape": shape}]}))
    split = {
        "files": {filename: sha256_file(data / filename)},
        "splits": {name: [[filename, "case", i]] for i, name in enumerate(("train", "val", "test"))},
    }
    config = {
        "normalizer_version": "normalizer.flow3d.si.v2",
        "encoder": {"base_channels": 4, "latent_channels": 8, "num_refine_blocks": 0},
        "spatial_transformer": {"num_heads": 2, "mlp_dim": 16},
        "latent_norm": {"num_groups": 2},
        "training": {"ema": ema},
    }
    if record_split:
        config["training"]["split_sha256"] = sha256_json(split)
    contract = {} if legacy else build_stage1_contract(config, split, data)
    manager = CheckpointManager(root / "training", model_name="dcae3d_head_a", contract=contract)
    model = Flow3DStage1Autoencoder.from_config(config)
    ema_state = {name: value + 0.5 for name, value in model.state_dict().items()} if ema else None
    path = manager.save_checkpoint(
        epoch=2, global_step=4, model=model, optimizer=torch.optim.Adam(model.parameters()),
        config=config, ema_state_dict=ema_state, metrics={"val_loss": 1.25}, is_best=True,
    )
    (manager.save_dir / "training_summary.json").write_text(json.dumps({"split_identity": split, "best_val_loss": 1.25}))
    return data, manager, path, split


@pytest.mark.parametrize(("legacy", "ema"), [(True, True), (False, False)])
def test_freeze_selected_weights_binds_actual_grid_and_preserves_original(tmp_path, legacy, ema):
    data, original, source, split = source_run(tmp_path, legacy=legacy, ema=ema)
    original_files = {p.name: sha256_file(p) for p in original.save_dir.iterdir()}
    manager, path, contract, weights, digest = prepare_frozen_checkpoint(
        original.save_dir, tmp_path / "export", split, dataset_dir=data,
    )
    assert path.parent == tmp_path / "export"
    assert contract["original_checkpoint_sha256"] == digest == original_files[source.name]
    assert contract["dataset_manifest_binding"] == ("legacy_export" if legacy else "training")
    assert contract["split_sha256"] == sha256_json(split)
    assert contract["dataset_manifest_sha256"] == sha256_file(data / "manifest.json")
    assert contract["latent_layout"] == "B,T,C,Z,Y,X"
    assert contract["latent_shape"] == [8, 1, 2, 3]
    assert weights == contract["weight_source"] == ("ema" if ema else "raw")
    training = original.verify_checkpoint("best")
    exported = manager.verify_checkpoint(path, expected_contract=contract)
    assert exported["checkpoint_kind"] == "weights" and exported["ema_state_dict"] is None
    expected = training["ema_state_dict"] if ema else training["model_state_dict"]
    assert expected.keys() == exported["model_state_dict"].keys()
    for name, value in expected.items():
        torch.testing.assert_close(exported["model_state_dict"][name], value, atol=0, rtol=0)
    model = Flow3DStage1Autoencoder.from_config(exported["config"]).eval()
    model.load_state_dict(exported["model_state_dict"])
    with torch.no_grad():
        encoded = model.encode(torch.zeros(1, 1, 4, 8, 16, 24), torch.ones(1, 2, 8, 16, 24))
    assert list(encoded.shape) == [1, 1, *contract["latent_shape"]]
    assert {p.name: sha256_file(p) for p in original.save_dir.iterdir()} == original_files


@pytest.mark.parametrize("change", ["split", "missing_digest", "file", "manifest", "pointer", "empty_pointer_digest"])
def test_freeze_rejects_unverified_training_or_source(tmp_path, change):
    data, manager, _, split = source_run(tmp_path, legacy=change == "missing_digest", record_split=change != "missing_digest")
    if change == "split":
        split["splits"]["train"] = []
    if change == "file":
        with h5py.File(data / "trajectory.h5", "r+") as h5:
            h5["flow_fields"][0, 0, 0, 0, 0] = 1
    if change == "manifest":
        with (data / "manifest.json").open("a") as manifest:
            manifest.write("\n")
    if change in {"pointer", "empty_pointer_digest"}:
        pointer = json.loads(manager.best_pointer_path.read_text())
        pointer["sha256"] = "0" * 64
        if change == "empty_pointer_digest":
            pointer.update(sha256="", checkpoint_sha256="", checkpoint_id="")
        manager.best_pointer_path.write_text(json.dumps(pointer))
    with pytest.raises((CheckpointContractError, CheckpointIntegrityError)):
        prepare_frozen_checkpoint(manager.save_dir, tmp_path / "export", split, dataset_dir=data)
    assert not (tmp_path / "export").exists()


@pytest.mark.parametrize("trusted_split", [True, False])
def test_legacy_resume_preserves_the_recorded_lineage(tmp_path, trusted_split):
    data, original, source, split = source_run(tmp_path, legacy=True, record_split=trusted_split)
    new = CheckpointManager(tmp_path / "resumed", model_name=original.model_name,
                            contract=build_stage1_contract({}, split, data))
    training = {"split_sha256": sha256_json(split)}
    assert configure_stage1_resume(new, original.best_pointer_path, training) == source
    assert new.contract == {}
    assert ("split_sha256" in training) == trusted_split
    if trusted_split:
        with pytest.raises(CheckpointContractError, match="different split"):
            configure_stage1_resume(new, source, {"split_sha256": "0" * 64})
