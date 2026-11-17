from __future__ import annotations

import random

import pytest

from stage1.io.checkpoint import (
    CheckpointContractError,
    CheckpointIntegrityError,
    CheckpointManager,
    CheckpointStateError,
    canonical_json,
    sha256_json,
)


def test_contract_hash_is_stable_for_mapping_order():
    left = {"latent_shape": [8, 12, 24], "normalizer": {"p": "v1", "u": "v1"}}
    right = {"normalizer": {"u": "v1", "p": "v1"}, "latent_shape": [8, 12, 24]}
    assert canonical_json(left) == canonical_json(right)
    assert sha256_json(left) == sha256_json(right)


def _torch_or_skip():
    return pytest.importorskip("torch")


def _make_objects(torch):
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
    return model, optimizer, scheduler


def test_roundtrip_restores_optimizer_scheduler_and_rng(tmp_path):
    torch = _torch_or_skip()
    contract = {
        "latent_layout": "B,T,C,Z,Y,X",
        "latent_shape": [256, 8, 12, 24],
        "normalizer_version": "norm-v1",
    }
    manager = CheckpointManager(tmp_path, contract=contract)
    model, optimizer, scheduler = _make_objects(torch)
    loss = model(torch.ones(4, 3)).square().mean()
    loss.backward()
    optimizer.step()
    scheduler.step()
    path = manager.save_checkpoint(3, 17, model, optimizer, scheduler, contract=contract, is_best=True)

    expected_py = random.random()
    expected_torch = torch.rand(3)

    # Perturb all states, then restore from the checkpoint.
    random.seed(9001)
    torch.manual_seed(9001)
    model_2, optimizer_2, scheduler_2 = _make_objects(torch)
    manager.load_checkpoint(path, model_2, optimizer_2, scheduler_2, expected_contract=contract)

    assert random.random() == expected_py
    torch.testing.assert_close(torch.rand(3), expected_torch)
    assert optimizer_2.state_dict()["state"]
    assert scheduler_2.state_dict()["last_epoch"] == scheduler.state_dict()["last_epoch"]


def test_best_pointer_survives_interrupted_write(tmp_path, monkeypatch):
    torch = _torch_or_skip()
    import stage1.io.checkpoint as checkpoint_module

    manager = CheckpointManager(tmp_path)
    model, optimizer, scheduler = _make_objects(torch)
    first = manager.save_checkpoint(1, 1, model, optimizer, scheduler, is_best=True)
    pointer_before = manager.best_pointer_path.read_text(encoding="utf-8")

    def fail_save(*args, **kwargs):
        raise OSError("simulated disk interruption")

    monkeypatch.setattr(checkpoint_module.torch, "save", fail_save)
    with pytest.raises(OSError):
        manager.save_checkpoint(2, 2, model, optimizer, scheduler, is_best=True)

    assert manager.best_pointer_path.read_text(encoding="utf-8") == pointer_before
    assert manager.verify_checkpoint("best")["epoch"] == 1
    assert first.exists()
    assert not list(tmp_path.glob("*.tmp"))


def test_contract_mismatch_and_missing_resume_state_fail_loudly(tmp_path):
    torch = _torch_or_skip()
    manager = CheckpointManager(tmp_path, contract={"latent_shape": [256, 8, 12, 24]})
    model, optimizer, scheduler = _make_objects(torch)
    path = manager.save_checkpoint(1, 1, model, optimizer, scheduler)

    with pytest.raises(CheckpointContractError):
        manager.load_checkpoint(path, model, optimizer, scheduler, expected_contract={"latent_shape": [1]})

    model_weights_only, _, _ = _make_objects(torch)
    weights_manager = CheckpointManager(tmp_path / "weights")
    weights = weights_manager.save_checkpoint(
        1, 1, model_weights_only, optimizer=None, include_rng=False
    )
    with pytest.raises(CheckpointStateError):
        weights_manager.load_checkpoint(weights, model_weights_only, optimizer=optimizer, restore_rng=False)


def test_best_pointer_detects_tampering(tmp_path):
    torch = _torch_or_skip()
    manager = CheckpointManager(tmp_path)
    model, optimizer, scheduler = _make_objects(torch)
    path = manager.save_checkpoint(1, 1, model, optimizer, scheduler, is_best=True)
    with path.open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(CheckpointIntegrityError):
        manager.load_checkpoint("best", model, optimizer, scheduler)


def test_frozen_manifest_binds_checkpoint_and_contract(tmp_path):
    torch = _torch_or_skip()
    contract = {"latent_layout": "B,T,C,Z,Y,X", "normalizer_version": "norm-v1"}
    manager = CheckpointManager(tmp_path, contract=contract)
    model, optimizer, scheduler = _make_objects(torch)
    path = manager.save_checkpoint(1, 1, model, optimizer, scheduler, is_best=True)
    manifest = manager.create_frozen_manifest(
        "best",
        latent_source="z_clean",
        latent_layout="B,T,256,8,12,24",
        normalizer_version="norm-v1",
        export_mode="eval",
        output_path="cache_manifest.json",
    )
    verified = manager.verify_frozen_manifest(manifest, expected_contract=contract)
    assert verified["checkpoint_sha256"] == manifest["checkpoint_sha256"]
    assert verified["checkpoint"] == path.name
