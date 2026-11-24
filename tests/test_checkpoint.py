from __future__ import annotations

import errno
import json
import random
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import numpy as np
import pytest

from stage1.io.checkpoint import (
    CheckpointContractError,
    CheckpointIntegrityError,
    CheckpointManager,
    CheckpointStateError,
    canonical_json,
    sha256_file,
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


def test_same_step_checkpoint_retries_preserve_files_and_best(tmp_path, monkeypatch):
    torch = _torch_or_skip()
    import stage1.io.checkpoint as checkpoint_module

    monkeypatch.chdir(tmp_path)
    manager = CheckpointManager("checkpoints")
    model, optimizer, _ = _make_objects(torch)
    first = manager.save_checkpoint(1, 7, model, optimizer, is_best=True)
    digest = sha256_file(first)
    with torch.no_grad():
        model.weight.add_(1)
    second = manager.save_checkpoint(1, 7, model, optimizer)
    assert second != first and sha256_file(first) == digest
    manager.verify_checkpoint(manager.best_pointer_path, expected_sha256=digest)

    # If publishing the new pointer fails, the previous best and its bytes survive.
    with monkeypatch.context() as patch:
        def fail_pointer(*args):
            raise OSError("interrupted pointer publication")
        patch.setattr(checkpoint_module, "_atomic_write_bytes", fail_pointer)
        with pytest.raises(OSError, match="interrupted pointer"):
            manager.save_checkpoint(1, 7, model, optimizer, is_best=True)
    manager.verify_checkpoint("best", expected_sha256=digest)
    final = manager.save_checkpoint(1, 7, model, optimizer, is_best=True)
    manager.verify_checkpoint("best", expected_sha256=sha256_file(final))
    assert first != final and sha256_file(first) == digest


def test_concurrent_same_step_writers_publish_complete_distinct_checkpoints(tmp_path):
    torch = _torch_or_skip()
    ready = Barrier(4)
    def save(writer):
        manager = CheckpointManager(tmp_path)
        model = torch.nn.Linear(1, 1)
        ready.wait(timeout=15)
        return manager.save_checkpoint(1, 1, model, None, include_rng=False,
                                       extra_state={"writer": writer}, is_best=True)

    with ThreadPoolExecutor(max_workers=4) as pool:
        paths = list(pool.map(save, range(4)))
    manager = CheckpointManager(tmp_path)
    assert len(set(paths)) == 4
    for writer, path in enumerate(paths):
        assert manager.verify_checkpoint(path)["extra_state"]["writer"] == writer
    pointer = json.loads(manager.best_pointer_path.read_text())
    winner = manager.verify_checkpoint("best")["extra_state"]["writer"]
    assert paths[winner].name == pointer["checkpoint"]
    assert sha256_file(paths[winner]) == pointer["sha256"]


def test_checkpoints_remain_immutable_when_filesystem_has_no_hard_links(tmp_path, monkeypatch):
    torch = _torch_or_skip()
    import stage1.io.checkpoint as checkpoint_module

    def unsupported(*args):
        raise OSError(errno.ENOTSUP, "hard links unavailable")

    monkeypatch.setattr(checkpoint_module.os, "link", unsupported)
    manager = CheckpointManager(tmp_path)
    model, optimizer, _ = _make_objects(torch)
    path = manager.save_checkpoint(2, 1, model, optimizer, is_best=True)
    before = sha256_file(path)
    second = manager.save_checkpoint(2, 1, model, optimizer, extra_state={"retry": 1})
    assert second != path and sha256_file(path) == before
    manager.verify_checkpoint("best", expected_sha256=before)


def test_numpy_resume_state_roundtrip_weights_only(tmp_path):
    torch = _torch_or_skip()
    manager = CheckpointManager(tmp_path)
    model, optimizer, _ = _make_objects(torch)
    indices = np.arange(18, dtype=">i8").reshape(3, 6)[:, ::2]
    state = {"indices": indices, "cursor": np.uint32(2),
             "__stage1_numpy_state_v1__": "literal key", "limits": (0, torch.Size([3, 3]))}
    loader = [{np.int64(4): np.empty((0, 2), dtype="float16"), "a": np.array(3.)}]
    path = manager.save_checkpoint(1, 1, model, optimizer, sampler_state=state, dataloader_state=loader)
    assert torch.load(path, weights_only=True)["format_version"] == 2
    for result in (manager.verify_checkpoint(path), manager.load_checkpoint(path, model, optimizer)):
        saved = result["sampler_state"]
        assert type(saved["cursor"]) is np.uint32
        assert saved["__stage1_numpy_state_v1__"] == "literal key"
        assert type(saved["limits"][1]) is torch.Size
        np.testing.assert_array_equal(saved["indices"], indices, strict=True)
        loaded = result["dataloader_state"][0]
        assert type(next(iter(loaded))) is np.int64
        np.testing.assert_array_equal(loaded[4], loader[0][4], strict=True)
        assert type(loaded["a"]) is np.ndarray and loaded["a"].shape == ()


def test_unsupported_sampler_state_rejected_before_publication(tmp_path):
    torch = _torch_or_skip()
    manager = CheckpointManager(tmp_path)
    model, optimizer, _ = _make_objects(torch)
    path = manager.save_checkpoint(1, 1, model, optimizer, is_best=True)
    with pytest.raises(CheckpointStateError, match="object dtype"):
        manager.save_checkpoint(1, 1, model, optimizer, sampler_state=np.array([object()]), is_best=True)
    manager.verify_checkpoint("best", expected_sha256=sha256_file(path))
    assert list(tmp_path.glob("*.pt")) == [path]


def test_legacy_safe_resume_state_remains_plain(tmp_path):
    torch = _torch_or_skip()
    manager = CheckpointManager(tmp_path)
    model, optimizer, _ = _make_objects(torch)
    path = manager.save_checkpoint(1, 1, model, optimizer)
    payload = torch.load(path, weights_only=True)
    del payload["data_state_format"]
    payload["sampler_state"] = {"__stage1_numpy_state_v1__": "legacy value"}
    old = tmp_path / "previous.pt"
    torch.save(payload, old)
    assert manager.load_checkpoint(old, model, optimizer)["sampler_state"] == payload["sampler_state"]


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


def test_frozen_manifest_verifies_checkpoint_index_and_shards(tmp_path):
    torch = _torch_or_skip()
    contract = {
        "latent_layout": "B,T,C,Z,Y,X", "normalizer_version": "norm-v1",
        "dataset_manifest_sha256": sha256_json(["trajectory"]),
    }
    manager = CheckpointManager(tmp_path, contract=contract)
    model, optimizer, scheduler = _make_objects(torch)
    checkpoint = manager.save_checkpoint(1, 1, model, optimizer, scheduler, is_best=True)
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    index_path = cache_dir / "index.json"
    names = ["last.bin", "first.bin"]
    for name in names:
        (cache_dir / name).write_bytes(name.encode())
    index = {
        "checkpoint_sha256": sha256_file(checkpoint), "contract_sha256": sha256_json(contract),
        "shards": names, "sample_ids": ["case2", "case1"],
    }
    index_path.write_text(json.dumps(index))
    path = cache_dir / "manifest.json"
    manifest = manager.create_frozen_manifest("best", output_path=path, cache_index="index.json")
    verified = manager.verify_frozen_manifest(path, expected_contract=contract)
    assert verified["checkpoint"] == checkpoint.name
    assert verified["dataset_manifest_sha256"] == contract["dataset_manifest_sha256"]
    assert verified["shard_paths"] == [str(cache_dir / name) for name in names]
    assert verified["cache_index_path"] == str(index_path)
    assert manager.verify_frozen_manifest(manifest, cache_root=cache_dir)["shard_paths"] == verified["shard_paths"]

    for key, value in {
        "latent_source": "z_noisy", "latent_layout": "B,X,Z", "normalizer_version": "other",
        "dataset_manifest_sha256": "b" * 64, "export_mode": "train",
    }.items():
        with pytest.raises(CheckpointContractError, match=key):
            manager.verify_frozen_manifest(dict(manifest, **{key: value}), cache_root=cache_dir)
        with pytest.raises(CheckpointContractError, match=key):
            manager.create_frozen_manifest("best", cache_index=index_path, **{key: value})
    with pytest.raises(CheckpointContractError, match="checkpoint contract mismatch"):
        manager.create_frozen_manifest("best", cache_index=index_path, contract={"latent_layout": "wrong"})
    with pytest.raises(CheckpointContractError, match="v2"):
        manager.verify_frozen_manifest(dict(manifest, manifest_version=1))
    with pytest.raises(CheckpointIntegrityError, match="shard hashes"):
        manager.verify_frozen_manifest(dict(manifest, shards_sha256={}), cache_root=cache_dir)

    index_path.write_text(json.dumps(dict(index, sample_ids=["case1", "case2"])))
    with pytest.raises(CheckpointIntegrityError, match="cache index or shard"):
        manager.verify_frozen_manifest(path)
    index_path.write_text(json.dumps(dict(index, contract_sha256="other")))
    with pytest.raises(CheckpointIntegrityError, match="different checkpoint or contract"):
        manager.create_frozen_manifest("best", cache_index=index_path)
    index_path.write_text(json.dumps(index))
    (cache_dir / names[0]).write_bytes(b"swapped cache shard")
    with pytest.raises(CheckpointIntegrityError, match="cache index or shard"):
        manager.verify_frozen_manifest(path)
    (cache_dir / names[0]).unlink()
    with pytest.raises(CheckpointIntegrityError, match="cannot verify cache index"):
        manager.verify_frozen_manifest(path)


def test_cache_export_needs_dataset_identity(tmp_path):
    torch = _torch_or_skip()
    manager = CheckpointManager(tmp_path, contract={"latent_layout": "B,T,C,Z,Y,X", "normalizer_version": "v1"})
    model, optimizer, _ = _make_objects(torch)
    path = manager.save_checkpoint(1, 1, model, optimizer)
    assert manager.verify_checkpoint(path)["epoch"] == 1
    with pytest.raises(CheckpointContractError, match="dataset_manifest_sha256"):
        manager.create_frozen_manifest(path, cache_index="index.json")
