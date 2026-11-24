"""Unit tests for Flow3DH5Dataset, collate function, and Flow3DStage1Trainer."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from flow3d.data.dataset import Flow3DH5Dataset, collate_flow3d_stage1
from flow3d.data.fixtures import make_analytical_divergence_free_field_3d, make_cylinder_geometry_3d
from flow3d.data.schema import load_yaml_or_json_config
from flow3d.models import Flow3DStage1Autoencoder, torch_available
from flow3d.training.trainer import Flow3DStage1Trainer
from stage1.io.checkpoint import CheckpointManager

if torch_available():
    import torch


@pytest.mark.skipif(not torch_available(), reason="PyTorch is required")
def test_dataset_loading_and_collate():
    dataset_dir = Path("/root/autodl-tmp/datasets/world_model")
    if not dataset_dir.is_dir():
        pytest.skip("Dataset directory /root/autodl-tmp/datasets/world_model not present")

    # Load with 1 explicit eligible file for fast unit test
    ds_train = Flow3DH5Dataset(
        dataset_dir=dataset_dir,
        eligible_files=["re200k_wm_50steps.h5"],
        split="train",
        purge_gap_frames=4,
    )
    assert len(ds_train) > 0

    item = ds_train[0]
    assert "norm_field" in item
    assert "phys_field" in item
    assert "geometry" in item
    assert "b" in item
    assert item["norm_field"].shape == (4, 64, 96, 192)
    assert item["geometry"].shape == (2, 64, 96, 192)
    assert item["b"].shape == (6,)

    # Test collate
    batch = collate_flow3d_stage1([ds_train[0], ds_train[1]])
    assert batch["norm_field"].shape == (2, 4, 64, 96, 192)
    assert batch["phys_field"].shape == (2, 4, 64, 96, 192)
    assert batch["geometry"].shape == (2, 2, 64, 96, 192)
    assert batch["b"].shape == (2, 6)
    assert len(batch["meta"]) == 2


@pytest.mark.skipif(not torch_available(), reason="PyTorch is required")
@pytest.mark.parametrize("frame_ids", [(7,), (7, 2)])
def test_collate_can_omit_only_the_physical_target(frame_ids):
    cells = torch.arange(24, dtype=torch.float32).view(4, 1, 2, 3)
    samples = [
        {
            "norm_field": cells + frame,
            "phys_field": cells * 3 - frame,
            "geometry": cells[:2] * frame,
            "b": torch.arange(6, dtype=torch.float32) + frame,
            "spacing_m": (0.01, 0.02, 0.03),
            "spacing_star": (0.1, 0.2, 0.3),
            "case_id": f"case-{frame}",
            "reynolds_number": frame * 10000,
            "u0_m_s": 0.2 * frame,
            "time_physical_s": frame * 0.1,
            "time_nondim_ctu": frame * 0.25,
            "step_index": frame,
        }
        for frame in frame_ids
    ]
    full = collate_flow3d_stage1(samples)
    assert torch.equal(full.pop("phys_field"), torch.stack([sample["phys_field"] for sample in samples]))
    training = collate_flow3d_stage1(samples, include_phys_field=False)
    assert training.keys() == full.keys()
    for key, value in full.items():
        if torch.is_tensor(value):
            assert torch.equal(training[key], value)
        else:
            assert training[key] == value


@pytest.mark.skipif(not torch_available(), reason="PyTorch is required")
def test_trainer_train_step_and_eval():
    cfg = load_yaml_or_json_config("configs/flow3d/stage1.yaml")
    # The trainer configures Torch globally; keep this test's caller-selected CPU budget.
    cfg["runtime"]["cpu_threads"] = torch.get_num_threads()
    # Mini shape for fast unit test (divisible by downsample_factor=8)
    nz, ny, nx = 8, 16, 24
    dz, dy, dx = 0.0625, 0.0625, 0.0625
    geom_np, _ = make_cylinder_geometry_3d((nz, ny, nx), (dz * 0.3, dy * 0.3, dx * 0.3))
    flow_np, _ = make_analytical_divergence_free_field_3d((nz, ny, nx), (dz, dy, dx))

    geom_t = torch.from_numpy(geom_np).unsqueeze(0).float()  # [1, 2, 8, 12, 24]
    field_t = torch.from_numpy(flow_np).unsqueeze(0).float()  # [1, 4, 8, 12, 24]
    b_t = torch.tensor([[0.0, 0.0, 0.0, 0.0, 1.0, 1.0]], dtype=torch.float32)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Flow3DStage1Autoencoder.from_config(cfg).to(device)

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_mgr = CheckpointManager(save_dir=tmpdir, model_name="test_dcae")
        trainer = Flow3DStage1Trainer(
            model=model,
            config=cfg,
            checkpoint_manager=ckpt_mgr,
            device=device,
            num_gpus=1 if device.type == "cuda" else 0,
            amp=False,  # Mini test in float32
            save_dir=tmpdir,
        )

        batch = {
            "norm_field": field_t,
            "geometry": geom_t,
            "b": b_t,
            "spacing_star": (dz, dy, dx),
            "spacing_m": (dz * 0.3, dy * 0.3, dx * 0.3),
        }

        # Train step
        metrics = trainer.train_step(batch)
        assert isinstance(metrics.total_loss, float)
        assert metrics.total_loss > 0.0
        assert metrics.grad_norm >= 0.0
        assert trainer.global_step == 1

        # Evaluate step
        eval_metrics = trainer.evaluate([{**batch, "phys_field": field_t}])
        assert isinstance(eval_metrics.val_loss, float)
        assert eval_metrics.rmse_uvw_ms >= 0.0
        assert eval_metrics.solid_interior_max_abs == 0.0

        # Checkpoint save
        saved_path = ckpt_mgr.save_checkpoint(
            epoch=1,
            global_step=1,
            model=model,
            optimizer=trainer.optimizer,
            is_best=True,
        )
        assert saved_path.is_file()
        assert ckpt_mgr.best_pointer_path.is_file()
