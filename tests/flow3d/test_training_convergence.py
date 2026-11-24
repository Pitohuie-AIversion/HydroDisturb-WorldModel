"""Regression tests for validation-driven training, selection weights and restarts."""

from dataclasses import replace
from math import sqrt

import pytest

from flow3d.training.trainer import Flow3DStage1Trainer

torch = pytest.importorskip("torch")


class ConstantField(torch.nn.Module):
    """One fitted value in all four channels, with the same solid mask as a head."""

    def __init__(self):
        super().__init__()
        self.value = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, fields, geometry, b=None, spacing_zyx=None):
        prediction = self.value * torch.ones_like(fields) * geometry[:, :1]
        return {"norm_fields": prediction, "phys_fields": prediction}


def make_trainer(save_dir, **training):
    cfg = {
        "runtime": {"num_gpus": 0, "cpu_threads": torch.get_num_threads()},
        "training": {
            "scheduler": "warmup_plateau",
            "learning_rate": 2e-4,
            "warmup_start_lr": 1e-5,
            "warmup_epochs": 2,
            "plateau_factor": 0.5,
            "plateau_patience": 2,
            "plateau_threshold": 0.001,
            "min_lr": 5e-6,
            "min_epochs": 10,
            "early_stop_patience": 6,
            "early_stop_threshold": 0.005,
            "ema": False,
            "ema_decay": 0.995,
            **training,
        },
    }
    return Flow3DStage1Trainer(
        ConstantField(), config=cfg, device="cpu", num_gpus=0, amp=False, save_dir=save_dir,
    )


def make_batch(values):
    fields = torch.tensor(values, dtype=torch.float32).view(-1, 1, 1, 1, 1)
    fields = fields.expand(-1, 4, 8, 8, 8).clone()
    geometry = torch.ones(len(values), 2, 8, 8, 8)
    geometry[:, 0, 3, 3, 3] = 0
    return {"norm_field": fields, "phys_field": fields.clone(), "geometry": geometry}


def test_validation_uses_all_fluid_cells_and_physical_first_layer(tmp_path):
    trainer = make_trainer(tmp_path)
    batch = make_batch([1, 2, 6])
    # This frame has fewer fluid cells, and the last minibatch has just one frame.
    batch["geometry"][2, 0, 0] = 0
    counts = [511, 511, 447]
    mse = sum(n * v**2 for n, v in zip(counts, [1, 2, 6])) / sum(counts)
    # The middle frame has a 3-4-5 velocity error at one of its six shell cells.
    # The extra solid plane of the third frame adds another 64 shell cells.
    batch["phys_field"].zero_()
    batch["phys_field"][1, :2, 3, 3, 4] = torch.tensor([3.0, 4.0])
    batch["phys_field"][1, 0, 6, 6, 6] = 12.0  # away from the wall
    batch["phys_field"][1, 3, 3, 3, 4] = 2.0
    uneven = [
        {k: value[:2] for k, value in batch.items()},
        {k: value[2:] for k, value in batch.items()},
    ]

    for batches in ([batch], uneven):
        metrics = trainer.evaluate(batches)
        assert metrics.val_loss == pytest.approx(4 * mse)
        assert metrics.rmse_uvw_ms == pytest.approx(sqrt((25 + 144) / sum(counts)))
        assert metrics.rmse_p_pa == pytest.approx(sqrt(4 / sum(counts)))
        assert metrics.near_wall_vel_rmse_ms == pytest.approx(sqrt(25 / (18 + 64)))
        assert metrics.div_rms_phys_s_inv == 0


def test_warmup_updates_and_plateau_early_stop(tmp_path, monkeypatch):
    trainer = make_trainer(tmp_path, min_epochs=12)
    batch = make_batch([1])
    fixed = replace(trainer.evaluate([batch]), val_loss=1.0)
    monkeypatch.setattr(trainer, "evaluate", lambda loader: fixed)
    used_lrs = []
    trainer.optimizer.register_step_pre_hook(
        lambda optimizer, args, kwargs: used_lrs.append(optimizer.param_groups[0]["lr"])
    )
    history = trainer.fit([batch, batch], [batch], epochs=30, eval_interval=1, save_interval=30)

    # Two epochs of two optimizer updates, including both warmup endpoints.
    assert used_lrs[:4] == pytest.approx([1e-5 + i * (2e-4 - 1e-5) / 3 for i in range(4)])
    assert used_lrs[4] == pytest.approx(2e-4)
    cuts = [row for row in history if row["next_learning_rate"] < row["learning_rate"]]
    assert cuts[0]["next_learning_rate"] == pytest.approx(1e-4)
    # Patience of two allows two bad validations; the third triggers a reduction.
    assert cuts[1]["epoch"] - cuts[0]["epoch"] == 3
    assert 12 <= trainer.current_epoch < 30
    assert history[-1]["next_learning_rate"] == history[-1]["learning_rate"]
    assert trainer.stop_reason == "CONVERGED"
    assert trainer.global_step == 2 * trainer.current_epoch


def test_budget_exhaustion_with_improving_validation(tmp_path, monkeypatch):
    trainer = make_trainer(tmp_path)
    batch = make_batch([1])
    baseline = trainer.evaluate([batch])
    # Each epoch improves by just 0.2%, but the progress accumulated during the
    # patience window exceeds 0.5%. A per-epoch comparison would stop this run.
    monkeypatch.setattr(
        trainer, "evaluate", lambda loader: replace(baseline, val_loss=0.998**trainer.current_epoch)
    )
    history = trainer.fit([batch], [batch], epochs=15, eval_interval=1, save_interval=15)
    assert len(history) == 15
    assert trainer.stop_reason == "NOT_CONVERGED"


def test_best_points_to_weights_used_for_ema_validation(tmp_path):
    trainer = make_trainer(tmp_path, ema=True)
    # Training drives the parameter away from the held-out target. The best must
    # stay at epoch 1 even as the training loss keeps dropping.
    train, val = [make_batch([2])], [make_batch([0])]
    history = trainer.fit(train, val, epochs=3, eval_interval=1, save_interval=3)
    evaluated = ConstantField()
    best = trainer.checkpoint_manager.load_checkpoint(
        "best", evaluated, use_ema=True, restore_rng=False
    )
    assert best["epoch"] == 1
    assert history[-1]["train_loss"] < history[0]["train_loss"]
    assert history[-1]["val_loss"] > history[0]["val_loss"]
    assert best["metrics"]["val_loss"] == pytest.approx(4 * evaluated.value.item() ** 2, abs=1e-15)
    raw = best["model_state_dict"]["value"].item()
    assert raw > 0
    assert evaluated.value.item() == pytest.approx(raw / 11, rel=1e-6, abs=0)


@pytest.mark.parametrize("interrupt_epoch", [1, 4], ids=["during-warmup", "during-plateau"])
def test_resume_reproduces_shuffled_training_and_best(tmp_path, interrupt_epoch):
    samples = [make_batch([v]) for v in (2.0, -1.0, 0.25, -3.0)]

    def make_loader(seed):
        sampler = torch.utils.data.RandomSampler(
            samples, generator=torch.Generator().manual_seed(seed)
        )
        return torch.utils.data.DataLoader(
            samples, batch_size=None, sampler=sampler, generator=torch.Generator().manual_seed(seed + 1)
        )

    val = [make_batch([0.25])]
    fit_options = {"epochs": 7, "eval_interval": 1, "save_interval": 1}
    full = make_trainer(tmp_path / "full", ema=True)
    expected = full.fit(make_loader(91), val, **fit_options)
    partial = make_trainer(tmp_path / "split", ema=True)

    class Interrupted(Exception):
        pass

    def interrupt(record):
        if record["epoch"] == interrupt_epoch:
            raise Interrupted

    with pytest.raises(Interrupted):
        partial.fit(make_loader(91), val, log_callback=interrupt, **fit_options)
    saved = next((tmp_path / "split").glob(f"*_epoch_{interrupt_epoch:06d}_step_*.pt"))
    resumed = make_trainer(tmp_path / "split", ema=True)
    loader = make_loader(17)
    resumed.resume(saved, loader)
    actual = resumed.fit(loader, val, **fit_options)
    for key in ("epoch", "global_step", "learning_rate", "next_learning_rate", "train_loss", "val_loss"):
        assert [row[key] for row in actual] == [row[key] for row in expected]
    torch.testing.assert_close(full.model.value, resumed.model.value, atol=0, rtol=0)
    torch.testing.assert_close(full.ema_model.value, resumed.ema_model.value, atol=0, rtol=0)
    assert resumed.scheduler.state_dict() == full.scheduler.state_dict()
    assert resumed.best_val_loss == full.best_val_loss
