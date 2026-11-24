"""Read back TensorBoard events written by Stage 1 training and checkpoint resumes."""

import importlib.util
import json
from pathlib import Path
import sys

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("tensorboard")
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from flow3d.training.tensorboard import Stage1TensorBoard
from flow3d.training.trainer import Flow3DStage1Trainer


class ScalarField(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.value = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, fields, geometry, **kwargs):
        prediction = self.value.expand_as(fields)
        return {"norm_fields": prediction, "phys_fields": prediction}


def sample(value):
    fields = torch.full((4, 4, 4, 4), float(value))
    geometry = torch.ones(2, 4, 4, 4)
    geometry[0, 1, 1, 1] = 0
    return {
        "norm_field": fields, "phys_field": fields * 2, "geometry": geometry,
        "b": torch.ones(6), "spacing_m": (1., 1., 1.), "spacing_star": (1., 1., 1.),
        "case_id": "fixture", "reynolds_number": 100, "u0_m_s": 1.,
        "step_index": value, "time_physical_s": value, "time_nondim_ctu": value,
    }


def events(log_dir):
    return EventAccumulator(str(log_dir), size_guidance={"scalars": 0}).Reload()


def assert_series(reader, tag, steps, values):
    scalars = reader.Scalars(tag)
    assert [scalar.step for scalar in scalars] == steps, tag
    assert [scalar.value for scalar in scalars] == pytest.approx(values), tag


def test_step_curves_skip_overflowed_optimizer_attempt(tmp_path):
    from flow3d.data.dataset import collate_flow3d_stage1

    model = ScalarField()
    optimizer = torch.optim.SGD(model.parameters(), lr=1 / 16)
    trainer = Flow3DStage1Trainer(
        model, config={"runtime": {"cpu_threads": 1}}, device="cpu", num_gpus=0,
        optimizer=optimizer, clip_grad_norm=100., amp=False, save_dir=tmp_path / "checkpoints",
    )
    trainer.scaler = torch.amp.GradScaler("cpu", init_scale=2.0)
    # A real scaler skips the middle optimizer update and halves its scale.
    multipliers = iter([1., float("inf"), 1.])
    model.value.register_hook(lambda grad: grad * next(multipliers))
    batches = [collate_flow3d_stage1([sample(i)]) for i in (1, 2, 3)]
    with Stage1TensorBoard(tmp_path / "events", "head_a") as tb:
        trainer.fit(batches, epochs=1, log_callback=tb.log_epoch, step_callback=tb.log_step)

    assert trainer.global_step == 2
    assert trainer.scaler.get_scale() == 1.
    reader = events(tb.log_dir)
    # The first update moves the prediction from 0 to 0.5. The third batch
    # still sees 0.5, giving a four-channel loss of 4 * (3 - 0.5)**2.
    assert_series(reader, "step/loss", [1, 2], [4., 25.])
    assert_series(reader, "step/grad_norm", [1, 2], [8., 20.])


def test_cli_resumes_checkpoint_history_once_in_separate_run(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[2] / "scripts/train_stage1_flow3d.py"
    spec = importlib.util.spec_from_file_location("tensorboard_stage1_cli", path)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    # Replace only the heavyweight CFD inputs and model, keeping the real loader,
    # trainer, checkpoint manager, CLI callbacks and TensorBoard event files.
    splits = ([sample(1), sample(2)], [sample(3)], [sample(4)])
    monkeypatch.setattr(cli.Flow3DH5Dataset, "create_splits", lambda **kwargs: splits)
    monkeypatch.setattr(cli, "stage1_split_identity", lambda *args: {"dataset": "fixture"})
    monkeypatch.setattr(cli, "build_stage1_contract", lambda *args: {"dataset": "fixture"})
    monkeypatch.setattr(cli.Flow3DStage1Autoencoder, "from_config", lambda cfg: ScalarField())
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"runtime": {"cpu_threads": 1, "num_gpus": 0}, "head": {"type": "head_a"}}))
    save_dir = tmp_path / "checkpoints"
    event_dir = tmp_path / "events"
    args = [
        str(path), "--config", str(config), "--save-dir", str(save_dir),
        "--no-amp", "--batch-size", "1", "--num-workers", "0", "--epochs", "3",
        "--eval-interval", "2", "--save-interval", "1", "--tensorboard-dir", str(event_dir),
    ]

    def run(*extra):
        monkeypatch.setattr(sys, "argv", [*args, *extra])
        cli.main()
        return json.loads((save_dir / "training_summary.json").read_text())

    original = run("--tensorboard")
    assert original["active_gpus"] == 0 and original["active_gpu_hardware"] == []
    assert original["detected_cpu_threads"] >= original["cpu_threads"] == 1
    assert original["available_ram_gib_at_start"] >= 0
    checkpoint, = save_dir.glob("*_epoch_000002_step_*.pt")
    resumed = run("--tensorboard", "--resume", str(checkpoint))
    original_dir, resumed_dir = (Path(record["tensorboard_log_dir"]) for record in (original, resumed))
    assert {original_dir, resumed_dir} == set(event_dir.iterdir())
    assert original_dir != resumed_dir
    assert resumed["history"][:2] == original["history"][:2]
    runs = [(original, original_dir, [1, 2, 3, 4, 5, 6]), (resumed, resumed_dir, [5, 6])]
    for record, log_dir, steps in runs:
        reader = events(log_dir)
        assert [row["epoch"] for row in record["history"]] == [1, 2, 3]
        assert [row["epoch"] for row in record["history"] if "val_loss" in row] == [2, 3]
        for tag, key in [
            ("train/loss", "train_loss"), ("validation/loss", "val_loss"),
            ("validation/pressure_rmse_pa", "rmse_p_pa"), ("optimization/learning_rate", "learning_rate"),
        ]:
            rows = [row for row in record["history"] if key in row]
            assert_series(reader, tag, [row["epoch"] for row in rows], [row[key] for row in rows])
        for tag in ("step/loss", "step/grad_norm"):
            assert [scalar.step for scalar in reader.Scalars(tag)] == steps

    disabled = run("--no-tensorboard")
    assert disabled["tensorboard_log_dir"] is None
    assert set(event_dir.iterdir()) == {original_dir, resumed_dir}
