import pytest

torch = pytest.importorskip("torch")

from flow3d.training.forecast import (
    ModeResForecaster,
    delta_z_rollout,
    load_forecaster,
    persistence_rollout,
    rollout_latents,
    save_forecaster,
    train_forecaster,
)


def history(values):
    z = torch.tensor(values, dtype=torch.float32).reshape(-1, 1, 1, 1, 1)
    b = torch.arange(6 * len(z), dtype=torch.float32).reshape(len(z), 6)
    return z, b


def test_fifo_mode_res_uses_all_observed_lags_then_its_own_predictions(tmp_path):
    model = ModeResForecaster(channels=1, history_length=3)
    with torch.no_grad():
        model.bias.fill_(2)
        model.lag_weights[:, 0, 0, 0, 0] = torch.tensor([1, 0.5])
    z, b = history([1, 3, 6])
    pred, stats = rollout_latents(model, z, b, 2, dt_ctu=0.5)
    torch.testing.assert_close(pred.flatten(), torch.tensor([10.5, 16.75]))
    torch.testing.assert_close(stats, b[-1].expand(2, -1))
    path = tmp_path / "forecast.pt"
    save_forecaster(model, path, metadata={"cache_sha256": "fixture"})
    restored, metadata = load_forecaster(path)
    assert metadata == {"cache_sha256": "fixture"}
    torch.testing.assert_close(rollout_latents(restored, z, b, 2, dt_ctu=0.5)[0], pred)
    with pytest.raises(ValueError, match="history length"):
        rollout_latents(restored, z[-2:], b[-2:], 1, dt_ctu=0.5)


def test_delta_and_real_si_persistence_have_distinct_inputs_and_no_future():
    z, b = history([1, 3, 6])
    pred, stats = delta_z_rollout(z, b, 3, dt_ctu=0.2)
    torch.testing.assert_close(pred.flatten(), torch.tensor([9.0, 12.0, 15.0]))
    torch.testing.assert_close(stats, b[-1].expand(3, -1))
    si = (z + 100).expand(-1, 4, -1, -1, -1)
    torch.testing.assert_close(persistence_rollout(si, 3), torch.full_like(si, 106.0))
    with pytest.raises(ValueError, match="at least two"):
        delta_z_rollout(z[-1:], b[-1:], 1, dt_ctu=1)
    with pytest.raises(ValueError, match="horizon"):
        delta_z_rollout(z, b, 0, dt_ctu=1)
    with pytest.raises(ValueError, match="dt_ctu"):
        delta_z_rollout(z, b, 1, dt_ctu=0)


def test_pushforward_training_windows_never_cross_segments_or_splits():
    z, b = history([0, 1, 2, 3, 4, 5])
    rows = [
        {"z": z, "b": b, "dt_ctu": 0.1},
        {"z_clean": z, "b": b, "conditions": {"dt_ctu": 0.2}, "frames": [{"split": "train"}] * len(z)},
    ]
    kwargs = dict(history_length=2, rollout_steps=2, epochs=6, batch_size=2, learning_rate=0.025, seed=3)
    model, info = train_forecaster(rows, **kwargs)
    # Each of two length-six segments has three valid L=2/H=2 windows.
    assert info["train_windows"] == 6 and info["train_segments"] == 2
    assert info["epoch_latent_mse"][-1] < info["epoch_latent_mse"][0]
    assert torch.all(rollout_latents(model, z[:2], b[:2], 2, dt_ctu=0.1)[0] > z[1])
    rows[1]["frames"][-1] = {"split": "test"}
    with pytest.raises(ValueError, match="only train"):
        train_forecaster(rows, **kwargs)
    with pytest.raises(ValueError, match="no train segment"):
        train_forecaster([{"z": z[:3], "b": b[:3], "dt_ctu": 1}] * 2, **kwargs)


def test_strided_validation_has_fixed_paired_origins_and_one_decode_per_method_and_target():
    import numpy as np
    from flow3d.cli.forecast import evaluate

    n = 50
    z, b = history(range(n))
    segment = {
        "trajectory_id": "run", "ancestor_id": "ancestor",
        "step_indices": list(range(100, 100 + n)), "z_clean": z.numpy(), "b": b.numpy(),
        "geometry": np.ones((2, 1, 1, 1), dtype=np.float32), "spacing_star": (1, 1, 1),
        "conditions": {"dt_ctu": 1.0}, "time_nondim_ctu": list(range(n)),
        "time_physical_s": list(range(n)),
    }

    class ValidationCache:
        def sequences(self, split):
            assert split == "val"
            yield segment

        def targets_si(self, seg, directory, indices):
            assert seg is segment
            assert indices == list(range(3, 44))
            return np.broadcast_to(z.numpy()[indices], (len(indices), 4, 1, 1, 1))

    class Decoder:
        calls = 0

        def decode(self, frame, geometry, **kwargs):
            self.calls += 1
            return {"phys_fields": frame.expand(-1, 4, -1, -1, -1)}

    decoder = Decoder()
    model = ModeResForecaster(1, history_length=4)
    rows = evaluate(ValidationCache(), None, model, decoder, 4, 10, "cpu", None,
                    split="val", origin_stride=10)
    assert decoder.calls == 80
    assert [(row["origin_step"], row["target_step"]) for row in rows] == [
        (origin, origin + lead) for origin in (103, 113, 123, 133) for lead in range(1, 11)
    ]
    for row in rows:
        scores = row["scores"]
        assert scores["delta_z"]["velocity_vector_rmse_m_s"] == 0
        assert scores["mode_res"]["pressure_rmse_pa"] == scores["persistence"]["pressure_rmse_pa"] == row["lead_steps"]
