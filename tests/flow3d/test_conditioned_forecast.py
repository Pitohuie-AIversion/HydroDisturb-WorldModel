import numpy as np
import pytest

torch = pytest.importorskip("torch")

from flow3d.cli.forecast import forecast_latents
from flow3d.training.dynamics import ConditionedForecaster, train_conditioned
from flow3d.training.forecast import load_forecaster, save_forecaster


@pytest.mark.parametrize(("conditions", "missing"), [
    ({"dt_phys_s": 0.1}, "dt_ctu"),
    ({"dt_phys_s": 0.1, "dt_ctu": 0.7, "known_mask": {"dt_ctu": False}}, "dt_ctu"),
    ({"dt_phys_s": 0.1, "dt_ctu": 0.7, "valid_mask": {"dt_phys_s": False}}, "dt_phys_s"),
])
def test_conditioned_rollout_requires_two_known_cadences(conditions, missing):
    model = ConditionedForecaster(8, width=16)
    with pytest.raises(ValueError, match=missing):
        model(torch.zeros(1, 4, 8, 1, 1, 1), torch.ones(1, 4, 6), [conditions], [2.], 1)


def test_conditioned_pushforward_roundtrip_and_future_isolation(tmp_path):
    rng = np.random.default_rng(42)
    row = {
        "split": "train", "frames": [{"split": "train"}] * 8,
        "b": rng.random((8, 6), dtype=np.float32) + 0.25,
        "z_clean": rng.normal(size=(8, 8, 2, 2, 2)).astype(np.float32),
        "conditions": {"dt_ctu": 0.1, "dt_phys_s": 0.01},
        "time_physical_s": np.arange(8) * 0.01,
    }
    settings = dict(history_length=4, rollout_steps=2, epochs=1, seed=42, batch_size=2,
                    learning_rate=2e-4, device="cpu", width=16)
    with pytest.raises(ValueError, match="train segments"):
        train_conditioned([dict(row, split="val")], **settings)
    model, info = train_conditioned([row], **settings)
    assert info["train_windows"] == 3
    before = forecast_latents(model, row, 0, 4, 5, "cpu")
    path = tmp_path/"forecaster.pt"
    save_forecaster(model, path, metadata={"cache_manifest_sha256": "fixture", "history_length": 4})
    loaded, metadata = load_forecaster(path)
    assert metadata["history_length"] == 4
    # The decoder must be able to advance beyond recorded data from the same four observations.
    truncated = {**row, "z_clean": row["z_clean"][:4], "b": row["b"][:4],
                 "time_physical_s": row["time_physical_s"][:4]}
    after = forecast_latents(loaded, truncated, 0, 4, 5, "cpu")
    for original, restored in zip(before, after, strict=True):
        torch.testing.assert_close(restored, original)
        assert torch.isfinite(restored).all()
    assert (after[1][:, 4:] > 0).all()
