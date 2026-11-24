from itertools import product

import numpy as np
import pytest

from flow3d.contracts import PhysicsContractError
from flow3d.metrics.rollout import score_forecast_frame_si


def test_iid_two_branch_forecast_beats_zero_baseline_in_distribution_score():
    # Enumerate the exact joint law of two independent members and the observation.
    shape = (4, 2, 1, 1)
    mask = np.ones(shape[1:])
    template = np.ones(shape)
    template[1:3] = 0
    crps, energy = [], []
    for a, b, y in product((-1, 1), repeat=3):
        observed = y * template
        score = score_forecast_frame_si(
            np.stack([a * template, b * template]), observed, mask
        )
        assert score["pressure_rmse_pa"] == abs((a + b) / 2 - y)
        crps.append(score["pressure_crps_pa"])
        energy.append(score["velocity_energy_score_m_s"])
    assert np.mean(crps) == np.mean(energy) == 0.5
    baseline = score_forecast_frame_si(np.zeros(shape), template, mask)
    assert baseline["pressure_crps_pa"] == baseline["velocity_energy_score_m_s"] == 1


def test_vector_norm_fluid_mask_and_single_member_rejection():
    mask = np.array([[[1, 0]], [[1, 0]]])
    observed = np.zeros((4, *mask.shape))
    pred = np.zeros_like(observed)
    pred[0, 0] = 3
    pred[1, 0] = 4
    pred[:, mask == 0] = np.nan
    score = score_forecast_frame_si(pred, observed, mask)
    assert score["velocity_vector_rmse_m_s"] == pytest.approx(5 / np.sqrt(2))
    assert score["velocity_energy_score_m_s"] == 2.5
    with pytest.raises(PhysicsContractError, match="at least two"):
        score_forecast_frame_si(pred[None], observed, mask)
    pred[3, 0, 0, 0] = np.inf
    with pytest.raises(PhysicsContractError, match="non-finite"):
        score_forecast_frame_si(pred, observed, mask)
