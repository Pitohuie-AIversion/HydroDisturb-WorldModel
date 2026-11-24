"""G3 point and distribution scores for one decoded forecast time and one ancestor."""

import numpy as np

from flow3d.contracts import PhysicsContractError


def score_forecast_frame_si(
    prediction: np.ndarray, observed: np.ndarray, fluid_mask: np.ndarray
) -> dict[str, float | int]:
    """Score a deterministic (4,Z,Y,X) field or K independent SI forecasts (K,4,Z,Y,X)."""
    obs = np.asarray(observed)
    mask = np.asarray(fluid_mask)
    members = np.asarray(prediction)
    if mask.ndim != 3 or obs.shape != (4, *mask.shape) or not np.isfinite(mask).all():
        raise PhysicsContractError(
            "expected a (4,Z,Y,X) observation and matching finite fluid mask"
        )
    if members.shape == obs.shape:
        members = members[None]
    elif members.ndim != 5 or members.shape[1:] != obs.shape or len(members) < 2:
        raise PhysicsContractError(
            "an ensemble requires at least two full (4,Z,Y,X) forecasts"
        )
    valid = mask > 0.5
    count = int(valid.sum())
    if count == 0:
        raise PhysicsContractError("no valid fluid cells for forecast scoring")
    k = len(members)
    velocity_squared = pressure_squared = 0.0
    velocity_energy = pressure_crps = 0.0
    # Accumulate in FP64 by depth slice. A full-volume K x K distance array is unnecessary.
    for z, plane in enumerate(valid):
        if not plane.any():
            continue
        if k == 1:
            # Deterministic methods dominate rollout evaluation. Their anomaly from
            # truth gives both point and distribution scores without member copies.
            error = np.subtract(members[0, :, z], obs[:, z], dtype=np.float64,
                                where=plane, out=np.zeros((4, *plane.shape), np.float64))
            if not np.isfinite(error).all():
                raise PhysicsContractError("non-finite forecast or observation in the fluid domain")
            speed_squared = np.einsum("cyx,cyx->yx", error[:3], error[:3])
            velocity_squared += speed_squared.sum()
            pressure_squared += np.square(error[3]).sum()
            velocity_energy += np.sqrt(speed_squared).sum()
            pressure_crps += np.abs(error[3]).sum()
            continue
        values = members[:, :, z][:, :, plane].astype(np.float64)
        target = obs[:, z][:, plane].astype(np.float64)
        if not np.isfinite(values).all() or not np.isfinite(target).all():
            raise PhysicsContractError(
                "non-finite forecast or observation in the fluid domain"
            )
        error = values.mean(axis=0) - target
        velocity_squared += np.square(error[:3]).sum()
        pressure_squared += np.square(error[3]).sum()
        member_errors = values - target
        velocity_energy += np.linalg.norm(member_errors[:, :3], axis=1).sum() / k
        pressure_crps += np.abs(member_errors[:, 3]).sum() / k
        # Each unordered pair occurs once; this gives half the mean distance
        # between distinct ordered pairs (the fair finite-ensemble estimator).
        for i in range(k - 1):
            differences = values[i + 1 :] - values[i]
            velocity_energy -= np.linalg.norm(differences[:, :3], axis=1).sum() / (
                k * (k - 1)
            )
            pressure_crps -= np.abs(differences[:, 3]).sum() / (k * (k - 1))
    return {
        "ensemble_size": k,
        "fluid_cell_count": count,
        "velocity_vector_rmse_m_s": float(np.sqrt(velocity_squared / count)),
        "pressure_rmse_pa": float(np.sqrt(pressure_squared / count)),
        "velocity_energy_score_m_s": float(velocity_energy / count),
        "pressure_crps_pa": float(pressure_crps / count),
    }
