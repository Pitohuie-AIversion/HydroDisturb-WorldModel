from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

import numpy as np

from flow3d.contracts import PhysicsContractError, SampleProvenance


@dataclass(frozen=True)
class SplitAssignment:
    train_indices: tuple[int, ...]
    val_indices: tuple[int, ...]
    test_indices: tuple[int, ...]
    group_by: str
    train_groups: tuple[str, ...]
    val_groups: tuple[str, ...]
    test_groups: tuple[str, ...]
    leakage_verified: bool = False
    # Starts of complete L+H windows, indexed in the supplied frame sequence.
    window_starts: tuple[tuple[int, ...], ...] | None = None


def group_split_by_provenance(
    provenances: Sequence[SampleProvenance],
    group_by: str = "ancestor_id",
    ratios: tuple[float, float, float] = (0.6, 0.2, 0.2),
) -> SplitAssignment:
    """Group-disjoint train/val/test split prioritizing `ancestor_id` then `trajectory_id`.

    Never performs random window splitting across groups. Fails closed if provenance is empty
    or if fewer than 3 distinct groups exist when non-empty val and test splits are requested.
    """
    if not provenances:
        raise PhysicsContractError("provenances list cannot be empty")
    if group_by not in ("ancestor_id", "trajectory_id"):
        raise PhysicsContractError(f"group_by must be 'ancestor_id' or 'trajectory_id', got '{group_by}'")
    if len(ratios) != 3 or any(r < 0 for r in ratios) or abs(sum(ratios) - 1.0) > 1e-5:
        raise PhysicsContractError(f"ratios must sum to 1.0, got {ratios}")

    # Preserve deterministic group order by first appearance
    groups_in_order: list[str] = []
    group_to_indices: dict[str, list[int]] = {}
    for idx, prov in enumerate(provenances):
        if not isinstance(prov, SampleProvenance):
            raise PhysicsContractError("all elements must be SampleProvenance instances; missing provenance rejected")
        if not prov.ground_truth_verified:
            raise PhysicsContractError(
                f"sample {idx} (case={prov.case_id}) has ground_truth_verified=False and must be quarantined before splitting"
            )
        g_key = getattr(prov, group_by)
        if not g_key or g_key.lower() in ("unknown", "none", "null"):
            raise PhysicsContractError(f"sample {idx} has missing/unknown {group_by}; refusing to claim zero leakage")
        if g_key not in group_to_indices:
            groups_in_order.append(g_key)
            group_to_indices[g_key] = []
        group_to_indices[g_key].append(idx)

    n_groups = len(groups_in_order)
    need_val = ratios[1] > 0
    need_test = ratios[2] > 0
    min_required = 1 + int(need_val) + int(need_test)
    if n_groups < min_required:
        raise PhysicsContractError(
            f"group_split_by_provenance requires at least {min_required} distinct '{group_by}' groups "
            f"to populate disjoint splits, but found only {n_groups}: {groups_in_order}. "
            f"Use temporal_block_split_with_purge for single-trajectory datasets."
        )

    n_test = max(1, int(round(n_groups * ratios[2]))) if need_test else 0
    n_val = max(1, int(round(n_groups * ratios[1]))) if need_val else 0
    while n_groups - n_val - n_test < 1:
        if n_val >= n_test and n_val > 1:
            n_val -= 1
        elif n_test > 1:
            n_test -= 1
        else:
            break
    n_train = n_groups - n_val - n_test

    train_g = tuple(groups_in_order[:n_train])
    val_g = tuple(groups_in_order[n_train : n_train + n_val])
    test_g = tuple(groups_in_order[n_train + n_val :])

    # Verify zero overlap across ancestor_id AND trajectory_id
    if set(train_g) & set(val_g) or set(train_g) & set(test_g) or set(val_g) & set(test_g):
        raise PhysicsContractError("internal error: group overlap detected")

    train_idx = tuple(i for g in train_g for i in group_to_indices[g])
    val_idx = tuple(i for g in val_g for i in group_to_indices[g])
    test_idx = tuple(i for g in test_g for i in group_to_indices[g])

    # Enforce BOTH ancestor_id AND trajectory_id disjointness across splits
    for attr in ("ancestor_id", "trajectory_id"):
        tr_set = {getattr(provenances[i], attr) for i in train_idx}
        va_set = {getattr(provenances[i], attr) for i in val_idx}
        te_set = {getattr(provenances[i], attr) for i in test_idx}
        overlap = (tr_set & va_set) | (tr_set & te_set) | (va_set & te_set)
        if overlap:
            raise PhysicsContractError(
                f"cross-split '{attr}' leakage detected when splitting by '{group_by}': {sorted(overlap)}"
            )

    return SplitAssignment(
        train_indices=train_idx,
        val_indices=val_idx,
        test_indices=test_idx,
        group_by=group_by,
        train_groups=train_g,
        val_groups=val_g,
        test_groups=test_g,
        leakage_verified=True,
    )


def temporal_block_split_with_purge(
    provenances: Sequence[SampleProvenance],
    *,
    history_frames: int,
    forecast_frames: int,
    purge_gap_frames: int | None = None,
    ratios: tuple[float, float, float] = (0.6, 0.2, 0.2),
) -> SplitAssignment:
    """Split one verified constant-step segment; frame indices and starts refer to its input order."""
    if (
        type(history_frames) is not int
        or history_frames < 1
        or type(forecast_frames) is not int
        or forecast_frames < 0
    ):
        raise PhysicsContractError("history_frames must be positive and forecast_frames nonnegative integers")
    if forecast_frames == 0 and history_frames != 1:
        raise PhysicsContractError("H=0 is reserved for single-frame Stage 1 reconstruction (L=1)")
    width = history_frames + forecast_frames
    gap = width if purge_gap_frames is None else purge_gap_frames
    if type(gap) is not int or gap < width:
        raise PhysicsContractError("purge_gap_frames must be >= history_frames + forecast_frames")
    if (
        len(ratios) != 3
        or not np.isfinite(ratios).all()
        or min(ratios) < 0
        or ratios[0] <= 0
        or not np.isclose(sum(ratios), 1)
    ):
        raise PhysicsContractError(
            "split ratios must be finite, nonnegative, sum to one and include training"
        )
    if not provenances or any(
        not isinstance(p, SampleProvenance) or not p.ground_truth_verified for p in provenances
    ):
        raise PhysicsContractError("verified per-frame provenance is required for temporal splitting")
    for key in ("case_id", "geometry_id", "ancestor_id", "trajectory_id", "segment_id"):
        labels = {getattr(p, key) for p in provenances}
        if len(labels) != 1 or next(iter(labels)).strip().casefold() in {"", "unknown", "none", "null"}:
            raise PhysicsContractError(f"temporal split requires one known {key}")
    active = [i for i, r in enumerate(ratios) if r > 0]
    usable = len(provenances) - gap * (len(active) - 1)
    if usable < width * len(active):
        raise PhysicsContractError(
            "insufficient frames for at least one complete window in each requested split"
        )
    if any(
        not isinstance(p.time_idx, (int, np.integer)) or isinstance(p.time_idx, bool) for p in provenances
    ):
        raise PhysicsContractError("time_idx must be an integer solver step")
    steps = np.diff([p.time_idx for p in provenances])
    if steps.size and ((steps <= 0).any() or not np.all(steps == steps[0])):
        raise PhysicsContractError("time_idx must increase at a constant stride")
    for key in ("time_phys_s", "time_ctu"):
        axis = np.array([getattr(p, key) for p in provenances], dtype=np.float64)
        deltas = np.diff(axis)
        tol = 2 * np.finfo(np.float32).eps * np.max(np.abs(axis))
        if (
            not np.isfinite(axis).all()
            or (deltas <= 0).any()
            or (deltas.size and not np.allclose(deltas, deltas.mean(), atol=tol, rtol=1e-5))
        ):
            raise PhysicsContractError(f"{key} must be increasing with constant spacing")
    # Reserve one full window for every requested block, then approach the requested frame ratios.
    sizes = [width if i in active else 0 for i in range(3)]
    for _ in range(usable - sum(sizes)):
        sizes[max(active, key=lambda i: usable * ratios[i] - sizes[i])] += 1
    blocks, starts, groups = [()] * 3, [()] * 3, [()] * 3
    offset = 0
    for i in active:
        end = offset + sizes[i]
        blocks[i] = tuple(range(offset, end))
        starts[i] = tuple(range(offset, end - width + 1))
        groups[i] = (f"{provenances[0].trajectory_id}:{provenances[0].segment_id}[{offset}:{end}]",)
        offset = end + gap
    return SplitAssignment(
        *blocks, f"temporal_block_purge_{gap}", *groups, leakage_verified=True, window_starts=tuple(starts)
    )
