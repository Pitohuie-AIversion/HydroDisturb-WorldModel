from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

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
    leakage_verified: bool = True


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
    num_frames: int,
    purge_gap_frames: int = 5,
    ratios: tuple[float, float, float] = (0.6, 0.2, 0.2),
    trajectory_id: str = "traj_0",
) -> SplitAssignment:
    """Contiguous temporal block split with mandatory purge gaps between train, val, and test."""
    if num_frames <= 0:
        raise PhysicsContractError("num_frames must be positive")
    if purge_gap_frames < 1:
        raise PhysicsContractError("purge_gap_frames must be >= 1 to prevent window boundary leakage")
    usable = num_frames - 2 * purge_gap_frames
    if usable < 3:
        raise PhysicsContractError(
            f"num_frames={num_frames} is too short for 3 contiguous blocks with purge_gap_frames={purge_gap_frames}"
        )
    n_val = max(1, int(round(usable * ratios[1])))
    n_test = max(1, int(round(usable * ratios[2])))
    n_train = usable - n_val - n_test
    if n_train < 1:
        raise PhysicsContractError("insufficient frames for train block after purge gaps")

    train_end = n_train
    val_start = train_end + purge_gap_frames
    val_end = val_start + n_val
    test_start = val_end + purge_gap_frames

    train_idx = tuple(range(0, train_end))
    val_idx = tuple(range(val_start, val_end))
    test_idx = tuple(range(test_start, num_frames))
    return SplitAssignment(
        train_indices=train_idx,
        val_indices=val_idx,
        test_indices=test_idx,
        group_by=f"temporal_block_purge_{purge_gap_frames}",
        train_groups=(f"{trajectory_id}:t[0..{train_end - 1}]",),
        val_groups=(f"{trajectory_id}:t[{val_start}..{val_end - 1}]",),
        test_groups=(f"{trajectory_id}:t[{test_start}..{num_frames - 1}]",),
        leakage_verified=True,
    )
