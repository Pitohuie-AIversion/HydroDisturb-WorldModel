"""G0 preflight for explicitly scheduled paired CFD branches."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from flow3d.contracts import PhysicsContractError

HISTORY_FRAMES = 4


def validated_intervention(row: dict, times_s: np.ndarray) -> dict | None:
    if "intervention" not in row:
        return None
    spec = row["intervention"]
    if not isinstance(spec, dict) or spec.get("exogenous") is not True:
        raise PhysicsContractError("intervention must be declared exogenous")
    for key in ("pair_id", "branch_id", "channel", "unit"):
        if not isinstance(spec.get(key), str) or not spec[key].strip():
            raise PhysicsContractError(f"intervention requires nonempty {key}")
    if not row.get("ancestor_id") or times_s.size <= HISTORY_FRAMES:
        raise PhysicsContractError("intervention requires an explicit ancestor_id and a post-history target")
    known_at = spec.get("known_at_time_s")
    if (
        isinstance(known_at, bool)
        or not isinstance(known_at, (int, float))
        or not np.isfinite(known_at)
        or known_at > float(times_s[HISTORY_FRAMES - 1])
    ):
        raise PhysicsContractError("schedule must be known by the forecast origin")
    for key in ("times_s", "values"):
        vector = np.asarray(spec.get(key))
        if vector.shape != times_s.shape or vector.dtype.kind not in "ifu" or not np.isfinite(vector).all():
            raise PhysicsContractError(f"intervention {key} must contain one finite number per frame")
    if not np.array_equal(np.asarray(spec["times_s"], dtype=times_s.dtype), times_s):
        raise PhysicsContractError("intervention times_s differ from the recorded physical time axis")
    return {
        key: spec[key] for key in ("pair_id", "branch_id", "channel", "unit", "known_at_time_s", "values", "exogenous")
    }


def paired_branch_report(rows: list[dict], eligible: list[dict]) -> dict[str, Any]:
    declared = defaultdict(list)
    admitted = {r["file_name"]: r for r in eligible if "intervention" in r}
    for row in rows:
        spec = row.get("intervention")
        if isinstance(spec, dict) and isinstance(spec.get("pair_id"), str):
            declared[spec["pair_id"]].append(row.get("file_name"))

    available, unmatched = [], []
    for pair_id, files in declared.items():
        branches = [admitted[name] for name in files if name in admitted]
        specs = [r["intervention"] for r in branches]
        identities = [s["branch_id"] for s in specs]
        common = {
            (
                r["ancestor_id"],
                s["channel"],
                s["unit"],
                s["history_and_grid_sha256"],
                tuple(s["values"][:HISTORY_FRAMES]),
            )
            for r, s in zip(branches, specs)
        }
        future = {tuple(s["values"][HISTORY_FRAMES:]) for s in specs}
        if len(branches) != len(files) or len(branches) < 2:
            reason = "fewer than two branches or declared branch failed admission"
        elif len(set(identities)) != len(identities):
            reason = "branch_id must be unique within pair"
        elif len(common) != 1:
            reason = "ancestor, physics, grid, time, history or action units differ"
        elif len(future) < 2:
            reason = "no distinct future schedules"
        else:
            available.append(
                {
                    "pair_id": pair_id,
                    "branch_ids": identities,
                    "file_names": files,
                    "ancestor_id": branches[0]["ancestor_id"],
                    "horizon": branches[0]["num_steps"] - HISTORY_FRAMES,
                }
            )
            continue
        unmatched.append({"pair_id": pair_id, "file_names": files, "reason": reason})
    return {
        "history_frames": HISTORY_FRAMES,
        "available_pair_count": len(available),
        "available_pairs": available,
        "unmatched_pairs": unmatched,
    }
