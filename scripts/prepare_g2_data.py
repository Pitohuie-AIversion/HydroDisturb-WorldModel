"""Audit the on-disk CFD set and freeze a downstream, ancestor-disjoint G2 split."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from tempfile import NamedTemporaryFile

import h5py

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from flow3d.data.inspect import inspect_single_source
from stage1.io.checkpoint import sha256_file, sha256_json


SIDECARS = ("manifest.json", "multi_re_tier1_certification_summary.json", "extraction_and_rampool_assessment.json")
HISTORY, HORIZON = 4, 10


def publish(path, data):
    text = json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", dir=path.parent, delete=False) as temp:
        temp.write(text)
    try:
        os.link(temp.name, path)
    except FileExistsError:
        if path.read_text() != text:
            raise ValueError(f"refusing to replace different artifact: {path}")
    finally:
        Path(temp.name).unlink()


def snapshot(directory, names):
    result = {}
    for name in names:
        path = directory / name
        if path.is_file():
            stat = path.stat()
            result[name] = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    return result


def prepare(dataset_dir, stage1_summary, output_dir, val_names, test_names, test_fallback):
    dataset_dir, output_dir = dataset_dir.resolve(), output_dir.resolve()
    stage1_identity = json.loads(stage1_summary.read_text())["split_identity"]
    admission_sha256 = {name: sha256_file(dataset_dir / name) for name in SIDECARS}
    inventory = json.loads((dataset_dir / "manifest.json").read_text())
    candidates = [t["file_name"] for t in inventory["trajectories"]]
    before = snapshot(dataset_dir, candidates)
    audit = inspect_single_source(dataset_dir)
    publish(output_dir / "g0_audit.json", audit)
    eligible = {t["file_name"]: t for t in audit["eligible_trajectories"]}
    legacy_files = stage1_identity["files"]
    if not legacy_files or not set(legacy_files).issubset(eligible):
        raise ValueError("every file from the Stage 1 split must be identified in this audit")
    legacy_ancestors = {eligible[name]["ancestor_id"] for name in legacy_files}
    for name, expected in legacy_files.items():
        if sha256_file(dataset_dir / name) != expected:
            raise ValueError(f"Stage 1 source has changed: {name}")

    held = {"val": set(val_names), "test": set(test_names)}
    novel = {name: row for name, row in eligible.items() if row["ancestor_id"] not in legacy_ancestors}
    missing_test = held["test"] - novel.keys()
    replacements = []
    if missing_test and test_fallback:
        reserved = {novel[name]["ancestor_id"] for name in set(val_names + test_names) & novel.keys()}
        for name in test_fallback:
            if name in novel and novel[name]["ancestor_id"] not in reserved:
                replacements.append(name)
                reserved.add(novel[name]["ancestor_id"])
            if len(replacements) == len(missing_test):
                break
        held["test"] = (held["test"] - missing_test) | set(replacements)
    # Reserve all branches of a specified ancestor together before any scores exist.
    assigned = {}
    for split, names in held.items():
        if not names or names - novel.keys():
            raise ValueError(f"{split}: held-out candidates unavailable: {sorted(names - novel.keys())}")
        for name in names:
            ancestor = novel[name]["ancestor_id"]
            if assigned.setdefault(ancestor, split) != split:
                raise ValueError(f"ancestor requested in multiple splits: {ancestor}")
    splits = {"train": [], "val": [], "test": []}
    rows = []
    for name, row in sorted(novel.items()):
        n = row["num_steps"]
        if n < HISTORY + HORIZON:
            raise ValueError(f"{name} has fewer than L+H consecutive frames")
        split = assigned.get(row["ancestor_id"], "train")
        with h5py.File(dataset_dir / name, "r") as f:
            steps = f["step_indices"][:].tolist()
        rows.append({
            "file_name": name, "trajectory_id": Path(name).stem, "ancestor_id": row["ancestor_id"],
            "source_sha256": sha256_file(dataset_dir / name), "split": split,
            "num_steps": n, "frame_indices": list(range(n)), "step_indices": steps,
            "dt_ctu": row["dt_ctu_mean"], "dt_phys_s": row["dt_phys_mean_s"],
            "reynolds_number": row["reynolds_number"], "tier1_status": row["tier1_limit_cycle_status"],
        })
        splits[split].append(name)
    if not all(splits.values()):
        raise ValueError("G2 requires at least one independent ancestor in each split")
    if before != snapshot(dataset_dir, candidates) or admission_sha256 != {
        name: sha256_file(dataset_dir / name) for name in SIDECARS
    }:
        raise RuntimeError("dataset changed during audit; rerun with a stable dataset")
    contract = {
        "schema_version": "flow3d.g2.split.v1", "dataset_dir": str(dataset_dir),
        "admission_sha256": admission_sha256, "g0_audit_sha256": sha256_file(output_dir / "g0_audit.json"),
        "stage1_split_identity_sha256": sha256_json(stage1_identity),
        "stage1_files": legacy_files, "stage1_ancestor_ids": sorted(legacy_ancestors),
        "policy": "Entire novel ancestors only; explicitly reserved ancestors go to validation or test.",
        "reserved_files": {split: sorted(names) for split, names in held.items()},
        "original_test_files": sorted(test_names), "test_fallback_order": test_fallback,
        "missing_original_test_files": sorted(missing_test), "test_replacements": replacements,
        "history_length": HISTORY, "horizon": HORIZON, "splits": splits, "trajectories": rows,
        "split_counts": {
            split: {
                "ancestors": len({row["ancestor_id"] for row in rows if row["split"] == split}),
                "frames": sum(row["num_steps"] for row in rows if row["split"] == split),
                "windows": sum(row["num_steps"] - HISTORY - HORIZON + 1 for row in rows if row["split"] == split),
            } for split in splits
        },
    }
    publish(output_dir / "split_manifest.json", contract)
    return contract


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--stage1-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--val-files", nargs="+", required=True)
    parser.add_argument("--test-files", nargs="+", required=True)
    parser.add_argument("--test-fallback", nargs="*", default=[], help="Ordered alternatives registered before scores")
    args = parser.parse_args()
    contract = prepare(args.dataset_dir, args.stage1_summary, args.output_dir, args.val_files, args.test_files, args.test_fallback)
    print(json.dumps(contract["split_counts"]))


if __name__ == "__main__":
    main()
