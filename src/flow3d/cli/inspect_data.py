from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

_SRC_DIR = str(Path(__file__).resolve().parents[2])
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from flow3d.data.inspect import inspect_single_source, run_g0_inspection_from_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.flow3d.cli.inspect_data",
        description="G0 3D CFD Data Inspector & Provenance Quarantine Auditor for FlowWorld-3D V1.3 (Route A).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/flow3d/data.yaml",
        help="Path to flow3d data YAML/JSON config file (default: configs/flow3d/data.yaml).",
    )
    parser.add_argument(
        "--path",
        type=str,
        default=None,
        help="Optional single dataset file or directory path to inspect directly.",
    )
    parser.add_argument(
        "--json-out",
        type=str,
        default=None,
        help="Optional path to save the structured JSON inspection report.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return non-zero exit code (1) if any inspected source is MISSING or has 0 eligible trajectories.",
    )
    return parser


def _has_strict_failure(report: dict) -> bool:
    if report.get("status") == "MISSING":
        return True
    if report.get("format") == "world_model_multi_re_hdf5_manifest" and report.get("eligible_trajectory_count", 0) <= 0:
        return True
    sources = report.get("sources")
    if isinstance(sources, dict):
        if not sources:
            return True
        return any(_has_strict_failure(v) for v in sources.values() if isinstance(v, dict))
    return False


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.path:
        report = inspect_single_source(args.path)
    else:
        report = run_g0_inspection_from_config(args.config)

    text = json.dumps(report, indent=2, ensure_ascii=False)
    print(text)
    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")
    if args.strict and _has_strict_failure(report):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
