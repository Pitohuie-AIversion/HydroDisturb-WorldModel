from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from flow3d.contracts import PhysicsContractError
from flow3d.data.adapters import (
    inspect_and_segment_c04_shedding_directory,
    inspect_world_model_dataset_manifest,
    load_c04_benchmark_npz,
)
from flow3d.data.schema import load_yaml_or_json_config
from flow3d.physics.boundary import evaluate_cylinder_boundary_conditions
from flow3d.physics.operators import div_h


def inspect_single_source(path: str | Path) -> dict[str, Any]:
    """Inspect a single 3D CFD data path (C04 NPZ file, shedding NPZ directory, or HDF5 dataset directory)."""
    p = Path(path)
    if not p.exists():
        return {"path": str(p), "status": "MISSING", "error": f"path does not exist: {p}"}

    if p.is_file() and p.suffix.lower() == ".npz":
        batch, summary = load_c04_benchmark_npz(p)
        f0 = batch.field[0, 0]
        fluid_mask = batch.geometry[0, 0]
        sdf_over_d = batch.geometry[0, 1]
        div_field = div_h(f0[0], f0[1], f0[2], batch.spacing, mask=fluid_mask)
        valid_cells = fluid_mask > 0.5
        bc_diag = evaluate_cylinder_boundary_conditions(f0, fluid_mask, sdf_over_d, batch.spacing)
        summary["physical_divergence_rms_s_inv"] = float(np.sqrt(np.mean(div_field[valid_cells] ** 2)))
        summary["physical_divergence_mae_s_inv"] = float(np.mean(np.abs(div_field[valid_cells])))
        summary["boundary_diagnostics"] = bc_diag.to_dict()
        summary["status"] = "VERIFIED_4FRAME_BENCHMARK"
        return summary

    if p.is_dir():
        if (p / "manifest.json").exists():
            return inspect_world_model_dataset_manifest(p)
        npz_files = list(p.glob("*.npz"))
        if npz_files:
            return inspect_and_segment_c04_shedding_directory(p)

    raise PhysicsContractError(f"unsupported or unknown 3D CFD dataset format at: {p}")


def run_g0_inspection_from_config(config_path: str | Path) -> dict[str, Any]:
    """Run G0 data inspection across all configured local 3D CFD data sources."""
    cfg = load_yaml_or_json_config(config_path)
    sources = cfg.get("sources", {})
    reports: dict[str, Any] = {}
    for name, src_path in sources.items():
        if isinstance(src_path, str):
            reports[name] = inspect_single_source(src_path)
    return {
        "schema_version": cfg.get("schema_version", "flow3d.data.v1"),
        "config_path": str(config_path),
        "sources": reports,
    }
