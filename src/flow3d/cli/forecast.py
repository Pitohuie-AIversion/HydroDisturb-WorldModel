"""Train a small latent forecast model, score a free rollout, or predict from saved history."""

from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import nullcontext
import json
from pathlib import Path
from tempfile import TemporaryFile

import numpy as np
import torch

from flow3d.contracts import condition_cadence
from flow3d.data.latent_cache import export_latent_cache, load_latent_cache
from flow3d.metrics.rollout import score_forecast_frame_si
from flow3d.training.forecast import (
    ModeResForecaster,
    delta_z_rollout,
    load_forecaster,
    rollout_latents,
    save_forecaster,
    train_forecaster,
)
from stage1.io.manifest import sha256_file


@torch.inference_mode()
def decode_frame(decoder, z, b, geometry, spacing, device):
    result = decoder.decode(z[None].to(device), geometry, b=b[None].to(device), spacing_zyx=spacing)
    field = result["phys_fields"][0].float().cpu().numpy()
    if not np.isfinite(field).all():
        raise ValueError("decoder produced a non-finite forecast in physical units")
    return field


@torch.inference_mode()
def forecast_latents(model, segment, start, length, horizon, device, *, baseline=False):
    history = slice(start, start + length)
    z = torch.as_tensor(segment["z_clean"][history], device=device)
    b = torch.as_tensor(segment["b"][history], device=device)
    dt_ctu = condition_cadence(segment["conditions"], "dt_ctu", required=True)
    if baseline:
        return delta_z_rollout(z, b, horizon, dt_ctu=dt_ctu)
    if isinstance(model, ModeResForecaster):
        return rollout_latents(model, z, b, horizon, dt_ctu=dt_ctu)
    origin = float(segment["time_physical_s"][start + length - 1])
    learned_z, learned_b = model(z[None], b[None], [segment["conditions"]], [origin], horizon)
    return learned_z[0], learned_b[0]


def forecast_window(model, decoder, segment, start, length, horizon, device, *, baseline=False):
    z, b = forecast_latents(model, segment, start, length, horizon, device, baseline=baseline)
    geometry = torch.as_tensor(segment["geometry"], device=device)[None]
    return np.stack([
        decode_frame(decoder, frame_z, frame_b, geometry, segment["spacing_star"], device)
        for frame_z, frame_b in zip(z, b)
    ])


def write_prediction(path, fields, segment, origin, lead_steps):
    # Forecast timestamps depend only on the origin and the known sampling interval.
    conditions = segment["conditions"]
    time_s = float(segment["time_physical_s"][origin])
    time_ctu = float(segment["time_nondim_ctu"][origin])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as output:
        np.savez_compressed(
            output, fields=fields, channels=["u", "v", "w", "p"], units=["m/s", "m/s", "m/s", "Pa"],
            axis_order="T,C,Z,Y,X", fluid_mask=np.asarray(segment["geometry"][0]) > 0.5,
            spacing_zyx_m=segment["spacing_m"], trajectory_id=segment["trajectory_id"],
            origin_step=segment["step_indices"][origin], origin_time_s=time_s, lead_steps=lead_steps,
            time_physical_s=time_s + lead_steps * condition_cadence(conditions, "dt_phys_s", required=True),
            time_nondim_ctu=time_ctu + lead_steps * condition_cadence(conditions, "dt_ctu", required=True),
        )


def summarize(rows):
    """Average over origins within each ancestor, then weight ancestors equally."""
    by_lead = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for row in rows:
        for name, score in row["scores"].items():
            by_lead[row["lead_steps"]][name][row["ancestor_id"]].append(score)
    result = []
    for lead, methods in sorted(by_lead.items()):
        means = {}
        for name, ancestors in methods.items():
            means[name] = {
                metric: float(np.mean([
                    np.mean([point[metric] for point in points]) for points in ancestors.values()
                ]))
                for metric in ("velocity_vector_rmse_m_s", "pressure_rmse_pa")
            }
        result.append({"lead_steps": lead, "methods": means})
    return result


def _score_origin(model, decoder, segment, start, history, horizon, device, observations, geometry, sample_path):
    origin = start + history - 1
    mask = segment["geometry"][0]
    learned = forecast_latents(model, segment, start, history, horizon, device)
    delta = forecast_latents(model, segment, start, history, horizon, device, baseline=True)
    rows = []
    # Keep the optional example on disk. Each method decodes just one SI field at a time.
    with (TemporaryFile() if sample_path is not None else nullcontext()) as sample_file:
        sample = None
        if sample_file is not None:
            sample = np.memmap(sample_file, dtype=np.float32, mode="w+",
                               shape=(horizon, *observations[origin].shape))
        for index in range(horizon):
            target = origin + index + 1
            observed = observations[target]
            scores = {"persistence": score_forecast_frame_si(observations[origin], observed, mask)}
            for method, (z, b) in (("mode_res", learned), ("delta_z", delta)):
                field = decode_frame(decoder, z[index], b[index], geometry, segment["spacing_star"], device)
                scores[method] = score_forecast_frame_si(field, observed, mask)
                if method == "mode_res" and sample is not None:
                    sample[index] = field
                del field
            rows.append({
                "trajectory_id": segment["trajectory_id"], "ancestor_id": segment["ancestor_id"],
                "origin_step": int(segment["step_indices"][origin]),
                "target_step": int(segment["step_indices"][target]), "lead_steps": index + 1,
                "lead_seconds": float(segment["time_physical_s"][target] - segment["time_physical_s"][origin]),
                "lead_ctu": float(segment["time_nondim_ctu"][target] - segment["time_nondim_ctu"][origin]),
                "scores": scores,
            })
        if sample is not None:
            write_prediction(sample_path, sample, segment, origin, np.arange(1, horizon + 1))
    return rows


@torch.inference_mode()
def evaluate(cache, dataset_dir, model, decoder, history, horizon, device, sample_path,
             *, split="test", origin_stride=1):
    rows = []
    for segment in cache.sequences(split):
        eligible = len(segment["b"]) - history - horizon + 1
        starts = range(0, eligible, origin_stride)
        if not starts:
            continue
        # Read each observed SI field once per segment across all selected origins and methods.
        indices = sorted({start + history - 1 + lead for start in starts for lead in range(horizon + 1)})
        fields = cache.targets_si(segment, dataset_dir, indices)
        observations = dict(zip(indices, fields, strict=True))
        geometry = torch.as_tensor(segment["geometry"], device=device)[None]
        for start in starts:
            rows.extend(_score_origin(model, decoder, segment, start, history, horizon, device,
                                      observations, geometry, sample_path if not rows else None))
        del observations, fields
    if not rows:
        raise ValueError(f"No {split} segment contains {history} history frames and {horizon} future frames")
    return rows


def run(args, device):
    out = args.out_dir.resolve()
    out.mkdir(exist_ok=True, parents=True)
    manifest = args.cache_manifest
    if manifest is None:
        print("Exporting the selected Stage 1 checkpoint and split-aligned latent sequences...", flush=True)
        manifest = export_latent_cache(args.checkpoint_dir, args.dataset_dir, out / "cache", device)
    cache = load_latent_cache(manifest, source_dir=args.dataset_dir)
    decoder = cache.load_model(device)
    fingerprint = sha256_file(manifest)
    eligible = sum(max(0, len(row["frames"]) - args.history - args.horizon + 1)
                   for row in cache.segments if row["split"] == args.eval_split)
    if not eligible:
        raise ValueError(f"No {args.eval_split} segment contains {args.history} history frames and "
                         f"{args.horizon} future frames")
    print("Fitting Stage 2 using training sequences...", flush=True)
    trainer = train_forecaster
    if args.model == "clean-query":
        from flow3d.training.dynamics import train_conditioned

        trainer = train_conditioned
    model, training = trainer(
        cache.sequences("train"), history_length=args.history, rollout_steps=args.train_horizon or args.horizon,
        epochs=args.epochs, learning_rate=args.lr, batch_size=args.batch_size, seed=args.seed, device=device,
    )
    model_path = out / "forecaster.pt"
    save_forecaster(model, model_path, metadata={"cache_manifest_sha256": fingerprint, "history_length": args.history})
    model, _ = load_forecaster(model_path, device=device)
    print(f"Scoring free rollouts and both baselines on the {args.eval_split} segments...", flush=True)
    rows = evaluate(cache, args.dataset_dir, model, decoder, args.history, args.horizon, device,
                    None if args.no_example else out / "forecast_example.npz",
                    split=args.eval_split, origin_stride=args.origin_stride)
    split_counts = {split: len(frames) for split, frames in cache.split_identity["splits"].items()}
    report = {
        "method": "Mode-Res", "history_length": args.history, "horizon": args.horizon,
        "stage1_weights": cache.weight_source, "stage1_split_strategy": cache.stage1_split_strategy,
        "evaluation_split_strategy": cache.split_strategy,
        "cache_manifest": str(Path(manifest).resolve()), "cache_manifest_sha256": fingerprint,
        "split_frame_counts": split_counts, "training": training,
        "eval_split": args.eval_split, "eligible_windows": eligible, "origin_stride": args.origin_stride,
        f"{args.eval_split}_windows": sum(row["lead_steps"] == 1 for row in rows),
        f"{args.eval_split}_ancestors": sorted({row["ancestor_id"] for row in rows}),
        "aggregation": "Mean of per-origin fluid-cell RMSE within each ancestor; equal weight per ancestor.",
        "background_statistics": (
            "Mode-Res predicts b from history; Delta-z persists the last observed b."
            if args.model == "clean-query" else "Both latent methods persist the last observed b."
        ),
        "by_horizon": summarize(rows), "records": rows,
    }
    if getattr(cache, "downstream_split_sha256", None):
        report["downstream_split_sha256"] = cache.downstream_split_sha256
    with (out / "rollout_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    print(f"Report: {out / 'rollout_report.json'}", flush=True)


def predict(args, device):
    cache = load_latent_cache(args.cache_manifest, source_dir=args.dataset_dir)
    model, metadata = load_forecaster(args.model, device=device)
    if metadata["cache_manifest_sha256"] != sha256_file(args.cache_manifest):
        raise ValueError("forecaster belongs to a different frozen latent cache")
    decoder = cache.load_model(device)
    history = metadata["history_length"]
    for segment in cache.sequences(args.split):
        if args.trajectory is not None and segment["trajectory_id"] != args.trajectory:
            continue
        steps = list(segment["step_indices"])
        if args.origin_step is not None and args.origin_step not in steps:
            continue
        origin = len(steps) - 1 if args.origin_step is None else steps.index(args.origin_step)
        if origin + 1 < history:
            continue
        fields = forecast_window(model, decoder, segment, origin + 1 - history, history, args.horizon, device)
        write_prediction(args.output, fields, segment, origin, np.arange(1, args.horizon + 1))
        print(f"Forecast from {segment['trajectory_id']} step {steps[origin]}: {args.output}", flush=True)
        return
    raise ValueError("no matching trajectory and origin with sufficient history in this split")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    fit = sub.add_parser("run", help="export, train, reload and evaluate a short rollout")
    source = fit.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint-dir", type=Path, help="Stage 1 run with a split-identified best checkpoint")
    source.add_argument("--cache-manifest", type=Path, help="reuse an existing frozen cache")
    fit.add_argument("--dataset-dir", type=Path, required=True)
    fit.add_argument("--out-dir", type=Path, required=True)
    fit.add_argument("--history", type=int, default=4)
    fit.add_argument("--model", choices=["channelwise", "clean-query"], default="channelwise")
    fit.add_argument("--train-horizon", type=int, help="pushforward loss steps (defaults to --horizon)")
    fit.add_argument("--eval-split", choices=["val", "test"], default="test")
    fit.add_argument("--origin-stride", type=int, default=1,
                     help="score every Nth eligible origin starting with the first in each segment")
    fit.add_argument("--no-example", action="store_true", help="skip writing the decoded example archive")
    fit.add_argument("--epochs", type=int, default=1)
    fit.add_argument("--batch-size", type=int, default=4)
    fit.add_argument("--lr", type=float, default=1e-3)
    fit.add_argument("--seed", type=int, default=42)
    infer = sub.add_parser("predict", help="decode future SI fields from cached history only")
    infer.add_argument("--cache-manifest", type=Path, required=True)
    infer.add_argument("--dataset-dir", type=Path, help="original G0 inputs for legacy caches missing action provenance")
    infer.add_argument("--model", type=Path, required=True)
    infer.add_argument("--split", choices=["train", "val", "test"], default="test")
    infer.add_argument("--trajectory", help="trajectory ID (default: first matching trajectory)")
    infer.add_argument("--origin-step", type=int, help="last observed step (default: last step of the segment)")
    infer.add_argument("--output", type=Path, required=True)
    for child in (fit, infer):
        child.add_argument("--horizon", type=int, default=2)
        child.add_argument("--device", default="cpu", help="a PyTorch device, e.g. cpu or cuda:0")
    args = parser.parse_args(argv)
    if args.horizon < 1:
        parser.error("--horizon must be positive")
    if args.command == "run" and (args.history < 2 or min(args.epochs, args.batch_size) < 1 or args.lr <= 0):
        parser.error("run needs --history >= 2 and positive epochs, batch size and learning rate")
    if args.command == "run" and (args.origin_stride < 1 or (args.train_horizon is not None and args.train_horizon < 1)):
        parser.error("--origin-stride and --train-horizon must be positive")
    device = torch.device(args.device)
    (run if args.command == "run" else predict)(args, device)


if __name__ == "__main__":
    main()
