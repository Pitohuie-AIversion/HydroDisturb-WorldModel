"""Freeze Stage 1 and encode an independently registered downstream train/validation split."""

import argparse
from pathlib import Path

from flow3d.data.latent_cache import export_downstream_cache


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint-dir", "stage1-data-dir", "dataset-dir", "split-manifest", "out-dir"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    path = export_downstream_cache(
        args.checkpoint_dir, args.stage1_data_dir, args.dataset_dir, args.split_manifest, args.out_dir,
        args.device, expected_checkpoint_sha256=args.expected_checkpoint_sha256,
    )
    print(path, flush=True)


if __name__ == "__main__":
    main()
