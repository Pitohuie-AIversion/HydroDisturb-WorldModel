from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator
import hashlib
import json

import numpy as np

from .normalization import FlowNormalizer, NormalizationStats


@dataclass(frozen=True)
class DatasetManifest:
    path: str
    sha256: str
    samples: int
    spatial_shape: tuple[int, int, int]
    channels: tuple[str, ...] = ("u", "v", "w", "p", "mask", "sdf")

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "samples": self.samples,
            "spatial_shape": list(self.spatial_shape),
            "channels": list(self.channels),
        }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class NpzFlowDataset:
    """Small deterministic adapter for a `.npz` array named `frames`.

    The adapter is intentionally NumPy-only. A Torch DataLoader can wrap it later
    without changing the on-disk data or normalization contract.
    """

    def __init__(self, path: str | Path, normalizer: FlowNormalizer | None = None):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        self._archive = np.load(self.path, allow_pickle=False, mmap_mode="r")
        if "frames" not in self._archive:
            raise ValueError("dataset archive must contain a `frames` array")
        self.frames = self._archive["frames"]
        if self.frames.ndim != 5 or self.frames.shape[1] != 6:
            raise ValueError("frames must have shape (N,6,D,H,W)")
        self.normalizer = normalizer or FlowNormalizer()
        self.manifest = DatasetManifest(str(self.path), _sha256(self.path), int(self.frames.shape[0]), tuple(int(v) for v in self.frames.shape[2:]))

    def __len__(self) -> int:
        return int(self.frames.shape[0])

    def __getitem__(self, index: int) -> np.ndarray:
        return np.asarray(self.frames[index], dtype=np.float32)

    def normalized(self, index: int) -> tuple[np.ndarray, NormalizationStats]:
        return self.normalizer.normalize(self[index])

    def iter_normalized(self, indices: list[int] | None = None) -> Iterator[tuple[np.ndarray, NormalizationStats]]:
        for index in indices if indices is not None else range(len(self)):
            yield self.normalized(int(index))

    def write_manifest(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.manifest.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
