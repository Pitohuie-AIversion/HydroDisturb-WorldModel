from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping


def sha256_file(path: str | os.PathLike[str], chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class FrozenArtifactManifest:
    checkpoint_sha256: str
    contract_sha256: str
    latent_source: str
    latent_layout: str
    normalizer_version: str
    dataset_manifest_sha256: str
    schema_version: str = "stage1.artifact.v1"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "FrozenArtifactManifest":
        required = {
            "checkpoint_sha256", "contract_sha256", "latent_source", "latent_layout",
            "normalizer_version", "dataset_manifest_sha256",
        }
        missing = sorted(required - payload.keys())
        if missing:
            raise ValueError(f"artifact manifest missing fields: {', '.join(missing)}")
        return cls(**{key: payload[key] for key in required}, schema_version=str(payload.get("schema_version", "stage1.artifact.v1")))


def write_manifest(path: str | os.PathLike[str], manifest: FrozenArtifactManifest) -> None:
    """Publish a manifest atomically so Stage 2 never sees a partial JSON file."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    with open(temp, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest.to_dict(), handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, target)


def validate_manifest(
    manifest: FrozenArtifactManifest,
    *,
    checkpoint_sha256: str,
    contract_sha256: str,
    latent_source: str = "z_clean",
    normalizer_version: str,
    dataset_manifest_sha256: str,
) -> None:
    expected = {
        "checkpoint_sha256": checkpoint_sha256,
        "contract_sha256": contract_sha256,
        "latent_source": latent_source,
        "normalizer_version": normalizer_version,
        "dataset_manifest_sha256": dataset_manifest_sha256,
    }
    actual = manifest.to_dict()
    mismatches = [f"{key}: expected {value!r}, got {actual.get(key)!r}" for key, value in expected.items() if actual.get(key) != value]
    if mismatches:
        raise ValueError("frozen artifact contract mismatch; " + "; ".join(mismatches))
