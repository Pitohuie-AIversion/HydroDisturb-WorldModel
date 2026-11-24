from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping


def hash_cache_index(
    index_path: Path, checkpoint_sha256: str, contract_sha256: str
) -> tuple[str, dict[str, str]]:
    """Hash the published JSON index and each shard it names, in index order."""
    contents = index_path.read_bytes()
    index = json.loads(contents)
    shards = index.get("shards") if isinstance(index, dict) else None
    if not isinstance(shards, list) or not shards:
        raise ValueError("cache index must declare a nonempty 'shards' list")
    if index.get("checkpoint_sha256") != checkpoint_sha256 or index.get("contract_sha256") != contract_sha256:
        raise ValueError("cache index belongs to a different checkpoint or contract")
    root = index_path.resolve().parent
    hashes: dict[str, str] = {}
    seen = {index_path.resolve()}
    for name in shards:
        if not isinstance(name, str) or not name or Path(name).is_absolute():
            raise ValueError("cache shard names must be relative paths")
        shard = (root / name).resolve()
        if not shard.is_relative_to(root) or shard in seen:
            raise ValueError(f"cache shard outside index directory or listed twice: {name}")
        seen.add(shard)
        hashes[name] = sha256_file(shard)
    return hashlib.sha256(contents).hexdigest(), hashes


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
    latent_layout: str = "B,T,C,Z,Y,X",
    normalizer_version: str,
    dataset_manifest_sha256: str,
) -> None:
    expected = {
        "checkpoint_sha256": checkpoint_sha256,
        "contract_sha256": contract_sha256,
        "latent_source": latent_source,
        "latent_layout": latent_layout,
        "normalizer_version": normalizer_version,
        "dataset_manifest_sha256": dataset_manifest_sha256,
    }
    actual = manifest.to_dict()
    mismatches = [f"{key}: expected {value!r}, got {actual.get(key)!r}" for key, value in expected.items() if actual.get(key) != value]
    if mismatches:
        raise ValueError("frozen artifact contract mismatch; " + "; ".join(mismatches))
