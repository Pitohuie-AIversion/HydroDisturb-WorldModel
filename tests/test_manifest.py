from pathlib import Path

import pytest

from stage1.io.manifest import FrozenArtifactManifest, sha256_file, validate_manifest, write_manifest


def test_manifest_roundtrip_and_hash(tmp_path: Path):
    weights = tmp_path / "weights.bin"
    weights.write_bytes(b"stage1")
    manifest = FrozenArtifactManifest(
        checkpoint_sha256=sha256_file(weights),
        contract_sha256="contract",
        latent_source="z_clean",
        latent_layout="B,C,D,H,W",
        normalizer_version="normalizer.v1",
        dataset_manifest_sha256="dataset",
    )
    path = tmp_path / "artifact.json"
    write_manifest(path, manifest)
    loaded = FrozenArtifactManifest.from_mapping(__import__("json").loads(path.read_text(encoding="utf-8")))
    validate_manifest(loaded, latent_layout=manifest.latent_layout, checkpoint_sha256=manifest.checkpoint_sha256, contract_sha256="contract", normalizer_version="normalizer.v1", dataset_manifest_sha256="dataset")


def test_manifest_rejects_wrong_checkpoint():
    manifest = FrozenArtifactManifest("a", "b", "z_clean", "B,C,D,H,W", "normalizer.v1", "d")
    with pytest.raises(ValueError, match="contract mismatch"):
        validate_manifest(manifest, checkpoint_sha256="wrong", contract_sha256="b", normalizer_version="normalizer.v1", dataset_manifest_sha256="d")


def test_manifest_checks_consumer_layout():
    manifest = FrozenArtifactManifest("a", "b", "z_clean", "B,X,Z", "norm", "d")
    with pytest.raises(ValueError, match="latent_layout"):
        validate_manifest(
            manifest, checkpoint_sha256="a", contract_sha256="b", normalizer_version="norm",
            dataset_manifest_sha256="d", latent_layout="B,T,C,Z,Y,X",
        )
