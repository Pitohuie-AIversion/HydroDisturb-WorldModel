"""Reliable Stage 1 checkpoints and frozen-artifact contracts.

The training code intentionally keeps checkpoint handling in a small, dependency
light module.  Checkpoints are written transactionally, carry all state needed
for a resume, and are bound to a versioned model/data contract.  Frozen latent
exports can use the same file hash to prove which encoder produced them.

PyTorch and NumPy are optional at import time so that configuration and
manifest tooling can still be used in a lightweight environment.  Operations
that need either dependency fail with a clear error when it is not installed.
"""

from __future__ import annotations

import copy
import datetime as _dt
import hashlib
import json
import os
import random
import tempfile
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence

try:  # Optional for tooling that only needs contracts/manifests.
    import numpy as np
except Exception:  # pragma: no cover - exercised only without numpy installed
    np = None  # type: ignore[assignment]

try:  # Optional so importing the package does not require a GPU/PyTorch image.
    import torch
except Exception:  # pragma: no cover - exercised only without torch installed
    torch = None  # type: ignore[assignment]


FORMAT_VERSION = 2
MANIFEST_VERSION = 1


class CheckpointError(RuntimeError):
    """Base class for checkpoint and artifact failures."""


class CheckpointDependencyError(CheckpointError):
    """Raised when a PyTorch/NumPy operation is requested without its package."""


class CheckpointIntegrityError(CheckpointError):
    """Raised when a checkpoint or manifest hash does not match its contents."""


class CheckpointContractError(CheckpointError):
    """Raised when a checkpoint is incompatible with the requested contract."""


class CheckpointStateError(CheckpointError):
    """Raised when a requested resume state is absent or cannot be restored."""


def _require_torch() -> Any:
    if torch is None:
        raise CheckpointDependencyError(
            "PyTorch is required for model checkpoint save/load operations"
        )
    return torch


def _utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat().replace("+00:00", "Z")


def _jsonable(value: Any) -> Any:
    """Convert contract/config values to deterministic JSON-safe primitives."""

    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, set):
        return sorted((_jsonable(v) for v in value), key=lambda v: repr(v))
    if np is not None and isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, bytes):
        return {"__bytes_hex__": value.hex()}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"value of type {type(value).__name__} is not JSON serializable")


def canonical_json(value: Any) -> str:
    """Return canonical JSON used for stable config/contract hashes."""

    return json.dumps(
        _jsonable(value),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json(value).encode("utf-8"))


def sha256_file(path: str | os.PathLike[str], chunk_size: int = 1024 * 1024) -> str:
    """Compute SHA-256 without loading a checkpoint into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while True:
            chunk = stream.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class CheckpointContract:
    """Versioned interface between Stage 1 and downstream latent consumers.

    ``extra`` allows a project to add fields without changing this helper.  For
    the broadest compatibility, :class:`CheckpointManager` also accepts a plain
    mapping as a contract and preserves all of its keys.
    """

    schema_version: int = 1
    latent_layout: str | None = "B,T,C,Z,Y,X"
    latent_shape: tuple[int, ...] | None = None
    normalizer_version: str | None = None
    normalization: Mapping[str, Any] | None = None
    dataset_manifest_sha256: str | None = None
    geometry_version: str | None = None
    code_version: str | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema_version": self.schema_version,
            "latent_layout": self.latent_layout,
            "latent_shape": list(self.latent_shape) if self.latent_shape is not None else None,
            "normalizer_version": self.normalizer_version,
            "normalization": _jsonable(self.normalization) if self.normalization is not None else None,
            "dataset_manifest_sha256": self.dataset_manifest_sha256,
            "geometry_version": self.geometry_version,
            "code_version": self.code_version,
        }
        result.update(_jsonable(self.extra))
        return result


def normalize_contract(contract: Any | None) -> dict[str, Any]:
    """Normalize a contract object/mapping to a deterministic dictionary."""

    if contract is None:
        return {}
    if isinstance(contract, CheckpointContract):
        return contract.to_dict()
    if is_dataclass(contract):
        return _jsonable(asdict(contract))
    if isinstance(contract, Mapping):
        return _jsonable(dict(contract))
    raise TypeError("contract must be a mapping, CheckpointContract, dataclass, or None")


def _capture_numpy_rng() -> dict[str, Any] | None:
    if np is None:
        return None
    algorithm, keys, pos, has_gauss, cached_gaussian = np.random.get_state()
    return {
        "algorithm": str(algorithm),
        "keys": [int(v) for v in keys.tolist()],
        "pos": int(pos),
        "has_gauss": int(has_gauss),
        "cached_gaussian": float(cached_gaussian),
    }


def _restore_numpy_rng(state: Mapping[str, Any]) -> None:
    if np is None:
        raise CheckpointDependencyError("NumPy is required to restore NumPy RNG state")
    np.random.set_state(
        (
            str(state["algorithm"]),
            np.asarray(state["keys"], dtype=np.uint32),
            int(state["pos"]),
            int(state["has_gauss"]),
            float(state["cached_gaussian"]),
        )
    )


def capture_rng_state() -> dict[str, Any]:
    """Capture Python, NumPy, CPU and CUDA RNG states."""

    t = _require_torch()
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": _capture_numpy_rng(),
        "torch_cpu": t.get_rng_state().clone(),
        "torch_cuda": None,
    }
    if t.cuda.is_available():
        state["torch_cuda"] = [s.clone() for s in t.cuda.get_rng_state_all()]
    return state


def restore_rng_state(state: Mapping[str, Any]) -> None:
    """Restore a state produced by :func:`capture_rng_state`."""

    t = _require_torch()
    if "python" in state:
        # ``random.setstate`` requires nested tuples; safe loading may return
        # lists, so convert recursively.
        def _tuplify(x: Any) -> Any:
            return tuple(_tuplify(v) for v in x) if isinstance(x, (list, tuple)) else x

        random.setstate(_tuplify(state["python"]))
    if state.get("numpy") is not None:
        _restore_numpy_rng(state["numpy"])
    if state.get("torch_cpu") is not None:
        t.set_rng_state(state["torch_cpu"])
    cuda_state = state.get("torch_cuda")
    if cuda_state is not None and t.cuda.is_available():
        t.cuda.set_rng_state_all(cuda_state)


def _fsync_directory(directory: Path) -> None:
    """Best-effort directory sync (not supported by all Windows filesystems)."""

    try:
        fd = os.open(str(directory), os.O_RDONLY)
    except (OSError, AttributeError):
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _atomic_write_bytes(data: bytes, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=str(destination.parent)
    )
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _atomic_torch_save(payload: Mapping[str, Any], destination: Path) -> None:
    t = _require_torch()
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=str(destination.parent)
    )
    try:
        with os.fdopen(fd, "wb") as stream:
            t.save(dict(payload), stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _safe_torch_load(path: Path, map_location: Any = "cpu") -> dict[str, Any]:
    t = _require_torch()
    try:
        payload = t.load(path, map_location=map_location, weights_only=True)
    except TypeError:  # Older PyTorch versions do not expose ``weights_only``.
        payload = t.load(path, map_location=map_location)
    if not isinstance(payload, dict):
        raise CheckpointError(f"checkpoint {path} does not contain a mapping payload")
    return payload


def _state_present(payload: Mapping[str, Any], key: str) -> bool:
    presence = payload.get("state_presence")
    if isinstance(presence, Mapping) and key in presence:
        return bool(presence[key])
    return payload.get(f"{key}_state_dict") is not None


class CheckpointManager:
    """Atomic, contract-checked Stage 1 checkpoint manager.

    A normal checkpoint is immutable once published.  ``best.json`` is a small
    atomic pointer to the immutable epoch/step file and includes its SHA-256.
    This avoids replacing a known-good ``best.pt`` in place after a failed
    write.
    """

    def __init__(
        self,
        save_dir: str | os.PathLike[str],
        model_name: str = "dcae3d_stage1",
        contract: Any | None = None,
        schema_version: int = FORMAT_VERSION,
    ) -> None:
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        self.contract = normalize_contract(contract)
        self.schema_version = int(schema_version)

    @property
    def best_pointer_path(self) -> Path:
        return self.save_dir / f"{self.model_name}_best.json"

    def _make_filename(self, epoch: int, global_step: int) -> Path:
        return self.save_dir / f"{self.model_name}_epoch_{int(epoch):06d}_step_{int(global_step):09d}.pt"

    def save_checkpoint(
        self,
        epoch: int,
        global_step: int,
        model: Any,
        optimizer: Any | None,
        scheduler: Any | None = None,
        scaler: Any | None = None,
        config: Mapping[str, Any] | None = None,
        metrics: Mapping[str, Any] | None = None,
        is_best: bool = False,
        contract: Any | None = None,
        sampler_state: Any | None = None,
        dataloader_state: Any | None = None,
        extra_state: Mapping[str, Any] | None = None,
        include_rng: bool = True,
    ) -> Path:
        """Publish a complete resume checkpoint and optionally update ``best``.

        The optimizer is intentionally a required argument (it may be ``None``
        for a weights-only export only when ``include_rng=False``).  A resume
        checkpoint with no optimizer is almost always an accidental partial
        save, so callers must make that choice explicit.
        """

        t = _require_torch()
        if model is None or not hasattr(model, "state_dict"):
            raise TypeError("model must provide state_dict()")
        if optimizer is None and include_rng:
            raise CheckpointStateError(
                "optimizer is required for a resumable checkpoint; pass include_rng=False for weights-only export"
            )

        contract_value = normalize_contract(contract if contract is not None else self.contract)
        config_value = _jsonable(dict(config or {}))
        metrics_value = _jsonable(dict(metrics or {}))
        payload: dict[str, Any] = {
            "format_version": self.schema_version,
            "checkpoint_kind": "resume" if include_rng else "weights",
            "created_at_utc": _utc_now(),
            "model_name": self.model_name,
            "epoch": int(epoch),
            "global_step": int(global_step),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
            "state_presence": {
                "optimizer": optimizer is not None,
                "scheduler": scheduler is not None,
                "scaler": scaler is not None,
            },
            "rng_state": capture_rng_state() if include_rng else None,
            "sampler_state": copy.deepcopy(sampler_state),
            "dataloader_state": copy.deepcopy(dataloader_state),
            "extra_state": _jsonable(dict(extra_state or {})),
            "config": config_value,
            "config_sha256": sha256_json(config_value),
            "metrics": metrics_value,
            "contract": contract_value,
            "contract_sha256": sha256_json(contract_value),
        }

        destination = self._make_filename(epoch, global_step)
        _atomic_torch_save(payload, destination)
        digest = sha256_file(destination)

        if is_best:
            pointer = {
                "pointer_version": 1,
                "model_name": self.model_name,
                "checkpoint": destination.name,
                "sha256": digest,
                "checkpoint_sha256": digest,
                "checkpoint_id": digest[:16],
                "contract_sha256": payload["contract_sha256"],
                "updated_at_utc": _utc_now(),
            }
            _atomic_write_bytes(
                (canonical_json(pointer) + "\n").encode("utf-8"), self.best_pointer_path
            )
        return destination

    def _resolve_path(self, checkpoint: str | os.PathLike[str]) -> tuple[Path, str | None]:
        value = str(checkpoint)
        if value in {"best", "latest_best"}:
            pointer_path = self.best_pointer_path
        else:
            candidate = Path(checkpoint)
            if candidate == self.best_pointer_path or candidate.name == self.best_pointer_path.name:
                pointer_path = candidate if candidate.is_absolute() else self.save_dir / candidate
            else:
                if not candidate.is_absolute() and candidate.parent == Path("."):
                    candidate = self.save_dir / candidate
                return candidate, None
        if not pointer_path.exists():
            raise FileNotFoundError(f"best checkpoint pointer not found: {pointer_path}")
        try:
            pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
            target = self.save_dir / str(pointer["checkpoint"])
            expected = str(pointer.get("sha256") or pointer["checkpoint_sha256"])
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise CheckpointIntegrityError(f"invalid checkpoint pointer: {pointer_path}") from exc
        if not target.exists():
            raise FileNotFoundError(f"checkpoint referenced by pointer is missing: {target}")
        actual = sha256_file(target)
        if actual != expected:
            raise CheckpointIntegrityError(
                f"checkpoint hash mismatch for {target}: expected {expected}, got {actual}"
            )
        return target, expected

    def verify_checkpoint(
        self,
        checkpoint: str | os.PathLike[str],
        expected_sha256: str | None = None,
        expected_contract: Any | None = None,
    ) -> dict[str, Any]:
        path, pointer_hash = self._resolve_path(checkpoint)
        digest = sha256_file(path)
        expected = expected_sha256 or pointer_hash
        if expected is not None and digest != expected:
            raise CheckpointIntegrityError(
                f"checkpoint hash mismatch for {path}: expected {expected}, got {digest}"
            )
        payload = _safe_torch_load(path)
        if int(payload.get("format_version", -1)) != self.schema_version:
            raise CheckpointContractError(
                f"unsupported checkpoint format {payload.get('format_version')}; expected {self.schema_version}"
            )
        effective_contract = expected_contract if expected_contract is not None else (self.contract or None)
        if effective_contract is not None:
            expected_value = normalize_contract(effective_contract)
            actual_hash = str(payload.get("contract_sha256", ""))
            if actual_hash != sha256_json(expected_value):
                raise CheckpointContractError(
                    f"checkpoint contract mismatch: expected {sha256_json(expected_value)}, got {actual_hash}"
                )
        return payload

    def load_checkpoint(
        self,
        checkpoint: str | os.PathLike[str],
        model: Any,
        optimizer: Any | None = None,
        scheduler: Any | None = None,
        scaler: Any | None = None,
        strict: bool = True,
        expected_contract: Any | None = None,
        expected_config: Mapping[str, Any] | None = None,
        restore_rng: bool = True,
        map_location: Any = "cpu",
    ) -> dict[str, Any]:
        """Load and strictly restore a checkpoint.

        Passing an optimizer/scheduler/scaler requests restoration of that
        state.  If the state was not saved, this raises instead of silently
        producing a partially resumed run.
        """

        path, pointer_hash = self._resolve_path(checkpoint)
        digest = sha256_file(path)
        if pointer_hash is not None and digest != pointer_hash:
            raise CheckpointIntegrityError(f"checkpoint hash mismatch for {path}")
        payload = _safe_torch_load(path, map_location=map_location)
        if int(payload.get("format_version", -1)) != self.schema_version:
            raise CheckpointContractError(
                f"unsupported checkpoint format {payload.get('format_version')}; expected {self.schema_version}"
            )

        effective_contract = expected_contract if expected_contract is not None else (self.contract or None)
        if effective_contract is not None:
            expected_value = normalize_contract(effective_contract)
            expected_hash = sha256_json(expected_value)
            if payload.get("contract_sha256") != expected_hash:
                raise CheckpointContractError(
                    f"checkpoint contract mismatch: expected {expected_hash}, got {payload.get('contract_sha256')}"
                )
        if expected_config is not None:
            expected_config_value = _jsonable(dict(expected_config))
            expected_hash = sha256_json(expected_config_value)
            if payload.get("config_sha256") != expected_hash:
                raise CheckpointContractError(
                    f"checkpoint config mismatch: expected {expected_hash}, got {payload.get('config_sha256')}"
                )

        if model is None or not hasattr(model, "load_state_dict"):
            raise TypeError("model must provide load_state_dict()")
        model.load_state_dict(payload["model_state_dict"], strict=strict)

        # Load the scheduler before optimizer so optimizer construction and
        # scheduler state are both present before training resumes.
        if scheduler is not None:
            if not _state_present(payload, "scheduler"):
                raise CheckpointStateError("scheduler state was requested but is absent")
            scheduler.load_state_dict(payload["scheduler_state_dict"])
        if optimizer is not None:
            if not _state_present(payload, "optimizer"):
                raise CheckpointStateError("optimizer state was requested but is absent")
            optimizer.load_state_dict(payload["optimizer_state_dict"])
        if scaler is not None:
            if not _state_present(payload, "scaler"):
                raise CheckpointStateError("GradScaler state was requested but is absent")
            scaler.load_state_dict(payload["scaler_state_dict"])

        if restore_rng:
            if payload.get("rng_state") is None:
                raise CheckpointStateError("RNG state was requested but is absent")
            restore_rng_state(payload["rng_state"])

        return payload

    def create_frozen_manifest(
        self,
        checkpoint: str | os.PathLike[str],
        latent_source: str = "z_clean",
        latent_layout: str = "B,T,C,Z,Y,X",
        normalizer_version: str | None = None,
        dataset_manifest_sha256: str | None = None,
        export_mode: str = "eval",
        contract: Any | None = None,
        output_path: str | os.PathLike[str] | None = None,
    ) -> dict[str, Any]:
        """Create a Stage 2 cache manifest bound to an immutable checkpoint."""

        path, pointer_hash = self._resolve_path(checkpoint)
        checkpoint_hash = sha256_file(path)
        if pointer_hash is not None and checkpoint_hash != pointer_hash:
            raise CheckpointIntegrityError(f"checkpoint hash mismatch for {path}")
        payload = _safe_torch_load(path)
        checkpoint_contract = payload.get("contract") or {}
        contract_value = normalize_contract(contract) if contract is not None else normalize_contract(checkpoint_contract)
        manifest = {
            "manifest_version": MANIFEST_VERSION,
            "checkpoint": path.name,
            "checkpoint_sha256": checkpoint_hash,
            "checkpoint_id": checkpoint_hash[:16],
            "contract_sha256": sha256_json(contract_value),
            "latent_source": str(latent_source),
            "latent_layout": str(latent_layout),
            "normalizer_version": normalizer_version,
            "dataset_manifest_sha256": dataset_manifest_sha256,
            "export_mode": str(export_mode),
            "created_at_utc": _utc_now(),
        }
        if output_path is not None:
            output = Path(output_path)
            if not output.is_absolute():
                output = self.save_dir / output
            _atomic_write_bytes((canonical_json(manifest) + "\n").encode("utf-8"), output)
        return manifest

    def verify_frozen_manifest(
        self,
        manifest: Mapping[str, Any] | str | os.PathLike[str],
        checkpoint: str | os.PathLike[str] | None = None,
        expected_contract: Any | None = None,
    ) -> dict[str, Any]:
        """Verify a frozen latent-cache manifest and its checkpoint identity."""

        if isinstance(manifest, (str, os.PathLike)):
            manifest_path = Path(manifest)
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
        else:
            manifest_path = None
            data = dict(manifest)
        if int(data.get("manifest_version", -1)) != MANIFEST_VERSION:
            raise CheckpointContractError("unsupported frozen manifest version")
        checkpoint_ref = checkpoint or data.get("checkpoint")
        if checkpoint_ref is None:
            raise CheckpointIntegrityError("frozen manifest does not identify a checkpoint")
        path, pointer_hash = self._resolve_path(checkpoint_ref)
        actual_hash = sha256_file(path)
        expected_hash = str(data.get("checkpoint_sha256", ""))
        if actual_hash != expected_hash:
            raise CheckpointIntegrityError(
                f"frozen manifest hash mismatch: expected {expected_hash}, got {actual_hash}"
            )
        if pointer_hash is not None and pointer_hash != actual_hash:
            raise CheckpointIntegrityError("best pointer hash does not match checkpoint")
        payload = _safe_torch_load(path)
        expected_contract_value = normalize_contract(expected_contract) if expected_contract is not None else None
        contract_hash = str(data.get("contract_sha256", ""))
        if expected_contract_value is not None and contract_hash != sha256_json(expected_contract_value):
            raise CheckpointContractError("frozen manifest contract mismatch")
        if payload.get("contract_sha256") != contract_hash:
            raise CheckpointContractError("frozen manifest is bound to a different contract")
        if manifest_path is not None:
            data["manifest_path"] = str(manifest_path)
        data["checkpoint_path"] = str(path)
        return data


__all__ = [
    "CheckpointContract",
    "CheckpointDependencyError",
    "CheckpointError",
    "CheckpointIntegrityError",
    "CheckpointContractError",
    "CheckpointManager",
    "CheckpointStateError",
    "MANIFEST_VERSION",
    "FORMAT_VERSION",
    "canonical_json",
    "capture_rng_state",
    "normalize_contract",
    "restore_rng_state",
    "sha256_bytes",
    "sha256_file",
    "sha256_json",
]
