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

import datetime as _dt
import errno
import hashlib
import io
import json
import os
import random
import shutil
import tempfile
from collections import OrderedDict
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from string import hexdigits
from typing import Any, Callable, Mapping

from .manifest import hash_cache_index


try:  # Optional for tooling that only needs contracts/manifests.
    import numpy as np
except Exception:  # pragma: no cover - exercised only without numpy installed
    np = None  # type: ignore[assignment]

try:  # Optional so importing the package does not require a GPU/PyTorch image.
    import torch
except Exception:  # pragma: no cover - exercised only without torch installed
    torch = None  # type: ignore[assignment]


FORMAT_VERSION = 2
MANIFEST_VERSION = 2


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
        cpu_rng = state["torch_cpu"]
        if hasattr(cpu_rng, "detach"):
            cpu_rng = cpu_rng.detach().to(device="cpu", dtype=t.uint8)
        t.set_rng_state(cpu_rng)
    cuda_state = state.get("torch_cuda")
    if cuda_state is not None and t.cuda.is_available():
        num_devices = int(t.cuda.device_count())
        seq = list(cuda_state) if isinstance(cuda_state, (list, tuple)) else [cuda_state]
        for dev_idx in range(min(len(seq), num_devices)):
            dev_rng = seq[dev_idx]
            if hasattr(dev_rng, "detach"):
                dev_rng = dev_rng.detach().to(device="cpu", dtype=t.uint8)
            t.cuda.set_rng_state(dev_rng, device=dev_idx)


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


def _atomic_torch_save(payload: Mapping[str, Any], destination: Path) -> tuple[Path, str]:
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
        digest = sha256_file(temporary)
        # Keep the traditional epoch/step name for its first publisher. Duplicate
        # steps (including concurrent saves) get an immutable content address.
        try:
            os.link(temporary, destination)
        except OSError as exc:
            if exc.errno not in (errno.EEXIST, errno.EPERM, errno.ENOTSUP, errno.ENOSYS):
                raise
            destination = destination.with_name(f"{destination.stem}_{digest}{destination.suffix}")
            os.replace(temporary, destination)
        _fsync_directory(destination.parent)
        return destination, digest
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


_NUMPY_STATE_TAG = "__stage1_numpy_state_v1__"


def _pack_data_state(value: Any) -> Any:
    """Keep sampler state readable by weights_only on every supported Torch version."""
    if np is not None and isinstance(value, (np.ndarray, np.generic)):
        if value.dtype.hasobject:
            raise CheckpointStateError("sampler/dataloader NumPy object dtype cannot be safely restored")
        if np.ma.isMaskedArray(value):
            raise CheckpointStateError("store the masked sampler array and its mask separately")
        buffer = io.BytesIO()
        np.save(buffer, value, allow_pickle=False)
        kind = "scalar" if isinstance(value, np.generic) else "array"
        return {_NUMPY_STATE_TAG: kind, "data": buffer.getvalue()}
    if type(value) in (dict, OrderedDict):
        # A caller's dictionary can itself contain the tag, or a NumPy scalar
        # key. Store pairs in that case so both keys and values round-trip.
        simple_keys = (str, int, float, bool, bytes, type(None))
        if type(value) is OrderedDict or _NUMPY_STATE_TAG in value or any(type(k) not in simple_keys for k in value):
            pairs = [(_pack_data_state(k), _pack_data_state(v)) for k, v in value.items()]
            kind = "ordered_mapping" if type(value) is OrderedDict else "mapping"
            return {_NUMPY_STATE_TAG: kind, "data": pairs}
        return {k: _pack_data_state(v) for k, v in value.items()}
    if type(value) in (list, tuple):
        items = [_pack_data_state(v) for v in value]
        return tuple(items) if isinstance(value, tuple) else items
    if value is None or type(value) in (str, bytes, int, float, bool):
        return value
    t = _require_torch()
    if type(value) is t.Tensor:
        return value.clone()
    # Preserve less common native Torch state types (e.g. Size or device)
    # exactly when this version's weights-only loader supports them.
    buffer = io.BytesIO()
    try:
        t.save(value, buffer)
        buffer.seek(0)
        return t.load(buffer, weights_only=True)
    except Exception as exc:
        raise CheckpointStateError(f"unsupported sampler/dataloader state type {type(value).__name__}") from exc


def _unpack_data_state(value: Any) -> Any:
    if isinstance(value, dict):
        if _NUMPY_STATE_TAG not in value:
            return {k: _unpack_data_state(v) for k, v in value.items()}
        kind = value[_NUMPY_STATE_TAG]
        if kind in ("mapping", "ordered_mapping"):
            cls = OrderedDict if kind == "ordered_mapping" else dict
            return cls((_unpack_data_state(k), _unpack_data_state(v)) for k, v in value["data"])
        if kind not in ("array", "scalar"):
            raise CheckpointStateError(f"unknown sampler/dataloader state tag {kind!r}")
        if np is None:
            raise CheckpointDependencyError("NumPy is required to restore NumPy sampler/dataloader state")
        raw = value["data"]
        arr = np.load(io.BytesIO(raw), allow_pickle=False, max_header_size=len(raw))
        return arr[()] if kind == "scalar" else arr
    if type(value) in (list, tuple):
        items = [_unpack_data_state(v) for v in value]
        return tuple(items) if isinstance(value, tuple) else items
    return value


def _safe_torch_load(path: Path, map_location: Any = "cpu") -> dict[str, Any]:
    payload = _require_torch().load(path, map_location=map_location, weights_only=True)
    if not isinstance(payload, dict):
        raise CheckpointError(f"checkpoint {path} does not contain a mapping payload")
    if "data_state_format" in payload:
        if payload["data_state_format"] != 1:
            raise CheckpointStateError("unsupported sampler/dataloader state encoding")
        for key in ("sampler_state", "dataloader_state"):
            payload[key] = _unpack_data_state(payload[key])
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
        ema_state_dict: Mapping[str, Any] | None = None,
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
            "ema_state_dict": ema_state_dict,
            "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
            "state_presence": {
                "optimizer": optimizer is not None,
                "scheduler": scheduler is not None,
                "scaler": scaler is not None,
            },
            "rng_state": capture_rng_state() if include_rng else None,
            "data_state_format": 1,
            "sampler_state": _pack_data_state(sampler_state),
            "dataloader_state": _pack_data_state(dataloader_state),
            "extra_state": _jsonable(dict(extra_state or {})),
            "config": config_value,
            "config_sha256": sha256_json(config_value),
            "metrics": metrics_value,
            "contract": contract_value,
            "contract_sha256": sha256_json(contract_value),
        }

        destination = self._make_filename(epoch, global_step)
        destination, digest = _atomic_torch_save(payload, destination)

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
            if candidate.name == self.best_pointer_path.name:
                pointer_path = self.best_pointer_path if candidate.parent == Path(".") else candidate
            else:
                if not candidate.is_absolute() and candidate.parent == Path("."):
                    candidate = self.save_dir / candidate
                return candidate, None
        if not pointer_path.exists():
            raise FileNotFoundError(f"best checkpoint pointer not found: {pointer_path}")
        try:
            pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
            target = pointer_path.parent / str(pointer["checkpoint"])
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

    def restore_best_pointer(self, path: Path, digest: str, contract_hash: str) -> None:
        """Carry an immutable historical best into the directory of a resumed run."""
        if self.best_pointer_path.is_file():
            _, current_hash = self._resolve_path("best")
            if current_hash == digest:
                return
        if sha256_file(path) != digest:
            raise CheckpointIntegrityError(f"historical best checkpoint hash mismatch: {path}")
        target = self.save_dir / path.name
        if target.resolve() != path.resolve():
            if target.exists() and sha256_file(target) != digest:
                target = target.with_name(f"{target.stem}_{digest}{target.suffix}")
            if not target.exists():
                # A cross-filesystem fork must finish its copy before the pointer is published.
                with tempfile.NamedTemporaryFile(dir=self.save_dir) as tmp:
                    with path.open("rb") as src:
                        shutil.copyfileobj(src, tmp)
                    tmp.flush()
                    os.fsync(tmp.fileno())
                    os.link(tmp.name, target)
            if sha256_file(target) != digest:
                raise CheckpointIntegrityError(f"copied best checkpoint hash mismatch: {target}")
        pointer = {
            "pointer_version": 1, "model_name": self.model_name, "checkpoint": target.name,
            "sha256": digest, "checkpoint_sha256": digest, "checkpoint_id": digest[:16],
            "contract_sha256": contract_hash, "updated_at_utc": _utc_now(),
        }
        _atomic_write_bytes((canonical_json(pointer) + "\n").encode(), self.best_pointer_path)

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
        use_ema: bool = False,
        validate_payload: Callable[[Mapping[str, Any]], None] | None = None,
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
        if validate_payload is not None:
            validate_payload(payload)
        weights = payload.get("ema_state_dict") if use_ema else None
        model.load_state_dict(payload["model_state_dict"] if weights is None else weights, strict=strict)

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

        payload["loaded_path"] = str(path.resolve())
        payload["loaded_sha256"] = digest
        return payload

    def _frozen_identity(
        self, checkpoint: str | os.PathLike[str], expected_contract: Any | None,
        expected_hash: str | None = None,
    ) -> tuple[Path, dict[str, str]]:
        path, pointer_hash = self._resolve_path(checkpoint)
        digest = sha256_file(path)
        if (pointer_hash is not None and digest != pointer_hash) or (expected_hash is not None and digest != expected_hash):
            raise CheckpointIntegrityError(f"checkpoint hash mismatch for {path}")
        payload = _safe_torch_load(path)
        if payload.get("format_version") != self.schema_version:
            raise CheckpointContractError("unsupported checkpoint format")
        contract = normalize_contract(payload.get("contract"))
        contract_hash = sha256_json(contract)
        if contract_hash != payload.get("contract_sha256"):
            raise CheckpointIntegrityError("checkpoint contract does not match its stored hash")
        expected = normalize_contract(expected_contract) if expected_contract is not None else (self.contract or None)
        if expected is not None and sha256_json(expected) != contract_hash:
            raise CheckpointContractError("frozen manifest checkpoint contract mismatch")
        for key in ("latent_layout", "normalizer_version", "dataset_manifest_sha256"):
            if not isinstance(contract.get(key), str) or not contract[key].strip():
                raise CheckpointContractError(f"Stage 2 export requires checkpoint contract field {key}")
        dataset_hash = contract["dataset_manifest_sha256"]
        if len(dataset_hash) != 64 or any(c not in hexdigits for c in dataset_hash):
            raise CheckpointContractError("dataset_manifest_sha256 must be a SHA-256 hex digest")
        return path, {
            "checkpoint_sha256": digest,
            "checkpoint_id": digest[:16],
            "contract_sha256": contract_hash,
            "latent_source": "z_clean",
            "latent_layout": contract["latent_layout"],
            "normalizer_version": contract["normalizer_version"],
            "dataset_manifest_sha256": dataset_hash,
            "export_mode": "eval",
        }

    @staticmethod
    def _cache_files(index: Path, identity: Mapping[str, str]) -> tuple[str, dict[str, str]]:
        try:
            return hash_cache_index(index, identity["checkpoint_sha256"], identity["contract_sha256"])
        except (OSError, ValueError) as exc:
            raise CheckpointIntegrityError(f"cannot verify cache index {index}: {exc}") from exc

    def create_frozen_manifest(
        self,
        checkpoint: str | os.PathLike[str],
        latent_source: str = "z_clean",
        latent_layout: str | None = None,
        normalizer_version: str | None = None,
        dataset_manifest_sha256: str | None = None,
        export_mode: str = "eval",
        contract: Any | None = None,
        output_path: str | os.PathLike[str] | None = None,
        *,
        cache_index: str | os.PathLike[str],
    ) -> dict[str, Any]:
        """Publish identities and file hashes for a completed z_clean export.

        Relative cache_index paths are resolved from the output manifest directory
        (save_dir for an in-memory manifest). Shards are relative to their index.
        """
        path, identity = self._frozen_identity(checkpoint, contract)
        for key, supplied in {
            "latent_source": latent_source, "latent_layout": latent_layout,
            "normalizer_version": normalizer_version, "dataset_manifest_sha256": dataset_manifest_sha256,
            "export_mode": export_mode,
        }.items():
            if supplied != identity[key] and (supplied is not None or key in ("latent_source", "export_mode")):
                raise CheckpointContractError(f"frozen export {key} must match {identity[key]!r}")
        output = self.save_dir / output_path if output_path is not None else None
        root = (output.parent if output is not None else self.save_dir).resolve()
        index = (root / cache_index).resolve()
        if not index.is_relative_to(root):
            raise CheckpointContractError("cache index must be under the manifest directory")
        index_digest, shards = self._cache_files(index, identity)
        manifest = {
            "manifest_version": MANIFEST_VERSION,
            "checkpoint": path.name,
            **identity,
            "cache_index": index.relative_to(root).as_posix(),
            "cache_index_sha256": index_digest,
            "shards_sha256": shards,
            "created_at_utc": _utc_now(),
        }
        if output is not None:
            inputs = {path.resolve(), self.best_pointer_path.resolve(), index}
            inputs.update((index.parent / name).resolve() for name in shards)
            if output.resolve() in inputs:
                raise CheckpointContractError("output manifest must be separate from the checkpoint, index and shards")
            _atomic_write_bytes((canonical_json(manifest) + "\n").encode("utf-8"), output)
        return manifest

    def verify_frozen_manifest(
        self,
        manifest: Mapping[str, Any] | str | os.PathLike[str],
        checkpoint: str | os.PathLike[str] | None = None,
        expected_contract: Any | None = None,
        *,
        cache_root: str | os.PathLike[str] | None = None,
    ) -> dict[str, Any]:
        """Validate an export and return the verified index and ordered shard paths.

        For an in-memory manifest, cache_root defaults to save_dir. For an on-disk
        manifest, paths are always anchored to its directory.
        """
        if isinstance(manifest, (str, os.PathLike)):
            manifest_path = Path(manifest)
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
            root = manifest_path.resolve().parent
        else:
            manifest_path = None
            data = dict(manifest)
            root = Path(cache_root if cache_root is not None else self.save_dir).resolve()
        if not isinstance(data, dict) or data.get("manifest_version") != MANIFEST_VERSION:
            raise CheckpointContractError("Stage 2 needs a v2 frozen manifest; re-export with a cache index")
        checkpoint_ref = checkpoint or data.get("checkpoint")
        if not checkpoint_ref:
            raise CheckpointIntegrityError("frozen manifest does not identify a checkpoint")
        path, identity = self._frozen_identity(checkpoint_ref, expected_contract, str(data.get("checkpoint_sha256", "")))
        for key, expected in identity.items():
            if data.get(key) != expected:
                raise CheckpointContractError(f"frozen manifest {key} mismatch")
        index_name = data.get("cache_index")
        if not isinstance(index_name, str) or not index_name or Path(index_name).is_absolute():
            raise CheckpointIntegrityError("cache_index must be a relative path")
        index = (root / index_name).resolve()
        if not index.is_relative_to(root):
            raise CheckpointIntegrityError("cache index is outside the manifest directory")
        index_digest, shards = self._cache_files(index, identity)
        if data.get("cache_index_sha256") != index_digest or data.get("shards_sha256") != shards:
            raise CheckpointIntegrityError("cache index or shard hashes do not match the frozen manifest")
        if manifest_path is not None:
            data["manifest_path"] = str(manifest_path)
        data["checkpoint_path"] = str(path)
        data["cache_index_path"] = str(index)
        data["shard_paths"] = [str(index.parent / name) for name in shards]
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
