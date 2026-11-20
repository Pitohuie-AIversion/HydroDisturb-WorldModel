from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence


def _parse_simple_yaml(text: str) -> dict[str, Any]:
    """Minimal zero-dependency YAML parser for nested mapping + scalar/list configs."""
    try:
        import yaml  # type: ignore[import-untyped]
        loaded = yaml.safe_load(text)
        return dict(loaded) if isinstance(loaded, Mapping) else {}
    except Exception:
        pass

    def _parse_scalar(val: str) -> Any:
        val = val.strip()
        if val.startswith("[") and val.endswith("]"):
            inner = val[1:-1].strip()
            if not inner:
                return []
            return [_parse_scalar(item) for item in inner.split(",")]
        if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
            return val[1:-1]
        low = val.lower()
        if low == "true":
            return True
        if low == "false":
            return False
        if low in ("null", "none", "~"):
            return None
        try:
            if "." in val or "e" in low:
                return float(val)
            return int(val)
        except ValueError:
            return val

    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        key, _, rest = line.strip().partition(":")
        key = key.strip()
        rest = rest.strip()
        while len(stack) > 1 and stack[-1][0] >= indent:
            stack.pop()
        parent = stack[-1][1]
        if not rest:
            child: dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = _parse_scalar(rest)
    return root


@dataclass(frozen=True)
class GridConfig:
    spatial_size: tuple[int, int, int] = (64, 96, 192)
    spacing: tuple[float, float, float] = (0.01875, 0.01875, 0.01875)
    mask_threshold: float = 0.5
    sdf_clip: float = 8.0


@dataclass(frozen=True)
class ModelConfig:
    latent_channels: int = 256
    downsample_factor: int = 8
    hard_mask_output: bool = True
    base_channels: int = 32


@dataclass(frozen=True)
class LossConfig:
    reconstruction_weight: float = 1.0
    divergence_weight: float = 0.0
    vorticity_weight: float = 0.0
    temporal_weight: float = 0.0
    probe_weight: float = 0.0


@dataclass(frozen=True)
class Stage1Config:
    schema_version: str = "stage1.v1"
    normalizer_version: str = "normalizer.v1"
    latent_contract_version: str = "latent.v1"
    grid: GridConfig = field(default_factory=GridConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "Stage1Config":
        grid_raw = raw.get("grid", {})
        model_raw = raw.get("model", {})
        loss_raw = raw.get("loss", {})
        grid = GridConfig(
            spatial_size=tuple(int(v) for v in grid_raw.get("spatial_size", GridConfig.spatial_size)),
            spacing=tuple(float(v) for v in grid_raw.get("spacing", GridConfig.spacing)),
            mask_threshold=float(grid_raw.get("mask_threshold", 0.5)),
            sdf_clip=float(grid_raw.get("sdf_clip", 8.0)),
        )
        if len(grid.spatial_size) != 3 or len(grid.spacing) != 3:
            raise ValueError("spatial_size and spacing must each contain exactly 3 values")
        model = ModelConfig(
            latent_channels=int(model_raw.get("latent_channels", 256)),
            downsample_factor=int(model_raw.get("downsample_factor", 8)),
            hard_mask_output=bool(model_raw.get("hard_mask_output", True)),
            base_channels=int(model_raw.get("base_channels", 32)),
        )
        if model.downsample_factor not in (1, 2, 4, 8):
            raise ValueError("downsample_factor must be one of 1, 2, 4, 8")
        loss = LossConfig(**{k: float(loss_raw.get(k, getattr(LossConfig(), k))) for k in (
            "reconstruction_weight", "divergence_weight", "vorticity_weight", "temporal_weight", "probe_weight"
        )})
        return cls(
            schema_version=str(raw.get("schema_version", "stage1.v1")),
            normalizer_version=str(raw.get("normalizer_version", "normalizer.v1")),
            latent_contract_version=str(raw.get("latent_contract_version", "latent.v1")),
            grid=grid,
            model=model,
            loss=loss,
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Stage1Config":
        text = Path(path).read_text(encoding="utf-8")
        return cls.from_mapping(_parse_simple_yaml(text))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "normalizer_version": self.normalizer_version,
            "latent_contract_version": self.latent_contract_version,
            "grid": {
                "spatial_size": list(self.grid.spatial_size),
                "spacing": list(self.grid.spacing),
                "mask_threshold": self.grid.mask_threshold,
                "sdf_clip": self.grid.sdf_clip,
            },
            "model": {
                "latent_channels": self.model.latent_channels,
                "downsample_factor": self.model.downsample_factor,
                "hard_mask_output": self.model.hard_mask_output,
                "base_channels": self.model.base_channels,
            },
            "loss": {
                "reconstruction_weight": self.loss.reconstruction_weight,
                "divergence_weight": self.loss.divergence_weight,
                "vorticity_weight": self.loss.vorticity_weight,
                "temporal_weight": self.loss.temporal_weight,
                "probe_weight": self.loss.probe_weight,
            },
        }


@dataclass(frozen=True)
class LatentContract:
    version: str
    shape: tuple[int, ...]
    layout: str = "B,T,C,Z,Y,X"
    source: str = "z_clean"
    normalizer_version: str = "normalizer.v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "shape": list(self.shape),
            "layout": self.layout,
            "source": self.source,
            "normalizer_version": self.normalizer_version,
        }

    def validate_shape(self, actual: Sequence[int]) -> None:
        if tuple(int(v) for v in actual) != self.shape:
            raise ValueError(f"latent shape mismatch: expected {self.shape}, got {tuple(actual)}")
