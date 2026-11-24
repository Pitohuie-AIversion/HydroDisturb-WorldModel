"""V1.3 Route A 3D Stage 1 models (`src/flow3d/models`)."""

from .dc_ae3d import (
    C2S3d,
    DCAE3DDecoder,
    DCAE3DEncoder,
    DownBlock3D,
    GeometryInject3D,
    LatentNorm3D,
    ResBlock3D,
    S2C3d,
    UpBlock3D,
    torch_available,
)
from .heads import (
    DirectFieldHead3D,
    ElasticRuntimePlan,
    Flow3DStage1Autoencoder,
    Flow3DStage1Model,
    PhysicalReadout3D,
    VectorPotentialCurlHead3D,
    configure_elastic_hardware,
    resolve_elastic_runtime_plan,
)
from .probe import LatentPhysicsProbe
from .spatial_transformer import SinusoidalPosEmb3D, SpatialTransformer3D, SpatialTransformerBlock3D

__all__ = [
    "C2S3d",
    "DCAE3DDecoder",
    "DCAE3DEncoder",
    "DirectFieldHead3D",
    "DownBlock3D",
    "ElasticRuntimePlan",
    "Flow3DStage1Autoencoder",
    "Flow3DStage1Model",
    "GeometryInject3D",
    "LatentNorm3D",
    "LatentPhysicsProbe",
    "PhysicalReadout3D",
    "ResBlock3D",
    "S2C3d",
    "SinusoidalPosEmb3D",
    "SpatialTransformer3D",
    "SpatialTransformerBlock3D",
    "UpBlock3D",
    "VectorPotentialCurlHead3D",
    "configure_elastic_hardware",
    "resolve_elastic_runtime_plan",
    "torch_available",
]
