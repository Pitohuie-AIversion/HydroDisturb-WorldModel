"""FlowWorld-3D V1.3 (Route A) core package."""

from .contracts import (
    CYLINDER3D_STATE_SPEC,
    Flow3DBatch,
    Flow3DConditions,
    PhysicsContractError,
    SampleProvenance,
    StateSpec3D,
)

__all__ = [
    "CYLINDER3D_STATE_SPEC",
    "Flow3DBatch",
    "Flow3DConditions",
    "PhysicsContractError",
    "SampleProvenance",
    "StateSpec3D",
]
