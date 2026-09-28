"""Layer 2: contact backends adapted to the shared frame contract."""

from .contracts import INTERACTION_STATE_SCHEMA, InteractionState
from .kernel_backend import (
    ALWAYS_AVAILABLE,
    KINEMATICS_DEPENDENT,
    OBSERVABLE_METHODS,
    OBSERVABLE_STATUS,
    OBSERVABLE_SUPPORT,
    PRESSURE_SEMANTICS,
    KernelInteractionStage,
    always_available,
    manifest_observables,
    observable_methods,
)

__all__ = [
    "ALWAYS_AVAILABLE",
    "INTERACTION_STATE_SCHEMA",
    "InteractionState",
    "KINEMATICS_DEPENDENT",
    "KernelInteractionStage",
    "OBSERVABLE_METHODS",
    "OBSERVABLE_STATUS",
    "OBSERVABLE_SUPPORT",
    "PRESSURE_SEMANTICS",
    "always_available",
    "manifest_observables",
    "observable_methods",
]
