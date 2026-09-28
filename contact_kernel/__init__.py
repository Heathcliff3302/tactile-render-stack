"""Backend-independent CPU contact kernel.

This package is the decoupled replacement for the Isaac Sim dynamics and
contact reporting used in Phase 1. It imports only the standard library: no
simulator, no contract package, no layer module. Adapters convert versioned
specifications into :class:`~contact_kernel.spec.KernelConfig` and convert
kernel frames into contract records.

Responsibility split follows ``docs/contact_kernel_design.md``:

===================  ===========================================
``geometry``         analytic gap, normal, tangent basis, grid
``solver``           semi-implicit integration, normal impulse
``tracking``         point IDs, cells, patches, episodes
``kernel``           per-step assembly, area layers, conservation
===================  ===========================================
"""

from .geometry import SurfaceGrid, TopFaceGeometry
from .kernel import AreaLayers, Conservation, KernelFrame, PairFrame, RigidContactKernel
from .solver import BodyState, ProbeSolution, RigidImpulseSolver, SubstepTrace
from .spec import (
    DRIVE_OVERWRITE_PER_CONTROL_STEP,
    GRAVITY_COMPENSATION_NONE,
    GRAVITY_COMPENSATION_PER_SUBSTEP,
    KERNEL_SCHEMA_VERSION,
    KINEMATICS_FINITE_DIFFERENCE,
    KINEMATICS_METHOD,
    KINEMATICS_RIGID_BODY,
    DriveSpec,
    GridSpec,
    KernelConfig,
    ProbeSpec,
    SamplingSpec,
    SolverSpec,
    SurfaceSpec,
    TimingSpec,
)
from .tracking import (
    GEOMETRIC_FOOTPRINT,
    POINT_OCCUPANCY,
    SWEPT_PATH,
    ContactCell,
    ContactEpisode,
    ContactPatch,
    ContactPoint,
)

__all__ = [
    "AreaLayers",
    "BodyState",
    "Conservation",
    "ContactCell",
    "ContactEpisode",
    "ContactPatch",
    "ContactPoint",
    "DRIVE_OVERWRITE_PER_CONTROL_STEP",
    "DriveSpec",
    "GEOMETRIC_FOOTPRINT",
    "GRAVITY_COMPENSATION_NONE",
    "GRAVITY_COMPENSATION_PER_SUBSTEP",
    "GridSpec",
    "KERNEL_SCHEMA_VERSION",
    "KINEMATICS_FINITE_DIFFERENCE",
    "KINEMATICS_METHOD",
    "KINEMATICS_RIGID_BODY",
    "KernelConfig",
    "KernelFrame",
    "POINT_OCCUPANCY",
    "PairFrame",
    "ProbeSolution",
    "ProbeSpec",
    "RigidContactKernel",
    "RigidImpulseSolver",
    "SWEPT_PATH",
    "SamplingSpec",
    "SolverSpec",
    "SubstepTrace",
    "SurfaceGrid",
    "SurfaceSpec",
    "TimingSpec",
    "TopFaceGeometry",
]
