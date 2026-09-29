"""Kernel configuration. Plain dataclasses with no contract or backend imports.

The kernel is a standalone CPU solver. Mapping a versioned ``ExperimentSpec``
onto these dataclasses is the job of the layer 1 adapter, so the solver stays
usable without the contract package.
"""

from __future__ import annotations

from dataclasses import dataclass, field

Vec3 = tuple[float, float, float]

KERNEL_SCHEMA_VERSION = "contact-kernel/k1"
SPHERE = "sphere"
BOX = "box"
EDGE_REJECT = "reject_outside_top_face"
EDGE_CLAMP = "phase1_clamp"
DRIVE_OVERWRITE_PER_CONTROL_STEP = "overwrite_linear_velocity_once_per_control_step"
GRAVITY_COMPENSATION_PER_SUBSTEP = "per_substep_feedforward"
GRAVITY_COMPENSATION_NONE = "none"
KINEMATICS_RIGID_BODY = "rigid_body"
KINEMATICS_FINITE_DIFFERENCE = "finite_difference"
#: Method name recorded on every point and quality entry, per declared source.
#: Phase 1 named these the same way, so a parity comparison can line them up.
KINEMATICS_METHOD = {
    KINEMATICS_RIGID_BODY: "rigid_body_twist",
    KINEMATICS_FINITE_DIFFERENCE: "contact_point_finite_difference",
}


def _vec3(values) -> Vec3:
    x, y, z = (float(value) for value in values)
    return (x, y, z)


@dataclass(frozen=True)
class ProbeSpec:
    """One controlled probe body. K1 implements the sphere case only."""

    id: str
    shape: str
    mass_kg: float
    initial_position_world_m: Vec3
    radius_m: float | None = None
    size_m: Vec3 | None = None
    angular_mode: str = "locked"
    initial_linear_velocity_mps: Vec3 = (0.0, 0.0, 0.0)

    def __post_init__(self):
        object.__setattr__(self, "initial_position_world_m", _vec3(self.initial_position_world_m))
        object.__setattr__(self, "initial_linear_velocity_mps", _vec3(self.initial_linear_velocity_mps))
        if self.size_m is not None:
            object.__setattr__(self, "size_m", _vec3(self.size_m))
        if self.mass_kg <= 0:
            raise ValueError(f"{self.id}: probe mass must be positive")
        if self.angular_mode != "locked":
            raise ValueError(f"{self.id}: K1 only solves rotation-locked probes")
        if self.shape == SPHERE:
            if not self.radius_m or self.radius_m <= 0:
                raise ValueError(f"{self.id}: a sphere probe needs a positive radius")
        elif self.shape == BOX:
            raise NotImplementedError(
                f"{self.id}: flat-bottom box probes arrive with the K3 multi-probe solver"
            )
        else:
            raise ValueError(f"{self.id}: unsupported probe shape {self.shape!r}")

    @property
    def support_offset_m(self) -> float:
        """Distance from the body origin to the lowest support point."""
        return float(self.radius_m)


@dataclass(frozen=True)
class SurfaceSpec:
    """Axis-aligned fixed box. Only its top face participates in K1 contact."""

    id: str
    center_world_m: Vec3
    size_m: Vec3
    motion: str = "fixed"

    def __post_init__(self):
        object.__setattr__(self, "center_world_m", _vec3(self.center_world_m))
        object.__setattr__(self, "size_m", _vec3(self.size_m))
        if self.motion != "fixed":
            raise NotImplementedError("K1 solves a fixed reference surface only")
        if any(value <= 0 for value in self.size_m):
            raise ValueError("Surface extents must be positive")

    @property
    def top_z_world_m(self) -> float:
        return self.center_world_m[2] + self.size_m[2] / 2.0

    @property
    def half_extent_m(self) -> tuple[float, float]:
        return (self.size_m[0] / 2.0, self.size_m[1] / 2.0)


@dataclass(frozen=True)
class GridSpec:
    """Output discretisation of the surface top face in surface-local XY."""

    rows: int
    cols: int
    cell_size_m: float
    origin_local_m: Vec3
    edge_policy: str = EDGE_REJECT
    connectivity: int = 4

    def __post_init__(self):
        object.__setattr__(self, "origin_local_m", _vec3(self.origin_local_m))
        if self.rows < 1 or self.cols < 1:
            raise ValueError("Grid needs at least one row and column")
        if self.cell_size_m <= 0:
            raise ValueError("Grid cell size must be positive")
        if self.edge_policy not in (EDGE_REJECT, EDGE_CLAMP):
            raise ValueError(f"Unknown edge policy {self.edge_policy!r}")
        if self.connectivity != 4:
            raise NotImplementedError("K1 keeps the Step 6 four-neighbour baseline")

    @property
    def cell_area_m2(self) -> float:
        return self.cell_size_m**2


@dataclass(frozen=True)
class SamplingSpec:
    """Thresholds that turn solver output into recorded contact evidence."""

    force_threshold_n: float = 0.05
    point_match_distance_m: float = 0.04
    max_gap_frames: int = 3
    slide_threshold_mps: float = 0.005
    direction_threshold_m: float = 1.0e-4
    filter_alpha: float = 0.5
    kinematics_source: str = KINEMATICS_RIGID_BODY

    def __post_init__(self):
        if self.kinematics_source not in (KINEMATICS_RIGID_BODY, KINEMATICS_FINITE_DIFFERENCE):
            raise ValueError(f"Unknown kinematics source {self.kinematics_source!r}")
        if not 0.0 < self.filter_alpha <= 1.0:
            raise ValueError("filter_alpha must lie in (0, 1]")


@dataclass(frozen=True)
class SolverSpec:
    """Physical and numerical choices of the rigid impulse mode."""

    gravity_mps2: Vec3 = (0.0, 0.0, -9.81)
    restitution: float = 0.0
    static_friction: float = 0.0
    dynamic_friction: float = 0.0
    linear_damping_per_s: float = 0.0
    contact_offset_m: float = 0.0
    rest_offset_m: float = 0.0
    max_depenetration_velocity_mps: float = 0.0
    solver_type: str = "single_sphere_plane_normal_impulse"

    def __post_init__(self):
        object.__setattr__(self, "gravity_mps2", _vec3(self.gravity_mps2))
        if self.gravity_mps2[0] or self.gravity_mps2[1]:
            raise NotImplementedError("K1 solves gravity along world Z only")
        if self.restitution:
            raise NotImplementedError("Restitution arrives with the K4 material modes")
        if self.static_friction or self.dynamic_friction:
            raise NotImplementedError("Coulomb friction arrives after the K1 baseline")
        if self.max_depenetration_velocity_mps:
            # Penetration is removed by geometric projection, recorded apart
            # from the contact impulse. A positive limit would mean impulse
            # driven depenetration, which is a different strategy and is not
            # implemented; accepting the value would make it look effective.
            raise NotImplementedError(
                "Velocity-based depenetration is not implemented; only "
                "max_depenetration_velocity_mps = 0 is supported"
            )
        if self.linear_damping_per_s < 0:
            raise ValueError("Linear damping must be non-negative")


@dataclass(frozen=True)
class TimingSpec:
    physics_dt_s: float
    control_dt_s: float
    output_dt_s: float
    substeps_per_control: int

    def __post_init__(self):
        if min(self.physics_dt_s, self.control_dt_s, self.output_dt_s) <= 0:
            raise ValueError("Every time step must be positive")
        if self.substeps_per_control < 1:
            raise ValueError("substeps_per_control must be at least one")
        expected = self.physics_dt_s * self.substeps_per_control
        if abs(expected - self.control_dt_s) > 1.0e-12:
            raise ValueError("Substeps must exactly cover one control interval")
        if abs(self.output_dt_s - self.control_dt_s) > 1.0e-12:
            raise NotImplementedError("K1 records one frame per control step")


@dataclass(frozen=True)
class DriveSpec:
    """How a velocity command reaches the probe.

    ``gravity_compensation_mode`` is an explicit kernel choice, not a PhysX
    default. ``per_substep_feedforward`` cancels gravity at every substep, so
    the realised trajectory and the contact impulse stay invariant under
    substep refinement. The commanded velocity itself is still overwritten once
    per control step, which is the Step 6 drive semantics.
    """

    semantics: str = DRIVE_OVERWRITE_PER_CONTROL_STEP
    gravity_compensation_mps2: float = 9.81
    gravity_compensation_mode: str = GRAVITY_COMPENSATION_PER_SUBSTEP

    def __post_init__(self):
        if self.semantics != DRIVE_OVERWRITE_PER_CONTROL_STEP:
            raise NotImplementedError(f"Unsupported drive semantics {self.semantics!r}")
        if self.gravity_compensation_mode not in (
            GRAVITY_COMPENSATION_PER_SUBSTEP,
            GRAVITY_COMPENSATION_NONE,
        ):
            raise ValueError(f"Unknown compensation mode {self.gravity_compensation_mode!r}")


@dataclass(frozen=True)
class KernelConfig:
    surface: SurfaceSpec
    probes: tuple[ProbeSpec, ...]
    timing: TimingSpec
    grid: GridSpec
    solver: SolverSpec = field(default_factory=SolverSpec)
    sampling: SamplingSpec = field(default_factory=SamplingSpec)
    drive: DriveSpec = field(default_factory=DriveSpec)
    backend_id: str = "rigid_cpu_v1"
    backend_version: str = "k1.0"

    def __post_init__(self):
        object.__setattr__(self, "probes", tuple(self.probes))
        if not self.probes:
            raise ValueError("At least one probe is required")
        ids = [probe.id for probe in self.probes]
        if len(set(ids)) != len(ids) or self.surface.id in ids:
            raise ValueError("Body IDs must be unique")
        expected = self.surface.size_m[0] / self.grid.cols
        if abs(self.grid.cell_size_m - expected) > 1.0e-12:
            raise ValueError("Grid cell size does not match the surface extent")

    @property
    def gravity_compensation_world(self) -> Vec3:
        if self.drive.gravity_compensation_mode == GRAVITY_COMPENSATION_NONE:
            return (0.0, 0.0, 0.0)
        return (0.0, 0.0, self.drive.gravity_compensation_mps2)

    def pair_ids(self) -> tuple[tuple[str, str], ...]:
        return tuple((probe.id, self.surface.id) for probe in self.probes)
