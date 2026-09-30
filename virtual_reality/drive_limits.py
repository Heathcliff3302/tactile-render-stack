"""Is a declared force target reachable under the declared drive semantics?

Layer 1 owns the drive, so it owns this question. With
``overwrite_linear_velocity_once_per_control_step`` the frame normal force is
set by the momentum the command injects:

    F = m * v_command / output_dt

so a force target implies a command speed, and a command-speed limit implies a
maximum reachable force. Declaring a target above that ceiling asks the
regulator for something its own limits forbid. The regulator would saturate,
never reach the target, and the run would fail its force gate with no
indication that the *specification* was inconsistent rather than the solver.

This is the same failure shape as the acceptance ledger and the field policy:
a declared number that nothing checks. So it is checked here, at load.

Phase 1 reached 1 N with a 0.25 m/s press limit because PhysX reports contact
impulse that also carries penetration recovery. This kernel removes
penetration by geometric projection and records it apart from the contact
impulse, so that contribution does not exist here. The two backends therefore
need different press limits for the same force target, and that is a real
difference between them rather than a parameter to be copied across.
"""

from __future__ import annotations

from contact_kernel import DRIVE_OVERWRITE_PER_CONTROL_STEP

#: How far above the target the press limit must reach. A limit that admits
#: exactly the target leaves the regulator no room at the top of its tolerance
#: band, so the requirement is the target plus its tolerance.
MIN_HEADROOM_FACTOR = 1.0

#: Headroom used when deriving a press limit for a new specification.
DERIVED_HEADROOM_FACTOR = 1.5


class UnreachableForceTarget(ValueError):
    """A declared force target its own declared limits cannot reach."""


def frame_force_n(*, mass_kg: float, speed_mps: float, output_dt_s: float) -> float:
    """Frame normal force produced by one commanded normal speed."""
    return mass_kg * speed_mps / output_dt_s


def command_speed_for_force_mps(*, mass_kg: float, force_n: float, output_dt_s: float) -> float:
    """Commanded normal speed that produces one frame normal force."""
    return force_n * output_dt_s / mass_kg


def derive_press_limit_mps(
    *, mass_kg: float, target_force_n: float, output_dt_s: float,
    headroom: float = DERIVED_HEADROOM_FACTOR,
) -> float:
    """Press limit admitting ``headroom`` times the target force."""
    if headroom < 1.0:
        raise ValueError("Headroom below 1.0 cannot reach the target at all")
    return command_speed_for_force_mps(
        mass_kg=mass_kg, force_n=headroom * target_force_n, output_dt_s=output_dt_s
    )


def assess_force_reachability(spec) -> dict | None:
    """Report whether a force target is reachable. ``None`` if not force mode."""
    data = spec.to_dict() if hasattr(spec, "to_dict") else spec
    force = data["controller"]["force"]
    if force is None:
        return None
    semantics = data["controller"]["drive_semantics"]
    if semantics != DRIVE_OVERWRITE_PER_CONTROL_STEP:
        raise NotImplementedError(
            f"Force reachability is only derived for "
            f"{DRIVE_OVERWRITE_PER_CONTROL_STEP!r}, not {semantics!r}"
        )
    mass = data["probes"][0]["mass_kg"]
    output_dt = data["timing"]["output_dt_s"]
    target = force["target_force_n"]
    tolerance = force["force_tolerance_n"]
    press_limit = force["max_press_speed_mps"]

    def speed(value):
        return command_speed_for_force_mps(mass_kg=mass, force_n=value, output_dt_s=output_dt)

    required = speed(target)
    required_with_tolerance = speed(target + MIN_HEADROOM_FACTOR * tolerance)
    ceiling = frame_force_n(mass_kg=mass, speed_mps=press_limit, output_dt_s=output_dt)
    return {
        "schema": "force-reachability/v1",
        "drive_semantics": semantics,
        "force_law": "F = mass * commanded_normal_speed / output_dt",
        "mass_kg": mass,
        "output_dt_s": output_dt,
        "target_force_n": target,
        "force_tolerance_n": tolerance,
        "required_press_speed_mps": required,
        "required_press_speed_with_tolerance_mps": required_with_tolerance,
        "declared_press_limit_mps": press_limit,
        "max_reachable_force_n": ceiling,
        "headroom_factor": ceiling / target if target else None,
        "target_reachable": press_limit >= required,
        "tolerance_band_reachable": press_limit >= required_with_tolerance,
        "note": (
            "the frame force is momentum injected by the once-per-control-step "
            "velocity overwrite, not a material contact force; changing the "
            "control rate changes the speed a given force needs"
        ),
    }


def enforce_force_reachability(spec) -> dict | None:
    """Reject a force target its declared press limit cannot reach."""
    report = assess_force_reachability(spec)
    if report is None:
        return None
    if not report["tolerance_band_reachable"]:
        raise UnreachableForceTarget(
            f"target_force_n={report['target_force_n']} with tolerance "
            f"{report['force_tolerance_n']} needs a press speed of "
            f"{report['required_press_speed_with_tolerance_mps']:.6f} m/s under "
            f"{report['force_law']}, but max_press_speed_mps is "
            f"{report['declared_press_limit_mps']}, which reaches at most "
            f"{report['max_reachable_force_n']:.6f} N"
        )
    return report
