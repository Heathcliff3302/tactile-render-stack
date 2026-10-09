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

There are **two** such ceilings, and clearing one says nothing about the
other:

``max_press_speed_mps``
    the clamp on the command the regulator may emit;

``force_ki_m_per_ns2 * force_integral_limit_ns``
    the command speed the integrator can supply once its own limit saturates.

The second one is easy to miss because it is a product of two separately
declared numbers, neither of which looks like a speed. See
``assess_regulator_authority`` for why it is the binding constraint at steady
state.

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

#: Largest closed-loop proportional gain admitted, as a fraction of the
#: stability boundary derived in ``assess_proportional_stability``. The
#: boundary itself is where the loop starts ringing instead of converging, so
#: a specification is required to sit below it rather than on it. Measured
#: behaviour at the boundary: ``a*kp = 0.99`` still settles, ``a*kp = 0.9999``
#: does not.
PROPORTIONAL_STABILITY_MARGIN = 0.9


class UnreachableForceTarget(ValueError):
    """A declared force target its own declared limits cannot reach."""


class ProportionalGainUnstable(ValueError):
    """A proportional gain that rings instead of converging at this rate.

    Not a subclass of :class:`UnreachableForceTarget`: the target is perfectly
    reachable, the loop just will not settle on it.
    """


class InsufficientRegulatorAuthority(UnreachableForceTarget):
    """A PI regulator whose integral limit cannot hold the declared target.

    Separate from :class:`UnreachableForceTarget` because the remedy differs:
    the press clamp is a drive limit, this one is a gain and an integral
    limit. Still a subclass, because the user-visible consequence is the same
    — the declared target cannot be reached.
    """


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


def derive_force_ki(
    *, mass_kg: float, target_force_n: float, output_dt_s: float,
    integral_limit_ns: float, headroom: float = DERIVED_HEADROOM_FACTOR,
) -> float:
    """Integral gain whose saturated authority admits ``headroom`` x the target.

    Inverts the steady-state condition derived in
    :func:`assess_regulator_authority`: ``ki * integral_limit`` must supply the
    command speed the target needs.
    """
    if headroom < 1.0:
        raise ValueError("Headroom below 1.0 cannot reach the target at all")
    if integral_limit_ns <= 0:
        raise ValueError("A non-positive integral limit gives the regulator no authority")
    return command_speed_for_force_mps(
        mass_kg=mass_kg, force_n=headroom * target_force_n, output_dt_s=output_dt_s
    ) / integral_limit_ns


def saturated_force_ceiling_n(
    *, mass_kg: float, output_dt_s: float, target_force_n: float,
    force_kp_m_per_ns: float, force_ki_m_per_ns2: float, integral_limit_ns: float,
) -> float:
    """Force the PI law settles on once its integral saturates.

    With ``a = m / dt`` the force per unit commanded speed, a saturated
    integral makes the loop a static map whose fixed point is::

        F = a * (kp * (T - F) + ki * I_limit)
          = a * (kp * T + ki * I_limit) / (1 + a * kp)

    Reported because it is the number a reader sees in a failing run, and
    because it is *not* the same quantity as the press-clamp ceiling even when
    the two happen to coincide numerically.
    """
    a = mass_kg / output_dt_s
    return a * (force_kp_m_per_ns * target_force_n + force_ki_m_per_ns2 * integral_limit_ns) / (
        1.0 + a * force_kp_m_per_ns
    )


def assess_proportional_stability(spec) -> dict | None:
    """Will the loop settle, or ring? ``None`` if not force mode.

    A third ceiling, and the only one that is not about reach. The force per
    unit commanded speed is ``a = mass / output_dt``, so with the declared
    one-step feedback delay the error recursion is::

        F_k = a * kp * (T - F_{k-1}) + (integral terms)

    whose homogeneous part is ``-a*kp``. It converges only while::

        a * kp < 1      i.e.     force_kp < output_dt / mass

    which makes the admissible gain *proportional to the output interval*.
    This is easy to miss because ``force_kp`` reads like a dimensionless
    tuning knob, and because the earlier authority derivation showed that kp
    does not bound the reachable force at all. It does not — it bounds
    whether the target is ever settled on.

    The declared specification is inside the bound at its declared rate and
    outside it at higher rates, which is the whole point of checking:

        60 Hz   a*kp = 0.600   settles
        120 Hz  a*kp = 1.200   rings, ends near 0.5 N
        240 Hz  a*kp = 2.400   rings, ends near 0.3 N

    Measured boundary: ``a*kp = 0.99`` still settles, ``0.9999`` does not, so
    the admitted gain is held at ``PROPORTIONAL_STABILITY_MARGIN`` of it
    rather than at the boundary itself.

    A pure-integral regulator (``force_kp = 0``) is unconditionally stable
    under this criterion and is reported as such.
    """
    data = spec.to_dict() if hasattr(spec, "to_dict") else spec
    force = data["controller"]["force"]
    if force is None:
        return None
    semantics = data["controller"]["drive_semantics"]
    if semantics != DRIVE_OVERWRITE_PER_CONTROL_STEP:
        raise NotImplementedError(
            f"Proportional stability is only derived for "
            f"{DRIVE_OVERWRITE_PER_CONTROL_STEP!r}, not {semantics!r}"
        )
    mass = data["probes"][0]["mass_kg"]
    output_dt = data["timing"]["output_dt_s"]
    kp = force["force_kp_m_per_ns"]
    force_per_speed = mass / output_dt
    loop_gain = force_per_speed * kp
    bound = output_dt / mass
    admitted = PROPORTIONAL_STABILITY_MARGIN * bound
    return {
        "schema": "proportional-stability/v1",
        "stability_law": "mass * force_kp / output_dt < 1",
        "force_kp_m_per_ns": kp,
        "force_per_commanded_speed_n_per_mps": force_per_speed,
        "proportional_loop_gain": loop_gain,
        "stability_boundary_force_kp": bound,
        "admitted_force_kp": admitted,
        "stability_margin_factor": PROPORTIONAL_STABILITY_MARGIN,
        "loop_gain_headroom": (
            (PROPORTIONAL_STABILITY_MARGIN / loop_gain) if loop_gain else None
        ),
        "unconditionally_stable": kp == 0,
        "stable": kp <= admitted,
        "feedback_delay_steps": data["controller"]["feedback_delay_steps"],
        "note": (
            "the admissible proportional gain is proportional to output_dt, so "
            "a gain that converges at one control rate can ring at another; "
            "this bounds convergence, not the reachable force"
        ),
    }


def enforce_proportional_stability(spec) -> dict | None:
    """Reject a proportional gain that will not settle at the declared rate."""
    report = assess_proportional_stability(spec)
    if report is None:
        return None
    if not report["stable"]:
        raise ProportionalGainUnstable(
            f"force_kp_m_per_ns={report['force_kp_m_per_ns']} gives a "
            f"proportional loop gain of {report['proportional_loop_gain']:.6f} "
            f"at output_dt={report['force_per_commanded_speed_n_per_mps']:.6g} N "
            f"per m/s. Convergence needs "
            f"{report['stability_law']}, so force_kp must be at most "
            f"{report['admitted_force_kp']:.6f} "
            f"({report['stability_margin_factor']} of the "
            f"{report['stability_boundary_force_kp']:.6f} boundary). Above it the "
            f"loop rings instead of settling and regulate_force times out. "
            f"{report['note']}"
        )
    return report


def assess_regulator_authority(spec) -> dict | None:
    """Can the PI regulator *hold* its target? ``None`` if not force mode.

    ``assess_force_reachability`` asks whether the drive could in principle
    emit a command large enough. This asks whether the regulator can still be
    emitting it once the error has gone to zero, which is a different question
    with a different answer.

    At steady state the proportional term vanishes with the error, so the
    command speed must come from the integrator alone, and the integrator is
    clamped at ``force_integral_limit_ns``. The requirement is therefore::

        force_ki * force_integral_limit  >=  F_target * output_dt / mass

    and it does not involve ``force_kp`` at all: solving the saturated fixed
    point for ``F >= T`` cancels every proportional term. A larger ``kp``
    raises the ceiling a failing regulator settles on without ever making the
    target reachable.

    Phase 1's inherited gains fail this by a factor of 1.67 under this
    kernel's drive semantics. They were never wrong for Phase 1: PhysX reports
    contact impulse that also carries penetration recovery, so the same
    integral authority bought more force there.

    One coincidence is worth disarming, because it invites a wrong diagnosis.
    The inherited gains settle at 0.75 N, and Phase 1's press limit of
    0.25 m/s *also* tops out at 0.75 N. Those are two unrelated ceilings that
    happen to meet: the first is the PI fixed point above, the second is
    ``frame_force_n`` at the clamp. The derived K2 press limit is 0.5 m/s, so
    the clamp is not what binds here.
    """
    data = spec.to_dict() if hasattr(spec, "to_dict") else spec
    force = data["controller"]["force"]
    if force is None:
        return None
    semantics = data["controller"]["drive_semantics"]
    if semantics != DRIVE_OVERWRITE_PER_CONTROL_STEP:
        raise NotImplementedError(
            f"Regulator authority is only derived for "
            f"{DRIVE_OVERWRITE_PER_CONTROL_STEP!r}, not {semantics!r}"
        )
    mass = data["probes"][0]["mass_kg"]
    output_dt = data["timing"]["output_dt_s"]
    target = force["target_force_n"]
    tolerance = force["force_tolerance_n"]
    integral_limit = force["force_integral_limit_ns"]
    ki = force["force_ki_m_per_ns2"]
    kp = force["force_kp_m_per_ns"]
    press_limit = force["max_press_speed_mps"]

    def speed(value):
        return command_speed_for_force_mps(mass_kg=mass, force_n=value, output_dt_s=output_dt)

    integral_authority = ki * integral_limit
    # The regulator can only use the smaller of what the integrator supplies
    # and what the drive clamp admits.
    effective = min(integral_authority, press_limit)
    required_with_tolerance = speed(target + MIN_HEADROOM_FACTOR * tolerance)
    ceiling = saturated_force_ceiling_n(
        mass_kg=mass, output_dt_s=output_dt, target_force_n=target,
        force_kp_m_per_ns=kp, force_ki_m_per_ns2=ki, integral_limit_ns=integral_limit,
    )
    return {
        "schema": "regulator-authority/v1",
        "drive_semantics": semantics,
        "steady_state_law": "ki * force_integral_limit >= F_target * output_dt / mass",
        "force_kp_m_per_ns": kp,
        "force_ki_m_per_ns2": ki,
        "force_integral_limit_ns": integral_limit,
        "integral_authority_mps": integral_authority,
        "declared_press_limit_mps": press_limit,
        "effective_press_authority_mps": effective,
        "binding_limit": "integral" if integral_authority <= press_limit else "press_clamp",
        "required_press_speed_mps": speed(target),
        "required_press_speed_with_tolerance_mps": required_with_tolerance,
        "saturated_force_ceiling_n": ceiling,
        "required_force_ki_m_per_ns2": derive_force_ki(
            mass_kg=mass, target_force_n=target, output_dt_s=output_dt,
            integral_limit_ns=integral_limit, headroom=1.0,
        ),
        "recommended_force_ki_m_per_ns2": derive_force_ki(
            mass_kg=mass, target_force_n=target, output_dt_s=output_dt,
            integral_limit_ns=integral_limit,
        ),
        "target_holdable": effective >= speed(target),
        "tolerance_band_holdable": effective >= required_with_tolerance,
        "kp_independence_note": (
            "force_kp does not appear in the condition: it cancels when the "
            "saturated fixed point is solved for F >= target. Raising kp "
            "raises saturated_force_ceiling_n without making the target "
            "reachable"
        ),
    }


def enforce_regulator_authority(spec) -> dict | None:
    """Reject a PI regulator that cannot hold its own declared target."""
    report = assess_regulator_authority(spec)
    if report is None:
        return None
    if not report["tolerance_band_holdable"]:
        raise InsufficientRegulatorAuthority(
            f"force_ki_m_per_ns2={report['force_ki_m_per_ns2']} with "
            f"force_integral_limit_ns={report['force_integral_limit_ns']} supplies at most "
            f"{report['integral_authority_mps']:.6f} m/s once the integrator saturates, and "
            f"the drive admits {report['declared_press_limit_mps']} m/s, so the regulator "
            f"commands at most {report['effective_press_authority_mps']:.6f} m/s. Holding "
            f"the declared target within its tolerance needs "
            f"{report['required_press_speed_with_tolerance_mps']:.6f} m/s. The loop would "
            f"settle at {report['saturated_force_ceiling_n']:.6f} N and time out in "
            f"regulate_force. Raise force_ki_m_per_ns2 to at least "
            f"{report['required_force_ki_m_per_ns2']:.6f} "
            f"(recommended {report['recommended_force_ki_m_per_ns2']:.6f}), or raise "
            f"force_integral_limit_ns. {report['kp_independence_note']}"
        )
    return report


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
