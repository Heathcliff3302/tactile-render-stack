"""Measure what the drive can actually deliver, instead of deriving it.

``drive_limits`` answers the force question from a closed form::

    F = mass * commanded_speed / output_dt

That law is exact for the K1/K2 CPU reference, and only because every term it
omits happens to be zero there: no linear damping, gravity exactly cancelled
by the declared compensation, one substep per control step, no solver decay.
It is a statement about an *idealised* drive, and it keeps reporting
``target_reachable: true`` for specifications where those terms are not zero.

Setting ``linear_damping_per_s`` to 30 is enough to show it. Damping scales
the body velocity each substep before the contact solve, so the speed that
reaches the constraint is lower than the speed commanded::

    damping 0   ->  press limit 0.5 m/s delivers 1.5000 N
    damping 1   ->                       delivers 1.4750 N
    damping 5   ->                       delivers 1.3750 N
    damping 10  ->                       delivers 1.2500 N
    damping 30  ->                       delivers 0.7500 N   <- below the 1 N target

Every one of those loads cleanly and every one reports a static ceiling of
1.5 N. ``linear_damping_per_s`` is classified ``enforced`` precisely because
the kernel reads it, so this is not an unsupported setting being smuggled in
— it is a supported setting the static check cannot see.

So this module asks the kernel instead. It runs the real solver with the real
configuration, commands the declared press limit against the surface, and
reports the normal force the sampling layer actually reported. That covers
damping, gravity and its compensation, the substep structure, the output
interval and the sampling threshold, because all of them are in the path.

What it still does not cover is the closed loop: feedback delay, the PI gains
and the correction clamp shape *how* the regulator approaches the target, not
whether the target is inside the drive's reach. Those are checked by running
the loop — see the stability sweep in the K2 tests — and the two together are
what justify a reachability claim. This probe deliberately reports a measured
ceiling and nothing about convergence.
"""

from __future__ import annotations

from contact_kernel import RigidContactKernel

#: Control steps to settle into contact before measuring. The approach only
#: needs to cross the declared gap; this is a generous bound on that.
_APPROACH_STEPS = 2000

#: Control steps to average the delivered force over once in contact.
_MEASURE_STEPS = 10


class UnreachableUnderRealisedDrive(ValueError):
    """The kernel does not deliver the declared target under its own settings."""


def measure_force_ceiling(spec, *, kernel_config=None) -> dict:
    """Largest normal force the kernel reports at the declared press limit.

    Runs the real solver rather than a model of it. Returns the measured
    ceiling alongside the ideal-law prediction, so a divergence between them
    is visible rather than implicit.
    """
    from .drive_limits import frame_force_n
    from .world import kernel_config_from_spec

    data = spec.to_dict() if hasattr(spec, "to_dict") else spec
    force = data["controller"]["force"]
    if force is None:
        return None
    config = kernel_config or kernel_config_from_spec(spec, probe_realised_drive=False)
    probe = config.probes[0]
    probe_id = probe.id
    key = RigidContactKernel.pair_key(probe_id, config.surface.id)
    press_limit = force["max_press_speed_mps"]
    approach_speed = data["controller"]["parameters"]["approach_speed_mps"]

    kernel = RigidContactKernel(config)
    contacted = False
    approach_steps = 0
    for _ in range(_APPROACH_STEPS):
        frame = kernel.step({probe_id: (0.0, 0.0, -approach_speed)}, phase="approach_force")
        approach_steps += 1
        if frame.pairs[key].contact_present:
            contacted = True
            break

    measured = []
    if contacted:
        for _ in range(_MEASURE_STEPS):
            frame = kernel.step({probe_id: (0.0, 0.0, -press_limit)}, phase="regulate_force")
            measured.append(frame.pairs[key].normal_force_n or 0.0)

    ceiling = max(measured) if measured else 0.0
    ideal = frame_force_n(
        mass_kg=probe.mass_kg, speed_mps=press_limit,
        output_dt_s=config.timing.output_dt_s,
    )
    target = force["target_force_n"]
    tolerance = force["force_tolerance_n"]
    required = target + tolerance
    return {
        "schema": "realised-drive-authority/v1",
        "method": "kernel_probe_at_declared_press_limit",
        "contact_established": contacted,
        "approach_steps": approach_steps,
        "measured_steps": len(measured),
        "declared_press_limit_mps": press_limit,
        "measured_force_ceiling_n": ceiling,
        "ideal_law_force_ceiling_n": ideal,
        "ideal_minus_measured_n": ideal - ceiling,
        "realised_fraction_of_ideal": (ceiling / ideal) if ideal else None,
        "target_force_n": target,
        "force_tolerance_n": tolerance,
        "required_force_n": required,
        "target_reachable_under_realised_drive": contacted and ceiling >= target,
        "tolerance_band_reachable_under_realised_drive": contacted and ceiling >= required,
        "accounted_for": [
            "linear_damping_per_s",
            "gravity_mps2 and the declared gravity compensation",
            "substeps_per_control and physics_dt_s",
            "output_dt_s and the force definition over the output interval",
            "rest_offset_m and the contact geometry",
            "sampling.force_threshold_n",
        ],
        "not_accounted_for": [
            "feedback_delay_steps",
            "force_kp_m_per_ns and force_ki_m_per_ns2",
            "max_correct_speed_mps",
            "force_integral_limit_ns",
        ],
        "scope": (
            "an open-loop ceiling measured on this kernel: whether the drive "
            "can deliver the target at all. It says nothing about whether the "
            "closed loop converges to it, which the loop gates measure"
        ),
    }


def enforce_realised_drive_authority(spec, *, kernel_config=None) -> dict | None:
    """Reject a target the kernel does not actually deliver. ``None`` if not force mode."""
    report = measure_force_ceiling(spec, kernel_config=kernel_config)
    if report is None:
        return None
    if not report["contact_established"]:
        raise UnreachableUnderRealisedDrive(
            f"the probe never reached contact in {report['approach_steps']} control "
            f"steps at the declared approach speed, so no force target is reachable. "
            f"Check linear_damping_per_s and the gravity compensation against the "
            f"declared approach_speed_mps"
        )
    if not report["tolerance_band_reachable_under_realised_drive"]:
        raise UnreachableUnderRealisedDrive(
            f"the kernel delivers at most "
            f"{report['measured_force_ceiling_n']:.6f} N at the declared press limit "
            f"of {report['declared_press_limit_mps']} m/s, but "
            f"target_force_n={report['target_force_n']} with tolerance "
            f"{report['force_tolerance_n']} needs {report['required_force_n']:.6f} N. "
            f"The ideal drive law predicts {report['ideal_law_force_ceiling_n']:.6f} N, "
            f"so {report['ideal_minus_measured_n']:.6f} N is lost to effects the law "
            f"omits — damping, gravity compensation, substepping or the sampling "
            f"threshold. Raise max_press_speed_mps, or remove what is absorbing it"
        )
    return report
