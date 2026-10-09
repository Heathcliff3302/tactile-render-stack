"""A PI regulator must be able to hold the target, not just reach for it.

``test_drive_limits.py`` covers the first ceiling: the press clamp. This file
covers the second, which clearing the first says nothing about.

At steady state the error has gone to zero, so the proportional term has gone
with it and the command speed must come from the integrator alone — bounded by
``force_ki * force_integral_limit``. Phase 1's inherited gains supply 0.2 m/s
where a 1 N target needs 0.3333 m/s on this kernel, so the loop settles short
of its target and ``regulate_force`` times out.

This was found by simulating the declared gains against the kernel's own force
law before implementing the controller. Without the guard, the specification
would load cleanly, the run would abort on a timeout, and nothing would say
the gains were the reason rather than the solver.
"""

import copy

import pytest

from virtual_reality import (
    DERIVED_HEADROOM_FACTOR,
    InsufficientRegulatorAuthority,
    UnreachableForceTarget,
    assess_regulator_authority,
    command_speed_for_force_mps,
    derive_force_ki,
    enforce_regulator_authority,
    kernel_config_from_spec,
    saturated_force_ceiling_n,
)

MASS = 0.05
DT = 1.0 / 60.0
TARGET = 1.0
INTEGRAL_LIMIT = 2.0
PHASE1_KI = 0.1
PHASE1_KP = 0.2


def mutate(spec, change):
    from tactile_contract import ExperimentSpec

    data = copy.deepcopy(spec.to_dict())
    change(data)
    return ExperimentSpec.build(**data)


def test_the_steady_state_requirement_is_a_speed_not_a_gain():
    # ki and the integral limit only matter as their product, which is a speed.
    required = command_speed_for_force_mps(mass_kg=MASS, force_n=TARGET, output_dt_s=DT)
    assert required == pytest.approx(1 / 3)
    assert PHASE1_KI * INTEGRAL_LIMIT == pytest.approx(0.2)
    assert PHASE1_KI * INTEGRAL_LIMIT < required


def test_derived_ki_supplies_the_declared_headroom():
    ki = derive_force_ki(
        mass_kg=MASS, target_force_n=TARGET, output_dt_s=DT, integral_limit_ns=INTEGRAL_LIMIT,
    )
    assert ki == pytest.approx(0.25)
    assert ki * INTEGRAL_LIMIT == pytest.approx(
        DERIVED_HEADROOM_FACTOR * command_speed_for_force_mps(
            mass_kg=MASS, force_n=TARGET, output_dt_s=DT
        )
    )
    # Trading the gain against the limit is the same requirement.
    halved = derive_force_ki(
        mass_kg=MASS, target_force_n=TARGET, output_dt_s=DT,
        integral_limit_ns=INTEGRAL_LIMIT / 2,
    )
    assert halved == pytest.approx(2 * ki)
    with pytest.raises(ValueError, match="cannot reach the target"):
        derive_force_ki(mass_kg=MASS, target_force_n=TARGET, output_dt_s=DT,
                        integral_limit_ns=INTEGRAL_LIMIT, headroom=0.9)
    with pytest.raises(ValueError, match="no authority"):
        derive_force_ki(mass_kg=MASS, target_force_n=TARGET, output_dt_s=DT,
                        integral_limit_ns=0.0)


def test_the_saturated_ceiling_matches_the_closed_loop_fixed_point():
    """Independent check: iterate the loop and compare with the closed form."""
    ceiling = saturated_force_ceiling_n(
        mass_kg=MASS, output_dt_s=DT, target_force_n=TARGET,
        force_kp_m_per_ns=PHASE1_KP, force_ki_m_per_ns2=PHASE1_KI,
        integral_limit_ns=INTEGRAL_LIMIT,
    )
    assert ceiling == pytest.approx(0.75)

    integral, force = 0.0, 0.0
    for _ in range(2000):
        error = TARGET - force
        integral = min(max(integral + error * DT, -INTEGRAL_LIMIT), INTEGRAL_LIMIT)
        speed = min(max(-(PHASE1_KP * error + PHASE1_KI * integral), -0.5), 0.2)
        force = max(0.0, -speed) * MASS / DT
    assert force == pytest.approx(ceiling, abs=1e-9)
    # And it never enters the tolerance band on the way.
    assert abs(TARGET - force) > 0.05


def test_raising_kp_does_not_make_the_target_reachable():
    # kp cancels in the condition, so a larger kp only lifts the ceiling the
    # regulator settles on. This is the tempting wrong fix.
    for kp in (0.2, 1.0, 5.0):
        ceiling = saturated_force_ceiling_n(
            mass_kg=MASS, output_dt_s=DT, target_force_n=TARGET,
            force_kp_m_per_ns=kp, force_ki_m_per_ns2=PHASE1_KI,
            integral_limit_ns=INTEGRAL_LIMIT,
        )
        assert ceiling < TARGET


def test_the_k2_reference_can_hold_its_target(k2_spec):
    report = enforce_regulator_authority(k2_spec)
    assert report["force_ki_m_per_ns2"] == pytest.approx(0.25)
    assert report["integral_authority_mps"] == pytest.approx(0.5)
    assert report["target_holdable"] is True
    assert report["tolerance_band_holdable"] is True
    assert report["saturated_force_ceiling_n"] >= TARGET


def test_phase_1_gains_are_rejected_at_load(k2_spec):
    spec = mutate(k2_spec, lambda d: d["controller"]["force"].update(
        force_ki_m_per_ns2=PHASE1_KI))
    report = assess_regulator_authority(spec)
    assert report["integral_authority_mps"] == pytest.approx(0.2)
    assert report["binding_limit"] == "integral"
    assert report["tolerance_band_holdable"] is False
    assert report["saturated_force_ceiling_n"] == pytest.approx(0.75)
    assert report["recommended_force_ki_m_per_ns2"] == pytest.approx(0.25)

    with pytest.raises(InsufficientRegulatorAuthority, match="would settle at"):
        enforce_regulator_authority(spec)
    # And the spec cannot be used to build a kernel either.
    with pytest.raises(InsufficientRegulatorAuthority):
        kernel_config_from_spec(spec)


def test_the_source_inventory_records_gains_that_cannot_hold_the_target():
    # The Phase 1 inventory is a faithful record and is not edited; the
    # assessment reports that its gains do not carry over to this kernel.
    import json
    from pathlib import Path

    from tactile_contract import ExperimentSpec

    path = Path(__file__).resolve().parents[2] / "scenarios" / "k0" / "step6_force_64.json"
    spec = ExperimentSpec.from_dict(json.loads(path.read_text(encoding="utf-8")))
    report = assess_regulator_authority(spec)
    assert report["force_ki_m_per_ns2"] == PHASE1_KI
    assert report["tolerance_band_holdable"] is False


def test_the_two_ceilings_are_reported_separately(k2_spec):
    """0.75 N appears for two unrelated reasons; they must not be conflated."""
    from virtual_reality import assess_force_reachability, frame_force_n

    # Phase 1's press clamp reaches 0.75 N.
    assert frame_force_n(mass_kg=MASS, speed_mps=0.25, output_dt_s=DT) == pytest.approx(0.75)
    # Phase 1's gains also settle at 0.75 N, under K2's 0.5 m/s clamp, which
    # is not what bound them.
    starved = mutate(k2_spec, lambda d: d["controller"]["force"].update(
        force_ki_m_per_ns2=PHASE1_KI))
    authority = assess_regulator_authority(starved)
    assert authority["declared_press_limit_mps"] == 0.5
    assert authority["saturated_force_ceiling_n"] == pytest.approx(0.75)
    assert authority["binding_limit"] == "integral"
    # The press-limit check passes on this spec; only the authority check fails.
    assert assess_force_reachability(starved)["tolerance_band_reachable"] is True


def test_a_press_clamp_below_the_integral_authority_becomes_binding(k2_spec):
    # Both guards are needed: this spec has enough integral authority but the
    # drive will not let the regulator use it.
    spec = mutate(k2_spec, lambda d: d["controller"]["force"].update(
        max_press_speed_mps=0.25))
    report = assess_regulator_authority(spec)
    assert report["integral_authority_mps"] == pytest.approx(0.5)
    assert report["effective_press_authority_mps"] == pytest.approx(0.25)
    assert report["binding_limit"] == "press_clamp"
    assert report["tolerance_band_holdable"] is False
    with pytest.raises(UnreachableForceTarget):
        kernel_config_from_spec(spec)


def test_trajectory_specs_are_not_subjected_to_an_authority_check(k1_spec):
    assert assess_regulator_authority(k1_spec) is None
    assert enforce_regulator_authority(k1_spec) is None


def test_the_declared_proportional_gain_is_stable_at_its_declared_rate(k2_spec):
    from virtual_reality import PROPORTIONAL_STABILITY_MARGIN, enforce_proportional_stability

    report = enforce_proportional_stability(k2_spec)
    assert report["force_kp_m_per_ns"] == PHASE1_KP
    assert report["proportional_loop_gain"] == pytest.approx(0.6)
    assert report["stability_boundary_force_kp"] == pytest.approx(1 / 3)
    assert report["admitted_force_kp"] == pytest.approx(PROPORTIONAL_STABILITY_MARGIN / 3)
    assert report["stable"] is True


def test_a_proportional_gain_past_the_stability_bound_is_rejected(k2_spec):
    from virtual_reality import ProportionalGainUnstable, assess_proportional_stability

    # The bound is output_dt / mass = 0.3333 at the declared 60 Hz.
    spec = mutate(k2_spec, lambda d: d["controller"]["force"].update(
        force_kp_m_per_ns=0.4))
    report = assess_proportional_stability(spec)
    assert report["proportional_loop_gain"] == pytest.approx(1.2)
    assert report["stable"] is False
    with pytest.raises(ProportionalGainUnstable, match="rings instead of settling"):
        kernel_config_from_spec(spec)


def test_the_stability_bound_scales_with_the_output_interval(k2_spec):
    """The declared gain is stable at 60 Hz and not at 120 Hz."""
    from virtual_reality import assess_proportional_stability

    faster = mutate(k2_spec, lambda d: d["timing"].update(
        output_dt_s=1 / 120, control_dt_s=1 / 120, physics_dt_s=1 / 120))
    report = assess_proportional_stability(faster)
    assert report["force_kp_m_per_ns"] == PHASE1_KP
    assert report["proportional_loop_gain"] == pytest.approx(1.2)
    assert report["stable"] is False
    # Nothing about the force target changed; only the rate did.
    assert faster.to_dict()["controller"]["force"]["target_force_n"] == TARGET


def test_a_pure_integral_regulator_is_unconditionally_stable(k2_spec):
    from virtual_reality import assess_proportional_stability

    spec = mutate(k2_spec, lambda d: d["controller"]["force"].update(
        force_kp_m_per_ns=0.0))
    report = assess_proportional_stability(spec)
    assert report["unconditionally_stable"] is True
    assert report["stable"] is True
    assert report["proportional_loop_gain"] == 0.0


def test_stability_and_authority_are_independent_questions(k2_spec):
    """One bounds convergence, the other bounds reach. Neither implies the other."""
    from virtual_reality import assess_proportional_stability

    # Ample authority, unstable gain.
    unstable = mutate(k2_spec, lambda d: d["controller"]["force"].update(
        force_kp_m_per_ns=0.4))
    assert assess_regulator_authority(unstable)["tolerance_band_holdable"] is True
    assert assess_proportional_stability(unstable)["stable"] is False

    # Stable gain, starved authority.
    starved = mutate(k2_spec, lambda d: d["controller"]["force"].update(
        force_ki_m_per_ns2=PHASE1_KI))
    assert assess_proportional_stability(starved)["stable"] is True
    assert assess_regulator_authority(starved)["tolerance_band_holdable"] is False


def test_an_unimplemented_drive_semantics_is_refused_not_guessed(k2_spec):
    # The derivation is specific to the declared drive. Another drive needs its
    # own derivation, not this one applied anyway.
    data = copy.deepcopy(k2_spec.to_dict())
    data["controller"]["drive_semantics"] = "force_at_contact"
    with pytest.raises(NotImplementedError, match="force_at_contact"):
        assess_regulator_authority(data)
