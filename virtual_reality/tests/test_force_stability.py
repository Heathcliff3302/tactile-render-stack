"""Does the loop converge, and not only have enough authority to?

The authority checks answer a static question: is the target inside the
drive's reach. They say nothing about the path taken to it. A regulator with
ample authority can still oscillate, or ring long enough to miss its timeout,
once the feedback delay and the control rate are varied.

So this sweep closes the loop. The plant is the kernel's own force law,
``F = mass * commanded_speed / output_dt``, with the declared feedback delay
inserted explicitly, and each combination is scored on three properties the
specification actually depends on:

* it does not diverge,
* it does not oscillate indefinitely,
* it enters the tolerance band before the declared regulate timeout.

The sweep is small on purpose. It exists to catch a combination that fails
qualitatively, not to tune gains, and a grid large enough to tune with would
make the suite slow without making it stronger.
"""

import math
from collections import deque

import pytest

from virtual_reality import (
    ForceConfig,
    ForceFeedbackStateMachine,
    assess_regulator_authority,
    derive_force_ki,
)
from tactile_contract.pipeline import LoopFeedback

MASS = 0.05
TARGET = 1.0
TOLERANCE = 0.05


def simulate(*, control_dt, feedback_delay_steps, kp, ki, settle_frames,
             press_limit, correct_limit, integral_limit=2.0, horizon_s=20.0):
    """Close the loop around the kernel's force law and report its behaviour.

    ``feedback_delay_steps`` is modelled as a queue rather than assumed to be
    one, because the point of the sweep is that the declared delay is a
    parameter the stability conclusion depends on.
    """
    config = ForceConfig(
        pre_settle_time_s=0.0,
        approach_timeout_s=5.0,
        regulate_timeout_s=horizon_s,
        hold_time_s=1.0,
        slide_time_s=1.0,
        release_time_s=0.5,
        approach_speed_mps=0.2,
        slide_speed_mps=0.05,
        retract_speed_mps=0.3,
        retract_distance_m=0.58,
        target_force_n=TARGET,
        force_tolerance_n=TOLERANCE,
        force_kp_m_per_ns=kp,
        force_ki_m_per_ns2=ki,
        force_integral_limit_ns=integral_limit,
        max_press_speed_mps=press_limit,
        max_correct_speed_mps=correct_limit,
        settle_frames=settle_frames,
        control_dt_s=control_dt,
    )
    controller = ForceFeedbackStateMachine(config)
    # Start in the regulating segment with contact already established; the
    # approach is not what this sweep is about.
    controller.phase = "regulate_force"

    delayed = deque([0.0] * max(1, feedback_delay_steps), maxlen=max(1, feedback_delay_steps))
    force = 0.0
    history = []
    steps = int(horizon_s / control_dt)
    for _ in range(steps):
        command = controller.command(
            LoopFeedback(contact_present=True, normal_force_n=delayed[0])
        )
        force = max(0.0, -command[2]) * MASS / control_dt
        delayed.append(force)
        history.append(force)

    finite = all(math.isfinite(value) for value in history)
    bounded = finite and max(history) <= 10 * TARGET
    settle_index = None
    run_length = 0
    for index, value in enumerate(history):
        run_length = run_length + 1 if abs(TARGET - value) <= TOLERANCE else 0
        if run_length >= settle_frames and settle_index is None:
            settle_index = index
    tail = history[-int(1.0 / control_dt):]
    # "Still oscillating" means the last second never stops leaving the band.
    tail_in_band = all(abs(TARGET - value) <= TOLERANCE for value in tail)
    return {
        "diverged": not bounded,
        "settled": settle_index is not None,
        "settle_time_s": settle_index * control_dt if settle_index is not None else None,
        "tail_in_band": tail_in_band,
        "final_force_n": history[-1],
        "peak_force_n": max(history),
        "history": history,
    }


def stable_kp(control_dt, *, fraction=0.6):
    """A proportional gain inside the stability bound for this control rate.

    The bound is ``output_dt / mass`` — see ``assess_proportional_stability``.
    Both gains are derived per rate rather than held fixed, because both
    scale with the output interval. Writing the declared 60 Hz numbers into
    every row of this sweep is what produced the first failing run of it.
    """
    return fraction * control_dt / MASS


#: Combinations that must all converge, each with gains derived for its own
#: control rate.
STABLE_CASES = [
    pytest.param(1 / 60, 1, 0.6, 5, id="declared-60hz"),
    pytest.param(1 / 120, 1, 0.6, 5, id="120hz"),
    pytest.param(1 / 240, 1, 0.6, 5, id="240hz"),
    pytest.param(1 / 60, 1, 0.0, 5, id="pure-integral"),
    pytest.param(1 / 60, 1, 0.9, 5, id="near-the-stability-bound"),
    pytest.param(1 / 60, 1, 0.6, 1, id="single-settle-frame"),
    pytest.param(1 / 60, 1, 0.6, 20, id="long-settle"),
    pytest.param(1 / 60, 2, 0.6, 5, id="two-step-delay"),
    pytest.param(1 / 60, 4, 0.6, 5, id="four-step-delay"),
]


@pytest.mark.parametrize("control_dt,delay,kp_fraction,settle_frames", STABLE_CASES)
def test_the_loop_converges_across_the_declared_envelope(
    control_dt, delay, kp_fraction, settle_frames
):
    ki = derive_force_ki(
        mass_kg=MASS, target_force_n=TARGET, output_dt_s=control_dt, integral_limit_ns=2.0,
    )
    result = simulate(
        control_dt=control_dt, feedback_delay_steps=delay,
        kp=stable_kp(control_dt, fraction=kp_fraction), ki=ki,
        settle_frames=settle_frames,
        press_limit=1.5 * TARGET * control_dt / MASS, correct_limit=0.2,
    )
    assert result["diverged"] is False
    assert result["settled"] is True, f"never entered the band: {result['final_force_n']}"
    assert result["tail_in_band"] is True, "still leaving the band at the horizon"
    assert result["settle_time_s"] < 20.0
    assert result["final_force_n"] == pytest.approx(TARGET, abs=TOLERANCE)


def test_holding_the_declared_gain_fixed_across_rates_is_what_rings():
    """Why the sweep derives kp per rate: the finding that produced the guard.

    The declared ``force_kp = 0.2`` is comfortably stable at its declared
    60 Hz and rings at higher rates, because the admissible gain scales with
    the output interval. Nothing checked this before
    ``assess_proportional_stability``.
    """
    declared_kp = 0.2
    outcomes = {}
    for control_dt in (1 / 60, 1 / 120, 1 / 240):
        ki = derive_force_ki(
            mass_kg=MASS, target_force_n=TARGET, output_dt_s=control_dt,
            integral_limit_ns=2.0,
        )
        result = simulate(
            control_dt=control_dt, feedback_delay_steps=1, kp=declared_kp, ki=ki,
            settle_frames=5,
            press_limit=1.5 * TARGET * control_dt / MASS, correct_limit=0.2,
        )
        outcomes[round(1 / control_dt)] = (
            MASS * declared_kp / control_dt, result["settled"]
        )
    assert outcomes[60] == (pytest.approx(0.6), True)
    assert outcomes[120][0] == pytest.approx(1.2)
    assert outcomes[120][1] is False
    assert outcomes[240][0] == pytest.approx(2.4)
    assert outcomes[240][1] is False


def test_no_case_in_the_envelope_overshoots_the_target():
    # Overshoot would mean pressing harder than asked, which on a real device
    # is the direction that matters.
    for control_dt, delay, kp_fraction in [
        (1 / 60, 1, 0.6), (1 / 120, 1, 0.6), (1 / 60, 4, 0.6),
    ]:
        ki = derive_force_ki(
            mass_kg=MASS, target_force_n=TARGET, output_dt_s=control_dt,
            integral_limit_ns=2.0,
        )
        result = simulate(
            control_dt=control_dt, feedback_delay_steps=delay,
            kp=stable_kp(control_dt, fraction=kp_fraction), ki=ki, settle_frames=5,
            press_limit=1.5 * TARGET * control_dt / MASS, correct_limit=0.2,
        )
        assert result["peak_force_n"] <= TARGET * 1.02, (control_dt, delay, kp_fraction)


def test_starved_integral_authority_never_settles_at_any_control_rate():
    # The failure the authority guard rejects at load, confirmed to be a
    # genuine non-convergence rather than merely slow.
    for control_dt in (1 / 60, 1 / 120):
        result = simulate(
            control_dt=control_dt, feedback_delay_steps=1, kp=0.2, ki=0.1,
            settle_frames=5, press_limit=0.5, correct_limit=0.2,
        )
        assert result["diverged"] is False
        assert result["settled"] is False
        assert result["final_force_n"] < TARGET - TOLERANCE


def test_an_excessive_gain_is_caught_as_oscillation_not_silently_passed():
    """The sweep must be able to fail, or it is not evidence.

    A very large integral gain rings instead of converging. The assertions in
    the envelope test above are what would catch it, so they are exercised
    here against a case that genuinely misbehaves.
    """
    result = simulate(
        control_dt=1 / 60, feedback_delay_steps=1, kp=0.2, ki=40.0,
        settle_frames=5, press_limit=1.5, correct_limit=0.2, integral_limit=2.0,
    )
    assert result["diverged"] is False
    assert result["tail_in_band"] is False
    assert result["peak_force_n"] > TARGET


def test_the_measured_boundary_matches_the_derived_one():
    """``a*kp = 1`` is where convergence stops, and it is checked, not assumed."""
    from virtual_reality import PROPORTIONAL_STABILITY_MARGIN

    control_dt = 1 / 60
    bound = control_dt / MASS
    ki = derive_force_ki(
        mass_kg=MASS, target_force_n=TARGET, output_dt_s=control_dt, integral_limit_ns=2.0,
    )

    def settles(kp):
        return simulate(
            control_dt=control_dt, feedback_delay_steps=1, kp=kp, ki=ki, settle_frames=5,
            press_limit=1.5 * TARGET * control_dt / MASS, correct_limit=0.2,
        )["settled"]

    # Inside the bound, including the admitted margin, converges.
    assert settles(PROPORTIONAL_STABILITY_MARGIN * bound) is True
    assert settles(0.99 * bound) is True
    # At and beyond it, it does not.
    assert settles(1.0 * bound) is False
    assert settles(1.1 * bound) is False


def test_the_declared_specification_sits_inside_the_stable_envelope(k2_spec):
    """The shipped numbers, not a nearby approximation of them."""
    data = k2_spec.to_dict()
    force = data["controller"]["force"]
    result = simulate(
        control_dt=data["timing"]["control_dt_s"],
        feedback_delay_steps=data["controller"]["feedback_delay_steps"],
        kp=force["force_kp_m_per_ns"],
        ki=force["force_ki_m_per_ns2"],
        settle_frames=force["settle_frames"],
        press_limit=force["max_press_speed_mps"],
        correct_limit=force["max_correct_speed_mps"],
        integral_limit=force["force_integral_limit_ns"],
        horizon_s=force["regulate_timeout_s"],
    )
    assert result["diverged"] is False
    assert result["settled"] is True
    assert result["tail_in_band"] is True
    # Convergence well inside the declared timeout, not just before it.
    assert result["settle_time_s"] < force["regulate_timeout_s"] / 2
    assert assess_regulator_authority(k2_spec)["tolerance_band_holdable"] is True
