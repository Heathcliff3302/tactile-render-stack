"""The closed-form reachability checks describe an idealised drive.

``F = mass * commanded_speed / output_dt`` is exact for the K1/K2 CPU
reference, and only because every term it omits is zero there. Set
``linear_damping_per_s`` to a value the kernel honours and the law keeps
reporting a reachable target while the kernel delivers less than the target.

That setting is classified ``enforced`` — the kernel reads it — so this is a
supported configuration the static check cannot see, not an unsupported one
being smuggled past. The probe below asks the kernel instead of modelling it.
"""

import copy
import json
from pathlib import Path

import pytest

from tactile_contract import ExperimentSpec
from virtual_reality import (
    UnreachableUnderRealisedDrive,
    assess_force_reachability,
    assess_regulator_authority,
    enforce_realised_drive_authority,
    kernel_config_from_spec,
    measure_force_ceiling,
)

K0 = Path(__file__).resolve().parents[2] / "scenarios" / "k0"


def damped(spec, damping):
    data = copy.deepcopy(spec.to_dict())
    data.pop("spec_sha256", None)
    data["runtime"]["linear_damping_per_s"]["value"] = damping
    return ExperimentSpec.build(**data)


def test_the_cpu_reference_measures_exactly_what_the_law_predicts(k2_spec):
    """The two agree here, which is worth recording rather than assuming."""
    report = measure_force_ceiling(k2_spec)
    assert report["contact_established"] is True
    assert report["measured_force_ceiling_n"] == pytest.approx(1.5)
    assert report["ideal_law_force_ceiling_n"] == pytest.approx(1.5)
    assert report["ideal_minus_measured_n"] == pytest.approx(0.0, abs=1e-12)
    assert report["realised_fraction_of_ideal"] == pytest.approx(1.0)
    assert report["tolerance_band_reachable_under_realised_drive"] is True


def test_damping_separates_the_measured_ceiling_from_the_predicted_one(k2_spec):
    # Monotone loss, all of it invisible to the closed-form checks.
    expected = {0.0: 1.5, 1.0: 1.475, 5.0: 1.375, 10.0: 1.25, 30.0: 0.75}
    for damping, ceiling in expected.items():
        spec = damped(k2_spec, damping)
        report = measure_force_ceiling(spec)
        assert report["measured_force_ceiling_n"] == pytest.approx(ceiling), damping
        assert report["ideal_law_force_ceiling_n"] == pytest.approx(1.5)
        # Both static checks are blind to every one of these.
        assert assess_force_reachability(spec)["tolerance_band_reachable"] is True
        assert assess_regulator_authority(spec)["tolerance_band_holdable"] is True


def test_a_target_the_kernel_cannot_deliver_is_rejected_at_load(k2_spec):
    """The finding this probe exists for."""
    spec = damped(k2_spec, 30.0)
    report = measure_force_ceiling(spec)
    assert report["measured_force_ceiling_n"] == pytest.approx(0.75)
    assert report["required_force_n"] == pytest.approx(1.05)
    assert report["target_reachable_under_realised_drive"] is False
    assert report["tolerance_band_reachable_under_realised_drive"] is False

    with pytest.raises(UnreachableUnderRealisedDrive, match="delivers at most"):
        enforce_realised_drive_authority(spec)
    # And the spec cannot be used to build a kernel.
    with pytest.raises(UnreachableUnderRealisedDrive):
        kernel_config_from_spec(spec)


def test_damping_that_prevents_contact_is_reported_as_such(k2_spec):
    # Enough damping and the probe never reaches the surface at all. That is a
    # different failure from "reached it but could not press hard enough", and
    # it gets its own message rather than a misleading zero-force ceiling.
    spec = damped(k2_spec, 400.0)
    report = measure_force_ceiling(spec)
    assert report["contact_established"] is False
    assert report["measured_force_ceiling_n"] == 0.0
    with pytest.raises(UnreachableUnderRealisedDrive, match="never reached contact"):
        enforce_realised_drive_authority(spec)


def test_the_probe_states_what_it_does_not_cover(k2_spec):
    # An open-loop ceiling is not a convergence claim, and the report says so
    # instead of letting a reader infer more than was measured.
    report = measure_force_ceiling(k2_spec)
    assert "linear_damping_per_s" in report["accounted_for"]
    assert "feedback_delay_steps" in report["not_accounted_for"]
    uncovered = " ".join(report["not_accounted_for"])
    assert "force_kp_m_per_ns" in uncovered
    assert "max_correct_speed_mps" in uncovered
    assert "says nothing about whether the closed loop converges" in report["scope"]


def test_a_trajectory_spec_is_not_probed(k1_spec):
    assert measure_force_ceiling(k1_spec) is None
    assert enforce_realised_drive_authority(k1_spec) is None


def test_the_phase_1_inventory_is_unreachable_on_both_counts():
    # The static check already rejects it; the measurement agrees. Neither
    # conclusion depends on the other.
    spec = ExperimentSpec.from_dict(
        json.loads((K0 / "step6_force_64.json").read_text(encoding="utf-8"))
    )
    assert assess_force_reachability(spec)["tolerance_band_reachable"] is False
    # It is a source_inventory with unresolved runtime settings, so it cannot
    # reach the kernel at all. That rejection comes first and is correct.
    with pytest.raises(ValueError, match="unresolved"):
        kernel_config_from_spec(spec)


def test_the_run_record_carries_the_measured_ceiling(k2_experiment):
    runtime = k2_experiment.effective_runtime()
    measured = runtime["realised_drive_authority"]
    assert measured["method"] == "kernel_probe_at_declared_press_limit"
    assert measured["tolerance_band_reachable_under_realised_drive"] is True
    # Both answers are recorded, so a divergence between them is visible in
    # the evidence rather than only at load time.
    assert runtime["regulator_authority"]["tolerance_band_holdable"] is True
