"""Regression against the stored Isaac Sim reference (captured 2026-11-25).

Every kernel change is re-scored against the four Isaac Sim 5.1.0 closed-loop
records in ``experiments/isaac_reference/``, without a GPU. The kernel is run
fresh from its spec; nothing is read from ``experiments/runs/``.

What is locked, for the ``aligned`` variant (same physics as the kernel):

* every frozen gate passes, with the contact point compared under the declared
  reporting convention (user ruling (a), 2026-11-25): PhysX reports a contact at
  the start of the step, the kernel at the end, so during sliding Isaac step n is
  compared with kernel step n-1;
* per-frame normal force stays within 1e-6 N of Isaac (observed: 1.6e-7 N).

The ``phase1_defaults`` records are kept as evidence that Isaac's default
materials change the result (49 mm slide drift); they are checked for integrity
only, never expected to match.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
from pathlib import Path

import pytest

from validation.isaac_parity import THRESHOLDS, _centroid, compare, load_kernel

REFERENCE = Path(__file__).resolve().parents[2] / "experiments" / "isaac_reference"
FORCE_MATCH_N = 1e-6

#: SHA-256 of the stored records. A changed file is a changed reference, which
#: needs a new capture on the GPU machine and the user's approval, not an edit.
REFERENCE_SHA256 = {
    "out_k1_single_probe_cpu_aligned.jsonl.gz": "7f217f1a2f87e7ab8a92fe7b81c4c1ae83f15db5f9ae3fc5fafb87cfb01c90a8",
    "out_k1_single_probe_cpu_phase1_defaults.jsonl.gz": "db2bde86b28daa970f96c4a9475bfe37acc4fb043bacd3739c19047f55cc23ae",
    "out_k2_force_cpu_aligned.jsonl.gz": "4d8638e526e0bb00943bf399454d2a0d5817aec2244ada013c78dffacde55fa6",
    "out_k2_force_cpu_phase1_defaults.jsonl.gz": "a87a375ce85e727ecb75fb5e2937f91ca32a7c43844c5c27e3c0b75ec92c06d4",
}


def _isaac(name: str) -> tuple[list[dict], dict]:
    with gzip.open(REFERENCE / name, "rt", encoding="utf-8") as handle:
        lines = [json.loads(line) for line in handle if line.strip()]
    return lines[:-1], lines[-1]["summary"]


@pytest.fixture(scope="module")
def kernel_frames(k1_run_directory, k2_run_directory):
    return {
        "k1_single_probe_cpu": load_kernel(k1_run_directory),
        "k2_force_cpu": load_kernel(k2_run_directory),
    }


@pytest.mark.parametrize("name", sorted(REFERENCE_SHA256))
def test_reference_records_are_unchanged(name):
    digest = hashlib.sha256((REFERENCE / name).read_bytes()).hexdigest()
    assert digest == REFERENCE_SHA256[name]
    frames, summary = _isaac(name)
    assert summary["errors"] == []
    assert summary["controller_done"] and frames


@pytest.mark.parametrize("spec", ["k1_single_probe_cpu", "k2_force_cpu"])
def test_kernel_matches_isaac_on_aligned_physics(kernel_frames, spec):
    isaac, _ = _isaac(f"out_{spec}_aligned.jsonl.gz")
    kernel = kernel_frames[spec]
    result = compare(kernel, isaac, spec)
    gates = result["gates"]
    for name, gate in gates.items():
        if name != "contact_point":
            assert gate["pass"], (name, gate)
    assert result["reported_only"]["phase_sequence_equal"]
    assert result["reported_only"]["frames"]["kernel"] == result["reported_only"]["frames"]["isaac"]
    assert result["reported_only"]["force_abs_diff_n"]["max"] <= FORCE_MATCH_N


@pytest.mark.parametrize("spec", ["k1_single_probe_cpu", "k2_force_cpu"])
def test_contact_points_match_under_the_declared_convention(kernel_frames, spec):
    """Static contact: same step. Sliding: Isaac step n against kernel step n-1."""
    isaac, _ = _isaac(f"out_{spec}_aligned.jsonl.gz")
    kernel = {f["control_step"]: f for f in kernel_frames[spec]}
    checked, worst = 0, 0.0
    for frame in isaac:
        if not (frame["contact_present"] and frame["contact_points"]):
            continue
        sliding = "slide" in frame["phase"]
        other = kernel.get(frame["control_step"] - 1 if sliding else frame["control_step"])
        if not (other and other["contact_present"] and ("slide" in other["phase"]) == sliding):
            continue
        worst = max(worst, math.dist(_centroid(frame["contact_points"]), _centroid(other["contact_points"])))
        checked += 1
    assert checked > 100
    assert worst <= THRESHOLDS["contact_point_max_m"], worst


def test_default_materials_are_kept_as_a_known_difference(kernel_frames):
    """Isaac's defaults (dynamic friction 1.0) drag the probe in the slide; this must stay visible."""
    isaac, _ = _isaac("out_k2_force_cpu_phase1_defaults.jsonl.gz")
    result = compare(kernel_frames["k2_force_cpu"], isaac, "k2_force_cpu")
    assert not result["gates"]["contact_point"]["pass"]
    assert result["gates"]["contact_point"]["max_m"] > 0.01
