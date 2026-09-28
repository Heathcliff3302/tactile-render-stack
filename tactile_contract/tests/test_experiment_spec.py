import json
from pathlib import Path

import pytest

from tactile_contract import ExperimentSpec, spec_comparison_gate, unresolved_settings
from tactile_contract._common import digest


ROOT = Path(__file__).parents[2]


def load(name):
    return json.loads((ROOT / "scenarios" / "k0" / name).read_text(encoding="utf-8"))


def test_k0_inventories_keep_unresolved_runtime_explicit():
    data = load("step6_trajectory_64.json")
    spec = ExperimentSpec.from_dict(data)
    assert spec.kind == "source_inventory"
    assert "gravity_mps2" in unresolved_settings(spec)
    assert data["spec_sha256"] == digest({k: v for k, v in data.items() if k != "spec_sha256"})


def test_cpu_reference_resolves_runtime_settings():
    spec = ExperimentSpec.from_dict(load("k1_single_probe_cpu.json"))
    assert spec.kind == "cpu_reference"
    assert unresolved_settings(spec) == []
    assert spec.backend_id == "rigid_cpu_v1"


def test_comparison_gate_only_allows_declared_refinement():
    left = ExperimentSpec.from_dict(load("step6_trajectory_32.json"))
    right = ExperimentSpec.from_dict(load("step6_trajectory_64.json"))
    same = spec_comparison_gate(left, right)
    assert not same["eligible"]
    assert "unresolved runtime settings" in " ".join(same["reasons"])
    refined = spec_comparison_gate(left, right, profile="grid_refinement")
    assert not refined["eligible"]
    assert any("unresolved" in reason for reason in refined["reasons"])


def test_spec_hash_mismatch_is_rejected():
    data = load("k1_single_probe_cpu.json")
    data["grid"]["rows"] = 32
    with pytest.raises(ValueError, match="spec_sha256 mismatch"):
        ExperimentSpec.from_dict(data)
