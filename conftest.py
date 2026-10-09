"""Shared fixtures. A full K1 run is executed once per session, not per test."""

import json
from pathlib import Path

import pytest

from mechanics.k1_report import write_run
from mechanics.k1_rigid import (
    K2_SCENARIO_DESCRIPTION,
    K2_SCENARIO_ID,
    K2_SCENARIO_VERSION,
    K1Experiment,
    RigidProbeExperiment,
)
from scenarios.k1_variants import time_refinement_specs
from tactile_contract import ExperimentSpec
from validation.k1_acceptance import acceptance_report

ROOT = Path(__file__).resolve().parent
K0 = ROOT / "scenarios" / "k0"
K1_SPEC_PATH = K0 / "k1_single_probe_cpu.json"
K2_SPEC_PATH = K0 / "k2_force_cpu.json"


def _load(path) -> ExperimentSpec:
    return ExperimentSpec.from_dict(json.loads(path.read_text(encoding="utf-8")))


@pytest.fixture(scope="session")
def k1_spec() -> ExperimentSpec:
    return _load(K1_SPEC_PATH)


@pytest.fixture(scope="session")
def k2_spec() -> ExperimentSpec:
    return _load(K2_SPEC_PATH)


@pytest.fixture(scope="session")
def k2_experiment(k2_spec) -> RigidProbeExperiment:
    return RigidProbeExperiment(
        k2_spec,
        run_id="k2-force-cpu",
        scenario_id=K2_SCENARIO_ID,
        scenario_version=K2_SCENARIO_VERSION,
        scenario_description=K2_SCENARIO_DESCRIPTION,
    )


@pytest.fixture(scope="session")
def k2_run(k2_experiment):
    return k2_experiment.run()


@pytest.fixture(scope="session")
def k1_experiment(k1_spec) -> K1Experiment:
    return K1Experiment(k1_spec)


@pytest.fixture(scope="session")
def k1_run(k1_experiment):
    return k1_experiment.run()


@pytest.fixture(scope="session")
def k1_refinement_runs(k1_spec):
    return [K1Experiment(variant).run() for variant in time_refinement_specs(k1_spec, (2, 4))]


@pytest.fixture(scope="session")
def k1_run_directory(k1_run, k1_experiment, tmp_path_factory) -> Path:
    directory = tmp_path_factory.mktemp("k1_run")
    report = acceptance_report(k1_run)
    write_run(k1_run, directory, report, k1_experiment.effective_runtime())
    return directory
