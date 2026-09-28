"""Shared fixtures. A full K1 run is executed once per session, not per test."""

import json
from pathlib import Path

import pytest

from mechanics.k1_report import write_run
from mechanics.k1_rigid import K1Experiment
from scenarios.k1_variants import time_refinement_specs
from tactile_contract import ExperimentSpec
from validation.k1_acceptance import acceptance_report

ROOT = Path(__file__).resolve().parent
K1_SPEC_PATH = ROOT / "scenarios" / "k0" / "k1_single_probe_cpu.json"


@pytest.fixture(scope="session")
def k1_spec() -> ExperimentSpec:
    return ExperimentSpec.from_dict(json.loads(K1_SPEC_PATH.read_text(encoding="utf-8")))


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
