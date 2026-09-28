"""Public contracts independent of simulation and actuator backends."""

from .interaction import INTERACTION_STATE_SCHEMA, InteractionState
from .experiment import EXPERIMENT_MANIFEST_SCHEMA, ExperimentManifest, comparison_gate, validate_run
from .experiment_spec import EXPERIMENT_SPEC_SCHEMA, ExperimentSpec, spec_comparison_gate, unresolved_settings

__all__ = ["InteractionState", "ExperimentManifest", "comparison_gate", "validate_run",
           "ExperimentSpec", "spec_comparison_gate", "unresolved_settings",
           "INTERACTION_STATE_SCHEMA", "EXPERIMENT_MANIFEST_SCHEMA", "EXPERIMENT_SPEC_SCHEMA"]
