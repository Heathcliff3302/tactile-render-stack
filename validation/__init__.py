"""Validation: contract gates, acceptance reports and cross-backend comparison."""

from .contracts import comparison_gate, validate_run
from .k1_acceptance import (
    ACCEPTANCE_KEY_GATES,
    DEFERRED_ACCEPTANCE_KEYS,
    RUN_GATE_ORDER,
    acceptance_key_ledger,
    acceptance_report,
    evaluate_replay,
    evaluate_run,
    evaluate_time_refinement,
    manifest_from_run,
)

__all__ = [
    "ACCEPTANCE_KEY_GATES",
    "DEFERRED_ACCEPTANCE_KEYS",
    "RUN_GATE_ORDER",
    "acceptance_key_ledger",
    "acceptance_report",
    "comparison_gate",
    "evaluate_replay",
    "evaluate_run",
    "evaluate_time_refinement",
    "manifest_from_run",
    "validate_run",
]
