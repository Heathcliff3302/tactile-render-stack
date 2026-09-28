"""Layer 2 re-exports the shared frame contract without duplicating it."""

from tactile_contract.interaction import INTERACTION_STATE_SCHEMA, InteractionState

__all__ = ["InteractionState", "INTERACTION_STATE_SCHEMA"]

