"""Layer 4 standard tactile target.

The target is device independent: it says what should be delivered to the
skin, not how any particular actuator delivers it. K1 declares one channel,
``normal_force``. Vibration and temperature stay ``None`` rather than zero,
because the K1 material baseline models neither transients nor heat, and a
zero would read as a real commanded value downstream.
"""

from __future__ import annotations

from tactile_contract.pipeline import STATUS_DERIVED, TactileTarget

K1_CHANNELS = ("normal_force",)


def contact_state_from_response(response) -> str:
    """Coarse interaction state used by the target layer and by replay."""
    if not response.contact_present:
        return "no_contact"
    phase = response.internal_state.get("motion_phase", "")
    if phase == "lateral_slide":
        return "sliding"
    if phase == "hold":
        return "hold"
    if phase == "retract":
        return "releasing"
    return "pressing"


class NormalForceTargetStage:
    """Layer 4: map a material response onto the declared target channels."""

    layer = 4
    stage_id = "normal_force_target_v1"

    def __init__(self, *, target_id: str, channels: tuple[str, ...] = K1_CHANNELS):
        self.target_id = target_id
        self.channels = tuple(channels)
        if self.channels != K1_CHANNELS:
            raise NotImplementedError("K1 declares the normal_force channel only")
        self._sequence: list[TactileTarget] = []

    def reset(self) -> None:
        self._sequence = []

    def status(self) -> str:
        return "ok"

    @property
    def sequence(self) -> tuple[TactileTarget, ...]:
        """Replayable target sequence for this run."""
        return tuple(self._sequence)

    def step(self, response) -> TactileTarget:
        target = TactileTarget(
            time_s=response.time_s,
            control_step=response.control_step,
            target_id=self.target_id,
            contact_state=contact_state_from_response(response),
            normal_force_n=response.normal_force_n,
            tangent_force_world_n=response.tangent_force_world_n,
            contact_area_m2=response.contact_area_m2,
            indentation_m=response.indentation_m,
            vibration_amplitude_n=None,
            vibration_frequency_hz=None,
            temperature_k=None,
            channels=self.channels,
            source_model_id=response.model_id,
            status=STATUS_DERIVED,
        )
        self._sequence.append(target)
        return target
