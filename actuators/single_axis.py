"""Layer 5 single-axis force actuator with a declared ideal transfer model.

No hardware is connected at K1. The measurement below is produced by a
declared model, not by a sensor, and every record says so through
``model_id`` and ``status``. Its purpose is to close the five-layer loop and
to make the command path, the measurement path, the inverse model, saturation,
latency and the watchdog exist as real code with real records, so K5 can
replace the transfer model with a calibrated one without touching layers 1 to 4.

Transfer model: ``measured(t) = gain * commanded(t - latency_steps) + bias``,
saturated to ``[0, max_force_n]``. The inverse model is its exact inverse, so
at K1 the only error visible is the one the declared latency creates.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from tactile_contract.pipeline import ActuatorCommand, ActuatorMeasurement

IDEAL_MODEL_ID = "single_axis_ideal_first_order_v0"
INVERSE_MODEL_ID = "single_axis_linear_inverse_v0"
MODEL_STATUS = "modelled_not_measured"


@dataclass(frozen=True)
class SingleAxisActuatorConfig:
    """Declared capability envelope of one force channel."""

    actuator_id: str
    channel: str = "normal_force"
    max_force_n: float = 5.0
    min_force_n: float = 0.0
    gain: float = 1.0
    bias_n: float = 0.0
    latency_steps: int = 1
    control_dt_s: float = 1.0 / 60.0
    watchdog_force_limit_n: float = 5.0

    def __post_init__(self):
        if self.max_force_n <= self.min_force_n:
            raise ValueError("Actuator range must be non-empty")
        if self.gain <= 0:
            raise ValueError("Transfer gain must be positive")
        if self.latency_steps < 0:
            raise ValueError("Latency cannot be negative")
        if self.watchdog_force_limit_n < self.max_force_n:
            raise ValueError("Watchdog limit must cover the declared range")

    @property
    def latency_s(self) -> float:
        return self.latency_steps * self.control_dt_s

    def to_dict(self) -> dict:
        return {
            "actuator_id": self.actuator_id,
            "channel": self.channel,
            "max_force_n": self.max_force_n,
            "min_force_n": self.min_force_n,
            "gain": self.gain,
            "bias_n": self.bias_n,
            "latency_steps": self.latency_steps,
            "latency_s": self.latency_s,
            "watchdog_force_limit_n": self.watchdog_force_limit_n,
            "transfer_model_id": IDEAL_MODEL_ID,
            "inverse_model_id": INVERSE_MODEL_ID,
            "status": MODEL_STATUS,
        }


class SingleAxisForceActuator:
    """Layer 5: command path, declared transfer model, measurement and error."""

    layer = 5
    stage_id = IDEAL_MODEL_ID

    def __init__(self, config: SingleAxisActuatorConfig):
        self.config = config
        self._pipeline: deque[float] = deque()
        self._watchdog_tripped = False
        self.reset()

    def reset(self) -> None:
        self._pipeline = deque([0.0] * self.config.latency_steps)
        self._watchdog_tripped = False

    def status(self) -> str:
        return "watchdog_tripped" if self._watchdog_tripped else "ok"

    def inverse(self, target_force_n: float) -> tuple[float, bool, float]:
        """Inverse transfer model with explicit saturation reporting."""
        config = self.config
        desired = (target_force_n - config.bias_n) / config.gain
        commanded = min(max(desired, config.min_force_n), config.max_force_n)
        return commanded, commanded != desired, desired - commanded

    def step(self, target) -> tuple[ActuatorCommand, ActuatorMeasurement]:
        config = self.config
        commanded, saturated, clipped = self.inverse(target.normal_force_n)
        if config.latency_steps:
            self._pipeline.append(commanded)
            delivered_command = self._pipeline.popleft()
        else:
            delivered_command = commanded
        measured = config.gain * delivered_command + config.bias_n
        if measured > config.watchdog_force_limit_n:
            self._watchdog_tripped = True
            measured = config.watchdog_force_limit_n
        command = ActuatorCommand(
            time_s=target.time_s,
            control_step=target.control_step,
            actuator_id=config.actuator_id,
            channel=config.channel,
            commanded_force_n=commanded,
            saturated=saturated,
            clipped_by_n=clipped,
            inverse_model_id=INVERSE_MODEL_ID,
        )
        measurement = ActuatorMeasurement(
            time_s=target.time_s,
            control_step=target.control_step,
            actuator_id=config.actuator_id,
            channel=config.channel,
            target_force_n=target.normal_force_n,
            measured_force_n=measured,
            error_n=measured - target.normal_force_n,
            latency_s=config.latency_s,
            model_id=IDEAL_MODEL_ID,
            status=MODEL_STATUS,
            watchdog_tripped=self._watchdog_tripped,
        )
        return command, measurement
