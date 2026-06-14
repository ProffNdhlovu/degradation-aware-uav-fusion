from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sdf_nav.filters import Measurement
from sdf_nav.quality import SensorQuality


@dataclass(frozen=True)
class SensorDecision:
    sensor: str
    action: str
    reliability: float
    covariance_scale: float
    reason: str


class DegradationAwarePolicy:
    """Decides whether a sensor measurement should be fused."""

    def __init__(self, reject_below: float = 0.2, downweight_below: float = 0.65) -> None:
        self.reject_below = reject_below
        self.downweight_below = downweight_below

    def decide(self, quality: SensorQuality) -> SensorDecision:
        if quality.reliability <= self.reject_below:
            return SensorDecision(
                quality.sensor,
                "reject",
                quality.reliability,
                np.inf,
                quality.reason,
            )
        if quality.reliability < self.downweight_below:
            scale = 1.0 / max(quality.reliability, 1e-3) ** 2
            return SensorDecision(
                quality.sensor,
                "downweight",
                quality.reliability,
                scale,
                quality.reason,
            )
        return SensorDecision(quality.sensor, "use", quality.reliability, 1.0, quality.reason)

    def gate_measurement(
        self,
        measurement: Measurement,
        quality: SensorQuality,
    ) -> tuple[Measurement | None, SensorDecision]:
        decision = self.decide(quality)
        if decision.action == "reject":
            return None, decision
        gated = Measurement(
            z=measurement.z,
            r=measurement.r * decision.covariance_scale,
            sensor=measurement.sensor,
        )
        return gated, decision

