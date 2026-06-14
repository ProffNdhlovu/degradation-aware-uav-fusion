import numpy as np

from sdf_nav.filters import Measurement
from sdf_nav.policy import DegradationAwarePolicy
from sdf_nav.quality import SensorQuality


def test_policy_rejects_bad_sensor() -> None:
    policy = DegradationAwarePolicy()
    measurement = Measurement(np.array([1.0, 2.0]), np.eye(2), "gnss")
    gated, decision = policy.gate_measurement(
        measurement,
        SensorQuality("gnss", 0.05, "no_fix"),
    )
    assert gated is None
    assert decision.action == "reject"


def test_policy_downweights_weak_sensor() -> None:
    policy = DegradationAwarePolicy()
    measurement = Measurement(np.array([1.0, 2.0]), np.eye(2), "vio")
    gated, decision = policy.gate_measurement(
        measurement,
        SensorQuality("vio", 0.5, "poor_visual_tracking"),
    )
    assert gated is not None
    assert decision.action == "downweight"
    assert gated.r[0, 0] > measurement.r[0, 0]

