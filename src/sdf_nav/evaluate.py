from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from sdf_nav.filters import ConstantVelocityKalman2D, ImuDrivenKalman3D, Measurement
from sdf_nav.policy import DegradationAwarePolicy
from sdf_nav.quality import gnss_quality, vio_quality


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    finite = np.isfinite(a).all(axis=1) & np.isfinite(b).all(axis=1)
    if not np.any(finite):
        return float("nan")
    return float(np.sqrt(np.mean(np.sum((a[finite] - b[finite]) ** 2, axis=1))))


def run_fusion(data: dict[str, np.ndarray]) -> tuple[np.ndarray, dict[str, int]]:
    if data["truth"].shape[1] == 3 and "imu_accel" in data:
        return run_fusion_3d(data)

    t = data["t"]
    truth = data["truth"]
    kf = ConstantVelocityKalman2D(initial_position=truth[0])
    policy = DegradationAwarePolicy()
    fused = np.zeros_like(truth)
    decisions: Counter[str] = Counter()

    for i in range(len(t)):
        dt = 0.1 if i == 0 else t[i] - t[i - 1]
        kf.predict(float(dt))
        predicted = kf.position

        candidates = []
        if np.isfinite(data["gnss"][i]).all():
            gnss_innov = float(np.linalg.norm(data["gnss"][i] - predicted))
            candidates.append(
                (
                    Measurement(data["gnss"][i], np.diag([0.7**2, 0.7**2]), "gnss"),
                    gnss_quality(
                        float(data["hdop"][i]),
                        int(data["sats"][i]),
                        bool(data["fix_ok"][i]),
                        gnss_innov,
                    ),
                )
            )
        else:
            decisions["gnss:reject:missing"] += 1

        if np.isfinite(data["vio"][i]).all():
            vio_innov = float(np.linalg.norm(data["vio"][i] - predicted))
            candidates.append(
                (
                    Measurement(data["vio"][i], np.diag([0.25**2, 0.25**2]), "vio"),
                    vio_quality(int(data["features"][i]), float(data["tracking_age"][i]), vio_innov),
                )
            )
        else:
            decisions["vio:reject:missing"] += 1
        for measurement, quality in candidates:
            gated, decision = policy.gate_measurement(measurement, quality)
            decisions[f"{decision.sensor}:{decision.action}:{decision.reason}"] += 1
            if gated is not None:
                kf.update(gated)
        fused[i] = kf.position

    return fused, dict(decisions)


def run_fusion_3d(data: dict[str, np.ndarray]) -> tuple[np.ndarray, dict[str, int]]:
    t = data["t"]
    truth = data["truth"]
    kf = ImuDrivenKalman3D(initial_position=truth[0])
    policy = DegradationAwarePolicy()
    fused = np.zeros_like(truth)
    decisions: Counter[str] = Counter()

    for i in range(len(t)):
        dt = 0.125 if i == 0 else t[i] - t[i - 1]
        kf.predict(float(dt), data["imu_accel"][i])
        predicted = kf.position

        if np.isfinite(data["gnss"][i]).all():
            gnss_innov = float(np.linalg.norm(data["gnss"][i] - predicted))
            quality = gnss_quality(
                float(data["hdop"][i]),
                int(data["sats"][i]),
                bool(data["fix_ok"][i]),
                gnss_innov,
            )
            _, decision = policy.gate_measurement(
                Measurement(data["gnss"][i, :2], np.eye(2), "gnss"),
                quality,
            )
            decisions[f"{decision.sensor}:{decision.action}:{decision.reason}"] += 1
            if decision.action != "reject":
                r = np.diag(np.clip(data["gnss_cov"][i], 0.25, 16.0)) * decision.covariance_scale
                kf.update_position(data["gnss"][i], r)
        else:
            decisions["gnss:reject:missing"] += 1

        if "vio_velocity" in data and np.isfinite(data["vio_velocity"][i]).all():
            vio_innov = float(np.linalg.norm(data["vio_velocity"][i, :2] - kf.x[3:5]))
            if vio_innov > 12.0:
                decisions["vio:reject:velocity_innovation"] += 1
                fused[i] = kf.position
                continue
            quality = vio_quality(
                int(data["features"][i]),
                float(data["tracking_age"][i]),
                vio_innov,
            )
            _, decision = policy.gate_measurement(
                Measurement(data["vio"][i, :2], np.eye(2), "vio"),
                quality,
            )
            decisions[f"{decision.sensor}:{decision.action}:{decision.reason}"] += 1
            if decision.action != "reject":
                velocity_var = np.maximum(data.get("vio_velocity_var", np.full_like(data["vio_velocity"], 2.0))[i], 0.05)
                r = np.diag(velocity_var) * decision.covariance_scale
                kf.update_velocity(data["vio_velocity"][i], r)

                if np.isfinite(data["vio"][i]).all():
                    position_innov = float(np.linalg.norm(data["vio"][i, :2] - kf.position[:2]))
                    if position_innov > 15.0:
                        decisions["vio:reject:position_innovation"] += 1
                        fused[i] = kf.position
                        continue
                    position_var = np.maximum(
                        data.get("vio_position_var", np.full_like(data["vio"], 4.0))[i],
                        0.05,
                    )
                    innovation_scale = max(1.0, (position_innov / 3.0) ** 2)
                    kf.update_position(
                        data["vio"][i],
                        np.diag(position_var) * decision.covariance_scale * innovation_scale,
                    )
        elif np.isfinite(data["vio"][i]).all():
            vio_innov = float(np.linalg.norm(data["vio"][i] - predicted))
            quality = vio_quality(
                int(data["features"][i]),
                float(data["tracking_age"][i]),
                vio_innov,
            )
            _, decision = policy.gate_measurement(
                Measurement(data["vio"][i, :2], np.eye(2), "vio"),
                quality,
            )
            decisions[f"{decision.sensor}:{decision.action}:{decision.reason}"] += 1
            if decision.action != "reject":
                kf.update_position(data["vio"][i], np.diag([1.0, 1.0, 1.0]) * decision.covariance_scale)
        else:
            decisions["vio:reject:missing"] += 1

        if "uwb_valid_count" in data:
            if data["uwb_valid_count"][i] > 0:
                decisions["uwb:observe:ranges_available"] += 1
            else:
                decisions["uwb:reject:missing"] += 1

        fused[i] = kf.position

    return fused, dict(decisions)


def evaluate_run(data: dict[str, np.ndarray]) -> dict[str, object]:
    truth = data["truth"]
    fused, decisions = run_fusion(data)
    return {
        "rmse_fused_m": rmse(fused, truth),
        "rmse_gnss_m": rmse(data["gnss"], truth),
        "rmse_vio_m": rmse(data["vio"], truth),
        "decisions": decisions,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    loaded = np.load(args.input, allow_pickle=False)
    metrics = evaluate_run({key: loaded[key] for key in loaded.files})
    for key, value in metrics.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
