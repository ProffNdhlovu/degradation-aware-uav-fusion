from __future__ import annotations

from dataclasses import dataclass

import numpy as np


FEATURE_NAMES = {
    "gnss": [
        "bias",
        "hdop",
        "cov_mean",
        "cov_max",
        "fix_ok",
        "gnss_vio_disagreement",
        "imu_accel_norm",
        "uwb_valid_count",
        "lidar_valid",
    ],
    "vio": [
        "bias",
        "feature_count",
        "tracking_age",
        "gnss_vio_disagreement",
        "vio_speed",
        "vio_position_var_mean",
        "imu_accel_norm",
        "uwb_valid_count",
        "lidar_valid",
    ],
}


@dataclass
class ReliabilityModel:
    sensor: str
    mean: np.ndarray
    scale: np.ndarray
    weights: np.ndarray

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        x = np.asarray(features, dtype=float)
        x = np.nan_to_num(x, nan=0.0, posinf=1e3, neginf=-1e3)
        z = (x - self.mean) / self.scale
        logits = np.clip(z @ self.weights, -40.0, 40.0)
        return 1.0 / (1.0 + np.exp(-logits))


def extract_sensor_features(data: dict[str, np.ndarray], sensor: str) -> np.ndarray:
    gnss = data["gnss"]
    vio = data["vio"]
    disagreement = np.linalg.norm(gnss - vio, axis=1)
    disagreement = np.nan_to_num(disagreement, nan=50.0, posinf=50.0)
    imu_norm = np.linalg.norm(data.get("imu_accel", np.zeros_like(gnss)), axis=1)
    uwb_count = data.get("uwb_valid_count", np.zeros(len(gnss)))
    lidar_valid = (data.get("lidar_range", np.zeros(len(gnss))) > 0.05).astype(float)

    if sensor == "gnss":
        cov = data.get("gnss_cov", np.ones_like(gnss))
        return np.column_stack(
            [
                np.ones(len(gnss)),
                data["hdop"],
                np.nanmean(cov, axis=1),
                np.nanmax(cov, axis=1),
                data["fix_ok"].astype(float),
                disagreement,
                imu_norm,
                uwb_count,
                lidar_valid,
            ]
        )
    if sensor == "vio":
        vio_velocity = data.get("vio_velocity", np.zeros_like(vio))
        vio_var = data.get("vio_position_var", np.ones_like(vio))
        return np.column_stack(
            [
                np.ones(len(vio)),
                data["features"],
                data["tracking_age"],
                disagreement,
                np.linalg.norm(np.nan_to_num(vio_velocity), axis=1),
                np.nanmean(vio_var, axis=1),
                imu_norm,
                uwb_count,
                lidar_valid,
            ]
        )
    raise ValueError(f"unsupported sensor: {sensor}")


def make_reliability_labels(
    data: dict[str, np.ndarray],
    sensor: str,
    threshold_m: float,
    horizon_steps: int = 4,
) -> np.ndarray:
    truth = data["truth"]
    measurement = data[sensor]
    error = np.linalg.norm(measurement - truth, axis=1)
    labels = np.zeros(len(error), dtype=float)
    for i in range(len(error)):
        j = min(len(error), i + horizon_steps + 1)
        window = error[i:j]
        finite = np.isfinite(window)
        labels[i] = float(np.any(finite) and np.nanmean(window[finite]) <= threshold_m)
    labels[~np.isfinite(error)] = 0.0
    return labels


def train_logistic_reliability(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    epochs: int = 800,
    lr: float = 0.08,
    l2: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(features, dtype=float)
    y = np.asarray(labels, dtype=float)
    valid = np.isfinite(x).all(axis=1) & np.isfinite(y)
    x = np.nan_to_num(x[valid], nan=0.0, posinf=1e3, neginf=-1e3)
    y = y[valid]
    if len(np.unique(y)) < 2:
        y = np.concatenate([y, 1.0 - y[:1]])
        x = np.vstack([x, x[:1]])

    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    mean[0] = 0.0
    scale[0] = 1.0
    scale = np.maximum(scale, 1e-6)
    z = (x - mean) / scale

    pos = max(np.sum(y == 1.0), 1.0)
    neg = max(np.sum(y == 0.0), 1.0)
    sample_weight = np.where(y > 0.5, len(y) / (2.0 * pos), len(y) / (2.0 * neg))
    weights = np.zeros(z.shape[1], dtype=float)
    for _ in range(epochs):
        logits = np.clip(z @ weights, -40.0, 40.0)
        pred = 1.0 / (1.0 + np.exp(-logits))
        grad = z.T @ ((pred - y) * sample_weight) / len(y)
        grad += l2 * np.r_[0.0, weights[1:]]
        weights -= lr * grad
    return mean, scale, weights


def fit_reliability_models(data_sets: list[dict[str, np.ndarray]]) -> dict[str, ReliabilityModel]:
    models = {}
    thresholds = {"gnss": 3.0, "vio": 2.0}
    for sensor in ("gnss", "vio"):
        features = np.vstack([extract_sensor_features(data, sensor) for data in data_sets])
        labels = np.concatenate(
            [make_reliability_labels(data, sensor, thresholds[sensor]) for data in data_sets]
        )
        mean, scale, weights = train_logistic_reliability(features, labels)
        models[sensor] = ReliabilityModel(sensor=sensor, mean=mean, scale=scale, weights=weights)
    return models


def save_models(path: str, models: dict[str, ReliabilityModel]) -> None:
    payload = {}
    for sensor, model in models.items():
        payload[f"{sensor}_mean"] = model.mean
        payload[f"{sensor}_scale"] = model.scale
        payload[f"{sensor}_weights"] = model.weights
    np.savez(path, **payload)


def load_models(path: str) -> dict[str, ReliabilityModel]:
    loaded = np.load(path, allow_pickle=False)
    models = {}
    for sensor in ("gnss", "vio"):
        models[sensor] = ReliabilityModel(
            sensor=sensor,
            mean=loaded[f"{sensor}_mean"],
            scale=loaded[f"{sensor}_scale"],
            weights=loaded[f"{sensor}_weights"],
        )
    return models

