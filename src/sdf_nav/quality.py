from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SensorQuality:
    sensor: str
    reliability: float
    reason: str


def clamp01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def gnss_quality(hdop: float, num_sats: int, fix_ok: bool, innovation_m: float) -> SensorQuality:
    if not fix_ok or num_sats < 4:
        return SensorQuality("gnss", 0.0, "no_fix")
    hdop_score = clamp01(1.0 - (hdop - 0.8) / 4.0)
    sats_score = clamp01((num_sats - 4) / 10.0)
    innovation_score = clamp01(1.0 - innovation_m / 8.0)
    reliability = 0.45 * hdop_score + 0.25 * sats_score + 0.30 * innovation_score
    return SensorQuality("gnss", reliability, "nominal" if reliability > 0.6 else "weak_geometry")


def vio_quality(feature_count: int, tracking_age_s: float, innovation_m: float) -> SensorQuality:
    feature_score = clamp01(feature_count / 120.0)
    age_score = clamp01(1.0 - tracking_age_s / 2.0)
    innovation_score = clamp01(1.0 - innovation_m / 5.0)
    reliability = 0.45 * feature_score + 0.25 * age_score + 0.30 * innovation_score
    reason = "nominal" if reliability > 0.55 else "poor_visual_tracking"
    return SensorQuality("vio", reliability, reason)


def lidar_quality(range_m: float, max_range_m: float = 40.0) -> SensorQuality:
    if range_m <= 0.05:
        return SensorQuality("lidar", 0.0, "zero_or_invalid_range")
    if range_m > max_range_m:
        return SensorQuality("lidar", 0.0, "out_of_range")
    reliability = clamp01(1.0 - max(0.0, range_m - 25.0) / 15.0)
    return SensorQuality("lidar", reliability, "nominal" if reliability > 0.5 else "long_range")


def magnetometer_quality(norm_ut: float, reference_norm_ut: float, innovation_rad: float) -> SensorQuality:
    norm_error = abs(norm_ut - reference_norm_ut) / max(reference_norm_ut, 1e-6)
    norm_score = clamp01(1.0 - norm_error / 0.35)
    innovation_score = clamp01(1.0 - innovation_rad / 0.8)
    reliability = 0.55 * norm_score + 0.45 * innovation_score
    reason = "nominal" if reliability > 0.55 else "field_distortion"
    return SensorQuality("mag", reliability, reason)

