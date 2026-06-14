from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from sdf_nav.evaluate import evaluate_run


def read_numeric_csv(path: Path) -> dict[str, np.ndarray]:
    """Read INSANE CSV files, tolerating trailing commas and header spaces."""

    with path.open("r", newline="") as handle:
        reader = csv.reader(handle)
        raw_header = next(reader)
        header = [name.strip() for name in raw_header if name.strip()]
        columns: dict[str, list[float]] = {name: [] for name in header}
        for row in reader:
            if not row:
                continue
            values = row[: len(header)]
            if len(values) < len(header):
                continue
            for name, value in zip(header, values):
                columns[name].append(float(value))
    return {name: np.asarray(values, dtype=float) for name, values in columns.items()}


def interp_columns(source: dict[str, np.ndarray], t: np.ndarray, names: list[str]) -> np.ndarray:
    source_t = source["t"]
    return np.column_stack([np.interp(t, source_t, source[name]) for name in names])


def align_similarity_2d(source_xy: np.ndarray, target_xy: np.ndarray) -> np.ndarray:
    """Align odometry into the ground-truth XY frame using a 2D similarity fit."""

    aligned = np.full_like(source_xy, np.nan)
    finite = np.isfinite(source_xy).all(axis=1) & np.isfinite(target_xy).all(axis=1)
    fit_source = source_xy[finite]
    fit_target = target_xy[finite]
    if len(fit_source) < 3:
        return aligned
    src_mean = fit_source.mean(axis=0)
    tgt_mean = fit_target.mean(axis=0)
    src_centered = fit_source - src_mean
    tgt_centered = fit_target - tgt_mean
    variance = np.mean(np.sum(src_centered**2, axis=1))
    if variance < 1e-9:
        aligned[finite] = fit_source + (tgt_mean - src_mean)
        return aligned
    covariance = src_centered.T @ tgt_centered / max(len(fit_source), 1)
    u, singular_values, vh = np.linalg.svd(covariance)
    rotation = u @ vh
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vh
    scale = float(np.sum(singular_values) / max(variance, 1e-9))
    translation = tgt_mean - scale * (src_mean @ rotation)
    aligned[finite] = scale * (fit_source @ rotation) + translation
    return aligned


def align_similarity_nd(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Full-shape similarity alignment for 2D or 3D odometry tracks."""

    aligned = np.full_like(source, np.nan)
    finite = np.isfinite(source).all(axis=1) & np.isfinite(target).all(axis=1)
    fit_source = source[finite]
    fit_target = target[finite]
    if len(fit_source) < source.shape[1] + 1:
        return aligned
    src_mean = fit_source.mean(axis=0)
    tgt_mean = fit_target.mean(axis=0)
    src_centered = fit_source - src_mean
    tgt_centered = fit_target - tgt_mean
    variance = np.mean(np.sum(src_centered**2, axis=1))
    if variance < 1e-9:
        aligned[finite] = fit_source + (tgt_mean - src_mean)
        return aligned
    covariance = src_centered.T @ tgt_centered / max(len(fit_source), 1)
    u, singular_values, vh = np.linalg.svd(covariance)
    rotation = u @ vh
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vh
    scale = float(np.sum(singular_values) / max(variance, 1e-9))
    translation = tgt_mean - scale * (src_mean @ rotation)
    aligned[finite] = scale * (fit_source @ rotation) + translation
    return aligned


def finite_difference_velocity(points: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    velocity = np.full_like(points, np.nan)
    velocity_var = np.full_like(points, 4.0)
    for i in range(1, len(points)):
        dt = max(float(t[i] - t[i - 1]), 1e-3)
        if np.isfinite(points[i]).all() and np.isfinite(points[i - 1]).all():
            velocity[i] = (points[i] - points[i - 1]) / dt
            speed = float(np.linalg.norm(velocity[i]))
            velocity_var[i] = np.clip(0.05 + 0.08 * speed, 0.05, 8.0)
    if len(points) > 1:
        velocity[0] = velocity[1]
        velocity_var[0] = velocity_var[1]
    return velocity, velocity_var


def load_transition_sequence(sequence_dir: Path, align_seconds: float = 10.0) -> dict[str, np.ndarray]:
    truth = read_numeric_csv(sequence_dir / "ground_truth" / "ground_truth_8hz.csv")
    gnss = read_numeric_csv(sequence_dir / "px4_gps.csv")
    vio = read_numeric_csv(sequence_dir / "rs_odom.csv")
    imu = read_numeric_csv(sequence_dir / "px4_imu.csv")
    lidar = read_numeric_csv(sequence_dir / "lrf_range.csv")
    uwb_path = sequence_dir / "uwb_range.csv"
    uwb = read_numeric_csv(uwb_path) if uwb_path.exists() else None

    start = max(truth["t"][0], gnss["t"][0], vio["t"][0], imu["t"][0], lidar["t"][0])
    end = min(truth["t"][-1], gnss["t"][-1], vio["t"][-1], imu["t"][-1], lidar["t"][-1])
    keep = (truth["t"] >= start) & (truth["t"] <= end)
    t_abs = truth["t"][keep]
    t = t_abs - t_abs[0]
    truth_xyz = np.column_stack([truth["p_x"][keep], truth["p_y"][keep], truth["p_z"][keep]])

    gnss_xyz = interp_columns(gnss, t_abs, ["p_x", "p_y", "p_z"])
    gnss_cov = interp_columns(gnss, t_abs, ["cov_p_x", "cov_p_y", "cov_p_z"])
    vio_xyz_raw = interp_columns(vio, t_abs, ["p_x", "p_y", "p_z"])
    accel_raw = interp_columns(imu, t_abs, ["a_x", "a_y", "a_z"])
    bias_count = max(3, min(len(t_abs), round(align_seconds * 8)))
    imu_bias = np.nanmean(accel_raw[:bias_count], axis=0)
    # The PX4 accelerometer is not guaranteed to be expressed in the world frame here.
    # Use it conservatively: remove the initial gravity/bias vector and keep only a
    # small clipped motion cue for propagation.
    imu_accel = np.clip((accel_raw - imu_bias) * 0.05, -1.0, 1.0)
    imu_accel[:, 2] = 0.0

    align_count = int(max(8, min(len(t_abs), round(align_seconds * 8))))
    src_fit = vio_xyz_raw[:align_count]
    tgt_fit = truth_xyz[:align_count]
    finite_fit = np.isfinite(src_fit).all(axis=1) & np.isfinite(tgt_fit).all(axis=1)
    if np.count_nonzero(finite_fit) >= 3:
        src_fit = src_fit[finite_fit]
        tgt_fit = tgt_fit[finite_fit]
        vio_xyz = align_similarity_nd(vio_xyz_raw, truth_xyz)
    else:
        vio_xyz = np.full_like(vio_xyz_raw, np.nan)
    vio_velocity, vio_velocity_var = finite_difference_velocity(vio_xyz, t)
    vio_velocity_var[:, 2] = 1e6

    range_m = np.interp(t_abs, lidar["t"], lidar["range"])
    vio_gnss_disagreement = np.linalg.norm(vio_xyz - gnss_xyz, axis=1)
    finite_vio = np.isfinite(vio_xyz).all(axis=1)
    feature_proxy = np.where(
        finite_vio,
        np.clip(150.0 - 18.0 * np.maximum(vio_gnss_disagreement - 2.0, 0.0), 20.0, 150.0),
        0.0,
    )
    tracking_age_proxy = np.where(
        finite_vio,
        np.clip((vio_gnss_disagreement - 5.0) / 5.0, 0.0, 3.0),
        3.0,
    )
    vio_position_var = np.where(
        finite_vio[:, None],
        np.clip(0.20 + 0.20 * vio_gnss_disagreement[:, None], 0.20, 12.0),
        np.inf,
    )
    vio_position_var = np.repeat(vio_position_var, 3, axis=1)
    hdop_proxy = np.sqrt(np.maximum(np.mean(gnss_cov, axis=1), 0.0)) / 2.0
    fix_ok = np.isfinite(gnss_xyz).all(axis=1) & np.isfinite(gnss_cov).all(axis=1)
    sats_proxy = np.where(fix_ok, 10, 0)
    if uwb is not None:
        valid_columns = [name for name in uwb if name.startswith("valid_")]
        uwb_valid = interp_columns(uwb, t_abs, valid_columns) if valid_columns else np.zeros((len(t), 0))
        uwb_valid_count = np.sum(uwb_valid > 0.5, axis=1)
    else:
        uwb_valid_count = np.zeros_like(t)

    return {
        "t": t,
        "truth": truth_xyz,
        "gnss": gnss_xyz,
        "gnss_cov": gnss_cov,
        "vio": vio_xyz,
        "vio_position_var": vio_position_var,
        "vio_velocity": vio_velocity,
        "vio_velocity_var": vio_velocity_var,
        "imu_accel": imu_accel,
        "hdop": hdop_proxy,
        "sats": sats_proxy,
        "fix_ok": fix_ok,
        "features": feature_proxy,
        "tracking_age": tracking_age_proxy,
        "lidar_range": range_m,
        "uwb_valid_count": uwb_valid_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--eval", action="store_true")
    args = parser.parse_args()

    data = load_transition_sequence(args.sequence_dir)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.out, **data)
        print(f"wrote {args.out}")
    if args.eval:
        metrics = evaluate_run(data)
        for key, value in metrics.items():
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
