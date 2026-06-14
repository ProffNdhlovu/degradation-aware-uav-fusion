# Aligned Sensor Feature Contract

Convert each INSANE sequence into one time-aligned table before model training.
Use a fixed rate such as 10 Hz or 20 Hz for the decision policy.

## Required Columns

Each row is one fusion decision timestamp.

| Column | Meaning |
| --- | --- |
| `t` | seconds from sequence start |
| `x_gt`, `y_gt`, `z_gt` | ground-truth position |
| `gnss_x`, `gnss_y`, `gnss_z` | GNSS-derived position, if available |
| `gnss_hdop` | GNSS horizontal dilution of precision |
| `gnss_num_sats` | number of satellites |
| `gnss_fix_ok` | boolean fix flag |
| `vio_x`, `vio_y`, `vio_z` | visual-inertial odometry position |
| `vio_feature_count` | tracked feature count or proxy |
| `vio_tracking_age_s` | time since tracking quality was last strong |
| `lidar_range_m` | downward laser range finder distance |
| `mag_norm_ut` | magnetometer field magnitude |
| `imu_accel_rms` | short-window acceleration RMS |
| `imu_gyro_rms` | short-window gyro RMS |

## Training Labels

Generate labels from future error against ground truth:

| Label | Meaning |
| --- | --- |
| `gnss_reliable` | `1` if GNSS error remains below threshold over the horizon |
| `vio_reliable` | `1` if VIO error remains below threshold over the horizon |
| `uwb_reliable` | same pattern for UWB when present |
| `mag_reliable` | `1` if heading innovation is acceptable |
| `safe_to_navigate` | `1` if fused position error is within mission tolerance |

Use task-relevant thresholds. For indoor/transition navigation, start with:

- Position reliable: error less than `1.5 m`.
- Heading reliable: error less than `10 deg`.
- Horizon: `0.5 s` to `2.0 s`.

## First INSANE Sequences

Prioritize these:

1. `transition_1`, `transition_2`, `transition_3`: main degradation and sensor switching cases.
2. `indoor_1`, `indoor_2`, `indoor_3`: controlled motion capture cases.
3. `outdoor_1`: clean outdoor GNSS/VIO baseline.
4. Mars sequences: domain transfer and visual texture stress testing.

## Adapter Boundary

The current code expects arrays like this:

```python
{
    "t": np.ndarray,
    "truth": np.ndarray,  # shape [n, 2] for the current 2D baseline
    "gnss": np.ndarray,
    "vio": np.ndarray,
    "hdop": np.ndarray,
    "sats": np.ndarray,
    "fix_ok": np.ndarray,
    "features": np.ndarray,
    "tracking_age": np.ndarray,
    "lidar_range": np.ndarray,
}
```

When the full INSANE adapter is added, keep this boundary stable and extend it to
3D plus attitude rather than changing the policy interface every time a sensor is
added.

