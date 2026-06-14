# Degradation-Aware Sensor Fusion Navigation

This project is a starter pipeline for UAV navigation when sensors degrade or drop out.
It is designed around the INSANE dataset, which includes indoor, outdoor, transition,
and Mars-analog UAV flights with redundant IMUs, cameras, GNSS, UWB, magnetometer,
laser range finder, and ground truth.

The implemented estimator is intentionally practical:

1. Compute per-sensor quality features.
2. Decide whether to use, down-weight, or reject each measurement.
3. Predict a 3D position/velocity state with conservative IMU propagation.
4. Fuse accepted GNSS, VIO, and odometry-derived updates with a Kalman filter.
5. Score navigation quality against ground truth under normal and degraded conditions.

This gives you a working safety baseline before moving to a learned policy.

## Why This Shape

For safe navigation, the model should not only estimate pose. It should also decide
which sensors are trustworthy enough to use. INSANE is especially useful because it
contains real degradation modes:

- GNSS degradation and unavailability during outdoor-to-indoor transitions.
- Magnetometer field changes near indoor metal structures.
- Camera lighting changes during transitions.
- Laser range finder zero readings at longer ranges or difficult surfaces.
- Vibration-heavy IMU signals, including a high-rate IMU for analysis.

## Quick Start

```bash
python3 -m sdf_nav.sim --out work/synthetic_run.npz
python3 -m sdf_nav.evaluate --input work/synthetic_run.npz
```

Expected output includes RMSE for raw GNSS, VIO, and fused navigation, plus counts of
sensor decisions.

## Run On Downloaded INSANE Sensor Data

After downloading and unpacking an INSANE sensor archive, run:

```bash
python3 -m sdf_nav.insane \
  --sequence-dir data/insane/raw/transition_1_sensors \
  --out work/transition_1_aligned.npz \
  --eval
```

The adapter currently uses ground truth, PX4 GNSS, RealSense odometry, and laser
range finder CSVs. It treats NaN-only odometry as a missing degraded sensor.

To create a static route plot and browser-based replay:

```bash
python3 -m sdf_nav.visualize \
  --input work/transition_1_aligned.npz \
  --svg outputs/transition_1_navigation.svg \
  --html outputs/transition_1_navigation_sim.html
```

To train and benchmark the learned reliability layer:

```bash
python3 -m sdf_nav.train_reliability \
  --sequence work/transition_1_aligned.npz \
  --sequence work/transition_2_aligned.npz \
  --sequence work/transition_3_aligned.npz \
  --model-out work/reliability_model.npz \
  --report-out outputs/reliability_benchmark.md
```

## Project Layout

- `src/sdf_nav/quality.py`: sensor health and reliability features.
- `src/sdf_nav/policy.py`: degradation-aware use/down-weight/reject decisions.
- `src/sdf_nav/ml_policy.py`: learned reliability model and feature extraction.
- `src/sdf_nav/filters.py`: 2D baseline filter and 3D IMU-driven Kalman filter.
- `src/sdf_nav/sim.py`: synthetic degraded sensor generator for fast iteration.
- `src/sdf_nav/insane.py`: INSANE transition-sequence CSV adapter.
- `src/sdf_nav/evaluate.py`: metrics for navigation performance and decisions.
- `src/sdf_nav/visualize.py`: dependency-free SVG plot and HTML navigation replay.
- `src/sdf_nav/train_reliability.py`: leave-one-sequence-out learned-policy benchmark.

## Current Fusion Behavior

For downloaded INSANE transition sensor data, the 3D path uses:

- IMU acceleration for conservative prediction after initial bias/gravity removal.
- PX4 GNSS position with the dataset covariance columns `cov_p_x/y/z`.
- RealSense odometry as relative velocity plus guarded absolute position correction.
- Innovation gates so VIO is skipped when it strongly disagrees with the predicted state.
- UWB valid-range availability as a degradation signal. Full UWB range fusion needs
  anchor/tag calibration before range residual updates can be added safely.

## Research Direction

See `docs/novelty_plan.md` for the current paper-oriented contribution plan and
`outputs/reliability_benchmark.md` for the learned reliability ablation.
- `docs/architecture.md`: recommended model roadmap for INSANE.

## INSANE Dataset Integration Plan

Start with the sensor-data archives for `transition_1`, `transition_2`, and
`transition_3`, because they stress GNSS availability, lighting, UWB, and magnetic
field changes. Use `indoor_1..3` for controlled validation and `outdoor_1` plus the
Mars sequences for domain transfer tests.

The dataset license is BSD-2-Clause with a non-commercial restriction, so keep trained
artifacts and redistribution plans aligned with that license.
