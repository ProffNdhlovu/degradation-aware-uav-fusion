# Novelty Plan

## Proposed Contribution

The project is moving from a rule-only sensor-fusion baseline toward:

**Learned degradation-aware covariance adaptation for UAV navigation under sensor
degradation.**

The current novelty layer trains per-sensor reliability estimators from INSANE data.
Each estimator predicts whether GNSS or VIO will remain reliable over a short future
horizon. The fusion system then uses this reliability as a residual advisor on top of
the interpretable safety rules.

## What Is Implemented

- Future-error labels for GNSS and VIO reliability.
- Dependency-free logistic reliability models implemented in NumPy.
- Leave-one-transition-out benchmarking.
- Hybrid fusion mode that combines:
  - rule-based safety gates,
  - learned reliability,
  - adaptive covariance scaling,
  - innovation rejection.

## Current Result

After downloading all available sensor archives and converting 26 flight sequences,
the learned VIO reliability layer is close to the rule baseline:

- Rule mean RMSE: about `5.47 m`.
- Learned residual mean RMSE: about `5.54 m`.

This is a useful research result: with more data, the learned layer becomes mostly
safe, but it still does not beat the corrected rule baseline. Many Mars sequences
lack usable VIO, so the next gain must
come from stronger temporal features, calibrated UWB range fusion, and better IMU
frame handling.

## Next Research Steps

1. Add calibrated UWB range fusion using anchor/tag positions.
2. Add attitude-compensated IMU propagation instead of conservative acceleration cues.
3. Replace logistic regression with a temporal reliability model over 1-3 seconds.
4. Add image-derived visual-quality features once camera archives are used.
5. Report ablations:
   - fixed covariance EKF,
   - rule-only adaptive EKF,
   - learned-only reliability,
   - hybrid rule plus learned reliability,
   - oracle sensor selection upper bound.

## Paper Framing

A credible paper contribution should not claim that the current learned layer already
solves degradation. The stronger framing is:

> We introduce an interpretable degradation-aware fusion pipeline and a learned
> reliability-ablation framework for cross-domain UAV navigation. Initial results
> show strong gains from degradation-aware gating and reveal the data requirements
> for learned reliability to generalize across transition domains.
