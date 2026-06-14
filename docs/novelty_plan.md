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

The learned residual policy is close to the rule baseline but does not yet outperform
it on the three downloaded transition sequences:

- Rule mean RMSE: about `12.06 m`.
- Learned residual mean RMSE: about `12.42 m`.

This is a useful research result: with only three transition sequences and proxy
features, the learned model is not yet sufficiently general. The code now supports
the experiment needed to improve it.

## Next Research Steps

1. Download more INSANE sequences, especially indoor, outdoor, and Mars traverses.
2. Train on many more sequences and keep transition sequences as held-out tests.
3. Add calibrated UWB range fusion using anchor/tag positions.
4. Add attitude-compensated IMU propagation instead of conservative acceleration cues.
5. Replace logistic regression with a temporal reliability model over 1-3 seconds.
6. Report ablations:
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

