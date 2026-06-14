from sdf_nav.evaluate import evaluate_run
from sdf_nav.sim import make_synthetic_run


def test_fused_navigation_beats_degraded_gnss() -> None:
    data = make_synthetic_run(n=240, seed=11)
    metrics = evaluate_run(data)
    assert metrics["rmse_fused_m"] < metrics["rmse_gnss_m"]
    assert metrics["rmse_fused_m"] < 2.5

