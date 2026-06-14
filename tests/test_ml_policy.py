import numpy as np

from sdf_nav.ml_policy import ReliabilityModel, train_logistic_reliability


def test_logistic_reliability_learns_separable_signal() -> None:
    features = np.array(
        [
            [1.0, 0.0],
            [1.0, 0.2],
            [1.0, 0.8],
            [1.0, 1.0],
        ]
    )
    labels = np.array([0.0, 0.0, 1.0, 1.0])
    mean, scale, weights = train_logistic_reliability(features, labels, epochs=500)
    model = ReliabilityModel("test", mean, scale, weights)

    low = model.predict_proba(np.array([[1.0, 0.1]]))[0]
    high = model.predict_proba(np.array([[1.0, 0.9]]))[0]

    assert high > low
    assert high > 0.5
    assert low < 0.5

