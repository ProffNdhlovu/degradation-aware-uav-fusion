from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from sdf_nav.evaluate import evaluate_run, rmse
from sdf_nav.insane import load_transition_sequence
from sdf_nav.ml_policy import (
    extract_sensor_features,
    fit_reliability_models,
    load_models,
    make_reliability_labels,
    save_models,
)


def load_npz(path: Path) -> dict[str, np.ndarray]:
    loaded = np.load(path, allow_pickle=False)
    return {key: loaded[key] for key in loaded.files}


def load_dataset(path: Path) -> dict[str, np.ndarray]:
    if path.suffix == ".npz":
        return load_npz(path)
    return load_transition_sequence(path)


def classification_summary(
    data: dict[str, np.ndarray],
    models: dict[str, object],
    sensor: str,
    threshold_m: float,
) -> dict[str, float]:
    features = extract_sensor_features(data, sensor)
    labels = make_reliability_labels(data, sensor, threshold_m)
    scores = models[sensor].predict_proba(features)
    pred = scores >= 0.5
    truth = labels >= 0.5
    tp = float(np.sum(pred & truth))
    fp = float(np.sum(pred & ~truth))
    tn = float(np.sum(~pred & ~truth))
    fn = float(np.sum(~pred & truth))
    return {
        "accuracy": (tp + tn) / max(tp + fp + tn + fn, 1.0),
        "precision": tp / max(tp + fp, 1.0),
        "recall": tp / max(tp + fn, 1.0),
        "positive_rate": float(np.mean(pred)),
    }


def benchmark(data_sets: list[dict[str, np.ndarray]], names: list[str]) -> str:
    lines = [
        "# Learned Reliability Benchmark",
        "",
        "| Test sequence | Rule fused RMSE | Learned fused RMSE | GNSS RMSE | VIO RMSE | GNSS reliability acc. | VIO reliability acc. |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    rule_scores = []
    learned_scores = []
    for test_idx, test_data in enumerate(data_sets):
        train_data = [data for idx, data in enumerate(data_sets) if idx != test_idx]
        models = fit_reliability_models(train_data)
        rule = evaluate_run(test_data)
        learned = evaluate_run(test_data, models)
        gnss_summary = classification_summary(test_data, models, "gnss", threshold_m=3.0)
        vio_summary = classification_summary(test_data, models, "vio", threshold_m=2.0)
        rule_scores.append(rule["rmse_fused_m"])
        learned_scores.append(learned["rmse_fused_m"])
        lines.append(
            "| "
            f"{names[test_idx]} | "
            f"{rule['rmse_fused_m']:.2f} | "
            f"{learned['rmse_fused_m']:.2f} | "
            f"{rule['rmse_gnss_m']:.2f} | "
            f"{rule['rmse_vio_m']:.2f} | "
            f"{gnss_summary['accuracy']:.2f} | "
            f"{vio_summary['accuracy']:.2f} |"
        )
    lines.extend(
        [
            "",
            f"Mean rule fused RMSE: {np.nanmean(rule_scores):.2f} m",
            f"Mean learned fused RMSE: {np.nanmean(learned_scores):.2f} m",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", type=Path, action="append", required=True)
    parser.add_argument("--model-out", type=Path, default=Path("work/reliability_model.npz"))
    parser.add_argument("--report-out", type=Path, default=Path("outputs/reliability_benchmark.md"))
    args = parser.parse_args()

    data_sets = [load_dataset(path) for path in args.sequence]
    names = [path.stem.replace("_aligned", "") for path in args.sequence]
    models = fit_reliability_models(data_sets)
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    save_models(str(args.model_out), models)
    # Reload once so the saved artifact path is validated too.
    load_models(str(args.model_out))

    report = benchmark(data_sets, names)
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(report, encoding="utf-8")
    print(f"wrote {args.model_out}")
    print(f"wrote {args.report_out}")
    print(report)


if __name__ == "__main__":
    main()

