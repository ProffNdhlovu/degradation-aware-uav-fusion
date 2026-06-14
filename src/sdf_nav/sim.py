from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def make_synthetic_run(n: int = 900, dt: float = 0.1, seed: int = 7) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    t = np.arange(n) * dt
    x = 18.0 * np.sin(0.045 * t) + 0.35 * t
    y = 10.0 * np.sin(0.07 * t + 0.7)
    truth = np.column_stack([x, y])

    gnss = truth + rng.normal(0.0, 0.7, size=(n, 2))
    vio = truth + rng.normal(0.0, 0.25, size=(n, 2))

    transition = (t > 32.0) & (t < 55.0)
    gnss[transition] += rng.normal(5.0, 3.5, size=(transition.sum(), 2))
    vio[transition] += np.column_stack(
        [
            np.linspace(0.0, 3.0, transition.sum()),
            np.linspace(0.0, -2.0, transition.sum()),
        ]
    )

    hdop = np.where(transition, rng.uniform(3.0, 8.0, n), rng.uniform(0.7, 1.8, n))
    sats = np.where(transition, rng.integers(2, 7, n), rng.integers(9, 16, n))
    fix_ok = sats >= 4
    features = np.where(transition, rng.integers(20, 80, n), rng.integers(90, 170, n))
    tracking_age = np.where(transition, rng.uniform(0.5, 2.8, n), rng.uniform(0.0, 0.4, n))
    lidar_range = 12.0 + 8.0 * np.sin(0.03 * t) + rng.normal(0.0, 0.2, n)
    lidar_range[(t > 62.0) & (t < 68.0)] = 0.0

    return {
        "t": t,
        "truth": truth,
        "gnss": gnss,
        "vio": vio,
        "hdop": hdop,
        "sats": sats,
        "fix_ok": fix_ok.astype(bool),
        "features": features,
        "tracking_age": tracking_age,
        "lidar_range": lidar_range,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("work/synthetic_run.npz"))
    parser.add_argument("--n", type=int, default=900)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **make_synthetic_run(n=args.n))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

