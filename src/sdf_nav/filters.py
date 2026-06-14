from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Measurement:
    """Position measurement in meters with a 2x2 covariance."""

    z: np.ndarray
    r: np.ndarray
    sensor: str


class ConstantVelocityKalman2D:
    """Small 2D constant-velocity Kalman filter for navigation baselines."""

    def __init__(
        self,
        initial_position: np.ndarray,
        initial_velocity: np.ndarray | None = None,
        process_accel_std: float = 1.5,
    ) -> None:
        velocity = np.zeros(2) if initial_velocity is None else initial_velocity
        self.x = np.array(
            [initial_position[0], initial_position[1], velocity[0], velocity[1]],
            dtype=float,
        )
        self.p = np.diag([2.0, 2.0, 5.0, 5.0])
        self.process_accel_std = float(process_accel_std)

    def predict(self, dt: float) -> np.ndarray:
        dt = max(float(dt), 1e-3)
        f = np.array(
            [
                [1.0, 0.0, dt, 0.0],
                [0.0, 1.0, 0.0, dt],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        q_base = self.process_accel_std**2
        q = q_base * np.array(
            [
                [dt**4 / 4, 0.0, dt**3 / 2, 0.0],
                [0.0, dt**4 / 4, 0.0, dt**3 / 2],
                [dt**3 / 2, 0.0, dt**2, 0.0],
                [0.0, dt**3 / 2, 0.0, dt**2],
            ]
        )
        self.x = f @ self.x
        self.p = f @ self.p @ f.T + q
        return self.x.copy()

    def update(self, measurement: Measurement) -> np.ndarray:
        h = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
        y = measurement.z - h @ self.x
        s = h @ self.p @ h.T + measurement.r
        k = self.p @ h.T @ np.linalg.inv(s)
        self.x = self.x + k @ y
        ident = np.eye(4)
        self.p = (ident - k @ h) @ self.p
        return self.x.copy()

    @property
    def position(self) -> np.ndarray:
        return self.x[:2].copy()


class ImuDrivenKalman3D:
    """3D position/velocity Kalman filter with acceleration control input."""

    def __init__(
        self,
        initial_position: np.ndarray,
        initial_velocity: np.ndarray | None = None,
        accel_noise_std: float = 1.2,
    ) -> None:
        velocity = np.zeros(3) if initial_velocity is None else initial_velocity
        self.x = np.array(
            [
                initial_position[0],
                initial_position[1],
                initial_position[2],
                velocity[0],
                velocity[1],
                velocity[2],
            ],
            dtype=float,
        )
        self.p = np.diag([2.0, 2.0, 2.0, 5.0, 5.0, 5.0])
        self.accel_noise_std = float(accel_noise_std)

    def predict(self, dt: float, accel: np.ndarray | None = None) -> np.ndarray:
        dt = max(float(dt), 1e-3)
        u = np.zeros(3) if accel is None else np.asarray(accel, dtype=float)
        if not np.isfinite(u).all():
            u = np.zeros(3)

        f = np.eye(6)
        f[0, 3] = dt
        f[1, 4] = dt
        f[2, 5] = dt
        b = np.array(
            [
                [0.5 * dt**2, 0.0, 0.0],
                [0.0, 0.5 * dt**2, 0.0],
                [0.0, 0.0, 0.5 * dt**2],
                [dt, 0.0, 0.0],
                [0.0, dt, 0.0],
                [0.0, 0.0, dt],
            ]
        )
        q_block = np.array([[dt**4 / 4, dt**3 / 2], [dt**3 / 2, dt**2]])
        q = np.zeros((6, 6))
        for axis in range(3):
            idx = [axis, axis + 3]
            q[np.ix_(idx, idx)] = self.accel_noise_std**2 * q_block

        self.x = f @ self.x + b @ u
        self.p = f @ self.p @ f.T + q
        return self.x.copy()

    def update_position(self, z: np.ndarray, r: np.ndarray) -> np.ndarray:
        h = np.zeros((3, 6))
        h[0, 0] = 1.0
        h[1, 1] = 1.0
        h[2, 2] = 1.0
        return self._update(z, r, h)

    def update_velocity(self, z: np.ndarray, r: np.ndarray) -> np.ndarray:
        h = np.zeros((3, 6))
        h[0, 3] = 1.0
        h[1, 4] = 1.0
        h[2, 5] = 1.0
        return self._update(z, r, h)

    def _update(self, z: np.ndarray, r: np.ndarray, h: np.ndarray) -> np.ndarray:
        z = np.asarray(z, dtype=float)
        if not np.isfinite(z).all():
            return self.x.copy()
        y = z - h @ self.x
        s = h @ self.p @ h.T + r
        k = self.p @ h.T @ np.linalg.inv(s)
        self.x = self.x + k @ y
        ident = np.eye(6)
        self.p = (ident - k @ h) @ self.p
        return self.x.copy()

    @property
    def position(self) -> np.ndarray:
        return self.x[:3].copy()
