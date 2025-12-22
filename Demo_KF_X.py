#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
KF for (x, y) using Scheme A:
- True contact position p = [x, y] follows random walk
- Two slow-varying biases b1, b2
- One shared first-order lag state s (soft layer + readout)

Observation model:
    z1 = s + b1 + noise
    z2 = s + b2 + noise

Public API:
    clean_xy(z1, z2, dt) -> (x_clean, y_clean)
    reset_filter_with_init(z_init=(x0, y0), params=XYKFParams(...))
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np


# =============================================================================
# Parameters and State
# =============================================================================

@dataclass
class XYKFParams:
    """Configuration parameters for the XY Kalman filter."""
    tau: float = 0.08

    # Process noise PSDs (scaled internally by dt)
    q_p: float = 1e-2
    q_b1: float = 1e-9
    q_b2: float = 1e-9
    q_s: float = 1e-6

    # Measurement noise covariances
    R1: np.ndarray = field(
        default_factory=lambda: np.diag([1e-3, 1e-3])
    )
    R2: np.ndarray = field(
        default_factory=lambda: np.diag([1e-3, 1e-3])
    )


@dataclass
class XYKFState:
    """
    Kalman filter state container.

    State vector layout:
        [p(2), b1(2), b2(2), s(2)] -> total dimension = 8
    """
    x: np.ndarray
    P: np.ndarray
    params: XYKFParams


# Global singleton state for simple functional API
_STATE: Optional[XYKFState] = None


# =============================================================================
# Core Kalman Filter Routines
# =============================================================================

def _predict(state: XYKFState, dt: float) -> None:
    """
    Time update:
        x_prior = F x
        P_prior = F P F^T + Q(dt)
    """
    params = state.params

    I2 = np.eye(2)
    Z2 = np.zeros((2, 2))

    dt_eff = max(float(dt), 1e-9)
    a = float(np.exp(-dt_eff / params.tau))

    # State transition matrix (8x8)
    F = np.block([
        [I2,         Z2,        Z2,        Z2],
        [Z2,         I2,        Z2,        Z2],
        [Z2,         Z2,        I2,        Z2],
        [(1 - a)*I2, Z2,        Z2,        a*I2],
    ])

    # Process noise covariance
    Q_diag = np.array(
        [
            params.q_p, params.q_p,
            params.q_b1, params.q_b1,
            params.q_b2, params.q_b2,
            params.q_s, params.q_s,
        ],
        dtype=float,
    )
    Q = np.diag(Q_diag) * dt_eff

    state.x = F @ state.x
    state.P = F @ state.P @ F.T + Q


def _update(state: XYKFState, z: np.ndarray, which: int) -> None:
    """
    Measurement update for a single channel.

    Args:
        which = 1 -> z1 = b1 + s
        which = 2 -> z2 = b2 + s
    """
    assert which in (1, 2), "which must be 1 or 2"

    I2 = np.eye(2)

    if which == 1:
        H = np.hstack([np.zeros((2, 2)), I2, np.zeros((2, 2)), I2])
        R = state.params.R1
    else:
        H = np.hstack([np.zeros((2, 4)), I2, I2])
        R = state.params.R2

    x_pred = state.x
    P_pred = state.P

    z = np.asarray(z, dtype=float).reshape(2,)
    y = z - H @ x_pred
    S = H @ P_pred @ H.T + R
    K = P_pred @ H.T @ np.linalg.inv(S)

    state.x = x_pred + K @ y
    state.P = (np.eye(P_pred.shape[0]) - K @ H) @ P_pred


# =============================================================================
# Public API
# =============================================================================

def clean_xy(
    z1: Optional[Tuple[float, float]],
    z2: Optional[Tuple[float, float]],
    dt: float,
) -> Tuple[float, float]:
    """
    One-step call into the XY Kalman filter.

    Args:
        z1 : (x, y) from channel-1 or None
        z2 : (x, y) from channel-2 or None
        dt : time step in seconds

    Returns:
        (x_clean, y_clean): de-biased, de-lagged position estimate
    """
    global _STATE

    if _STATE is None:
        reset_filter_with_init(z_init=(0.0, 0.0))

    _predict(_STATE, dt)

    if z1 is not None:
        _update(_STATE, np.asarray(z1, dtype=float), which=1)

    if z2 is not None:
        _update(_STATE, np.asarray(z2, dtype=float), which=2)

    x_clean = float(_STATE.x[0])
    y_clean = float(_STATE.x[1])

    return x_clean, y_clean


def reset_filter_with_init(
    z_init: Tuple[float, float] = (0.0, 0.0),
    params: Optional[XYKFParams] = None,
) -> None:
    """
    Reset the global filter with Scheme A priors.

    Args:
        z_init : initial (x, y) measurement
        params : optional XYKFParams override
    """
    global _STATE

    if params is None:
        params = XYKFParams()

    x0 = np.zeros(8, dtype=float)
    z0 = np.asarray(z_init, dtype=float).reshape(2,)

    # Initialize p and s near the first measurement
    x0[0:2] = z0
    x0[6:8] = z0

    P0 = np.diag([
        1.0, 1.0,      # p
        1e-3, 1e-3,    # b1
        1e-3, 1e-3,    # b2
        1e-2, 1e-2,    # s
    ])

    _STATE = XYKFState(x=x0, P=P0, params=params)


# =============================================================================
# Demo
# =============================================================================

def main():
    """
    Demo:
    - Both channels observe a fixed contact point.
    - Replace z1 / z2 with real sensor data in practice.
    """
    np.set_printoptions(precision=4, suppress=True)

    reset_filter_with_init(z_init=(9.0, 18.0))

    T = 300.0
    dt = 0.01
    N = int(T / dt)

    z1 = (0.0, 0.0)
    z2 = (0.0, 0.0)

    print("t(s)\tx_clean\ty_clean")
    t = 0.0

    for k in range(N):
        x_clean, y_clean = clean_xy(z1, z2, dt)
        if k % 10 == 0:
            print(f"{t:5.2f}\t{x_clean:8.4f}\t{y_clean:8.4f}")
        t += dt


if __name__ == "__main__":
    main()
