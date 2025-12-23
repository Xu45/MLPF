#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Demo_KF_F.py
-----------

Force Kalman Filter example with:
- Shared first-order lag state
- Two slow-varying bias states
- Full executable demo

State definition (12D):
    x = [F(3), b1(3), b2(3), s(3)]

Process model:
    F_k  = F_{k-1} + w_F
    b1_k = b1_{k-1} + w_b1
    b2_k = b2_{k-1} + w_b2
    s_k  = a*s_{k-1} + (1-a)*F_{k-1} + w_s
    a    = exp(-dt / tau)

Measurement model:
    z1 = s + b1 + v1
    z2 = s + b2 + v2

Usage:
- Assign z1, z2 in main() as your 3D force measurements or None
- Call fuse_force_step() to obtain F_clean (bias-free, de-lagged force)
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


# =============================================================================
# Parameters and State Containers
# =============================================================================

@dataclass
class ForceKFParams:
    """Kalman filter configuration parameters."""
    qF: float              # Force process noise power (N^2/s)
    qb1: float             # Bias-1 process noise power
    qb2: float             # Bias-2 process noise power
    qs: float              # Sensor lag state process noise power
    R1: np.ndarray         # (3x3) Measurement noise covariance (channel 1)
    R2: np.ndarray         # (3x3) Measurement noise covariance (channel 2)
    tau: float             # First-order lag time constant (seconds)


@dataclass
class ForceKFState:
    """Kalman filter internal state."""
    x: np.ndarray          # (12,) state vector
    P: np.ndarray          # (12,12) covariance matrix


# =============================================================================
# Initialization
# =============================================================================

def init_force_kf(
    params: ForceKFParams,
    F0: Optional[np.ndarray] = None,
    b10: Optional[np.ndarray] = None,
    b20: Optional[np.ndarray] = None,
    s0: Optional[np.ndarray] = None,
    P0_diag: Tuple[float, float, float, float] = (1e0, 1e-4, 1e-4, 1e-3),
) -> ForceKFState:
    """
    Initialize Kalman filter state.

    P0_diag defines the diagonal block variances for:
        (F, b1, b2, s), each multiplied by I3.
    """
    x = np.zeros(12, dtype=float)

    if F0 is not None:
        x[0:3] = np.asarray(F0, dtype=float)
    if b10 is not None:
        x[3:6] = np.asarray(b10, dtype=float)
    if b20 is not None:
        x[6:9] = np.asarray(b20, dtype=float)
    if s0 is not None:
        x[9:12] = np.asarray(s0, dtype=float)

    PF0, Pb10, Pb20, Ps0 = P0_diag
    I3 = np.eye(3)

    P = np.zeros((12, 12), dtype=float)
    P[0:3, 0:3] = PF0 * I3
    P[3:6, 3:6] = Pb10 * I3
    P[6:9, 6:9] = Pb20 * I3
    P[9:12, 9:12] = Ps0 * I3

    return ForceKFState(x=x, P=P)


# =============================================================================
# Kalman Filter Step
# =============================================================================

def fuse_force_step(
    z1: Optional[np.ndarray],
    z2: Optional[np.ndarray],
    dt: float,
    state: ForceKFState,
    params: ForceKFParams,
) -> Tuple[np.ndarray, ForceKFState]:
    """
    Perform one Kalman filter fusion step.

    Inputs:
        z1, z2 : (3,) np.ndarray or None
        dt     : time step (seconds)
        state  : ForceKFState
        params : ForceKFParams

    Returns:
        F_clean : (3,) estimated true force
        state   : updated filter state
    """
    # Unpack
    x, P = state.x, state.P
    qF, qb1, qb2, qs = params.qF, params.qb1, params.qb2, params.qs
    R1, R2, tau = params.R1, params.R2, params.tau

    # -------------------------------------------------------------------------
    # Prediction
    # -------------------------------------------------------------------------
    dt = float(max(dt, 1e-12))
    tau = float(max(tau, 1e-12))

    a = float(np.exp(-dt / tau))
    a = 0.1  # NOTE: kept exactly as original behavior

    I3 = np.eye(3)
    Z3 = np.zeros((3, 3))

    # State transition matrix (12x12)
    Fk = np.block([
        [I3,        Z3,        Z3,        Z3],
        [Z3,        I3,        Z3,        Z3],
        [Z3,        Z3,        I3,        Z3],
        [(1 - a)*I3, Z3,       Z3,        a*I3],
    ])

    # Process noise covariance
    Qk = np.diag(
        [qF]*3 + [qb1]*3 + [qb2]*3 + [qs]*3
    ) * dt

    x = Fk @ x
    P = Fk @ P @ Fk.T + Qk

    # -------------------------------------------------------------------------
    # Update
    # -------------------------------------------------------------------------
    def kf_update(x, P, H, R, z):
        """Standard Kalman update step."""
        z = np.asarray(z, dtype=float).reshape(3)
        y = z - H @ x
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)
        x = x + K @ y
        P = (np.eye(P.shape[0]) - K @ H) @ P
        return x, P

    # Measurement matrices
    H1 = np.block([Z3, I3, Z3, I3])  # z1 = b1 + s
    H2 = np.block([Z3, Z3, I3, I3])  # z2 = b2 + s

    if z1 is not None:
        x, P = kf_update(x, P, H1, R1, z1)

    if z2 is not None:
        x, P = kf_update(x, P, H2, R2, z2)

    state.x, state.P = x, P
    F_clean = x[0:3].copy()

    return F_clean, state


# =============================================================================
# Demo
# =============================================================================

def main():
    np.set_printoptions(precision=6, suppress=True)

    # 1) Filter parameters
    params = ForceKFParams(
        qF=1e-2,
        qb1=1e-8,
        qb2=1e-8,
        qs=1e-6,
        R1=np.eye(3) * 2e-3,
        R2=np.eye(3) * 2e-3,
        tau=0.01,
    )

    # 2) Initialization
    z_init = np.array([0.0, 0.0, 0.0], dtype=float)

    state = init_force_kf(
        params,
        F0=z_init,
        s0=z_init,
        b10=np.zeros(3),
        b20=np.zeros(3),
        P0_diag=(1e0, 1e-4, 1e-4, 1e-3),
    )

    # 3) Demo loop
    dt = 0.01
    steps = 50

    print("Running demo with constant measurements z1/z2 ...")

    for k in range(steps):
        z1 = np.array([1.0, -2.0, 3.0], dtype=float)
        z2 = np.array([1.0, -2.0, 4.0], dtype=float)

        F_clean, state = fuse_force_step(z1, z2, dt, state, params)
        print(f"step {k:02d}  F_clean = {F_clean}")


if __name__ == "__main__":
    main()
