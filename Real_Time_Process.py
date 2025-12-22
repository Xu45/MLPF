from __future__ import annotations

import math
import os
import time
import warnings
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from labml_nn.resnet import ResNetBase
from matplotlib.gridspec import GridSpec
from numpy.ma.core import arccos
from PIL import Image
from sympy.codegen.ast import continue_  # Unused, kept to match original code.

# Assume ResNet module has been adapted for 12-axis data.
from ResNet import *  # noqa: F401,F403


# =========================
# Geometry / utility functions
# =========================
def intersect_half_cylinder(
    x1: float,
    y1: float,
    Fx: float,
    Fy: float,
    Fz: float,
    R: float = 9.0,
    H: float = 28.0,
    eps: float = 1e-12,
) -> Tuple[float, float, float]:
    if Fz < 0:
        raise ValueError("Fz must be > 0.")

    A = Fx * Fx + Fz * Fz
    if A <= eps:
        raise ValueError("Degenerate direction: Fx and Fz cannot both be zero.")

    dx = x1 - R
    B = 2.0 * dx * Fx
    C = dx * dx - R * R

    disc = B * B - 4.0 * A * C
    if disc < -eps:
        raise RuntimeError(
            "No real intersection with cylindrical side (discriminant < 0)."
        )

    disc = max(0.0, disc)
    sqrt_disc = math.sqrt(disc)

    t_candidates = [
        (-B - sqrt_disc) / (2.0 * A),
        (-B + sqrt_disc) / (2.0 * A),
    ]

    feasible = []
    for t in t_candidates:
        if t >= -eps:
            y = y1 + t * Fy
            if -eps <= y <= H + eps:
                x = x1 + t * Fx
                z = t * Fz
                if z >= -eps:
                    feasible.append((t, x, y, z))

    if not feasible:
        return (0, 0, 0)

    t_star, x_hit, y_hit, z_hit = min(feasible, key=lambda tup: tup[0])

    if abs(y_hit) < eps:
        y_hit = 0.0
    if abs(H - y_hit) < eps:
        y_hit = H
    if abs(z_hit) < eps:
        z_hit = 0.0

    return (x_hit, y_hit, z_hit)


def unwrap_to_uv(x: float, y: float, z: float, R: float = 9.0) -> Tuple[float, float]:
    theta = math.atan2(x - R, z)  # atan2(sinθ, cosθ)
    s = R * theta
    u = s + 0.5 * math.pi * R
    v = y
    return (u, v)


# =========================
# KF_X parameters and state
# =========================
@dataclass
class XYKFParams:
    # Shared one-pole lag time constant (seconds). Tune by experiment.
    tau: float = 0.08

    # Process noise PSDs (scaled by dt internally)
    # p needs to be agile to follow motion; biases drift VERY slowly; s has tiny slack.
    q_p: float = 1e-1  # for p (x,y)
    q_b1: float = 1e-9  # for bias1 (x,y)  << very small
    q_b2: float = 1e-9  # for bias2 (x,y)  << very small
    q_s: float = 1e-6  # for shared s (x,y)

    # Measurement noise covariances (diagonal to start; replace with your stats)
    R1: np.ndarray = field(default_factory=lambda: np.diag([1e-3, 1e-3]))
    R2: np.ndarray = field(default_factory=lambda: np.diag([1e-3, 1e-3]))


@dataclass
class XYKFState:
    # State vector: [p(2), b1(2), b2(2), s(2)] -> shape (8,)
    x: np.ndarray
    # State covariance -> shape (8,8)
    P: np.ndarray
    # Params
    params: XYKFParams


_STATE: Optional[XYKFState] = None


# =========================
# Core Kalman routines
# =========================
def _predict(state: XYKFState, dt: float) -> None:
    """Time update: x- = F x,  P- = F P F^T + Q(dt)."""
    p = state.params
    I2 = np.eye(2)
    Z = np.zeros((2, 2))

    dt_eff = max(float(dt), 1e-9)
    a = float(np.exp(-dt_eff / p.tau))  # shared lag factor in [0,1)

    # State order: [p(2), b1(2), b2(2), s(2)]
    F = np.block(
        [
            [I2, Z, Z, Z],
            [Z, I2, Z, Z],
            [Z, Z, I2, Z],
            [(1 - a) * I2, Z, Z, a * I2],
        ]
    )

    # Process noise (scaled by dt)
    Q_diag = np.array(
        [p.q_p, p.q_p, p.q_b1, p.q_b1, p.q_b2, p.q_b2, p.q_s, p.q_s], dtype=float
    )
    Q = np.diag(Q_diag) * dt_eff

    state.x = F @ state.x
    state.P = F @ state.P @ F.T + Q


def _update(state: XYKFState, z: np.ndarray, which: int) -> None:
    """
    Measurement update for one channel.

    which = 1 => z1 = b1 + s
    which = 2 => z2 = b2 + s
    """
    assert which in (1, 2), "which must be 1 or 2"

    I2 = np.eye(2)
    if which == 1:
        # H1 = [0_p, I2_b1, 0_b2, I2_s]  (2x8)
        H = np.hstack([np.zeros((2, 2)), I2, np.zeros((2, 2)), I2])
        R = state.params.R1
    else:
        # H2 = [0_p, 0_b1, I2_b2, I2_s]  (2x8)
        H = np.hstack([np.zeros((2, 4)), I2, I2])
        R = state.params.R2

    x_pred, P_pred = state.x, state.P
    z = np.asarray(z, dtype=float).reshape(
        2,
    )
    y = z - H @ x_pred
    S = H @ P_pred @ H.T + R
    K = P_pred @ H.T @ np.linalg.inv(S)

    state.x = x_pred + K @ y
    state.P = (np.eye(P_pred.shape[0]) - K @ H) @ P_pred


# =========================
# Public API
# =========================
def clean_xy(
    z1: Optional[Tuple[float, float]],
    z2: Optional[Tuple[float, float]],
    dt: float,
) -> Tuple[float, float]:
    """
    One-step call into the filter.

    Args:
        z1: (x,y) from channel-1 (force-solving), or None if missing
        z2: (x,y) from channel-2 (deep-learning),  or None if missing
        dt: time since last call in seconds (can vary)

    Returns:
        (x_clean, y_clean): clean, de-biased, de-lagged current position estimate
    """
    global _STATE
    if _STATE is None:
        # If user didn't explicitly reset with an init, start from zeros (conservative).
        reset_filter_with_init(z_init=(0.0, 0.0))

    _predict(_STATE, dt)

    if z1 is not None:
        _update(_STATE, np.asarray(z1, dtype=float), which=1)
    if z2 is not None:
        _update(_STATE, np.asarray(z2, dtype=float), which=2)

    x_clean, y_clean = float(_STATE.x[0]), float(_STATE.x[1])
    return x_clean, y_clean


def reset_filter_with_init(
    z_init: Tuple[float, float] = (0.0, 0.0),
    params: Optional[XYKFParams] = None,
) -> None:
    """
    Reset the global filter with Scheme A priors.

    We pin both p and shared s close to the first measurement, and make biases tight.

    Args:
        z_init: initial measurement (x,y) used to initialize p and s near the data
        params: XYKFParams to override defaults (tau, noises, R1/R2)
    """
    global _STATE
    if params is None:
        params = XYKFParams()

    x0 = np.zeros(8, dtype=float)  # [p(2), b1(2), b2(2), s(2)]
    z0 = np.asarray(z_init, float).reshape(
        2,
    )

    # Place p and s near the initial observed point; biases start at 0
    x0[0:2] = z0  # p
    x0[6:8] = z0  # s

    # Covariances: p medium, biases tiny (we believe biases small), s small
    P0 = np.diag(
        [
            1.0,
            1.0,  # p
            1e-3,
            1e-3,  # b1
            1e-3,
            1e-3,  # b2
            1e-2,
            1e-2,  # s
        ]
    )

    _STATE = XYKFState(x=x0, P=P0, params=params)


# =========================
# KF_F parameters and state
# =========================
@dataclass
class ForceKFParams:
    qF: float  # 真实力过程噪声功率 (N^2/s)  —— 越大越“活”
    qb1: float  # 通道1偏置过程噪声（很小表示慢变）
    qb2: float  # 通道2偏置过程噪声
    qs: float  # 传感器一阶滞后内部状态过程噪声（很小）
    R1: np.ndarray  # (3x3) 通道1测量噪声协方差
    R2: np.ndarray  # (3x3) 通道2测量噪声协方差
    tau: float  # 一阶滞后时间常数 (s)


@dataclass
class ForceKFState:
    x: np.ndarray  # (12,) 状态向量 [F(3), b1(3), b2(3), s(3)]
    P: np.ndarray  # (12,12) 协方差


def init_force_kf(
    params: ForceKFParams,
    F0: Optional[np.ndarray] = None,
    b10: Optional[np.ndarray] = None,
    b20: Optional[np.ndarray] = None,
    s0: Optional[np.ndarray] = None,
    P0_diag: Tuple[float, float, float, float] = (1e0, 1e-4, 1e-4, 1e-3),
) -> ForceKFState:
    """
    Initialize KF state.
    P0_diag: Initial covariance blocks (PF, Pb1, Pb2, Ps), each multiplied by I3.
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
    P = np.zeros((12, 12), dtype=float)
    I3 = np.eye(3)
    P[0:3, 0:3] = PF0 * I3
    P[3:6, 3:6] = Pb10 * I3
    P[6:9, 6:9] = Pb20 * I3
    P[9:12, 9:12] = Ps0 * I3

    return ForceKFState(x=x, P=P)


def fuse_force_step(
    z1: Optional[np.ndarray],
    z2: Optional[np.ndarray],
    dt: float,
    state: ForceKFState,
    params: ForceKFParams,
) -> Tuple[np.ndarray, ForceKFState]:
    """
    One-step fusion: two observations -> output clean 3D force estimate.

    Args:
        z1, z2: (3,) np.ndarray or None   # two observations (measure s + b_i)
        dt: float, time interval (seconds)
        state: ForceKFState
        params: ForceKFParams

    Returns:
        F_clean: (3,) de-lagged, de-biased 3D force estimate at current step
        state: updated filter state
    """
    x, P = state.x, state.P
    qF, qb1, qb2, qs = params.qF, params.qb1, params.qb2, params.qs
    R1, R2, tau = params.R1, params.R2, params.tau

    dt = float(max(dt, 1e-12))
    tau = float(max(tau, 1e-12))
    a = float(np.exp(-dt / tau))  # one-pole lag factor

    I3 = np.eye(3)
    Z = np.zeros((3, 3))

    # State transition matrix F_k (12x12): x=[F, b1, b2, s]
    Fk = np.block(
        [
            [I3, Z, Z, Z],  # F_k = F_{k-1}
            [Z, I3, Z, Z],  # b1_k = b1_{k-1}
            [Z, Z, I3, Z],  # b2_k = b2_{k-1}
            [(1 - a) * I3, Z, Z, a * I3],  # s_k = a*s_{k-1} + (1-a)*F_{k-1}
        ]
    )

    # Process noise Q_k (12x12), scaled by dt
    Qk = np.diag([qF] * 3 + [qb1] * 3 + [qb2] * 3 + [qs] * 3) * dt

    x = Fk @ x
    P = Fk @ P @ Fk.T + Qk

    def kf_update(x_, P_, H_, R_, z_):
        z_ = np.asarray(z_, dtype=float).reshape(3)
        y_ = z_ - H_ @ x_
        S_ = H_ @ P_ @ H_.T + R_
        K_ = P_ @ H_.T @ np.linalg.inv(S_)
        x_ = x_ + K_ @ y_
        P_ = (np.eye(P_.shape[0]) - K_ @ H_) @ P_
        return x_, P_

    # H1/H2: z_i = b_i + s
    H1 = np.block([Z, I3, Z, I3])  # (3x12)
    H2 = np.block([Z, Z, I3, I3])  # (3x12)

    if z1 is not None:
        x, P = kf_update(x, P, H1, R1, z1)
    if z2 is not None:
        x, P = kf_update(x, P, H2, R2, z2)

    state.x, state.P = x, P
    F_clean = x[0:3].copy()
    return F_clean, state


# =========================
# Models and prediction
# =========================
class MLP(nn.Module):
    def __init__(self, input_dim: int = 12, hidden_dim: int = 128, output_dim: int = 5):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.model(x)


def ResNetModel(
    n_blocks: List[int],
    n_channels: List[int],
    bottlenecks_channels: Optional[List[int]] = None,
    in_channels: int = 12,
    first_kernel_size: int = 3,
):
    """Create a 4-class ResNet model."""
    base = ResNetBase(
        n_blocks,
        n_channels,
        bottlenecks_channels,
        in_channels=in_channels,
        first_conv_kernel_size=first_kernel_size,
    )
    classification = nn.Linear(n_channels[-1], 4)
    model = nn.Sequential(base, classification)
    return model


def predict_CNN(data_in, CNN_model, CNN_in_channels):
    """Predict which of the 4 classes the data belongs to."""
    data_in_CNN = data_in.copy()
    data_in_CNN = np.array(data_in_CNN)

    data_in_CNN = torch.tensor(data_in_CNN, dtype=torch.float32)
    data_in_CNN = data_in_CNN.unsqueeze(0)
    data_in_CNN = data_in_CNN.unsqueeze(1)
    data_in_CNN = data_in_CNN.permute(0, 2, 1)

    with torch.no_grad():
        predicted = CNN_model(data_in_CNN)

    _, region_predicted = torch.max(predicted.data, 1)
    region_predicted = int(region_predicted.item())
    return region_predicted


def predict_MLP(data_in, mlp_model, scaler_inputs, scaler_targets):
    """Predict force and position using the MLP model."""
    data_in_MLP = np.array(data_in)
    data_in_MLP = pd.DataFrame(
        data=data_in_MLP.reshape(1, -1),
        columns=[
            "fx1",
            "fy1",
            "fz1",
            "fx2",
            "fy2",
            "fz2",
            "fx3",
            "fy3",
            "fz3",
            "fx4",
            "fy4",
            "fz4",
        ],
    )

    new_inputs = scaler_inputs.transform(data_in_MLP)
    new_inputs = torch.tensor(new_inputs, dtype=torch.float32)

    with torch.no_grad():
        data_predicted = mlp_model(new_inputs)

    data_predicted = scaler_targets.inverse_transform(data_predicted.numpy()).squeeze()
    predicted_targets = np.array(data_predicted)
    return predicted_targets


def inverse_process(processed_inputs, rotation_matrices, num):
    # 1) Split into 4x3D vectors
    rot = [
        processed_inputs[0:3],
        processed_inputs[3:6],
        processed_inputs[6:9],
        processed_inputs[9:12],
    ]

    # 2) Inverse rotation: vec = R^T @ rot
    vecs = []
    for i in range(4):
        R = rotation_matrices[i]
        RT = [
            [R[0][0], R[1][0], R[2][0]],
            [R[0][1], R[1][1], R[2][1]],
            [R[0][2], R[1][2], R[2][2]],
        ]
        v = rot[i]
        x = v[0] * RT[0][0] + v[1] * RT[0][1] + v[2] * RT[0][2]
        y = v[0] * RT[1][0] + v[1] * RT[1][1] + v[2] * RT[1][2]
        z = v[0] * RT[2][0] + v[1] * RT[2][1] + v[2] * RT[2][2]
        vecs.append([x, y, z])

    # 3) Merge back to 12D
    flat = []
    for v in vecs:
        flat.extend(v)

    # 4) Re-apply physical coefficients
    inputsF_restored = [flat[i] * num[i] for i in range(12)]
    return inputsF_restored


# =========================
# Main
# =========================
if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    num_channels = 12  # 12 channels (4 sensors x 3 axes)
    adc_to_uv = 0.3338  # ADC to microvolt conversion factor

    # ---------------------- Rectangle partition params ----------------------
    length = 28.28
    total_width = 28
    num_segments = 4
    segment_width = total_width / num_segments
    spacing = 1
    x_min, x_max = 0, length
    z_min, z_max = 0, total_width

    # Data file path
    file_dir = "C:\\Users\\86135\\Desktop\\NdtToolbox-v1.0.18\\Data"
    file_name = os.listdir(file_dir)[0]
    file_path = os.path.join(file_dir, file_name)

    CNN_in_channels = 12

    # Load MLP model and scalers
    mlp_path = r"train_12axis_4regions/6/init_train/Only_MLP"

    # Keep exec usage to avoid behavior differences vs. original code
    exec(f'scaler_inputs_all = joblib.load("{mlp_path}/scaler_inputs_all.pkl")')
    exec(f'scaler_targets_all = joblib.load("{mlp_path}/scaler_targets_all.pkl")')
    exec("mlp_model_all = MLP(input_dim=12, hidden_dim=128, output_dim=5)")
    exec(f'mlp_model_all.load_state_dict(torch.load("{mlp_path}/mlp_model_all.pth"))')
    exec("mlp_model_all.eval()")

    # Load ResNet classifier (4 classes)
    cnn_path = f"train_{CNN_in_channels}axis_4regions/5/init_train/CNN"
    deg_model = ResNetModel(
        n_blocks=[4, 4, 4, 4],
        n_channels=[32, 64, 128, 256],
        in_channels=CNN_in_channels,
        first_kernel_size=6,
    )
    deg_model.load_state_dict(
        torch.load(f"{cnn_path}/model_deg.pth", map_location=torch.device("cpu"))
    )
    deg_model.eval()

    # Real-time visualization initialization
    fig = plt.figure(figsize=(10, 10))
    gs = GridSpec(12, 10)
    plt.ion()

    # 12-axis sensor signals plot
    ax_signals = fig.add_subplot(gs[0:4, 0:5])
    lines_signals = [
        ax_signals.plot([], [], label=f"Sensor{i // 3 + 1}-Ch{i % 3 + 1}")[0]
        for i in range(12)
    ]
    ax_signals.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax_signals.set_xlabel("Time(Step)")
    ax_signals.set_ylabel("Value(μV)")
    ax_signals.set_title("12-Channels Signals (4 Sensors × 3 Axes)")
    ax_signals.set_ylim(-2000, 2000)

    # Forces plot
    ax_forces = fig.add_subplot(gs[0:4, 5:-1])
    lines_forces = [ax_forces.plot([], [], label=i)[0] for i in ["Fx", "Fy", "Fz"]]
    ax_forces.legend()
    ax_forces.set_xlabel("Time(Step)")
    ax_forces.set_ylabel("Force(N)")
    ax_forces.set_title("Real-time Forces")
    ax_forces.set_ylim(-15, 15)

    # Position plot: full rectangle (X vertical, Z horizontal)
    ax_position = fig.add_subplot(gs[5:-1, 0:-1])

    rect_full = plt.Rectangle(
        (z_min, x_min),
        z_max - z_min,
        x_max - x_min,
        fill=False,
        edgecolor="black",
        linewidth=2,
    )
    ax_position.add_patch(rect_full)

    for seg in range(1, num_segments):
        z_sep = seg * segment_width
        if z_min <= z_sep <= z_max:
            ax_position.axvline(
                x=z_sep,
                ymin=x_min / length,
                ymax=x_max / length,
                color="gray",
                linestyle="--",
                alpha=0.7,
            )
            ax_position.text(
                z_sep,
                x_min - 0.5,
                f"region{seg}",
                ha="center",
                va="top",
                fontsize=10,
                color="gray",
            )

    pos_point, = ax_position.plot([], [], "ro", markersize=10, label="pos")
    pos_text = ax_position.text(
        0.02,
        0.95,
        "",
        transform=ax_position.transAxes,
        va="top",
        ha="left",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        fontsize=11,
    )

    ax_position.set_xlabel("Z (mm)", fontsize=12)
    ax_position.set_ylabel("X (mm)", fontsize=12)
    ax_position.set_title("Z-X", fontsize=14, pad=20)
    ax_position.set_xlim(z_min - 1, z_max + 1)
    ax_position.set_ylim(x_min - 1, x_max + 1)
    ax_position.grid(True, alpha=0.3)
    ax_position.legend(loc="upper right", fontsize=10)

    # Sliding window stores last 100 points
    signals = [deque(maxlen=100) for _ in range(12)]
    forces = [deque(maxlen=100) for _ in range(3)]

    plt.tight_layout()

    # Initialization variables
    k = 0
    signals_50 = []
    calibrate = np.array([0] * 12)
    pred_x_pass = 0
    pred_z_pass = 0
    F_k = 0.01 * 0.3  # * 0.43

    # Sensor conversion coefficients
    num0 = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    num = [5, 5, 5, 2.42, 2.42, 2.42, 2.87, 2.87, 2.87, 4, 4, 4]
    num2 = [1, 1, 1, 1.5, 1.5, 1.5, 1, 1, 1, 0.4, 0.4, 0.4]

    pass_input = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    # KF_F initialization
    np.set_printoptions(precision=6, suppress=True)
    params = ForceKFParams(
        qF=2e-1,
        qb1=1e-8,
        qb2=1e-8,
        qs=1e-6,
        R1=np.eye(3) * 2e-3,
        R2=np.eye(3) * 2e-3,
        tau=0.01,
    )

    z_const = np.array([0.0, 0.0, 0.0], dtype=float)
    state = init_force_kf(
        params,
        F0=z_const,
        s0=z_const,
        b10=np.zeros(3),
        b20=np.zeros(3),
        P0_diag=(1e0, 1e-4, 1e-4, 1e-3),
    )

    dt = 0.1
    steps = 10

    # KF_X initialization
    np.set_printoptions(precision=4, suppress=True)
    reset_filter_with_init(z_init=(0.0, 0.0))

    while True:
        start_time = time.time()

        # Read latest data
        with open(file_path, "rb") as file:
            file.seek(-150, 2)
            inputs_last100 = file.readlines()
            inputs_last_line = inputs_last100[-1]
            inputs_last_line = inputs_last_line.split()
            inputs_last_line.pop(0)
            init_adc_inputs = [float(inputs_last_line[j]) for j in range(12)]

        # First 10 reads for calibration
        if k < 10:
            k += 1
            signals_50.append(init_adc_inputs)
            continue

        if k == 10:
            signals_50 = np.array(signals_50)
            signals_50 = np.stack(signals_50, 1)
            for idx in range(num_channels):
                calibrate[idx] = np.average(signals_50[idx])
            k = 100

        Dataset_calibrate = [
            -1434,
            -1761,
            -1894,
            -1821,
            -1690,
            -1779,
            -1766,
            -1805,
            -1823,
            -1875,
            -1913,
            -1943,
        ]

        # Data processing: calibration
        inputs = [
            (float(init_adc_inputs[j]) - calibrate[j] + Dataset_calibrate[j])
            for j in range(12)
        ]
        inputsF = [(float(init_adc_inputs[j]) - calibrate[j]) for j in range(12)]

        # Physical correction
        for i in range(12):
            inputsF[i] = inputsF[i] / num[i]

        # Rotation matrices
        rotation_matrices = [
            [
                [0.36388561, -0.86660445, 0.34144399],
                [-0.86660445, -0.18061041, 0.465163],
                [-0.34144399, -0.465163, -0.81672479],
            ],
            [
                [0.99299462, 0.11335093, 0.03336541],
                [0.11335093, -0.83408096, -0.53987084],
                [-0.03336541, 0.53987084, -0.84108633],
            ],
            [
                [0.97098258, -0.23533816, -0.04252983],
                [-0.23533816, -0.90864809, -0.34492696],
                [0.04252983, 0.34492696, -0.93766551],
            ],
            [
                [-0.20532487, -0.8935475, 0.39926752],
                [-0.8935475, 0.33758345, 0.29599032],
                [-0.39926752, -0.29599032, -0.86774142],
            ],
        ]

        rotation_matrices2 = [
            [
                [-0.39652662, -0.685532, 0.61058374],
                [-0.685532, 0.66348359, 0.29972554],
                [-0.61058374, -0.29972554, -0.73304304],
            ],
            [
                [-0.42192132, 0.66830845, 0.61265505],
                [0.66830845, 0.68589247, -0.28795022],
                [-0.61265505, 0.28795022, -0.73602884],
            ],
            [
                [-0.73045052, 0.27507161, 0.6251221],
                [0.27507161, 0.95627474, -0.09936912],
                [-0.6251221, 0.09936912, -0.77417578],
            ],
            [
                [-0.39288511, -0.74439434, 0.53992441],
                [-0.74439434, 0.60217614, 0.28854976],
                [-0.53992441, -0.28854976, -0.79070897],
            ],
        ]

        rotation_matrices0 = [
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        ]

        # Split 12D into four 3D vectors
        vectors = [inputsF[0:3], inputsF[3:6], inputsF[6:9], inputsF[9:12]]

        rotated_vectors = []
        for i in range(4):
            vec = vectors[i]
            mat = rotation_matrices[i]

            rotated_x = vec[0] * mat[0][0] + vec[1] * mat[0][1] + vec[2] * mat[0][2]
            rotated_y = vec[0] * mat[1][0] + vec[1] * mat[1][1] + vec[2] * mat[1][2]
            rotated_z = vec[0] * mat[2][0] + vec[1] * mat[2][1] + vec[2] * mat[2][2]
            rotated_vectors.append([rotated_x, rotated_y, rotated_z])

        processed_inputs = []
        for vec in rotated_vectors:
            processed_inputs.extend(vec)

        print("第一个:", processed_inputs[0], processed_inputs[1], processed_inputs[2])
        print("第二个:", processed_inputs[3], processed_inputs[4], processed_inputs[5])
        print("第三个:", processed_inputs[6], processed_inputs[7], processed_inputs[8])
        print("第四个:", processed_inputs[9], processed_inputs[10], processed_inputs[11])

        F_y = -processed_inputs[0] - processed_inputs[3] + processed_inputs[6] + processed_inputs[9]
        F_x = processed_inputs[1] + processed_inputs[4] - processed_inputs[7] - processed_inputs[10]
        F_z = processed_inputs[2] + processed_inputs[5] + processed_inputs[8] + processed_inputs[11]

        if F_x > 0:
            F_x = F_x * 0.5
        if F_y < 0:
            F_y = F_y * 1.5

        d_xhd = 4
        M_x = (processed_inputs[2] + processed_inputs[5]) * d_xhd + (
            processed_inputs[8] + processed_inputs[11]
        ) * (28 - d_xhd)
        M_y = -(processed_inputs[2] + processed_inputs[8]) * d_xhd - (
            processed_inputs[5] + processed_inputs[11]
        ) * (18 - d_xhd)

        if (F_z * F_k) < -0.3:
            x_F = M_y / F_z
            y_F = -M_x / F_z
        else:
            x_F = 0
            y_F = 0

        x_F = -x_F
        y_F = -y_F

        if x_F < 0:
            x_F = 0
        if x_F > 18:
            x_F = 18
        if y_F < 0:
            y_F = 0
        if y_F > 28:
            y_F = 28

        if F_z < 0:
            xh, yh, zh = intersect_half_cylinder(x_F, y_F, F_x, F_y, -F_z, R=9.0, H=28.0)
        else:
            xh = 0
            yh = 0
            zh = 0

        aaa = (3.1415926 / 20) * arccos(1 - xh / 9) * 57.3
        X_FB, Y_FB = unwrap_to_uv(xh, yh, zh, R=9.0)

        # Two-stage prediction: classify then regress
        region_predicted = predict_CNN(inputs, deg_model, CNN_in_channels) + 1
        exec("prediction = predict_MLP(inputs, mlp_model_all, scaler_inputs_all, scaler_targets_all)")

        prediction[0] = prediction[1]
        prediction[1] = -1.0 * prediction[0]
        prediction[2] = prediction[2]

        end_time = time.time()

        # KF_F
        z1 = np.array([F_x * F_k, F_y * F_k, F_z * F_k], dtype=float)
        z2 = np.array([prediction[0], prediction[1], prediction[2]], dtype=float)
        F_clean, state = fuse_force_step(z1, z2, dt, state, params)

        # KF_X
        if abs(prediction[2]) < 0.5:
            prediction[3] = 0
            prediction[4] = 0
            x_F = 0
            y_F = 0

        z1_xy = (x_F, y_F)
        z2_xy = (prediction[3], prediction[4])
        x_clean, y_clean = clean_xy(z1_xy, z2_xy, dt)

        # Update signals plot with processed_inputs
        signals[0].append(processed_inputs[0])
        signals[1].append(processed_inputs[1])
        signals[2].append(processed_inputs[2])
        signals[3].append(processed_inputs[3])
        signals[4].append(processed_inputs[4])
        signals[5].append(processed_inputs[5])
        signals[6].append(processed_inputs[6])
        signals[7].append(processed_inputs[7])
        signals[8].append(processed_inputs[8])
        signals[9].append(processed_inputs[9])
        signals[10].append(processed_inputs[10])
        signals[11].append(processed_inputs[11])

        x = np.arange(len(signals[0]))
        for i, _line in enumerate(lines_signals):
            lines_signals[i].set_data(x, signals[i])
        ax_signals.relim()
        ax_signals.autoscale_view()

        # Update forces plot
        forces[0].append(F_clean[0])
        forces[1].append(F_clean[1])
        forces[2].append(F_clean[2])

        for i, _line in enumerate(lines_forces):
            lines_forces[i].set_data(x, forces[i])
        ax_forces.relim()
        ax_forces.autoscale_view()

        # Position plot update (single point, X vertical, Z horizontal)
        pred_x = pred_x_pass * 0.5 + x_clean * 0.5
        pred_z = pred_z_pass * 0.5 + y_clean * 0.5
        pred_x_pass = pred_x
        pred_z_pass = pred_z

        if abs(F_clean[2]) < 0.5:
            pred_z = 0
            pred_x = 0

        pos_text_content = (
            f"\nX = {pred_x:.2f} mm\nZ = {pred_z:.2f} mm\n"
            f'{region_predicted if region_predicted != 0 else "nono"}'
        )

        # NOTE: Original behavior: plot uses (y_F, x_F) rather than (pred_z, pred_x)
        pos_point.set_data([y_F], [x_F])

        pos_text.set_text(pos_text_content)

        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        fig.show()
