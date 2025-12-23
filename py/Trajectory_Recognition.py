from __future__ import annotations

import math
import os
import time
import warnings
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from labml_nn.resnet import ResNetBase
from matplotlib.patches import Circle, FancyArrowPatch
from numpy.ma.core import arccos
from sympy.codegen.ast import continue_  # Unused, kept to match original code.

# Assume ResNet module has been adapted for 12-axis data.
from ResNet import *  # noqa: F401,F403


# =========================
# Dial: circle + arrow (supports external ax)
# =========================
class AngleDial:
    def __init__(
        self,
        radius: float = 1.0,
        offset_deg: float = 0.0,
        clockwise: bool = False,
        ax: Optional[plt.Axes] = None,
        fig: Optional[plt.Figure] = None,
    ):
        """
        Parameters
        ----------
        radius
            Dial radius.
        offset_deg
            Angle zero offset (degrees). 0 means 0° points to +X; 90 means 0° points up.
        clockwise
            If True, positive angles are clockwise; otherwise counterclockwise.
        ax / fig
            Optional: draw the dial into an existing Axes/Figure (shared window with other plots).
        """
        self.radius = float(radius)
        self.offset_deg = float(offset_deg)
        self.clockwise = bool(clockwise)

        if ax is None or fig is None:
            plt.ion()
            self.fig, self.ax = plt.subplots(figsize=(5, 5))
        else:
            self.fig, self.ax = fig, ax

        self.ax.set_aspect("equal", adjustable="box")
        self.ax.set_xlim(-1.25 * self.radius, 1.25 * self.radius)
        self.ax.set_ylim(-1.25 * self.radius, 1.25 * self.radius)
        self.ax.axis("off")

        # Dial circle and center point.
        self.ax.add_patch(Circle((0, 0), self.radius, fill=False, lw=2))
        self.ax.add_patch(Circle((0, 0), 0.02 * self.radius, color="k"))

        # Major ticks (0/90/180/270).
        # Labels follow the display convention used with offset=90, clockwise=True.
        tick_r1 = self.radius * 0.92
        tick_r2 = self.radius * 1.00
        for deg, lbl in [(0, "90°"), (90, "0°"), (180, "270°"), (270, "180°")]:
            th = np.deg2rad(deg)
            x1, y1 = tick_r1 * np.cos(th), tick_r1 * np.sin(th)
            x2, y2 = tick_r2 * np.cos(th), tick_r2 * np.sin(th)
            self.ax.plot([x1, x2], [y1, y2], lw=2)
            tx, ty = 1.10 * self.radius * np.cos(th), 1.10 * self.radius * np.sin(th)
            self.ax.text(tx, ty, lbl, ha="center", va="center", fontsize=10)

        # Minor ticks (every 30°).
        tick_r1_minor = self.radius * 0.95
        for deg in range(0, 360, 30):
            if deg in (0, 90, 180, 270):
                continue
            th = np.deg2rad(deg)
            x1, y1 = tick_r1_minor * np.cos(th), tick_r1_minor * np.sin(th)
            x2, y2 = tick_r2 * np.cos(th), tick_r2 * np.sin(th)
            self.ax.plot([x1, x2], [y1, y2], lw=1)

        # Arrow (initially 0°).
        self.arrow = FancyArrowPatch(
            posA=(0, 0),
            posB=(self.radius * 0.9, 0),
            arrowstyle="-|>",
            mutation_scale=20,
            lw=3,
        )
        self.ax.add_patch(self.arrow)

        # Readout display (kept as in original code, but commented out).
        # self.label = self.ax.text(
        #     0, -1.18*self.radius, "0.0°",
        #     ha='center', va='center', fontsize=12,
        #     bbox=dict(boxstyle='round', facecolor='white', alpha=0.85)
        # )

        _cw_ccw = "CW" if self.clockwise else "CCW"
        _zero_dir = (
            "right(+X)" if self.offset_deg == 0 else f"offset {self.offset_deg:.0f}°"
        )
        self.ax.set_title(" ", pad=12)

        if fig is None:
            self.fig.tight_layout()
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            plt.show(block=False)

    def update(self, angle_deg) -> None:
        """Update arrow direction using the given angle in degrees (recommended 0~360)."""
        if angle_deg is None:
            return
        try:
            angle_in = float(angle_deg) % 360.0
        except Exception:
            return

        signed = (-angle_in) if self.clockwise else angle_in
        a = (signed + self.offset_deg) % 360.0
        th = np.deg2rad(a)

        x = self.radius * 0.9 * np.cos(th)
        y = self.radius * 0.9 * np.sin(th)

        self.arrow.set_positions((0, 0), (x, y))
        # self.label.set_text(f"{angle_in:.1f}°")

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        plt.pause(0.001)


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
    tau: float = 0.08
    q_p: float = 1e-1
    q_b1: float = 1e-9
    q_b2: float = 1e-9
    q_s: float = 1e-6
    R1: np.ndarray = field(default_factory=lambda: np.diag([1e-3, 1e-3]))
    R2: np.ndarray = field(default_factory=lambda: np.diag([1e-3, 1e-3]))


@dataclass
class XYKFState:
    x: np.ndarray
    P: np.ndarray
    params: XYKFParams


_STATE: Optional[XYKFState] = None


def _predict(state: XYKFState, dt: float) -> None:
    p = state.params
    I2 = np.eye(2)
    Z = np.zeros((2, 2))

    dt_eff = max(float(dt), 1e-9)
    a = float(np.exp(-dt_eff / p.tau))

    F = np.block(
        [
            [I2, Z, Z, Z],
            [Z, I2, Z, Z],
            [Z, Z, I2, Z],
            [(1 - a) * I2, Z, Z, a * I2],
        ]
    )

    Q_diag = np.array(
        [p.q_p, p.q_p, p.q_b1, p.q_b1, p.q_b2, p.q_b2, p.q_s, p.q_s], dtype=float
    )
    Q = np.diag(Q_diag) * dt_eff

    state.x = F @ state.x
    state.P = F @ state.P @ F.T + Q


def _update(state: XYKFState, z: np.ndarray, which: int) -> None:
    assert which in (1, 2), "which must be 1 or 2"
    I2 = np.eye(2)

    if which == 1:
        H = np.hstack([np.zeros((2, 2)), I2, np.zeros((2, 2)), I2])
        R = state.params.R1
    else:
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


def clean_xy(
    z1: Optional[Tuple[float, float]],
    z2: Optional[Tuple[float, float]],
    dt: float,
) -> Tuple[float, float]:
    global _STATE

    if _STATE is None:
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
    global _STATE

    if params is None:
        params = XYKFParams()

    x0 = np.zeros(8, dtype=float)
    z0 = np.asarray(z_init, float).reshape(
        2,
    )
    x0[0:2] = z0
    x0[6:8] = z0

    P0 = np.diag([1.0, 1.0, 1e-3, 1e-3, 1e-3, 1e-3, 1e-2, 1e-2])
    _STATE = XYKFState(x=x0, P=P0, params=params)


# =========================
# KF_F parameters and state
# =========================
@dataclass
class ForceKFParams:
    qF: float
    qb1: float
    qb2: float
    qs: float
    R1: np.ndarray
    R2: np.ndarray
    tau: float


@dataclass
class ForceKFState:
    x: np.ndarray  # (12,)
    P: np.ndarray  # (12,12)


def init_force_kf(
    params: ForceKFParams,
    F0: Optional[np.ndarray] = None,
    b10: Optional[np.ndarray] = None,
    b20: Optional[np.ndarray] = None,
    s0: Optional[np.ndarray] = None,
    P0_diag: Tuple[float, float, float, float] = (1e0, 1e-4, 1e-4, 1e-3),
) -> ForceKFState:
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
    x, P = state.x, state.P
    qF, qb1, qb2, qs = params.qF, params.qb1, params.qb2, params.qs
    R1, R2, tau = params.R1, params.R2, params.tau

    dt = float(max(dt, 1e-12))
    tau = float(max(tau, 1e-12))
    a = float(np.exp(-dt / tau))

    I3 = np.eye(3)
    Z = np.zeros((3, 3))

    Fk = np.block(
        [
            [I3, Z, I3 * 0, Z],
            [Z, I3, Z, Z],
            [Z, Z, I3, Z],
            [(1 - a) * I3, Z, Z, a * I3],
        ]
    )

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
    data_in_CNN = np.array(data_in)
    data_in_CNN = torch.tensor(data_in_CNN, dtype=torch.float32)
    data_in_CNN = data_in_CNN.unsqueeze(0).unsqueeze(1).permute(0, 2, 1)
    with torch.no_grad():
        predicted = CNN_model(data_in_CNN)
    _, region_predicted = torch.max(predicted.data, 1)
    return int(region_predicted.item())


def predict_MLP(data_in, mlp_model, scaler_inputs, scaler_targets):
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
    return np.array(data_predicted)


def inverse_process(processed_inputs, rotation_matrices, num):
    rot = [
        processed_inputs[0:3],
        processed_inputs[3:6],
        processed_inputs[6:9],
        processed_inputs[9:12],
    ]

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
        z = v[0] * RT[2][0] + v[1] * RT[2][1] + v[2] * RT[2][2]  # Fixed z component
        vecs.append([x, y, z])

    flat = []
    for v in vecs:
        flat.extend(v)

    inputsF_restored = [flat[i] * num[i] for i in range(12)]
    return inputsF_restored


# Global variables in the original script (kept)
draw_begin_cut = 0
draw_begin_flag = 0
draw_flag = 0
ddd = 30 * 0.15  # step = 0.01 mm
lastpoint = [0, 0]
nowpoint = [0, 0]


# =========================
# Main
# =========================
if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    num_channels = 12
    adc_to_uv = 0.3338

    # Region parameters (kept; not used for visualization)
    length = 28.28
    total_width = 28
    num_segments = 4
    segment_width = total_width / num_segments
    spacing = 1
    x_min, x_max = 0, length
    z_min, z_max = 0, total_width

    # Data file path (per local setup)
    file_dir = "C:\\Users\\86135\\Desktop\\NdtToolbox-v1.0.18\\Data"
    file_name = os.listdir(file_dir)[0]
    file_path = os.path.join(file_dir, file_name)

    CNN_in_channels = 12

    # Load MLP model and scalers
    mlp_path = r"train_12axis_4regions/13/init_train/Only_MLP"
    scaler_inputs_all = joblib.load(f"{mlp_path}/scaler_inputs_all.pkl")
    scaler_targets_all = joblib.load(f"{mlp_path}/scaler_targets_all.pkl")
    mlp_model_all = MLP(input_dim=12, hidden_dim=128, output_dim=5)
    mlp_model_all.load_state_dict(
        torch.load(f"{mlp_path}/mlp_model_all.pth", map_location="cpu")
    )
    mlp_model_all.eval()

    # Load ResNet classification model (4 classes)
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

    # Trajectory image saving settings
    SAVE_DIR = r"C:\Users\86135\Desktop\tu\guizi"
    os.makedirs(SAVE_DIR, exist_ok=True)

    def save_traj_axes(ax, fig):
        """Save only the right-side trajectory subplot (ax) into SAVE_DIR. Filename includes timestamp."""
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        bbox = ax.get_tightbbox(renderer).expanded(1.02, 1.06)
        bbox_inches = bbox.transformed(fig.dpi_scale_trans.inverted())
        ts = time.strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(SAVE_DIR, f"trajectory_{ts}.png")
        fig.savefig(out_path, dpi=300, bbox_inches=bbox_inches)
        return out_path

    # Trajectory viewport "overall centering" settings
    WINDOW_W = 180  # viewport width (same unit as trajectory)
    WINDOW_H = 180  # viewport height
    MARGIN = 0

    def center_view_on(ax: plt.Axes, cx: float, cy: float):
        """
        Move the trajectory axes view to be centered at (cx, cy).
        (cx, cy) are in the display coordinate system (see xn/yn swapping below).
        """
        half_w = WINDOW_W / 2.0
        half_h = WINDOW_H / 2.0
        ax.set_xlim(cx - half_w - MARGIN, cx + half_w + MARGIN)
        ax.set_ylim(cy - half_h - MARGIN, cy + half_h + MARGIN)

    def recenter_to_trajectory(ax: plt.Axes, xs_disp: List[float], ys_disp: List[float]):
        """Center the view on the bounding-box center of the full trajectory (display coordinate system)."""
        if not xs_disp or not ys_disp:
            return
        xmin, xmax = min(xs_disp), max(xs_disp)
        ymin, ymax = min(ys_disp), max(ys_disp)
        cx = 0.5 * (xmin + xmax)
        cy = 0.5 * (ymin + ymax)
        center_view_on(ax, cx, cy)

    # Shared window visualization: left (dial) + right (trajectory)
    plt.ion()
    fig, (ax_dial, ax_traj) = plt.subplots(1, 2, figsize=(10, 5))

    # Draw dial on ax_dial
    dial = AngleDial(radius=1.0, offset_deg=90, clockwise=True, ax=ax_dial, fig=fig)

    # Initialize trajectory axes (right)
    ax_traj.set_aspect("equal", adjustable="box")
    ax_traj.set_xlabel("X (mm)")
    ax_traj.set_ylabel("Y (mm)")
    ax_traj.set_title("Trajectory")
    ax_traj.set_xlim(-80, 80)
    ax_traj.set_ylim(-80, 80)
    traj_line, = ax_traj.plot([], [], "-", lw=2)

    traj_points: List[Tuple[float, float]] = []  # accumulated points (mechanical coordinates)
    drawing = False  # whether in drawing mode
    lastpoint_xy = (0.0, 0.0)

    fig.tight_layout()
    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.show(block=False)

    # Sliding calibration initialization
    k = 0
    signals_50 = []
    calibrate = np.array([0] * 12)

    pred_x_pass = 0
    pred_z_pass = 0
    F_k = 0.01 * 0.3

    num0 = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    pass_input = [0] * 12

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

    # KF_X initialization
    np.set_printoptions(precision=4, suppress=True)
    reset_filter_with_init(z_init=(0.0, 0.0))

    dt_fixed = 0.1  # original setting; use real loop interval if needed

    while True:
        # Read the latest line of data
        with open(file_path, "rb") as file:
            try:
                file.seek(-150, 2)
            except OSError:
                file.seek(0, 0)

            inputs_last100 = file.readlines()
            inputs_last_line = inputs_last100[-1]
            inputs_last_line = inputs_last_line.split()
            inputs_last_line.pop(0)  # remove timestamp
            init_adc_inputs = [float(inputs_last_line[j]) for j in range(12)]

        # Zero-drift for the first 10 cycles
        if k < 10:
            k += 1
            signals_50.append(init_adc_inputs)
            time.sleep(0.005)
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

        # Two streams: inputsF for geometry/rough calc; inputs_mdl for model
        inputsF = [(float(init_adc_inputs[j]) - calibrate[j]) for j in range(12)]
        inputs_mdl = [
            (float(init_adc_inputs[j]) - calibrate[j] + Dataset_calibrate[j])
            for j in range(12)
        ]

        # Physical correction (currently num0=1)
        for i in range(12):
            inputsF[i] = inputsF[i] / num0[i]

        # Rotation matrices (currently identity)
        rotation_matrices0 = [
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        ]

        # Split into four 3D vectors and apply rotation (currently identity)
        vectors = [inputsF[0:3], inputsF[3:6], inputsF[6:9], inputsF[9:12]]
        rotated_vectors = []
        for i in range(4):
            vec = vectors[i]
            mat = rotation_matrices0[i]
            rx = vec[0] * mat[0][0] + vec[1] * mat[0][1] + vec[2] * mat[0][2]
            ry = vec[0] * mat[1][0] + vec[1] * mat[1][1] + vec[2] * mat[1][2]
            rz = vec[0] * mat[2][0] + vec[1] * mat[2][1] + vec[2] * mat[2][2]
            rotated_vectors.append([rx, ry, rz])

        processed_inputs = []
        for vec in rotated_vectors:
            processed_inputs.extend(vec)

        # Rough force/torque estimation
        F_y = -processed_inputs[0] - processed_inputs[3] + processed_inputs[6] + processed_inputs[9]
        F_x = processed_inputs[1] + processed_inputs[4] - processed_inputs[7] - processed_inputs[10]
        F_z = processed_inputs[2] + processed_inputs[5] + processed_inputs[8] + processed_inputs[11]
        print(F_z)

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
        x_F = min(max(x_F, 0), 18)
        y_F = min(max(y_F, 0), 28)

        if F_z < 0:
            try:
                xh, yh, zh = intersect_half_cylinder(
                    x_F, y_F, F_x, F_y, -F_z, R=9.0, H=28.0
                )
            except Exception:
                xh = yh = zh = 0
        else:
            xh = yh = zh = 0

        # This variable is unused; kept as in original code
        aaa = (3.1415926 / 20) * arccos(max(-1.0, min(1.0, 1 - xh / 9))) * 57.3
        X_FB, Y_FB = unwrap_to_uv(xh, yh, zh, R=9.0)

        # Classification + regression
        region_predicted = predict_CNN(inputs_mdl, deg_model, CNN_in_channels) + 1
        prediction = predict_MLP(inputs_mdl, mlp_model_all, scaler_inputs_all, scaler_targets_all)

        # KF_F fusion
        z1 = np.array([F_x * F_k, F_y * F_k, F_z * F_k], dtype=float)
        z2 = np.array([prediction[0], prediction[1], prediction[2]], dtype=float)
        F_clean, state = fuse_force_step(z1, z2, dt_fixed, state, params)

        # KF_X fusion
        if abs(prediction[2]) < 0.5:
            prediction[3] = 0
            prediction[4] = 0
            x_F = 0
            y_F = 0

        z1_xy = (x_F, y_F)
        z2_xy = (prediction[3], prediction[4])
        x_clean, y_clean = clean_xy(z1_xy, z2_xy, dt_fixed)

        # Shared-window dynamic visualization
        # 1) Dial (left): use prediction[0] as the angle
        dial.update(prediction[0])

        # 2) Trajectory (right): enter/exit drawing mode based on thresholds
        if F_z > 500:
            if not drawing:
                # Enter drawing mode: start at (0,0)
                drawing = True
                draw_begin_flag = 1
                traj_points = [(0.0, 0.0)]
                lastpoint_xy = (0.0, 0.0)

                # Clear and reset the right plot
                ax_traj.cla()
                ax_traj.set_aspect("equal", adjustable="box")
                ax_traj.set_xlabel("X (mm)")
                ax_traj.set_ylabel("Y (mm)")
                ax_traj.set_title("Trajectory")
                traj_line, = ax_traj.plot([], [], "-", lw=6)

                # Center view on current (single-point) trajectory
                recenter_to_trajectory(ax_traj, [0.0], [0.0])
            else:
                # Already drawing: advance along prediction[0] direction by 0.01 mm
                draw_begin_flag = 1
                ang_deg = float(prediction[0]) if prediction is not None else 0.0
                ang_rad = math.radians(ang_deg)
                step = ddd  # 0.01 mm (per original comment)

                nx = lastpoint_xy[0] + step * math.cos(ang_rad)
                ny = lastpoint_xy[1] + step * math.sin(ang_rad)
                lastpoint_xy = (nx, ny)
                traj_points.append(lastpoint_xy)

            # Refresh trajectory (no autoscale)
            if traj_points:
                xs, ys = zip(*traj_points)  # mechanical coordinates
            else:
                xs, ys = [], []

            # Original display swap: display X <- y, Y <- x
            xn = ys
            yn = xs
            traj_line.set_data(xn, yn)

            # Center on the bounding-box center of the full trajectory
            xs_disp = list(xn)
            ys_disp = list(yn)
            recenter_to_trajectory(ax_traj, xs_disp, ys_disp)

            fig.canvas.draw_idle()
            fig.canvas.flush_events()
            plt.pause(0.001)

        elif drawing and F_z <= 200:
            # Save current trajectory subplot before clearing
            try:
                if len(traj_points) >= 2:
                    saved_path = save_traj_axes(ax_traj, fig)
                    print(f"[轨迹已保存] {saved_path}")
            except Exception as e:
                print(f"[保存轨迹失败] {e}")

            # Exit and clear until next entry
            drawing = False
            draw_begin_flag = 0
            traj_points.clear()

            ax_traj.cla()
            ax_traj.set_aspect("equal", adjustable="box")
            ax_traj.set_xlabel("X (mm)")
            ax_traj.set_ylabel("Y (mm)")
            ax_traj.set_title("Trajectory (cleared)")
            traj_line, = ax_traj.plot([], [], "-", lw=2)

            fig.canvas.draw_idle()
            fig.canvas.flush_events()

        # Reduce CPU usage
        time.sleep(0.005)
