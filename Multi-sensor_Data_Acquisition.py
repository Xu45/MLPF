# -*- coding: utf-8 -*-

"""
dual_ch341_read_12ch_9ch_60hz_gui.py

Features:
- Two CH341 devices reading 12-channel and 9-channel sensors respectively
- Fixed 60 Hz sampling
- Global thread-safe buffers for visualization
- Single TkAgg window with two subplots:
    - Top: 12-channel signals
    - Bottom: 9-channel signals
"""

import os
import sys
import time
import threading
import ctypes as ct
from ctypes import wintypes
from collections import deque

import numpy as np

# GUI must run in main thread (TkAgg on Windows)
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.animation import FuncAnimation


# =============================================================================
# User Configuration
# =============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DLL_PATH = os.path.join(BASE_DIR, "CH341DLLA64.DLL")

SAMPLE_RATE_HZ = 60.0
MODE_VALUE = 0x10

I2C_ADDR_7BIT_A = 0x50      # Sensor A (12-channel)
I2C_ADDR_7BIT_B = 0x50      # Sensor B (9-channel)

DEV_INDEX_A = 0
DEV_INDEX_B = 1

FRAME_BYTES_A = 24          # 12 channels × 2 bytes
FRAME_BYTES_B = 18          # 9 channels  × 2 bytes

PRINT_PREFIX_A = "[S12]"
PRINT_PREFIX_B = "[S09]"

BUF_LEN = 600               # ~10 seconds @ 60 Hz
Y_RANGE_A = (-3000, 1000)
Y_RANGE_B = (-3000, 1000)

ANIM_INTERVAL_MS = 100


# =============================================================================
# Global Buffers (Thread-Safe)
# =============================================================================

buf_lock = threading.Lock()

buf_A = [deque(maxlen=BUF_LEN) for _ in range(12)]
buf_B = [deque(maxlen=BUF_LEN) for _ in range(9)]

step_A = deque(maxlen=BUF_LEN)
step_B = deque(maxlen=BUF_LEN)

_ctr_A = 0
_ctr_B = 0


def _append_vals_to_buffer(is_A: bool, vals):
    """Append one frame of values into the global buffers."""
    global _ctr_A, _ctr_B

    with buf_lock:
        if is_A:
            _ctr_A += 1
            step_A.append(_ctr_A)
            for i, v in enumerate(vals):
                if i < len(buf_A):
                    buf_A[i].append(v)
        else:
            _ctr_B += 1
            step_B.append(_ctr_B)
            for i, v in enumerate(vals):
                if i < len(buf_B):
                    buf_B[i].append(v)


# =============================================================================
# CH341 DLL Loading and Binding
# =============================================================================

DLL = ct.WinDLL(DLL_PATH)
print(f"[INFO] Loaded DLL: {DLL_PATH}")

DLL.CH341OpenDevice.argtypes = [wintypes.ULONG]
DLL.CH341OpenDevice.restype = wintypes.HANDLE

DLL.CH341CloseDevice.argtypes = [wintypes.ULONG]
DLL.CH341CloseDevice.restype = None

DLL.CH341SetTimeout.argtypes = [wintypes.ULONG, wintypes.ULONG, wintypes.ULONG]
DLL.CH341SetTimeout.restype = wintypes.BOOL

HAS_SETSTREAM = True
try:
    DLL.CH341SetStream.argtypes = [wintypes.ULONG, wintypes.ULONG]
    DLL.CH341SetStream.restype = wintypes.BOOL
except AttributeError:
    HAS_SETSTREAM = False
    print("[WARN] CH341SetStream not available; using default I2C speed.")

DLL.CH341StreamI2C.argtypes = [
    wintypes.ULONG,
    wintypes.ULONG, wintypes.LPVOID,
    wintypes.ULONG, wintypes.LPVOID,
]
DLL.CH341StreamI2C.restype = wintypes.BOOL


# =============================================================================
# Low-Level I2C Helpers
# =============================================================================

def _addr_byte(addr7: int, read: bool) -> int:
    """Compose I2C address byte."""
    return (addr7 << 1) | (1 if read else 0)


def ch341_open(idx: int):
    """Open CH341 device."""
    h = DLL.CH341OpenDevice(idx)
    if h == 0:
        raise OSError(f"CH341OpenDevice({idx}) failed.")
    DLL.CH341SetTimeout(idx, 500, 500)

    if HAS_SETSTREAM:
        if not DLL.CH341SetStream(idx, 0x01):
            print(f"[WARN] Device {idx}: SetStream failed.")
    return h


def ch341_close(idx: int):
    """Close CH341 device."""
    DLL.CH341CloseDevice(idx)


def i2c_write(idx: int, addr7: int, data: bytes):
    """I2C write."""
    buf = bytes([_addr_byte(addr7, False)]) + data
    wb = (ct.c_ubyte * len(buf))(*buf)
    if not DLL.CH341StreamI2C(idx, len(buf), wb, 0, None):
        raise OSError(f"Device {idx}: I2C write failed.")


def i2c_write_then_read(idx: int, addr7: int, wr: bytes, rd_len: int) -> bytes:
    """I2C write followed by read."""
    buf = bytes([_addr_byte(addr7, False)]) + wr
    wb = (ct.c_ubyte * len(buf))(*buf)
    rb = (ct.c_ubyte * rd_len)()

    if not DLL.CH341StreamI2C(idx, len(buf), wb, rd_len, rb):
        raise OSError(f"Device {idx}: I2C read failed.")
    return bytes(rb)


def read_reg(idx: int, addr7: int, reg: int, n: int = 1) -> bytes:
    """Read device register."""
    return i2c_write_then_read(idx, addr7, bytes([reg & 0xFF]), n)


def write_reg(idx: int, addr7: int, reg: int, val: int):
    """Write device register."""
    i2c_write(idx, addr7, bytes([reg & 0xFF, val & 0xFF]))


def arm_and_wait_ready(
    idx: int,
    addr7: int,
    mode_val: int = MODE_VALUE,
    timeout_s: float = 1.0,
    poll_interval_s: float = 0.001,
):
    """Arm sensor and poll until data ready."""
    write_reg(idx, addr7, 0x86, mode_val)
    write_reg(idx, addr7, 0x87, 0x00)

    t0 = time.time()
    while True:
        if read_reg(idx, addr7, 0x87, 1)[0] != 0:
            return
        if time.time() - t0 > timeout_s:
            raise TimeoutError(f"Device {idx}: ready timeout")
        time.sleep(poll_interval_s)


def read_frame(idx: int, addr7: int, frame_bytes: int) -> bytes:
    """Read one sensor frame."""
    raw = i2c_write_then_read(idx, addr7, b"\x88", frame_bytes)
    write_reg(idx, addr7, 0x87, 0x00)
    return raw


def parse_vals_int16_le(raw: bytes):
    """Parse little-endian int16 values."""
    if len(raw) % 2 != 0:
        raise ValueError("Frame length must be even.")

    vals = []
    for i in range(0, len(raw), 2):
        lo, hi = raw[i], raw[i + 1]
        v = (hi << 8) | lo
        if v & 0x8000:
            v -= 0x10000
        vals.append(v)
    return vals


# =============================================================================
# Sensor Reader Thread
# =============================================================================

class SensorReader(threading.Thread):
    """Thread for continuous sensor acquisition."""

    def __init__(
        self,
        name: str,
        dev_index: int,
        addr7: int,
        frame_bytes: int,
        sample_hz: float,
        is_A: bool,
    ):
        super().__init__(daemon=True)
        self.name = name
        self.dev = dev_index
        self.addr = addr7
        self.nbyt = frame_bytes
        self.period = 1.0 / sample_hz
        self.is_A = is_A

        self._stop_event = threading.Event()
        self._opened = False
        self._next_time = None

    def stop(self):
        """Request thread stop."""
        self._stop_event.set()

    def run(self):
        try:
            ch341_open(self.dev)
            self._opened = True
            self._next_time = time.perf_counter()

            while not self._stop_event.is_set():
                self._next_time += self.period

                try:
                    arm_and_wait_ready(self.dev, self.addr)
                    raw = read_frame(self.dev, self.addr, self.nbyt)
                    vals = parse_vals_int16_le(raw)
                    _append_vals_to_buffer(self.is_A, vals)
                except Exception as e:
                    print(f"[ERROR] {self.name}: {e}")
                    time.sleep(0.05)

                delay = self._next_time - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
                else:
                    self._next_time = time.perf_counter()

        finally:
            if self._opened:
                ch341_close(self.dev)


# =============================================================================
# GUI
# =============================================================================

def launch_gui():
    """Launch TkAgg GUI with two subplots."""
    fig = plt.figure(figsize=(12, 7))
    gs = GridSpec(6, 6, figure=fig)

    ax_A = fig.add_subplot(gs[0:3, :])
    lines_A = [ax_A.plot([], [], label=f"Ch{i + 1}")[0] for i in range(12)]
    ax_A.set_title("12-Channel Signals (A)")
    ax_A.set_ylim(*Y_RANGE_A)
    ax_A.legend(ncol=6, fontsize=8)

    ax_B = fig.add_subplot(gs[3:6, :])
    lines_B = [ax_B.plot([], [], label=f"Ch{i + 1}")[0] for i in range(9)]
    ax_B.set_title("9-Channel Signals (B)")
    ax_B.set_ylim(*Y_RANGE_B)
    ax_B.legend(ncol=6, fontsize=8)

    def update(_):
        with buf_lock:
            ys_A = [np.array(ch) for ch in buf_A]
            ys_B = [np.array(ch) for ch in buf_B]

        for i, line in enumerate(lines_A):
            if ys_A[i].size:
                line.set_data(np.arange(len(ys_A[i])), ys_A[i])
        ax_A.set_xlim(0, max(100, len(ys_A[0])))

        for i, line in enumerate(lines_B):
            if ys_B[i].size:
                line.set_data(np.arange(len(ys_B[i])), ys_B[i])
        ax_B.set_xlim(0, max(100, len(ys_B[0])))

        ax_A.relim()
        ax_A.autoscale_view(scaley=False)
        ax_B.relim()
        ax_B.autoscale_view(scaley=False)

        return lines_A + lines_B

    FuncAnimation(fig, update, interval=ANIM_INTERVAL_MS, blit=False)
    plt.tight_layout()
    plt.show()


# =============================================================================
# Main
# =============================================================================

def main():
    print("[INFO] Starting dual sensor acquisition (A=12ch, B=9ch @ 60Hz)")

    th_A = SensorReader(
        PRINT_PREFIX_A,
        DEV_INDEX_A,
        I2C_ADDR_7BIT_A,
        FRAME_BYTES_A,
        SAMPLE_RATE_HZ,
        is_A=True,
    )
    th_B = SensorReader(
        PRINT_PREFIX_B,
        DEV_INDEX_B,
        I2C_ADDR_7BIT_B,
        FRAME_BYTES_B,
        SAMPLE_RATE_HZ,
        is_A=False,
    )

    th_A.start()
    th_B.start()

    try:
        launch_gui()
    finally:
        th_A.stop()
        th_B.stop()
        th_A.join(timeout=2.0)
        th_B.join(timeout=2.0)
        print("[INFO] Acquisition stopped.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("[FATAL]", e)
        sys.exit(1)
