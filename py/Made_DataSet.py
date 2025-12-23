#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Made_DataSet.py

Ethernet-based force sensor data acquisition and dataset generation tool.

Features:
- TCP communication with M4313MXB Ethernet sensor
- Real-time GSD data parsing (6 channels)
- Baseline subtraction (auto-record first valid frame)
- Keyboard-triggered data saving
- Concurrent acquisition and monitoring threads
- Excel dataset generation (12-channel + calibrated 3-axis data)
"""

import os
import time
import socket
import struct
import threading
from typing import List, Tuple, Optional

import keyboard
import numpy as np
import pandas as pd


# Global incremental ID used when saving Excel records
ID = 0


# =============================================================================
# Sensor Class
# =============================================================================

class M4313MXB_Ethernet_Sensor:
    """
    Ethernet interface for M4313MXB force sensor.
    """

    def __init__(
        self,
        target_ip: str = "192.168.0.108",
        target_port: int = 4008,
        local_ip: str = "192.168.0.1",
        sample_rate: int = 60,
    ):
        self.target_ip = target_ip
        self.target_port = target_port
        self.local_ip = local_ip
        self.sample_rate = sample_rate

        self.tcp_socket: Optional[socket.socket] = None
        self.last_package_no = -1
        self.is_running = False

        # Baseline calibration
        self.baseline: Optional[List[float]] = None
        self.baseline_recorded = False

        # Latest calibrated sensor data cache (for monitor thread)
        self.latest_calibrated_data: Optional[List[float]] = None

    # -------------------------------------------------------------------------

    def open_tcp_connection(self) -> bool:
        """Establish TCP connection to sensor."""
        try:
            self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.tcp_socket.bind((self.local_ip, 0))
            self.tcp_socket.connect((self.target_ip, self.target_port))
            self.tcp_socket.settimeout(0.5)
            print(f"TCP连接成功：{self.local_ip} -> {self.target_ip}:{self.target_port}")
            return True
        except Exception as e:
            print(f"TCP连接失败：{str(e)}")
            if self.tcp_socket:
                self.tcp_socket.close()
            return False

    # -------------------------------------------------------------------------

    def send_command(self, cmd: str) -> Tuple[bool, str]:
        """Send AT command to sensor."""
        if not self.tcp_socket:
            return False, "TCP连接未建立"

        full_cmd = f"AT+{cmd}\r\n"
        try:
            self.tcp_socket.sendall(full_cmd.encode("ascii"))
            time.sleep(0.1)

            try:
                response = self.tcp_socket.recv(1024).decode("ascii", errors="ignore").strip()
            except socket.timeout:
                if cmd == "GSD":
                    print("GSD指令已发送，开始连续接收数据（无ACK响应）")
                    return True, "GSD command sent"
                return False, "接收响应超时"

            if "OK" in response:
                print(f"指令成功：{full_cmd.strip()} | 响应：{response}")
                return True, response
            if "ERROR" in response:
                print(f"指令失败：{full_cmd.strip()} | 响应：{response}")
                return False, response

            if cmd == "GSD":
                print("GSD指令已发送，收到非标准响应，继续接收数据")
                return True, "GSD command sent (non-standard response)"

            return False, f"无效响应：{response}"

        except Exception as e:
            print(f"指令发送异常：{str(e)}")
            return False, str(e)

    # -------------------------------------------------------------------------

    def set_sample_rate(self) -> bool:
        """Configure sensor sample rate."""
        success, _ = self.send_command(f"SMPR={self.sample_rate}")
        if success:
            _, response = self.send_command("SMPR=?")
            if str(self.sample_rate) in response:
                print(f"采样率已确认：{self.sample_rate}Hz")
                return True
            print(f"采样率验证失败，响应：{response}")
        return False

    # -------------------------------------------------------------------------

    def parse_gsd_data(self, data: bytes) -> Tuple[bool, List[float] | str]:
        """Parse a single 31-byte GSD data packet."""
        if len(data) != 31:
            return False, f"数据包总长度异常：{len(data)}字节（需31字节）"

        if data[:2] != b"\xaa\x55":
            return False, f"帧头错误：{data[:2].hex().upper()}"

        package_len = (data[2] << 8) | data[3]
        if package_len != 27:
            return False, f"包长度异常：{package_len}字节（需27字节）"

        package_no = (data[4] << 8) | data[5]
        if self.last_package_no != -1:
            expected = (self.last_package_no + 1) % 65536
            if package_no != expected:
                lost = (package_no - self.last_package_no - 1) % 65536
                print(f"警告：丢包{lost}个")
        self.last_package_no = package_no

        channels: List[float] = []
        for i in range(6):
            start = 6 + i * 4
            raw = data[start:start + 4][::-1]
            try:
                value = struct.unpack(">f", raw)[0]
                channels.append(round(value, 6))
            except Exception as e:
                return False, f"通道{i + 1}解析失败：{e}"

        checksum_calc = sum(data[6:30]) & 0xFF
        checksum_recv = data[30]
        if checksum_calc != checksum_recv:
            return False, "校验和不匹配"

        return True, channels

    # -------------------------------------------------------------------------

    def start_gsd_acquisition(self, duration: Optional[int] = None) -> None:
        """Start continuous GSD acquisition."""
        if not self.tcp_socket:
            print("请先建立TCP连接")
            return

        success, _ = self.send_command("GSD")
        if not success:
            print("GSD指令发送失败")
            return

        self.is_running = True
        buffer = b""
        start_time = time.time()

        try:
            while self.is_running:
                if duration and time.time() - start_time > duration:
                    print("采集完成")
                    break

                try:
                    chunk = self.tcp_socket.recv(1024)
                    buffer += chunk
                except socket.timeout:
                    continue

                while len(buffer) >= 31:
                    idx = buffer.find(b"\xaa\x55")
                    if idx == -1:
                        buffer = b""
                        break
                    buffer = buffer[idx:]

                    packet, buffer = buffer[:31], buffer[31:]
                    ok, result = self.parse_gsd_data(packet)

                    if ok:
                        ts = time.time()
                        if not self.baseline_recorded:
                            self.baseline = result
                            self.baseline_recorded = True
                            self.latest_calibrated_data = [0.0] * 6
                            print(f"【基准值已记录】{result}")
                        else:
                            calibrated = [
                                round(result[i] - self.baseline[i], 6)
                                for i in range(6)
                            ]
                            self.latest_calibrated_data = calibrated
                            print(f"[{ts:.3f}] 校准后数据: {calibrated}")
                    else:
                        print(f"解析失败: {result}")

                time.sleep(max(0, 1 / self.sample_rate - 0.0005))

        finally:
            self.stop_gsd_acquisition()

    # -------------------------------------------------------------------------

    def stop_gsd_acquisition(self) -> None:
        """Stop acquisition."""
        self.is_running = False
        if self.tcp_socket:
            try:
                self.tcp_socket.sendall(b"AT+SFWV=?\r\n")
                time.sleep(0.2)
                self.tcp_socket.recv(1024)
                print("GSD采集已停止")
            except Exception as e:
                print(f"停止异常：{e}")

    # -------------------------------------------------------------------------

    def close_tcp_connection(self) -> None:
        """Close TCP connection."""
        if self.tcp_socket:
            self.tcp_socket.close()
            self.tcp_socket = None
            print("TCP连接已关闭")


# =============================================================================
# Data Utilities
# =============================================================================

def get_latest_channel_data(file_path: str) -> Optional[List[float]]:
    """Read last valid 12-channel record from file."""
    try:
        with open(file_path, "rb") as f:
            f.seek(-150, os.SEEK_END)
            lines = f.readlines()
            if not lines:
                return None

            parts = lines[-1].decode("utf-8", errors="ignore").split()
            if len(parts) < 13:
                return None

            return [float(parts[i]) for i in range(1, 13)]
    except Exception:
        return None


def save_to_excel(data: List[float], excel_file: str, data_type: str) -> None:
    """Append data to Excel file."""
    global ID

    if os.path.exists(excel_file):
        df_old = pd.read_excel(excel_file)
        new_id = len(df_old)
    else:
        new_id = 0

    ID = new_id
    record = {"id": new_id}

    if data_type == "12channel":
        for i in range(12):
            record[f"ch{i + 1}"] = data[i]
    elif data_type == "3channel":
        record["fx"], record["fy"], record["fz"] = data
    else:
        return

    df_new = pd.DataFrame([record])
    os.makedirs(os.path.dirname(excel_file), exist_ok=True)

    if os.path.exists(excel_file):
        pd.concat([df_old, df_new]).to_excel(excel_file, index=False)
    else:
        df_new.to_excel(excel_file, index=False)


# =============================================================================
# Threads
# =============================================================================

def data_monitor(sensor: M4313MXB_Ethernet_Sensor) -> None:
    """Keyboard-triggered data saving thread."""
    file_dir = r"C:\Users\86135\Desktop\NdtToolbox-v1.0.18\Data"
    excel_dir = r"C:\Users\86135\Desktop\CNN-MLP\truedata\xy"

    files = os.listdir(file_dir)
    if not files:
        return

    file_path = os.path.join(file_dir, files[0])
    excel_12 = os.path.join(excel_dir, "channel_data_records.xlsx")
    excel_3 = os.path.join(excel_dir, "sensor_3axis_records.xlsx")

    while True:
        if keyboard.is_pressed("enter"):
            data12 = get_latest_channel_data(file_path)
            if data12:
                save_to_excel(data12, excel_12, "12channel")

            if sensor.latest_calibrated_data:
                save_to_excel(sensor.latest_calibrated_data[:3], excel_3, "3channel")

            time.sleep(0.5)

        if keyboard.is_pressed("esc"):
            break

        time.sleep(0.05)


def sensor_acquisition(sensor: M4313MXB_Ethernet_Sensor) -> None:
    """Sensor acquisition thread."""
    if sensor.open_tcp_connection() and sensor.set_sample_rate():
        threading.Thread(
            target=sensor.start_gsd_acquisition,
            daemon=True,
        ).start()


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("启动数据采集系统...")

    shared_sensor = M4313MXB_Ethernet_Sensor(sample_rate=100)

    threading.Thread(
        target=data_monitor,
        args=(shared_sensor,),
        daemon=True,
    ).start()

    threading.Thread(
        target=sensor_acquisition,
        args=(shared_sensor,),
        daemon=True,
    ).start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        shared_sensor.close_tcp_connection()
        print("程序已退出")
