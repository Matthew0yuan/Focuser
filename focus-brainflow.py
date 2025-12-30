#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Theta/Beta Ratio Neurofeedback — BrainFlow 版本
- 兼容 OpenBCI Cyton / Cyton Daisy
- 实时计算 TBR (Theta/Beta) 比率 + 动态阈值训练
- 终端进度条反馈 + CSV 日志

request manual config
"""

import time
import csv
import sys
import numpy as np
from datetime import datetime

from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import (
    DataFilter, FilterTypes, DetrendOperations, WindowOperations
)

# ========= 基本设置 =========
# 1) 选择你的板卡：CYTON_BOARD（8ch）或 CYTON_DAISY_BOARD（16ch）
BOARD_ID = BoardIds.CYTON_DAISY_BOARD.value    # 你有16通道就用这个；8通道请改成 BoardIds.CYTON_BOARD.value

# 2) 串口（USB Dongle）：Windows 多为 COMx；Mac/Linux 多为 /dev/ttyUSB* 或 /dev/tty.* 
SERIAL_PORT = 'COM3' if sys.platform.startswith('win') else '/dev/ttyUSB0'

# 3) 训练参数
BASELINE_SEC   = 60            # 基线时长
TRAIN_SEC      = 20 * 60       # 训练时长（20分钟）
REFRESH_HZ     = 1             # 每秒更新一次反馈
WINDOW_SEC     = 4             # 频谱窗口长度（秒）
STEP_SEC       = 1             # 滑动步长（秒）

# 4) 频段
THETA_BAND     = (4.0, 8.0)
BETA_BAND      = (13.0, 20.0)

# 5) 动态阈值（维持在“最优学习带”）
TARGET_SUCCESS = (0.60, 0.70)  # 目标命中率区间
ADJUST_EVERY   = 30            # 每 30 秒调整一次阈值
UP_HARDER      = 0.97          # 命中率高 -> 阈值更紧
DOWN_EASIER    = 1.03          # 命中率低 -> 阈值放松

# 6) 反馈/日志
BAR_LEN        = 40
LOG_CSV        = f"tbr_brainflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

# 7) 选择用于计算 TBR 的 EEG 通道索引（基于 BrainFlow 的 eeg_channels 排序）
#    建议将你的 Fz / Cz 分别接到 Cyton 的 ch1/ch2（或 Daisy 顺延），
#    然后在运行时打印的 eeg_channels 中确认索引。这里默认用前两个 EEG 通道：
USE_EEG_INDEX_MODE = 'mean'     # 'first'（仅第1个通道）, 'second'（仅第2个）, 'mean'（前两个的平均）
USE_IDX_A = 0                   # eeg_channels 列表中的第 1 个
USE_IDX_B = 1                   # eeg_channels 列表中的第 2 个

# 8) 伪迹裁剪（振幅阈值，单位：伏特；100 µV = 100e-6）
CLIP_UV = 100e-6


# ========= 工具函数 =========
def setup_board(board_id=BOARD_ID, serial_port=SERIAL_PORT):
    params = BrainFlowInputParams()
    params.serial_port = serial_port
    board = BoardShim(board_id, params)
    return board

def preprocess_1_40_notch50(sig, sfreq):
    """预处理：去趋势 + 1-40Hz 带通 + 50Hz 陷波 + 振幅裁剪"""
    DataFilter.detrend(sig, DetrendOperations.CONSTANT.value)
    # 1-40 Hz 带通（通过 center_freq 和 bandwidth 实现）
    DataFilter.perform_bandpass(sig, sfreq, 20.5, 19.5, 4, FilterTypes.BUTTERWORTH.value, 0)
    # 50 Hz 陷波
    DataFilter.perform_bandstop(sig, sfreq, 50.0, 2.0, 4, FilterTypes.BUTTERWORTH.value, 0)
    # 振幅裁剪，抑制眨眼/肌电爆点
    sig[:] = np.clip(sig, -CLIP_UV, CLIP_UV)
    return sig

def bandpower_welch(sig, sfreq, fmin, fmax):
    """Welch PSD + 频段积分"""
    nfft = DataFilter.get_nearest_power_of_two(len(sig))
    psd, freqs = DataFilter.get_psd_welch(
        sig, nfft, nfft // 2, sfreq, WindowOperations.HANNING.value
    )
    idx = np.where((freqs >= fmin) & (freqs <= fmax))[0]
    if idx.size == 0:
        return 0.0
    return float(np.trapz(psd[idx], freqs[idx]))

def compute_tbr(block_2d, sfreq):
    """
    block_2d: shape (n_channels_used, n_samples) —— 注意与 BrainFlow 的 data 矩阵取法一致
    返回：多个通道（如 Fz/Cz）TBR 的平均
    """
    ratios = []
    for ch in range(block_2d.shape[0]):
        sig = np.ascontiguousarray(block_2d[ch, :].copy())
        preprocess_1_40_notch50(sig, sfreq)
        tpow = bandpower_welch(sig, sfreq, *THETA_BAND)
        bpow = bandpower_welch(sig, sfreq, *BETA_BAND)
        ratio = (tpow / bpow) if bpow > 1e-20 else np.inf
        ratios.append(ratio)
    return float(np.mean(ratios))

def render_bar(ratio, threshold):
    """终端进度条，目标：ratio < threshold（越低越好）"""
    good = (ratio < threshold)
    margin = max(0.0, min(2.0, threshold / max(ratio, 1e-9)))  # >1 代表“好”
    filled = int(min(BAR_LEN, round(BAR_LEN * min(margin, 1.5) / 1.5)))
    bar = '█' * filled + '·' * (BAR_LEN - filled)
    flag = '✔' if good else ' '
    print(f"\rRatio={ratio:5.2f}  Thr={threshold:5.2f}  [{bar}] {flag}", end='', flush=True)
    return good


# ========= 主程序 =========
def main():
    print(">> 初始化 BrainFlow ...")
    BoardShim.enable_dev_board_logger()
    board = setup_board()
    sfreq = BoardShim.get_sampling_rate(BOARD_ID)

    # 获取 EEG 通道索引（BrainFlow 给的是实际板卡的物理序号）
    eeg_channels = BoardShim.get_eeg_channels(BOARD_ID)
    print(f">> 采样率: {sfreq} Hz")
    print(f">> EEG 通道索引（按板卡定义）: {eeg_channels}")

    # 选择用于 TBR 的通道行索引（相对于 eeg_channels）
    use_rows = []
    if USE_EEG_INDEX_MODE == 'first' and len(eeg_channels) >= 1:
        use_rows = [0]
    elif USE_EEG_INDEX_MODE == 'second' and len(eeg_channels) >= 2:
        use_rows = [1]
    elif USE_EEG_INDEX_MODE == 'mean':
        # 默认用前两个 EEG 通道（建议把 Fz/Cz 接在它们上）
        rows = []
        if len(eeg_channels) > USE_IDX_A:
            rows.append(USE_IDX_A)
        if len(eeg_channels) > USE_IDX_B:
            rows.append(USE_IDX_B)
        if not rows:
            rows = [0]
        use_rows = rows
    else:
        use_rows = [0]

    print(f">> 用于 TBR 的相对行索引（相对 eeg_channels）：{use_rows}")

    # 打开会话并开始拉流
    board.prepare_session()
    board.start_stream(45000)  # ring buffer
    time.sleep(2.0)

    # 计算窗口
    window_samples = int(WINDOW_SEC * sfreq)
    step_samples = int(STEP_SEC * sfreq)

    # 日志
    f = open(LOG_CSV, 'w', newline='', encoding='utf-8')
    writer = csv.writer(f)
    writer.writerow(['time', 'phase', 'ratio', 'threshold', 'success'])

    try:
        # ---------- Baseline ----------
        print(">> Baseline 开始：自然睁眼放松，不要刻意控制脑电 ...")
        baseline_ratios = []
        baseline_start = time.time()
        last_emit = 0.0

        while time.time() - baseline_start < BASELINE_SEC:
            buf = board.get_current_board_data(window_samples)
            if buf.shape[1] < window_samples:
                time.sleep(0.01)
                continue

            now = time.time()
            if now - last_emit < STEP_SEC:
                time.sleep(0.005)
                continue
            last_emit = now

            # 从 board 矩阵中取出 EEG 行，再取使用的相对行
            eeg_mat = buf[eeg_channels, :]                # shape: (n_eeg, n_samples)
            use_mat = eeg_mat[use_rows, :]                # e.g. (1 or 2, n_samples)

            r = compute_tbr(use_mat, sfreq)
            baseline_ratios.append(r)

            thr_tmp = np.median(baseline_ratios) if baseline_ratios else r
            render_bar(r, thr_tmp)
            writer.writerow([now, 'baseline', f"{r:.6f}", f"{thr_tmp:.6f}", ""])

        baseline_ratio = float(np.median(baseline_ratios))
        threshold = baseline_ratio * 0.95
        print(f"\n>> Baseline 完成：median={baseline_ratio:.3f} → 初始阈值={threshold:.3f}")

        # ---------- Training ----------
        print(">> 训练开始：放松但清醒，尽量让进度条更满（Ratio < Thr 即“得分”） ...")
        train_start = time.time()
        last_emit = 0.0
        last_adjust = time.time()
        succ, tot = 0, 0

        while time.time() - train_start < TRAIN_SEC:
            buf = board.get_current_board_data(window_samples)
            if buf.shape[1] < window_samples:
                time.sleep(0.01)
                continue

            now = time.time()
            if now - last_emit < STEP_SEC:
                time.sleep(0.005)
                continue
            last_emit = now

            eeg_mat = buf[eeg_channels, :]
            use_mat = eeg_mat[use_rows, :]

            r = compute_tbr(use_mat, sfreq)
            good = render_bar(r, threshold)
            tot += 1
            if good:
                succ += 1
            writer.writerow([now, 'train', f"{r:.6f}", f"{threshold:.6f}", int(good)])

            # 动态阈值（每 ADJUST_EVERY 秒）
            if now - last_adjust >= ADJUST_EVERY and tot > 0:
                sr = succ / tot
                if sr > TARGET_SUCCESS[1]:
                    threshold *= UP_HARDER
                elif sr < TARGET_SUCCESS[0]:
                    threshold *= DOWN_EASIER
                print(f"\n>> 阈值调整：成功率={sr:.2f} → 新阈值={threshold:.3f}")
                succ, tot = 0, 0
                last_adjust = now

        print("\n>> 训练结束。日志文件:", LOG_CSV)

    except KeyboardInterrupt:
        print("\n>> 手动中断。")
    finally:
        try:
            board.stop_stream()
        except Exception:
            pass
        board.release_session()
        f.close()
        print(">> 已断开并保存。")
        

if __name__ == "__main__":
    main()
