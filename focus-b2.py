#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Theta/Beta Ratio Neurofeedback — BrainFlow + 自动通道映射 + 提示音(Windows零依赖)
- 兼容 OpenBCI Cyton / Cyton Daisy
- 自动根据 pin_map 锁定 Fz / Cz（N6p / N7p），并用于 TBR 计算（可改为只用其中一个）
- Baseline -> 训练（动态阈值维持 60–70% 命中）-> CSV 日志
- 提示音：阶段提示、节拍、命中提示（带冷却）、阈值调整上/下行音阶
- 预处理：去趋势 + 高通1Hz + 低通40Hz + 安全50Hz陷波 + 振幅裁剪
"""

import os
import time
import csv
import sys
import threading
import numpy as np
from datetime import datetime

from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import (
    DataFilter, FilterTypes, DetrendOperations, WindowOperations
)

# ========== 基本设置 ==========
BOARD_ID = BoardIds.CYTON_DAISY_BOARD.value   # 16通道：CYTON_DAISY_BOARD；8通道：CYTON_BOARD
SERIAL_PORT = 'COM3' if sys.platform.startswith('win') else '/dev/ttyUSB0'

# 训练参数
BASELINE_SEC   = 60            # 基线时长
TRAIN_SEC      = 20 * 60       # 训练时长（20分钟）
WINDOW_SEC     = 3             # 频谱窗口长度（秒）
STEP_SEC       = 1             # 滑动步长（秒）

# 频段
THETA_BAND     = (4.0, 8.0)
BETA_BAND      = (13.0, 20.0)

# 动态阈值
TARGET_SUCCESS = (0.65, 0.8)  # 目标命中率区间
ADJUST_EVERY   = 15            # 每 30 秒调整一次阈值
UP_HARDER      = 0.98          # 命中率高 -> 阈值更紧
DOWN_EASIER    = 1.10          # 命中率低 -> 阈值放松


THR_MIN_FACTOR = 0.30  # 阈值不低于基线的 30%
THR_MAX_FACTOR = 3.00  # 阈值不高于基线的 3 倍


# 反馈/日志
BAR_LEN        = 40
LOG_CSV        = f"tbr_brainflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

# 伪迹裁剪（振幅阈值，单位：伏特；100 µV = 100e-6）
CLIP_UV        = 60e-6

# ========= 针脚映射（Cyton Daisy 推荐布线，可改）=========
pin_map = {
    'N1p': 'Fp1','N2p': 'Fp2','N3p': 'C3','N4p': 'C4',
    'N5p': 'P7','N6p': 'Fz','N7p': 'Cz','N8p': 'Pz',
    'N9p': 'F7','N10p': 'F8','N11p':'F3','N12p':'F4',
    'N13p':'T7','N14p':'T8','N15p':'P3','N16p':'P4'
}
# 用于 TBR 的脑区（默认 Fz + Cz 的平均；可改为 ['Fz'] 或 ['Cz']）
tbr_regions = ['Fz', 'Cz']

# ========= 声音设置（Windows零依赖）=========
SOUND_ENABLED          = True
SOUND_USE_WAV_IF_EXIST = True      # 若目录 sounds/ 下存在 wav，则优先用 wav
SOUND_DIR              = os.path.join(os.path.dirname(__file__), 'sounds')

# 节拍提示（每步轻微 tick）
METRONOME_ENABLED      = True

# 命中提示冷却（秒）：避免连续 ding 过密
SUCCESS_COOLDOWN_SEC   = 1.5

# 合成音默认频率与时长（Beep）
TONE_MS_DEFAULT        = 120
TONE_MS_LONG           = 300

# 事件 -> wav 文件名（放在 sounds/ 目录）；若找不到则用合成 Beep
WAV_FILES = {
    'baseline_start': 'baseline_start.wav',
    'baseline_end'  : 'baseline_end.wav',
    'train_start'   : 'train_start.wav',
    'train_end'     : 'train_end.wav',
    'tick'          : 'tick.wav',
    'success'       : 'success.wav',
    'adjust_up'     : 'adjust_up.wav',
    'adjust_down'   : 'adjust_down.wav',
}

# 合成 Beep 配置（Hz）
SYNTH_TONES = {
    'baseline_start': (660, TONE_MS_LONG),
    'baseline_end'  : (523, TONE_MS_LONG),
    'train_start'   : (784, TONE_MS_LONG),
    'train_end'     : (392, 450),
    'tick'          : (1000, 60),
    'success'       : (1175, TONE_MS_DEFAULT),
}
# 上/下行音阶（用于阈值调整反馈）
ADJUST_UP_SCALE   = [660, 784, 988]   # 难度↑
ADJUST_DOWN_SCALE = [988, 784, 660]   # 难度↓

# ========= 声音工具（winsound零依赖）=========
class Sounder:
    def __init__(self):
        self._last_success_ts = 0.0
        if sys.platform.startswith('win') and SOUND_ENABLED:
            self._backend = 'winsound'
        else:
            self._backend = 'mute'

    def _wav_path(self, name):
        if not SOUND_USE_WAV_IF_EXIST:
            return None
        fname = WAV_FILES.get(name)
        if not fname:
            return None
        full = os.path.join(SOUND_DIR, fname)
        return full if os.path.isfile(full) else None

    def play_event(self, name):
        if self._backend != 'winsound':
            return
        import winsound
        p = self._wav_path(name)
        if p is not None:
            try:
                winsound.PlaySound(p, winsound.SND_FILENAME | winsound.SND_ASYNC)
                return
            except Exception:
                pass
        # 合成 Beep
        freq, ms = SYNTH_TONES.get(name, (880, TONE_MS_DEFAULT))
        try:
            winsound.Beep(int(freq), int(ms))
        except Exception:
            pass

    def play_tone(self, freq, ms=TONE_MS_DEFAULT):
        if self._backend != 'winsound':
            return
        import winsound
        try:
            winsound.Beep(int(freq), int(ms))
        except Exception:
            pass

    def _play_scale(self, freqs, per_ms=120):
        for f in freqs:
            self.play_tone(f, per_ms)
            time.sleep(per_ms / 1000.0 * 0.85)

    def metronome(self):
        if self._backend != 'winsound' or not METRONOME_ENABLED:
            return
        self.play_event('tick')

    def success(self):
        now = time.time()
        if now - self._last_success_ts < SUCCESS_COOLDOWN_SEC:
            return
        self._last_success_ts = now
        self.play_event('success')

    def adjust_up(self):
        threading.Thread(target=self._play_scale, args=(ADJUST_UP_SCALE, 120), daemon=True).start()

    def adjust_down(self):
        threading.Thread(target=self._play_scale, args=(ADJUST_DOWN_SCALE, 120), daemon=True).start()


# ========= BrainFlow 工具 =========
def setup_board(board_id=BOARD_ID, serial_port=SERIAL_PORT):
    params = BrainFlowInputParams()
    params.serial_port = serial_port
    board = BoardShim(board_id, params)
    return board

def map_regions_to_indices(regions):
    """
    根据 pin_map 反查脑区到 N{num}p，再转为 0-based 行索引（相对 eeg_channels）
    retunr: [idx0, idx1, ...] （0-based）
    """
    n_list = []
    for r in regions:
        found = None
        for pin, lab in pin_map.items():
            if lab.lower() == r.lower():
                found = pin
                break
        if not found:
            raise ValueError(f"在 pin_map 中找不到脑区 {r}")
        idx0 = int(found[1:-1]) - 1  # 'N6p' -> 6 -> idx 5
        n_list.append(idx0)
    return n_list

# 可选引入 NoiseTypes（旧版 BrainFlow 可能没有）
try:
    from brainflow.data_filter import NoiseTypes
except Exception:
    NoiseTypes = None

def safe_notch(sig, sfreq, mains=50.0, bw=4.0):
    """
    对 50/60 Hz 做“安全陷波”，自动避开 Nyquist 边界；
    若不可行则尝试 BrainFlow 的环境电噪去除；仍不可行则跳过。
    """
    nyq = sfreq / 2.0

    # 1) 优先使用 remove_environmental_noise（更鲁棒）
    if NoiseTypes is not None and mains in (50.0, 60.0):
        try:
            noise_enum = NoiseTypes.FIFTY.value if mains == 50.0 else NoiseTypes.SIXTY.value
            DataFilter.remove_environmental_noise(sig, sfreq, noise_enum)
            return
        except Exception:
            pass  # 回退到手工 bandstop

    # 2) 手工带阻：确保 (center ± bw/2) 落在 (0, Nyquist) 内
    if nyq <= 1.0:
        return  # 采样过低，放弃

    center = mains
    if center >= nyq * 0.98:
        center = nyq * 0.95

    half = bw / 2.0
    low = max(0.5, center - half)
    high = min(nyq * 0.98, center + half)
    bw_eff = max(0.5, high - low)
    center_eff = (low + high) / 2.0

    try:
        DataFilter.perform_bandstop(
            sig, sfreq,
            center_eff, bw_eff,
            4, FilterTypes.BUTTERWORTH.value, 0
        )
    except Exception:
        pass  # 仍失败则跳过

def preprocess_1_40_notch50(sig, sfreq):
    """预处理：去趋势 + 高通1Hz + 低通40Hz + 安全50Hz陷波 + 振幅裁剪"""
    # 保证 float64 & 连续内存
    if sig.dtype != np.float64:
        sig[:] = sig.astype(np.float64, copy=False)
    sig.setflags(write=True)

    DataFilter.detrend(sig, DetrendOperations.CONSTANT.value)

    # 高通 1 Hz
    DataFilter.perform_highpass(sig, sfreq, 1.0, 4, FilterTypes.BUTTERWORTH.value, 0)

    # 低通 40 Hz（避免贴 Nyquist）
    nyq = sfreq / 2.0
    lp = 40.0
    if lp >= nyq * 0.98:
        lp = nyq * 0.95
    if lp > 1.5:  # 防止极低采样率下无效
        DataFilter.perform_lowpass(sig, sfreq, lp, 4, FilterTypes.BUTTERWORTH.value, 0)

    # 安全陷波（AU 是 50Hz；若在 60Hz 地区把 mains 改成 60.0）
    safe_notch(sig, sfreq, mains=50.0, bw=4.0)

    # 振幅裁剪（去大伪迹）
    sig[:] = np.clip(sig, -CLIP_UV, CLIP_UV)
    return sig


def bandpower_welch(sig, sfreq, fmin, fmax):
    """
    更稳的 Welch：保证至少两段（50% 重叠），否则回退到 NumPy periodogram。
    """
    n = len(sig)
    if n < 32:  # 太短直接放弃
        return 0.0

    # 选取 <= n//2 的最大 2 的幂，保证至少两段
    max_base = max(32, n // 2)
    nfft = 1 << int(np.floor(np.log2(max_base)))
    overlap = nfft // 2

    psd = None
    freqs = None

    # 先尝试 BrainFlow Welch
    try:
        psd, freqs = DataFilter.get_psd_welch(
            sig, int(nfft), int(overlap), float(sfreq), WindowOperations.HANNING.value
        )
    except Exception:
        # 回退：用 NumPy 计算简单功率谱（带汉宁窗）
        win = np.hanning(n)
        # 避免全零或NaN
        if not np.any(np.isfinite(sig)):
            return 0.0
        x = np.asarray(sig) * win
        spec = np.fft.rfft(x)
        freqs = np.fft.rfftfreq(n, d=1.0 / float(sfreq))
        # PSD 归一化：等价于 periodogram（单位近似 V^2/Hz）
        psd = (np.abs(spec) ** 2) / (float(sfreq) * (win**2).sum())

    # 频段积分
    idx = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(idx):
        return 0.0
    bp = float(np.trapz(psd[idx], freqs[idx]))
    if not np.isfinite(bp):
        return 0.0
    return bp


def compute_tbr(block_2d, sfreq):
    """
    block_2d: shape (n_used_channels, n_samples) —— 每个通道计算 theta/beta，再取平均
    """
    ratios = []
    for ch in range(block_2d.shape[0]):
        sig = np.ascontiguousarray(block_2d[ch, :].copy())
        preprocess_1_40_notch50(sig, sfreq)
        tpow = bandpower_welch(sig, sfreq, *THETA_BAND)
        bpow = bandpower_welch(sig, sfreq, *BETA_BAND)
        ratio = (tpow / bpow) if bpow > 1e-20 else np.inf
        # 防御 NaN/Inf
        if not np.isfinite(ratio):
            ratio = 0.0
        ratios.append(ratio)
    return float(np.mean(ratios)) if ratios else 0.0

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
    eeg_channels = BoardShim.get_eeg_channels(BOARD_ID)  # 逻辑行号（1..8/16）
    print(f">> 采样率: {sfreq} Hz")
    print(f">> EEG 逻辑通道编号（与 N1p..N16p 对齐）: {eeg_channels}")

    # 声音器
    snd = Sounder()

    # 自动把 tbr_regions（Fz/Cz）转成相对 eeg_channels 的 0-based 行索引
    rel_idx = map_regions_to_indices(tbr_regions)  # 0-based：N6p->5, N7p->6
    # 校验这些索引在 eeg_channels 长度范围内
    max_possible = len(eeg_channels)
    rel_idx = [i for i in rel_idx if i < max_possible]
    if not rel_idx:
        rel_idx = [0]  # 兜底
    print(f">> 用于 TBR 的脑区: {tbr_regions} -> 相对行索引(0-based): {rel_idx}")

    # 打开会话并开始拉流
    board.prepare_session()
    board.start_stream(45000)  # ring buffer
    time.sleep(2.0)

    window_samples = int(WINDOW_SEC * sfreq)
    step_samples   = int(STEP_SEC * sfreq)

    f = open(LOG_CSV, 'w', newline='', encoding='utf-8')
    writer = csv.writer(f)
    writer.writerow(['time', 'phase', 'ratio', 'threshold', 'success'])

    try:
        # ---------- Baseline ----------
        print(">> Baseline 开始：自然睁眼放松，不要刻意控制脑电 ...")
        snd.play_event('baseline_start')

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

            eeg_mat = buf[eeg_channels, :]   # shape: (n_eeg, n_samples)
            use_mat = eeg_mat[rel_idx, :]    # 取 Fz/Cz 对应的行
            r = compute_tbr(use_mat, sfreq)

            baseline_ratios.append(r)
            thr_tmp = np.median(baseline_ratios) if baseline_ratios else r
            render_bar(r, thr_tmp)
            writer.writerow([now, 'baseline', f"{r:.6f}", f"{thr_tmp:.6f}", ""])

            snd.metronome()

        baseline_ratio = float(np.median(baseline_ratios)) if baseline_ratios else 1.0
        threshold = baseline_ratio * 1.15
        threshold = max(baseline_ratio * THR_MIN_FACTOR, min(threshold, baseline_ratio * THR_MAX_FACTOR))
        print(f"\n>> Baseline 完成：median={baseline_ratio:.3f} → 初始阈值={threshold:.3f}")
        snd.play_event('baseline_end')

        # ---------- Training ----------
        print(">> 训练开始：放松但清醒，尽量让进度条更满（Ratio < Thr 即“得分”） ...")
        snd.play_event('train_start')

        train_start = time.time()
        last_emit = 0.0
        last_adjust = time.time()
        succ, tot = 0, 0
        prev_good = False

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
            use_mat = eeg_mat[rel_idx, :]
            r = compute_tbr(use_mat, sfreq)

            good = render_bar(r, threshold)
            tot += 1
            if good:
                succ += 1
            writer.writerow([now, 'train', f"{r:.6f}", f"{threshold:.6f}", int(good)])

            # 声音反馈
            snd.metronome()
            if good and not prev_good:
                snd.success()
            prev_good = good

            # 动态阈值
            if now - last_adjust >= ADJUST_EVERY and tot > 0:
                sr = succ / tot
                if sr > TARGET_SUCCESS[1]:
                    threshold *= UP_HARDER
                    print(f"\n>> 阈值调整：成功率={sr:.2f} → 新阈值={threshold:.3f}（↑更难）")
                    snd.adjust_up()
                elif sr < TARGET_SUCCESS[0]:
                    threshold *= DOWN_EASIER
                    print(f"\n>> 阈值调整：成功率={sr:.2f} → 新阈值={threshold:.3f}（↓放松）")
                    snd.adjust_down()
                else:
                    print(f"\n>> 阈值保持：成功率={sr:.2f}（在目标区间）")
                succ, tot = 0, 0
                last_adjust = now

        print("\n>> 训练结束。日志文件:", LOG_CSV)
        snd.play_event('train_end')

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
