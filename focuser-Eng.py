#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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

# =========================
# Basic settings
# =========================
BOARD_ID = BoardIds.CYTON_DAISY_BOARD.value   # 16ch: CYTON_DAISY_BOARD; 8ch: CYTON_BOARD
SERIAL_PORT = 'COM3'
# Training settings
BASELINE_SEC   = 60#baseline duration(sec)
TRAIN_SEC      = 20 * 60 #training duration
WINDOW_SEC     = 3  #PSD window length
STEP_SEC       = 1 #update step
#frequency bands
THETA_BAND     = (4.0, 8.0)
BETA_BAND      = (13.0, 20.0)
EMG_BAND       = (20.0, 45.0) #simple EMG proxy band

#dynamic threshold (lower is better: ratio < threshold = hit)
TARGET_SUCCESS = (0.65, 0.80)   #target hit rate range
ADJUST_EVERY   = 15   #adjust every 15s
UP_HARDER      = 0.9   # too easy > tighten threshold
DOWN_EASIER    = 1.10 #too hard> loosen threshold
THR_MIN_FACTOR = 0.30#clamp >= baseline * 0.30
THR_MAX_FACTOR = 3.00 #clamp <= baseline * 3.00

#smoothing
R_EWMA_ALPHA   = 0.30           # 0.2–0.4 is stable; higher is more responsive

#artifact clip(preprocessing stability)
CLIP_UV        = 60e-6

#Adaptive blink detection (replaces fixed thresholds)
BLINK_BAND       = (1.0, 4.0)
BLINK_REF_LP     = 8.0     #low-pass Fp to 8 Hz to keep eye activity
BLINK_Z_PP       = 4.0 #peak-to-peak Z-score threshold
BLINK_Z_BP       = 3.0 #1–4 Hz power Z-score threshold
BLINK_REF_ALPHA  = 0.01    #slow EWMA update of baseline mean during training
BLINK_REFRACT    = 0.30  #blink refractory period (sec)

#Noise veto (EMG/Beta only; check after regression)
EMG_FACTOR     = 8.0        #if EMG/Beta > 8, mark as noisy (muscle)
AMP_REJECT_UV  = None       #absolute amplitude veto disabled; set e.g. 150e-6 to enable

#Feedback & logging
BAR_LEN        = 40
LOG_CSV        = f"tbr_brainflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

# =========================
# HUD settings
# =========================
SHOW_HUD             = True
HUD_WIDTH            = 260
HUD_HEIGHT           = 140
HUD_UPDATE_HZ        = 15
ATTN_DROP_COOLDOWN_S = 3.0   # attention-drop tone cooldown
DROP_GRACE_STEPS     = 1     # grace steps for OK -> drop (avoid one-off jitter)

#pin map(TODO please EDIT HERE BASED ON YOUR PIN)
pin_map = {
    'N1p': 'Fp1','N2p': 'Fp2','N3p': 'C3','N4p': 'C4',
    'N5p': 'P7','N6p': 'Fz','N7p': 'Cz','N8p': 'Pz',
    'N9p': 'F7','N10p': 'F8','N11p':'F3','N12p':'F4',
    'N13p':'T7','N14p':'T8','N15p':'P3','N16p':'P4'
}

# Regions used for TBR (default: average of Fz + Cz; you can also use ['Cz'])
tbr_regions = ['Fz', 'Cz']

# Regions used for EOG reference / blink detection
eog_regions = ['Fp1', 'Fp2']

# =========================
#Audio setting

SOUND_ENABLED          = True
SOUND_USE_WAV_IF_EXIST = True
SOUND_DIR              = os.path.join(os.path.dirname(__file__), 'sounds')
METRONOME_ENABLED      = True
SUCCESS_COOLDOWN_SEC   = 1.5
TONE_MS_DEFAULT        = 120
TONE_MS_LONG           = 300

WAV_FILES = {
    'baseline_start': 'baseline_start.wav',
    'baseline_end'  : 'baseline_end.wav',
    'train_start'   : 'train_start.wav',
    'train_end'     : 'train_end.wav',
    'tick'          : 'tick.wav',
    'success'       : 'success.wav',
    'adjust_up'     : 'adjust_up.wav',
    'adjust_down'   : 'adjust_down.wav',
    'drop'          : 'drop.wav',          # new: attention-drop tone
}

SYNTH_TONES = {
    'baseline_start': (660, TONE_MS_LONG),
    'baseline_end'  : (523, TONE_MS_LONG),
    'train_start'   : (784, TONE_MS_LONG),
    'train_end'     : (392, 450),
    'tick'          : (1000, 60),
    'success'       : (1175, TONE_MS_DEFAULT),
    'drop'          : (420, 180),          # new: attention-drop tone
}

ADJUST_UP_SCALE   = [660, 784, 988]   # harder
ADJUST_DOWN_SCALE = [988, 784, 660]   # easier

# =========================
# Audio helper (winsound)
# =========================
class Sounder:
    def __init__(self):
        self._last_success_ts = 0.0
        self._last_drop_ts = 0.0
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

    def drop(self):
        now = time.time()
        if now - self._last_drop_ts < ATTN_DROP_COOLDOWN_S:
            return
        self._last_drop_ts = now
        self.play_event('drop')


#mini HUD (Tkinter window)

class MiniHUD:
    """
    Minimal HUD: shows Ratio / Thr / status bar
    - Green: OK; Red: below target; Yellow: gated (blink/noise)
    - Non-blocking, runs in a background thread
    """
    def __init__(self, width=HUD_WIDTH, height=HUD_HEIGHT, update_hz=HUD_UPDATE_HZ):
        self.width = width
        self.height = height
        self.period = 1.0 / max(1, update_hz)
        self._thread = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._state = dict(ratio=0.0, thr=0.0, good=False, blink=False, noisy=False)

    def start(self):
        if not SHOW_HUD:
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()

    def update(self, ratio, thr, good, blink, noisy):
        if not SHOW_HUD:
            return
        with self._lock:
            self._state.update(ratio=ratio, thr=thr, good=good, blink=blink, noisy=noisy)

    def _run(self):
        try:
            import tkinter as tk
        except Exception:
            return  # no Tk available

        root = tk.Tk()
        root.title("EEG Focus HUD")
        root.geometry(f"{self.width}x{self.height}+40+40")
        root.resizable(False, False)
        root.attributes("-topmost", True)

        ratio_var = tk.StringVar(value="Ratio: --")
        thr_var   = tk.StringVar(value="Thr: --")
        status_var= tk.StringVar(value="Status: --")

        frm = tk.Frame(root, padx=8, pady=6)
        frm.pack(fill="both", expand=True)

        lbl1 = tk.Label(frm, textvariable=ratio_var, font=("Consolas", 12))
        lbl1.pack(anchor="w")
        lbl2 = tk.Label(frm, textvariable=thr_var, font=("Consolas", 12))
        lbl2.pack(anchor="w")
        lbl3 = tk.Label(frm, textvariable=status_var, font=("Consolas", 12))
        lbl3.pack(anchor="w", pady=(2, 6))

        canvas = tk.Canvas(frm, width=self.width-32, height=22, bg="#222222", highlightthickness=0)
        canvas.pack()
        bar = canvas.create_rectangle(0, 0, 0, 22, fill="#00c853", width=0)

        def tick():
            if self._stop.is_set():
                try:
                    root.destroy()
                except Exception:
                    pass
                return

            with self._lock:
                s = dict(self._state)

            ratio_var.set(f"Ratio: {s['ratio']:.2f}")
            thr_var.set(f"Thr:   {s['thr']:.2f}")

            # Color priority: gated (yellow) > ok (green) > drop (red)
            if s['blink'] or s['noisy']:
                color = "#ffd600"
                status_var.set("Status: gated (blink/noise)")
            else:
                if s['good']:
                    color = "#13b556"
                    status_var.set("Status: OK")
                else:
                    color = "#ff5252"
                    status_var.set("Status: drop")

            # Bar length maps to thr/ratio (clamped)
            try:
                margin = max(0.0, min(1.5, s['thr'] / max(s['ratio'], 1e-9)))
            except Exception:
                margin = 0.0
            L = int((self.width - 32) * (margin / 1.5))
            canvas.coords(bar, 0, 0, max(2, L), 22)
            canvas.itemconfig(bar, fill=color)

            root.after(int(self.period * 1000), tick)

        root.after(200, tick)
        root.protocol("WM_DELETE_WINDOW", lambda: self._stop.set())
        root.mainloop()

# =========================
# BrainFlow helpers
# =========================
def setup_board(board_id=BOARD_ID, serial_port=SERIAL_PORT):
    params = BrainFlowInputParams()
    params.serial_port = serial_port
    board = BoardShim(board_id, params)
    return board

def map_regions_to_indices(regions):
    n_list = []
    for r in regions:
        found = None
        for pin, lab in pin_map.items():
            if lab.lower() == r.lower():
                found = pin
                break
        if not found:
            raise ValueError(f"Region not found in pin_map: {r}")
        idx0 = int(found[1:-1]) - 1  # 'N6p' -> 6 -> idx 5
        n_list.append(idx0)
    return n_list

# Optional NoiseTypes import (older BrainFlow versions may not have it)
try:
    from brainflow.data_filter import NoiseTypes
except Exception:
    NoiseTypes = None

def safe_notch(sig, sfreq, mains=50.0, bw=4.0):
    nyq = sfreq / 2.0
    if NoiseTypes is not None and mains in (50.0, 60.0):
        try:
            noise_enum = NoiseTypes.FIFTY.value if mains == 50.0 else NoiseTypes.SIXTY.value
            DataFilter.remove_environmental_noise(sig, sfreq, noise_enum)
            return
        except Exception:
            pass
    if nyq <= 1.0:
        return
    center = mains
    if center >= nyq * 0.98:
        center = nyq * 0.95
    half = bw / 2.0
    low = max(0.5, center - half)
    high = min(nyq * 0.98, center + half)
    bw_eff = max(0.5, high - low)
    center_eff = (low + high) / 2.0
    try:
        DataFilter.perform_bandstop(sig, sfreq, center_eff, bw_eff, 4, FilterTypes.BUTTERWORTH.value, 0)
    except Exception:
        pass

def preprocess_1_40_notch50(sig, sfreq):
    if sig.dtype != np.float64:
        sig[:] = sig.astype(np.float64, copy=False)
    sig.setflags(write=True)
    DataFilter.detrend(sig, DetrendOperations.CONSTANT.value)
    DataFilter.perform_highpass(sig, sfreq, 1.0, 4, FilterTypes.BUTTERWORTH.value, 0)
    nyq = sfreq / 2.0
    lp = 40.0 if 40.0 < nyq * 0.98 else nyq * 0.95
    if lp > 1.5:
        DataFilter.perform_lowpass(sig, sfreq, lp, 4, FilterTypes.BUTTERWORTH.value, 0)
    safe_notch(sig, sfreq, mains=50.0, bw=4.0)
    sig[:] = np.clip(sig, -CLIP_UV, CLIP_UV)
    return sig

def bandpower_welch(sig, sfreq, fmin, fmax):
    n = len(sig)
    if n < 32:
        return 0.0
    max_base = max(32, n // 2)
    nfft = 1 << int(np.floor(np.log2(max_base)))
    overlap = nfft // 2
    try:
        psd, freqs = DataFilter.get_psd_welch(
            sig, int(nfft), int(overlap), float(sfreq), WindowOperations.HANNING.value
        )
    except Exception:
        win = np.hanning(n)
        if not np.any(np.isfinite(sig)):
            return 0.0
        x = np.asarray(sig) * win
        spec = np.fft.rfft(x)
        freqs = np.fft.rfftfreq(n, d=1.0 / float(sfreq))
        psd = (np.abs(spec) ** 2) / (float(sfreq) * (win**2).sum())
    idx = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(idx):
        return 0.0
    bp = float(np.trapz(psd[idx], freqs[idx]))
    return bp if np.isfinite(bp) else 0.0

# =========================
#Blink detection/  EOG regression/ noise veto
def _robust_center_scale(x):
    x = np.asarray(x, dtype=np.float64)
    med = np.median(x)
    mad = np.median(np.abs(x - med)) + 1e-20
    sigma = 1.4826 * mad  #approx. standard deviation
    return med, max(sigma, 1e-12)

def compute_fp_features(sig, sfreq):
    s = np.ascontiguousarray(sig.copy()).astype(np.float64, copy=False)
    DataFilter.detrend(s, DetrendOperations.CONSTANT.value)
    DataFilter.perform_lowpass(s, sfreq, BLINK_REF_LP, 4, FilterTypes.BUTTERWORTH.value, 0)
    pp = float(np.ptp(s))
    bp = bandpower_welch(s, sfreq, *BLINK_BAND)
    return pp, bp
#------blink detection--------
class BlinkGate:
    def __init__(self, z_pp=BLINK_Z_PP, z_bp=BLINK_Z_BP, alpha=BLINK_REF_ALPHA, refractory=BLINK_REFRACT):
        self.z_pp_th = z_pp
        self.z_bp_th = z_bp
        self.alpha = alpha
        self.refractory = refractory
        self.last_blink_ts = -1e9
        self.pp_mu = None; self.pp_sigma = None
        self.bp_mu = None; self.bp_sigma = None

    def fit_baseline(self, fp_pp_list, fp_bp_list):
        pp_mu, pp_sigma = _robust_center_scale(fp_pp_list)
        bp_mu, bp_sigma = _robust_center_scale(fp_bp_list)
        self.pp_mu, self.pp_sigma = float(pp_mu), float(pp_sigma)
        self.bp_mu, self.bp_sigma = float(bp_mu), float(bp_sigma)

    def _update_ref(self, pp, bp):
        self.pp_mu = (1 - self.alpha) * self.pp_mu + self.alpha * pp
        self.bp_mu = (1 - self.alpha) * self.bp_mu + self.alpha * bp

    def is_blink(self, fp_mat, sfreq, now_ts):
        if self.pp_mu is None or self.bp_mu is None:
            return False
        pp_vals, bp_vals = [], []
        for i in range(fp_mat.shape[0]):
            pp, bp = compute_fp_features(fp_mat[i, :], sfreq)
            pp_vals.append(pp); bp_vals.append(bp)
        pp = float(np.max(pp_vals))   #take the more blink-like side
        bp = float(np.max(bp_vals))

        z_pp = (pp - self.pp_mu) / self.pp_sigma
        z_bp = (bp - self.bp_mu) / self.bp_sigma
        blink = (z_pp >= self.z_pp_th) or (z_bp >= self.z_bp_th)
#refractory period
        if blink and (now_ts - self.last_blink_ts) < self.refractory:
            blink = False
        if blink:
            self.last_blink_ts = now_ts

        #slowly update reference mean
        self._update_ref(pp, bp)
        return blink

def eog_regress_out(use_mat, eeg_mat, fp_rel_idx):
    if not fp_rel_idx or max(fp_rel_idx) >= eeg_mat.shape[0]:
        return use_mat
    X = np.ascontiguousarray(eeg_mat[fp_rel_idx, :].astype(np.float64, copy=False))  # (2, n)
    Xc = np.vstack([X, np.ones((1, X.shape[1]))])  # add constant term (3, n)
    Xt = Xc.T
    Y = use_mat.astype(np.float64, copy=True)
    try:
        XtX_inv = np.linalg.pinv(Xt.T @ Xt)  # (3,3)
        W = XtX_inv @ Xt.T                   # (3,n)
        for i in range(Y.shape[0]):
            beta = W @ Y[i, :]
            Y[i, :] = Y[i, :] - (Xt @ beta)
    except Exception:
        pass
    return Y

def is_noisy_after_regressed(use_mat_regressed, sfreq):
    if AMP_REJECT_UV is not None:
        if np.max(np.abs(use_mat_regressed)) > AMP_REJECT_UV:
            return True
    emg, bet = [], []
    for ch in range(use_mat_regressed.shape[0]):
        sig = np.ascontiguousarray(use_mat_regressed[ch, :].copy())
        preprocess_1_40_notch50(sig, sfreq)
        emg.append(bandpower_welch(sig, sfreq, *EMG_BAND))
        bet.append(bandpower_welch(sig, sfreq, *BETA_BAND))
    emg = np.mean(emg); bet = np.mean(bet) + 1e-20
    return (emg / bet) > EMG_FACTOR

#TBR
def compute_tbr(block_2d, sfreq):
    ratios = []
    for ch in range(block_2d.shape[0]):
        sig = np.ascontiguousarray(block_2d[ch, :].copy())
        preprocess_1_40_notch50(sig, sfreq)
        tpow = bandpower_welch(sig, sfreq, *THETA_BAND)
        bpow = bandpower_welch(sig, sfreq, *BETA_BAND)
        ratio = (tpow / bpow) if bpow > 1e-20 else np.inf
        if not np.isfinite(ratio):
            ratio = 0.0
        ratios.append(ratio)
    return float(np.mean(ratios)) if ratios else 0.0

def render_bar(ratio, threshold, blink=False, noisy=False):
    good = (ratio < threshold)
    margin = max(0.0, min(2.0, threshold / max(ratio, 1e-9)))
    filled = int(min(BAR_LEN, round(BAR_LEN * min(margin, 1.5) / 1.5)))
    bar = '█' * filled + '·' * (BAR_LEN - filled)

    # If blink/noise is detected, do not show ✔ (even if ratio < threshold).
    show_check = (good and not blink and not noisy)
    flag = 'Good' if show_check else ' '

    mark = ''
    if blink: mark += ' 👁 '
    if noisy: mark += ' ⚡ '
    print(f"\rRatio={ratio:5.2f}  Thr={threshold:5.2f}  [{bar}] {flag}{mark}", end='', flush=True)
    return good

# =========================
# Main
# =========================
def main():
    print("[INFO] Starting BrainFlow...")
    BoardShim.enable_dev_board_logger()
    board = setup_board()
    sfreq = BoardShim.get_sampling_rate(BOARD_ID)
    eeg_channels = BoardShim.get_eeg_channels(BOARD_ID)  # channel row indices in BrainFlow buffer (aligned with N1p..N16p order)

    print(f"[INFO] Sampling rate: {sfreq} Hz")
    print(f"[INFO] EEG channel rows (aligned with N1p..N16p): {eeg_channels}")

    snd = Sounder()

    #indices are relative to: eeg_mat = buf[eeg_channels, :]
    rel_idx = map_regions_to_indices(tbr_regions)    # e.g., Fz/Cz -> [5,6]
    fp_rel_idx = map_regions_to_indices(eog_regions) # Fp1/Fp2 -> [0,1]
    max_possible = len(eeg_channels)
    #keep only indices that are within range
    rel_idx = [i for i in rel_idx if i < max_possible]
    fp_rel_idx = [i for i in fp_rel_idx if i < max_possible]
    #ensure rel_idx is never empty
    if not rel_idx:
        rel_idx = [0]


    print(f"[INFO] TBR regions: {tbr_regions} -> relative indices (0-based): {rel_idx}")
    print(f"[INFO] EOG/blink regions: {eog_regions} -> relative indices (0-based): {fp_rel_idx}")

    #start HUD
    hud = MiniHUD()
    hud.start()

    board.prepare_session()
    board.start_stream(45000) #ring buffer
    time.sleep(2.0)

    window_samples = int(WINDOW_SEC * sfreq)
    step_samples   = int(STEP_SEC * sfreq)

    f = open(LOG_CSV, 'w', newline='', encoding='utf-8')
    writer = csv.writer(f)
    writer.writerow(['time', 'phase', 'ratio_raw', 'ratio_ewma', 'threshold', 'success', 'blink', 'noisy'])

    try:
        # ---------- Baseline ----------
        print("[BASELINE] Start: relax with eyes open. Don't try to control your mind")
        snd.play_event('baseline_start')

        baseline_ratios = []
        fp_baseline_pp, fp_baseline_bp = [], []
        baseline_start = time.time()
        last_emit = 0.0

        while time.time() - baseline_start < BASELINE_SEC:
            buf = board.get_current_board_data(window_samples)
            if buf.shape[1] < window_samples:
                time.sleep(0.01); continue

            now = time.time()
            if now - last_emit < STEP_SEC:
                time.sleep(0.005); continue
            last_emit = now

            eeg_mat = buf[eeg_channels, :]

            # Collect baseline Fp features (pp/bp) for blink gating
            if fp_rel_idx and max(fp_rel_idx) < eeg_mat.shape[0]:
                fp_mat = eeg_mat[fp_rel_idx, :]
                pp_list, bp_list = [], []
                for i in range(fp_mat.shape[0]):
                    pp, bp = compute_fp_features(fp_mat[i, :], sfreq)
                    pp_list.append(pp); bp_list.append(bp)

                # Use the more blink-like side to build a conservative baseline
                fp_baseline_pp.append(np.max(pp_list))
                fp_baseline_bp.append(np.max(bp_list))

            #Regress out EOG before compute TBR(baseline should be cleaned)
            use_mat_raw = eeg_mat[rel_idx, :]
            use_mat = eog_regress_out(use_mat_raw, eeg_mat, fp_rel_idx)
            r = compute_tbr(use_mat, sfreq)

            baseline_ratios.append(r)
            thr_tmp = np.median(baseline_ratios) if baseline_ratios else r
            good = render_bar(r, thr_tmp)
            writer.writerow([now, 'baseline', f"{r:.6f}", f"{r:.6f}", f"{thr_tmp:.6f}", "", 0, 0])

            # HUD update
            hud.update(ratio=r, thr=thr_tmp, good=good, blink=False, noisy=False)

            snd.metronome()

        baseline_ratio = float(np.median(baseline_ratios)) if baseline_ratios else 1.0
        threshold = baseline_ratio * 1.15
        threshold = max(baseline_ratio * THR_MIN_FACTOR, min(threshold, baseline_ratio * THR_MAX_FACTOR))

        #Build blink gates
        blink_gate = BlinkGate()
        if fp_baseline_pp and fp_baseline_bp:
            blink_gate.fit_baseline(fp_baseline_pp, fp_baseline_bp)

        print(f"\n[BASELINE] Done: median={baseline_ratio:.3f} -> initial threshold={threshold:.3f}")
        snd.play_event('baseline_end')

    #----------training-------------------
        print("[TRAIN] Start: stay relaxed but alert. Aim for Ratio < Thr (that's a hit).")
        snd.play_event('train_start')

        train_start = time.time()
        last_emit = 0.0
        last_adjust = time.time()
        succ, tot = 0, 0
        prev_good = False
        r_ewma = None

    #attention-drop detection buffer
        drop_buffer = 0

        while time.time() - train_start < TRAIN_SEC:
            buf = board.get_current_board_data(window_samples)
            if buf.shape[1] < window_samples:
                time.sleep(0.01); continue

            now = time.time()
            if now - last_emit < STEP_SEC:
                time.sleep(0.005); continue
            last_emit = now

            eeg_mat = buf[eeg_channels, :]
            use_mat_raw = eeg_mat[rel_idx, :]

            #EOG regression clean the data
            use_mat = eog_regress_out(use_mat_raw, eeg_mat, fp_rel_idx)

            #Adaptive blink detection (Z-score + refractory period)
            if fp_rel_idx and max(fp_rel_idx) < eeg_mat.shape[0]:
                fp_mat = eeg_mat[fp_rel_idx, :]
                blink = blink_gate.is_blink(fp_mat, sfreq, now)
            else:
                blink = False

            #noise detection on regressed signal (EMG/Beta only)
            noisy = is_noisy_after_regressed(use_mat, sfreq)

            #compute TBR (use regressed signal)
            r_raw = compute_tbr(use_mat, sfreq)

            #EWMA smoothing for display/scoring
            r_ewma = r_raw if r_ewma is None else (R_EWMA_ALPHA * r_raw + (1 - R_EWMA_ALPHA) * r_ewma)
            good = render_bar(r_ewma, threshold, blink=blink, noisy=noisy)

            #HUD
            hud.update(ratio=r_ewma, thr=threshold, good=(good and not blink and not noisy), blink=blink, noisy=noisy)

            counted = (not blink) and (not noisy)
            if counted:
                tot += 1
                if good:
                    succ += 1
                    if not prev_good:
                        snd.success()
                    prev_good = True
                    drop_buffer = 0
                else:
            #if drop from OK -> not OK, play an "attention drop" tone (with grace/cooldown)
                    if prev_good:
                        drop_buffer += 1
                        if drop_buffer > DROP_GRACE_STEPS:
                            snd.drop()
                            prev_good = False
                            drop_buffer = 0
                    else:
                        drop_buffer = 0

            writer.writerow([now, 'train', f"{r_raw:.6f}", f"{r_ewma:.6f}", f"{threshold:.6f}",
                             int(good and counted), int(blink), int(noisy)])

            snd.metronome()

            #dynamic threshold control
            if now - last_adjust >= ADJUST_EVERY and tot > 0:
                sr = succ / tot
                if sr > TARGET_SUCCESS[1]:
                    threshold *= UP_HARDER
                    print(f"\n[THRESHOLD] Adjust: hit rate={sr:.2f} -> new threshold={threshold:.3f} harder")
                    snd.adjust_up()
                elif sr < TARGET_SUCCESS[0]:
                    threshold *= DOWN_EASIER
                    print(f"\n[THRESHOLD] Adjust: hit rate={sr:.2f} -> new threshold={threshold:.3f} easier")
                    snd.adjust_down()
                else:
                    print(f"\n[THRESHOLD] Keep: hit rate={sr:.2f} (within target range)")

                threshold = max(baseline_ratio * THR_MIN_FACTOR, min(threshold, baseline_ratio * THR_MAX_FACTOR))
                succ, tot = 0, 0
                last_adjust = now

        print("\n[INFO] Training finished. Log file:", LOG_CSV)
        snd.play_event('train_end')

    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user (Ctrl+C).")
    finally:
        try:
            board.stop_stream()
        except Exception:
            pass
        board.release_session()
        try:
            f.close()
        except Exception:
            pass
        try:
            hud.stop()
            time.sleep(0.2)
        except Exception:
            pass
        print("[INFO] Disconnected and saved.")

if __name__ == "__main__":
    main()
