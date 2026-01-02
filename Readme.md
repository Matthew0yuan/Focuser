

# Theta/Beta Ratio (TBR) Neurofeedback — BrainFlow + OpenBCI
Choose the version you want to use, please do prepair openBCI board and headset before use, this is Daisy board
Mainly focuser-Eng.py, you will need to modify the code, **REMEMBER TO EDIT PIN MAP BASED ON YOUR PIN MAP****

A real-time EEG neurofeedback script that computes **Theta/Beta Ratio (TBR)** on selected channel (default: **Fz/Cz**) and provides feedback via a **terminal progress bar + audio cues**. Includes basic handling for **blinks/eye movements** using **Fp1/Fp2** as an EOG reference.

---

## What this does

### Session flow
1. **Baseline**
   - You relax with eyes open.
   - The script measures your typical TBR distribution.
   - It sets an initial training threshold based on your baseline median.
   - It also learns your blink baseline from Fp1/Fp2.

2. **Training**
   - for every update step:
     - reads a rolling EEG window (e.g., 3 seconds)
     - regresses out EOG (Fp1/Fp2) from target channels
     - filters the EEG (1–40 Hz + notch)
     - estimates bandpower using Welch PSD
     - computes TBR = theta(4–8 Hz) / beta(13–20 Hz)
     - smooths the ratio with EWMA
     - counts success if `ratio < threshold`
   - The threshold is auto-adjusted to keep success rate in a target range.

### Artifact handling
- **Blink detection** (from Fp1/Fp2):
  - features: peak-to-peak + low-frequency power (1–4 Hz)
  - robust baseline stats (median/MAD) + refractory period
  - blink windows are **not counted** in success-rate stats and do not trigger “success” sound

- **EOG regression**:
  - subtracts the component explained by Fp1/Fp2 (+ constant) from target channels
  - TBR is computed on the cleaned signal

- **Noise detection (EMG-like)**:
  - after regression, checks if high-frequency power (20–45 Hz) is unusually strong
  - noisy windows are excluded from success stats

---

## Requirements

### Python packages
- `brainflow`
- `numpy`

Install:
```bash
pip install brainflow numpy
