"""
Script: tfr_evoked_analysis.py
Version: 1.0
Author: Philip Press
Date: [20-03-2025]
License: MIT License (https://opensource.org/licenses/MIT)

Description:
    This script loads preprocessed execution and imagery epochs from multiple sessions,
    merges them, performs time-frequency analysis, compares conditions, and computes
    evoked responses for MEG data.

Usage:
    Update the epoch file paths and save directory before running.

Dependencies:
    - MNE-Python
    - NumPy
    - Matplotlib
"""

import os
import numpy as np
import mne
import matplotlib

matplotlib.use("QtAgg")
import matplotlib.pyplot as plt

# ========================== Load Epochs ========================== #

print("Loading Execution Epochs...")
s1_exe = mne.read_epochs("/path/to/f_epoexe_s0_raw.fif", preload=True)
s2_exe = mne.read_epochs("/path/to/f_epoexe_s1_raw.fif", preload=True)
s3_exe = mne.read_epochs("/path/to/f_epoexe_s2_raw.fif", preload=True)

print("Loading Imagery Epochs...")
s1_ima = mne.read_epochs("/path/to/g_epoima_s0_raw.fif", preload=True)
s2_ima = mne.read_epochs("/path/to/g_epoima_s1_raw.fif", preload=True)
s3_ima = mne.read_epochs("/path/to/g_epoima_s2_raw.fif", preload=True)

# ========================== Merge Sessions ========================== #

exe_epo_comb = mne.epochs.concatenate_epochs([s1_exe, s2_exe, s3_exe])
ima_epo_comb = mne.epochs.concatenate_epochs([s1_ima, s2_ima, s3_ima])

# ========================== Fix Channel Names ========================== #

for epochs in [exe_epo_comb, ima_epo_comb]:
    epochs.rename_channels({
        ch: ch.replace("-", "") for ch in epochs.info["ch_names"]
    })

# ========================== Time-Frequency Analysis: Low Frequencies ========================== #

freqs = np.arange(2, 31, 1)
n_cycles = freqs / 2
time_bandwidth = 2.0

print("Computing low-frequency TFR (2–30 Hz)...")
tfr_exe = mne.time_frequency.tfr_multitaper(
    exe_epo_comb, freqs=freqs, n_cycles=n_cycles, time_bandwidth=time_bandwidth,
    picks="mag", use_fft=True, return_itc=False, average=True, decim=2, n_jobs=-1
)

tfr_ima = mne.time_frequency.tfr_multitaper(
    ima_epo_comb, freqs=freqs, n_cycles=n_cycles, time_bandwidth=time_bandwidth,
    picks="mag", use_fft=True, return_itc=False, average=True, decim=2, n_jobs=-1
)

# Plot TFR results
tfr_exe.plot_topo(tmin=-1, tmax=3, baseline=[-1, 0], mode="percent", title="Execution: <30 Hz")
tfr_ima.plot_topo(tmin=-1, tmax=3, baseline=[-1, 0], mode="percent", title="Imagery: <30 Hz")

# Condition difference
tfr_diff = tfr_exe.copy()
tfr_diff.data = (tfr_exe.data - tfr_ima.data) / (tfr_exe.data + tfr_ima.data)
tfr_diff.plot_topo(tmin=-2, tmax=3, title="Difference: Execution - Imagery (<30 Hz)")

# ========================== Time-Frequency Analysis: Gamma Band ========================== #

freqs = np.arange(30, 101, 2)
n_cycles = freqs / 4
time_bandwidth = 4.0

print("Computing high-frequency TFR (30–100 Hz)...")
tfr_gamma_exe = mne.time_frequency.tfr_multitaper(
    exe_epo_comb, freqs=freqs, n_cycles=n_cycles, time_bandwidth=time_bandwidth,
    picks="mag", use_fft=True, return_itc=False, average=True, decim=2, n_jobs=-1
)

tfr_gamma_ima = mne.time_frequency.tfr_multitaper(
    ima_epo_comb, freqs=freqs, n_cycles=n_cycles, time_bandwidth=time_bandwidth,
    picks="mag", use_fft=True, return_itc=False, average=True, decim=2, n_jobs=-1
)

# Plot TFR gamma results
tfr_gamma_exe.plot_topo(tmin=-1, tmax=3, baseline=[-1, 0], mode="percent",
                        title="Execution: Gamma (>30 Hz)", vmin=-0.5, vmax=0.5)
tfr_gamma_ima.plot_topo(tmin=-1, tmax=3, baseline=[-1, 0], mode="percent",
                        title="Imagery: Gamma (>30 Hz)", vmin=-1, vmax=1)

# Condition difference
tfr_diff_gamma = tfr_gamma_exe.copy()
tfr_diff_gamma.data = (tfr_gamma_exe.data - tfr_gamma_ima.data) / (
    tfr_gamma_exe.data + tfr_gamma_ima.data
)
tfr_diff_gamma.plot_topo(tmin=-2, tmax=3, title="Difference: Execution - Imagery (>30 Hz)")

# ========================== Save Results ========================== #

save_dir = r"/path/to/save/directory/"  # <-- Update this
tfr_exe.save(os.path.join(save_dir, input("Filename for Execution TFR (<30 Hz): ")), overwrite=True)
tfr_ima.save(os.path.join(save_dir, input("Filename for Imagery TFR (<30 Hz): ")), overwrite=True)
tfr_gamma_exe.save(os.path.join(save_dir, input("Filename for Execution TFR (>30 Hz): ")), overwrite=True)
tfr_gamma_ima.save(os.path.join(save_dir, input("Filename for Imagery TFR (>30 Hz): ")), overwrite=True)

# ========================== Evoked Responses ========================== #

print("Computing evoked responses (1–30 Hz)...")
exe_epo_comb.filter(l_freq=1, h_freq=30, n_jobs=-1)
ima_epo_comb.filter(l_freq=1, h_freq=30, n_jobs=-1)

evoked_exe = exe_epo_comb.average()
evoked_ima = ima_epo_comb.average()

# Plot evoked responses
evoked_exe.plot(title="Evoked: Execution")
evoked_exe.plot_topo(title="Evoked Topo: Execution")
evoked_ima.plot(title="Evoked: Imagery")
evoked_ima.plot_topo(title="Evoked Topo: Imagery")

# ========================== END OF SCRIPT ========================== #
