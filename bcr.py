"""
Script: bcr.py
Version: 1.0
Author: Philip Press
Date: [20-03-2025]
License: MIT License (https://opensource.org/licenses/MIT)

Description:
    Preprocessing script for MEG data.
    This includes bad channel marking, notch filtering, and bandpass filtering.

Usage:
    Set the appropriate file paths before running the script.
    Warning: 'n_jobs=-1' utilizes all available CPU cores.

Dependencies:
    - MNE-Python
    - NumPy
    - Matplotlib
"""

import os
import numpy as np
import mne
import matplotlib

# Set matplotlib backend before importing pyplot
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt

# Optional: Import ICA if planning artifact removal
from mne.preprocessing import ICA

# ========================== Load Raw Data ========================== #

# Load raw MEG data file (update path as needed)
raw = mne.io.read_raw_fif(
    r"/path/to/your/data/sub-XX/b_raw.fif",  # <--- Replace with your data path
    preload=True,
)

# Make a backup copy of the raw data
orig_raw = raw.copy()

# Print basic info
print(raw)
print(raw.info)

# ========================== Initial Inspection ========================== #

# Power Spectral Density plot to inspect noise profile
raw.compute_psd(fmax=300).plot(picks="meg")

# Sensor location plot
raw.plot_sensors()

# Visual inspection of raw signal time courses
raw.plot(duration=10, title="Raw").show()

# ========================== Mark Bad Channels ========================== #

# Manually add known bad channels (example list – adapt per subject)
raw.info["bads"] += [
    "CH001", "CH002", "CH003",  # Replace with actual bad channel names
]

# Save the raw data with bad channels marked
raw.load_data()
save_dir = r"/path/to/save/preprocessed_data/"
save_filename = input("Enter the filename to save (bcr): ")
raw.save(os.path.join(save_dir, save_filename), overwrite=True)

# ========================== Preprocessing ========================== #

# Keep only magnetometer data and exclude bad channels
raw_good = raw.copy().pick(picks="mag", exclude="bads").load_data()

# Re-check PSD after bad channel removal
raw.compute_psd(fmax=300).plot(picks="meg")
raw_good.compute_psd(fmax=300).plot(picks="meg")

# Apply notch filter to remove line noise and harmonics
raw_notch = raw_good.copy().notch_filter(
    freqs=[50, 100, 120, 150, 188], notch_widths=2
)

# Apply bandpass filter to retain relevant frequency components
raw_bandpass = raw_notch.copy().filter(l_freq=1, h_freq=100)

# ========================== Final Inspection ========================== #

# Inspect PSD after filtering
raw_notch.compute_psd(fmax=300).plot(picks="meg")
raw_bandpass.compute_psd(fmax=300).plot(picks="meg")

# Plot filtered data
raw_notch.plot(duration=10, title="Post-Notch Filter").show()
raw_bandpass.plot(duration=10, title="Post-Bandpass Filter").show()

# Save preprocessed data
raw_bandpass.load_data()
save_filename = input("Enter the filename to save (nbf): ")
raw_bandpass.save(os.path.join(save_dir, save_filename), overwrite=True)

# ========================== END OF SCRIPT ========================== #
