"""
Script rica.py
Version: 1.0
Author: Philip Press
Date: [20-03-2025]
License: MIT License (https://opensource.org/licenses/MIT)

Description:
    ICA-based artifact rejection, manual annotation, event detection,
    and epoching for MEG data using MNE-Python.

Usage:
    Set your file paths accordingly before running.
    Press 'a' during raw plot for annotation mode.

Dependencies:
    - MNE-Python
    - NumPy
    - Matplotlib
    - SciPy
"""

import os
import numpy as np
import mne
import matplotlib

matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
from mne.preprocessing import ICA

# ========================== Load Preprocessed Data ========================== #

# Load filtered raw data for ICA
raw_bandpass = mne.io.read_raw_fif(
    r"/path/to/preprocessed/nbf-sub-XX-s0.fif",  # <-- Replace with your file path
    preload=True,
)

# ========================== ICA for Artifact Rejection ========================== #

# Fit ICA model
ica = ICA(n_components=20, method="fastica", random_state=23).fit(raw_bandpass)

# Plot ICA sources and components
ica.plot_sources(raw_bandpass, title="ICA Sources")
ica.plot_components()

# Reject selected ICA components (eye & heart artifacts)
ica.exclude = [0, 1, 2, 3, 4, 8, 14, 16]  # <-- Update based on visual inspection

# Apply ICA and visualize before vs. after
raw_ica = ica.apply(raw_bandpass.copy())
raw_bandpass.plot(duration=10, title="Before ICA").show()
raw_ica.plot(duration=10, title="After ICA").show()

# Save cleaned data after ICA
raw_ica.load_data()
raw_ica.save(
    r"/path/to/save/rica-sub-XX-s0.fif",  # <-- Replace with save location
    overwrite=True,
)

# ========================== Manual Annotation ========================== #

# Plot and manually annotate muscle artifacts
raw_clean = raw_ica.copy()
raw_clean.plot(duration=10, title="Annotate: Press 'a' for annotation mode").show()

# After annotating, you can access the annotations here:
interactive_annot = raw_clean.annotations

# Save annotations and cleaned raw data
folder_name = r"/path/to/save/"  # <-- Replace with save directory
raw_clean.annotations.save(os.path.join(folder_name, "annotations_muscle.csv"), overwrite=True)
raw_clean.load_data()
raw_clean.save(os.path.join(folder_name, "raw_clean.fif"), overwrite=True)

# ========================== Event Detection ========================== #

# Reload data for event detection
raw = mne.io.read_raw_fif(os.path.join(folder_name, "raw_clean.fif"), preload=True)

# Pick stimulation channels
stim_raw = raw.copy().pick(picks=["ai122", "ai123"])  # <-- Check actual stim channels in raw.ch_names
stim_raw.plot()

# Detect events (onsets) from analog channel
event_cross = mne.find_events(
    stim_raw, stim_channel="ai122", min_duration=0.02, output="onset"
)

# Separate execution and imagery events
event_exe = mne.pick_events(event_cross, exclude=3)  # Event ID 2: Execution
event_ima = mne.pick_events(event_cross, exclude=2)  # Event ID 3: Imagery

# Define event dictionary
event_dict = {"motor/execution": 2, "motor/imagery": 3}

# Plot events and raw data
mne.viz.plot_events(
    event_cross,
    sfreq=raw.info["sfreq"],
    first_samp=raw.first_samp,
    event_id=event_dict,
)
raw.plot(
    events=event_cross,
    start=5,
    duration=20,
    color="gray",
    event_color={2: "y", 3: "r"},
)

# ========================== Epoching ========================== #

# Epoch around execution events
epochs_exe = mne.Epochs(
    raw,
    event_exe,
    tmin=-3,
    tmax=3,
    reject_tmax=0,
    reject_by_annotation=True,
    preload=True,
)

# Epoch around imagery events
epochs_ima = mne.Epochs(
    raw,
    event_ima,
    tmin=-3,
    tmax=3,
    reject_tmax=0,
    reject_by_annotation=True,
    preload=True,
)

# Visualize some example epochs
epochs_exe.plot(n_epochs=10, events=True)
epochs_ima.plot(n_epochs=10, events=True)

# Save epochs
epochs_exe.save(os.path.join(folder_name, "Exe_epoch.fif"), overwrite=True)
epochs_ima.save(os.path.join(folder_name, "Ima_epoch.fif"), overwrite=True)

# ========================== END OF SCRIPT ========================== #
