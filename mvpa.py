"""
Script: mvpa.py
Version: 1.0
Author: bohem-dev
Date: [20-03-2025]
License: MIT License (https://opensource.org/licenses/MIT)

Description:
    This script performs time-resolved decoding on MEG data using machine learning.
    It loads execution and imagery epochs, preprocesses the data, and applies classifiers
    to assess decoding accuracy over time. It also visualizes decoding performance and
    spatial patterns of activity.

Usage:
    Set the data file paths in the script. Warning: 'n_jobs=-1' uses all available CPUs & SVM is slow.

Dependencies:
    - MNE-Python
    - NumPy
    - scikit-learn
    - Matplotlib
"""

import matplotlib

# Set the Matplotlib backend
matplotlib.use("QtAgg")  # Set before importing pyplot
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
import numpy as np
import mne
from mne.decoding import (
    SlidingEstimator,
    GeneralizingEstimator,
    cross_val_multiscore,
    LinearModel,
    get_coef,
    Vectorizer,
)
import sklearn.svm
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from mne import pick_info


# ========================== Load Participant Data ========================== #

# Define participant file paths (change the participant number in the file names)
exe_epochs_path = r""  # Path to execution epochs file (.fif)
ima_epochs_path = r""  # Path to imagery epochs file (.fif)

# Load execution and imagery epochs
exe_epochs = mne.read_epochs(exe_epochs_path, preload=True, verbose=True)
ima_epochs = mne.read_epochs(ima_epochs_path, preload=True, verbose=True)

# Combine execution and imagery epochs
combined_epochs = mne.concatenate_epochs([exe_epochs, ima_epochs])


# ========================== Preprocessing ========================== #

# Select only magnetometers, apply filtering, cropping, and resampling
combined_epochs = combined_epochs.pick(picks="mag").filter(
    0, 30, n_jobs=-1, fir_design="firwin", verbose=True
)
combined_epochs = combined_epochs.resample(100)
resting_epochs = combined_epochs.copy().crop(tmin=-1.0, tmax=0.0)
combined_epochs = combined_epochs.crop(tmin=-0.1, tmax=0.5)

# Extract the data
X = combined_epochs.get_data(picks="mag")

# Prepare labels: Convert event codes to 1 (Motor Execution) and 2 (Motor Imagery)
temp = combined_epochs.events[:, 2]
temp[combined_epochs.events[:, 2] == 2] = 1  # Motor Execution
temp[combined_epochs.events[:, 2] == 3] = 2  # Motor Imagery
y = temp


# ========================== Time Decoding (Sliding Estimator) ========================== #

# Define classifier pipeline (Vectorization, Scaling, and Linear SVM)
clf = make_pipeline(
    Vectorizer(), StandardScaler(), LinearModel(sklearn.svm.SVC(kernel="linear"))
)

# Perform time decoding using a sliding estimator
time_decod = SlidingEstimator(clf, n_jobs=-1, scoring="roc_auc", verbose=True)

# Cross-validation to assess decoding accuracy over time
scores = cross_val_multiscore(time_decod, X, y, cv=5, n_jobs=-1)
scores = np.mean(scores, axis=0)

# Plot decoding accuracy over time
fig, ax = plt.subplots()
plt.ylim([0.40, 0.90])
ax.plot(combined_epochs.times, scores, label="Decoding Accuracy (AUC)")
ax.axhline(0.5, color="k", linestyle="--", label="Chance Level")
ax.set_xlabel("Time (s)")
ax.set_ylabel("AUC Score")
ax.legend()
ax.axvline(0.0, color="k", linestyle="-")
ax.set_title(f"Sensor Space Decoding Over Time")
plt.show()


# ========================== Extract & Visualize Spatial Patterns ========================== #

# Fit classifier to obtain spatial patterns
time_decod.fit(X, y)

# Get only the good channels (exclude marked bads)
good_channels = [
    ch for ch in combined_epochs.ch_names if ch not in combined_epochs.info["bads"]
]

# Create new info object with only the good channels
info = pick_info(
    combined_epochs.info,
    sel=[combined_epochs.ch_names.index(ch) for ch in good_channels],
)

# Retrieve spatial patterns from the classifier
coef = get_coef(time_decod, "patterns_", inverse_transform=True)

# Create an EvokedArray to visualize spatial patterns over time
evoked_time_gen = mne.EvokedArray(coef, info, tmin=combined_epochs.times[0])

# Define plotting parameters
joint_kwargs = dict(ts_args=dict(time_unit="s"), topomap_args=dict(time_unit="s"))

# Plot spatial patterns at selected time points
evoked_time_gen.plot_joint(
    times=np.arange(0.0, 0.500, 0.050),
    title=f"Spatial Patterns from Temporal Decoding",
    **joint_kwargs,
)


# ========================== Temporal Generalization ========================== #

# Define the temporal generalization estimator
time_gen = GeneralizingEstimator(clf, n_jobs=-1, scoring="roc_auc", verbose=True)

# Perform cross-validation for generalization across time
scores = cross_val_multiscore(time_gen, X, y, cv=3, n_jobs=-1)
scores = np.mean(scores, axis=0)  # Average across cross-validation folds

# Plot diagonal (self-generalization, same as sliding estimator)
fig, ax = plt.subplots()
ax.plot(combined_epochs.times, np.diag(scores), label="Self-Generalization Score")
ax.axhline(0.5, color="k", linestyle="--", label="Chance Level")
ax.set_xlabel("Time (s)")
ax.set_ylabel("AUC Score")
ax.legend()
ax.axvline(0.0, color="k", linestyle="-")
ax.set_title(f"Decoding Accuracy Over Time")
plt.show()

# Plot the full generalization matrix (Training Time vs. Testing Time)
fig, ax = plt.subplots(1, 1)
im = ax.imshow(
    scores,
    interpolation="lanczos",
    origin="lower",
    cmap="RdBu_r",
    extent=combined_epochs.times[[0, -1, 0, -1]],
    vmin=0.0,
    vmax=1.0,
)
ax.set_xlabel("Testing Time (s)")
ax.set_ylabel("Training Time (s)")
ax.set_title(f"Temporal Generalization Matrix")
ax.axvline(0, color="k")
ax.axhline(0, color="k")
cbar = plt.colorbar(im, ax=ax)
cbar.set_label("AUC Score")
plt.show()

# ========================== END OF SCRIPT ========================== #
