# opm-mvpa
---

Set of scripts for MVPA analysis.

Pipeline - 
  1. BCR & Filtering (bcr.py) - Selecting good channels, bandpass and notch filtering.
  2. ICA, Muscle Annot. & Epoching (rica.py) - Independent Component Analysis, muscle artefact annotation (manual) & epoching
  3. TFA (tfa.py) - Time-frequency analysis (final check for noisy sensors)
  4. MVPA (mvpa.py) - Multivariate pattern analysis
