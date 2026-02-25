# TIDA PLV Project

A Python pipeline for Phase Locking Value (PLV) analysis of calcium imaging recording traces, developed for the study of TIDA (Tuberoinfundibular Dopaminergic) neuron synchronisation.

## What it does

The pipeline takes raw fluorescence trace CSV files exported from Inscopix Data Processing software and computes pairwise PLV between all recorded cells. PLV measures the consistency of the phase difference between two signals over time — a value of 1 indicates perfect phase locking, 0 indicates no locking.

The main steps are:

1. Load a recording and interactively select a time interval of interest
2. Bandpass filter the traces and extract instantaneous phase via the Hilbert transform
3. Compute a pairwise PLV matrix for all cell pairs
4. Test significance against a surrogate null distribution (circular shift or phase randomisation)
5. Correct for multiple comparisons using the Benjamini-Hochberg procedure
6. Visualise significant connections overlaid on the cell map

## Input file format

### Traces CSV

Row 0 is a header containing cell status labels (` accepted`, ` undecided`, ` rejected`). Rows 1+ contain time in seconds in the first column and fluorescence values in the remaining columns.

### Props CSV

Expected at `<recording_stem>-props.csv` in the same folder as the recording. Must contain at least `CentroidX` and `CentroidY` columns for spatial analysis.

### Map image

A PNG file in the same folder as the recording, used as background for spatial connectivity plots.


## Statistical notes

Two surrogate methods are available. `circular_shift_surrogate` is faster but may accidentally preserve phase relationships if the shift is close to a multiple of the dominant oscillation period. `phase_randomisation_surrogate` randomises FFT phases independently for each frequency bin and is the recommended null for slow oscillations such as TIDA neurons (~0.05–0.1 Hz).



## Author

Andrea Locarno