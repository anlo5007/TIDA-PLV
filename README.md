# TIDA-PLV

**TIDA-PLV** is a Python toolkit for Phase Locking Value (PLV) analysis of neuronal calcium imaging traces, developed for Tuberoinfundibular Dopaminergic (TIDA) neuron recordings.

Developed by **Andrea Locarno** — [Christian Broberger Lab](https://www.su.se/english/research/research-catalogue/research-groups/c/christian-brobergers-research-group)

---

## Overview

TIDA-PLV provides a complete pipeline for quantifying synchrony between neurons from calcium imaging data:

- Bandpass filtering and instantaneous phase extraction
- Pairwise PLV matrix computation
- Hierarchical clustering of PLV matrices
- Surrogate-based permutation testing (circular shift and phase randomisation)
- FDR correction of p-values (Benjamini-Hochberg / Benjamini-Yekutieli)
- Spatial mapping of significant phase-locking connections
- Interactive interval selection from raw traces

---

## Installation

### Requirements

- Python ≥ 3.8
- numpy
- pandas
- scipy
- matplotlib
- seaborn

### Install dependencies

```bash
pip install numpy pandas scipy matplotlib seaborn
```

### Clone the repository

```bash
git clone https://github.com/anlo5007/Phase_locking-TIDA.git
cd TIDA-PLV
```

No package installation is required — import the `functions` folder directly from your working directory or notebook.

---

## Input Data Format

TIDA-PLV expects a `.csv` file structured as follows:

| (ignored)   | Cell_1     | Cell_2     | ... |
|-------------|------------|------------|-----|
| accepted    | undecided  | rejected   | ... |
| 0.000       | 0.412      | 0.387      | ... |
| 0.100       | 0.415      | 0.391      | ... |

- **Row 0**: cell status labels (`accepted`, `undecided`, `rejected`)
- **Row 1+**: first column is time in seconds; remaining columns are fluorescence traces

A companion `-props.csv` file with columns `Name`, `CentroidX`, `CentroidY` is required for spatial mapping. A `.png` map image is optional but recommended.

---

## Usage

The recommended way to run an analysis is to open the demo notebook:

```bash
phase_locking_TIDA.ipynb
```

The notebook walks through the full pipeline step by step — from loading a recording to exporting results — and is the best starting point for new analyses. Each cell is annotated with comments explaining the parameters and expected outputs.

### Interval selection methods

| Method | Description |
|---|---|
| `'single_click'` | Click once to set interval start; duration set by `window` (seconds) |
| `'double_click'` | Click twice to define start and end |
| `'manual'` | Pass `start_time` and `end_time` directly |
| `'block'` | Click within a segment of a concatenated recording |

---

## API Reference

### `Recording` class (`io.py`)

The recommended entry point for loading data.

| Method / Attribute | Description |
|---|---|
| `Recording.from_dialog(keep_status)` | Load a recording via file dialog |
| `Recording(path, keep_status)` | Load from a known path |
| `.load()` | Load traces, props and map from disk |
| `.select_interval(method, **kwargs)` | Interactively trim the recording |
| `.data` | DataFrame of fluorescence traces |
| `.time` | Time axis in seconds |
| `.sampling_rate` | Sampling rate in Hz |
| `.cell_names` | Cell column labels |
| `.props` | Cell properties DataFrame |
| `.map_img` | Cell map image array |

### Signal processing (`utils.py`)

| Function | Description |
|---|---|
| `bandpass(signal, fs, fmin, fmax)` | Zero-phase Butterworth bandpass filter |
| `get_phase(signal)` | Instantaneous phase via Hilbert transform |
| `poly_detrend(signal, order)` | Polynomial baseline removal |
| `dominant_frequency(signal, fs)` | FFT-based dominant frequency extraction |

### PLV computation (`utils.py`)

| Function | Description |
|---|---|
| `plv_einsum(phases)` | Fast pairwise PLV matrix and mean phase differences |

### Surrogate generation (`utils.py`)

| Function | Description |
|---|---|
| `circular_shift_surrogate(signals, min_interval, max_interval)` | Circular-shift null dataset |
| `phase_randomisation_surrogate(signals)` | FFT phase-randomisation null dataset |

### Significance testing (`utils.py`)

| Function | Description |
|---|---|
| `permutation_test(signals, plv, phase_diff, surrogate_fn, n_permutations)` | Surrogate permutation test |
| `correct_p_values(p_values, FDR_correction, cell_labels, method)` | FDR correction (BH or BY) |

### Spatial analysis (`utils.py`)

| Function | Description |
|---|---|
| `compute_pairwise_distances(props, pair_labels)` | Euclidean distances between cell centroids |

### Plotting (`plots.py`)

| Function | Description |
|---|---|
| `plot_traces(time, data, status)` | Stacked trace plot, optionally colour-coded by status |
| `plot_plv_matrix(plv_matrix, p_val_matrix)` | PLV heatmap with significance annotations |
| `plot_plv_map(props, plv_vector, p_val_vector, map)` | Spatial map of phase-locking connections |
| `plot_surrogate_vs_empirical(empirical, surrogate)` | Surrogate null distributions vs empirical values |

---

## Output

Running the full pipeline produces a `results_PLV.csv` file and a `Results_PLV_figures.pdf` saved in the same folder as the input recording. The CSV contains one row per cell pair with the following columns:

| Column | Description |
|---|---|
| `PLV Value` | Pairwise PLV |
| `p_value` | BH-corrected p-value |
| `p_value_uncorrected` | Uncorrected p-value |
| `Phase Diff (radians)` | Mean phase difference |
| `distance_px` | Euclidean distance between cell centroids (pixels) |

---


## License

This project is licensed under the **GNU General Public License v3.0**. See [LICENSE](LICENSE) for details.
