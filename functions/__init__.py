"""
TIDA PLV Analysis Package
=========================
Tools for Phase Locking Value (PLV) analysis of neuronal calcium imaging
traces, developed for TIDA neuron recordings.

Submodules
----------
io
    Loading calcium imaging recordings and associated metadata files.
    File selection dialogs, CSV parsing, props and map image loading.
    The ``Recording`` class is the recommended high-level interface.

utils
    Signal processing, PLV computation, surrogate generation and
    permutation-based significance testing.

plots
    Visualization utilities for traces, PLV matrices, spatial maps,
    and surrogate distributions.

Quick Start
-----------
Typical usage in a Jupyter notebook::

    import functions as plv

    # 1. Load data (high-level interface)
    recording = plv.Recording.from_dialog(keep_status=('accepted', 'undecided'))

    # 2. Select interval
    recording.select_interval(method='single_click', window=300)

    # 3. Preprocess
    filtered = recording.data.apply(lambda x: plv.bandpass(x, recording.sampling_rate, 0.01, 1.0))
    phases   = filtered.apply(plv.get_phase)

    # 4. PLV
    PLV, phase_diff = plv.plv_einsum(phases.T.to_numpy())

    # 5. Permutation test
    surr_plv, p_vals, surr_phase, _ = plv.permutation_test(
        filtered.T.to_numpy(), PLV, phase_diff,
        surrogate_fn=plv.circular_shift_surrogate,
        n_permutations=1000,
        min_interval=100, max_interval=500
    )

    # 6. Correct p-values
    p_matrix, p_pairs = plv.correct_p_values(p_vals, cell_labels=recording.cell_names)

See Also
--------
utils.py, io.py, plots.py
"""

# ---------------------------------------------------------------------------
# IO — high-level Recording class
# ---------------------------------------------------------------------------
from .io import (
    Recording,
)

# ---------------------------------------------------------------------------
# IO — low-level file loading and selection
# ---------------------------------------------------------------------------
from .io import (
    select_file,
    load_recording,
    locate_props,
    locate_map,
    load_props,
    load_map,
    select_interval_manual,
    select_click_interval,
    select_double_click,
    select_block,
    save_open_figs,
)

# ---------------------------------------------------------------------------
# Signal conditioning
# ---------------------------------------------------------------------------
from .utils import (
    bandpass,
    get_phase,
    poly_detrend,
)

# ---------------------------------------------------------------------------
# Spectral analysis
# ---------------------------------------------------------------------------
from .utils import (
    dominant_frequency,
)

# ---------------------------------------------------------------------------
# Phase Locking Value
# ---------------------------------------------------------------------------
from .utils import (
    plv_einsum,
)

# ---------------------------------------------------------------------------
# Surrogate generators
# ---------------------------------------------------------------------------
from .utils import (
    circular_shift_surrogate,
    phase_randomisation_surrogate,
)

# ---------------------------------------------------------------------------
# Significance testing
# ---------------------------------------------------------------------------
from .utils import (
    permutation_test,
    correct_p_values,
)

# ---------------------------------------------------------------------------
# Spatial analysis
# ---------------------------------------------------------------------------
from .utils import (
    compute_pairwise_distances,
)

# ---------------------------------------------------------------------------
# Time discontinuities
# ---------------------------------------------------------------------------
from .utils import (
    detect_time_jumps,
)

# ---------------------------------------------------------------------------
# Plotting and visualization
# ---------------------------------------------------------------------------
from .plots import (
    plot_traces,
    plot_plv_matrix,
    plot_plv_map,
    plot_surrogate_vs_empirical,
)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
__all__ = [
    # IO — high-level
    'Recording',

    # IO — low-level
    'select_file',
    'load_recording',
    'locate_props',
    'locate_map',
    'load_props',
    'load_map',
    'select_interval_manual',
    'select_click_interval',
    'select_double_click',
    'select_block',
    'save_open_figs',

    # Signal conditioning
    'bandpass',
    'get_phase',
    'poly_detrend',

    # Spectral
    'dominant_frequency',

    # PLV
    'plv_einsum',

    # Surrogates
    'circular_shift_surrogate',
    'phase_randomisation_surrogate',

    # Significance
    'permutation_test',
    'correct_p_values',

    # Spatial
    'compute_pairwise_distances',

    # Time discontinuities
    'detect_time_jumps',

    # Plotting
    'plot_traces',
    'plot_plv_matrix',
    'plot_plv_map',
    'plot_surrogate_vs_empirical',
]