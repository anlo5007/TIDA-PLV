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

utils
    Signal processing, PLV computation, surrogate generation and
    permutation-based significance testing.

Quick Start
-----------
Typical usage in a Jupyter notebook::

    import functions as plv

    # 1. Load data
    recording_path          = plv.select_file()
    time, data, labels, fs  = plv.load_recording(recording_path)
    props                   = plv.load_props(plv.locate_props(recording_path))
    map_img                 = plv.load_map(plv.locate_map(recording_path))

    # 1. Select interval (optional)
    time, data = plv.select_interval_manual(time, data, start_time=10, end_time=60, status=labels)
    # or
    time, data = plv.select_click_interval(time, data, status=labels, window=50)
    # or
    time, data = plv.select_double_click(time, data, status=labels)

    # 3. Preprocess
    filtered = [plv.bandpass(data[col].values, fs, 0.01, 1.0)
                for col in data.columns]
    phases   = np.array([plv.get_phase(sig) for sig in filtered])

    # 4. PLV
    PLV = plv.plv_einsum(phases)

    # 5. Permutation test
    from functools import partial
    surrogate_fn = partial(plv.circular_shift_surrogate,
                           min_interval=100, max_interval=500)
    surr_plv, p_vals, surr_phases = plv.permutation_test(
        np.array(filtered), PLV, surrogate_fn
    )

See Also
--------
utils.py, io.py
"""

# ---------------------------------------------------------------------------
# IO — file loading and selection
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
)

# ---------------------------------------------------------------------------
# Plotting and visualization
# ---------------------------------------------------------------------------
from .plots import (
    plot_traces,
)

# ---------------------------------------------------------------------------
# Finding time discountinuities
# ---------------------------------------------------------------------------
from .utils import (
    detect_time_jumps,
)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
__all__ = [
    # IO
    'select_file',
    'load_recording',
    'locate_props',
    'locate_map',
    'load_props',
    'load_map',
    'select_interval_manual',
    'select_click_interval',
    'select_double_click',
    
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
    
    # Plotting
    'plot_traces',

    # Time discontinuities
    'detect_time_jumps',
]