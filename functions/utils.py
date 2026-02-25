
"""
utils.py
========
Signal processing utilities for Phase Locking Value (PLV) analysis
of neuronal calcium imaging traces.

This module provides a complete pipeline from raw signal conditioning
through phase extraction, PLV computation, and surrogate-based
significance testing.

Routine Listings
----------------
Signal conditioning
    bandpass                      : Zero-phase Butterworth bandpass filter.
    get_phase                     : Instantaneous phase via Hilbert transform.
    poly_detrend                  : Polynomial baseline removal.

Spectral analysis
    dominant_frequency            : FFT-based dominant frequency extraction.

Phase Locking Value
    plv_einsum                    : Fast pairwise PLV matrix (einsum).

Surrogate generators
    circular_shift_surrogate      : Circular-shift null dataset.
    phase_randomisation_surrogate : FFT phase-randomisation null dataset.

Significance testing
    permutation_test              : Method-agnostic surrogate permutation test.

Notes
-----
Recommended pipeline order::

    raw signal
        → bandpass
        → get_phase
        → plv_einsum
        → permutation_test

See Also
--------
scipy.signal.hilbert, scipy.signal.butter
"""

from scipy.signal import hilbert, butter, filtfilt
import numpy as np
from matplotlib import pyplot as plt


# =============================================================================
# Signal conditioning
# =============================================================================

def bandpass(signal, fs, fmin, fmax, order=3):
    """
    Apply a zero-phase Butterworth bandpass filter.

    Uses ``scipy.signal.filtfilt`` for forward-backward filtering, which
    produces zero phase distortion and doubles the effective filter order.

    Parameters
    ----------
    signal : array_like, shape (n_timepoints,)
        Input signal to filter.
    fs : float
        Sampling frequency in Hz.
    fmin : float
        Lower cutoff frequency in Hz.
    fmax : float
        Upper cutoff frequency in Hz.
    order : int, optional
        Filter order. Default is 3. The effective order after forward-backward
        filtering is ``2 * order``.

    Returns
    -------
    filtered : ndarray, shape (n_timepoints,)
        Bandpass-filtered signal, same length as input.

    Notes
    -----
    For calcium imaging data with slow oscillations (0.01-1 Hz), order=3
    provides a good balance between roll-off steepness and ringing artefacts.

    Examples
    --------
    >>> filtered = bandpass(raw_signal, fs=10.0, fmin=0.01, fmax=1.0)
    """
    nyq = fs / 2
    b, a = butter(order, [fmin / nyq, fmax / nyq], btype='band')
    return filtfilt(b, a, signal)


def get_phase(signal):
    """
    Compute the instantaneous phase of a signal via the Hilbert transform.

    The analytic signal is constructed as ``x(t) + i*H{x(t)}``, where
    ``H{}`` is the Hilbert transform. The instantaneous phase is the angle
    of this complex signal at each timepoint.

    Parameters
    ----------
    signal : array_like, shape (n_timepoints,)
        Input signal. Should be bandpass-filtered before calling this
        function to ensure the analytic signal is well-defined.

    Returns
    -------
    phase : ndarray, shape (n_timepoints,)
        Instantaneous phase in radians, wrapped to the range ``[-pi, pi]``.

    Notes
    -----
    The Hilbert transform assumes the input is a narrowband analytic signal.
    Broadband or non-stationary signals will produce unreliable phase estimates.
    Always bandpass filter first.

    Do **not** apply polynomial detrending to the extracted phase — detrending
    should be applied to the raw signal before this step.

    Examples
    --------
    >>> phase = get_phase(bandpass(signal, fs=10.0, fmin=0.01, fmax=1.0))
    """
    analytic_signal = hilbert(signal)
    instantaneous_phase = np.angle(analytic_signal)
    return instantaneous_phase


def poly_detrend(signal, order):
    """
    Remove a polynomial trend from a signal.

    Fits a polynomial of the given order to the signal as a function of
    sample index, then subtracts it. Useful for removing slow baseline
    drifts before spectral analysis.

    Parameters
    ----------
    signal : array_like, shape (n_timepoints,)
        Input signal with trend to remove.
    order : int
        Polynomial order. Use 1 for linear detrending, 2 for quadratic, etc.
        Higher orders remove more complex drifts but risk distorting the
        signal of interest if set too high.

    Returns
    -------
    detrended : ndarray, shape (n_timepoints,)
        Signal with polynomial trend subtracted, same length as input.

    Notes
    -----
    This function is used internally by ``dominant_frequency`` to reduce
    spectral leakage before FFT computation. It is **not** needed before
    ``get_phase`` — bandpass filtering already removes slow drifts in that
    context.

    Examples
    --------
    >>> detrended = poly_detrend(signal, order=4)
    """
    t = np.arange(len(signal))
    p = np.polyfit(t, signal, order)
    return signal - np.polyval(p, t)


# =============================================================================
# Spectral analysis
# =============================================================================

def dominant_frequency(signal, fs, hann=True, poly_order=4, plot=False):
    """
    Find the dominant (peak power) frequency of a signal via FFT.

    Preprocessing is applied in the following order before computing the
    spectrum:

    1. DC removal (subtract mean)
    2. Optional Hann windowing (reduces spectral leakage)
    3. Optional polynomial detrending (removes residual slow trends)

    Parameters
    ----------
    signal : array_like, shape (n_timepoints,)
        Input signal, typically a bandpass-filtered calcium trace.
    fs : float
        Sampling frequency in Hz.
    hann : bool, optional
        If True, apply a Hann window before computing the FFT. Reduces
        spectral leakage at the cost of slightly broadened peaks.
        Default is True.
    poly_order : int or None, optional
        Polynomial order for detrending. Set to None to skip detrending.
        Default is 4.
    plot : bool, optional
        If True, plot the power spectrum on the current axes. The caller
        is responsible for creating the figure beforehand.
        Default is False.

    Returns
    -------
    max_freq : float
        Dominant frequency in Hz (frequency bin with highest power,
        excluding DC at 0 Hz).
    freqs : ndarray, shape (n_freqs,)
        Frequency axis in Hz corresponding to each power value.
    power : ndarray, shape (n_freqs,)
        Power spectrum (squared magnitude of FFT coefficients).
    processed_signal : ndarray, shape (n_timepoints,)
        Signal after all preprocessing steps (DC removal, windowing,
        detrending). Useful for sanity checking preprocessing effects.

    Notes
    -----
    The DC bin (0 Hz) is always excluded from peak detection regardless
    of windowing or detrending settings.

    Plotting is the caller's responsibility::

        fig, ax = plt.subplots()
        freq, freqs, power, _ = dominant_frequency(sig, fs, plot=True)
        ax.set_xlim(0, 1)

    Examples
    --------
    >>> freq, freqs, power, _ = dominant_frequency(signal, fs=10.0)
    >>> print(f"Dominant frequency: {freq:.4f} Hz, period: {1/freq:.1f} s")
    """
    # Remove DC offset
    signal = signal - np.mean(signal)

    # Hann window
    if hann:
        window = np.hanning(len(signal))
        signal = signal * window

    # Optional polynomial detrending
    if poly_order is not None:
        signal = poly_detrend(signal, order=poly_order)

    # FFT
    fft_vals = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(len(signal), d=1 / fs)

    # Power spectrum
    power = np.abs(fft_vals) ** 2

    # Ignore DC bin (index 0)
    idx = np.argmax(power[1:]) + 1

    if plot:
        plt.plot(freqs, power, alpha=0.7)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power')
        plt.xlim(0, 1)

    max_freq = freqs[idx]
    return max_freq, freqs, power, signal


# =============================================================================
# Phase Locking Value
# =============================================================================

def plv_einsum(phases):
    """
    Compute the full pairwise PLV matrix using einsum (vectorised).

    The Phase Locking Value between two signals i and j is defined as::

        PLV(i, j) = | (1/T) * sum_t( exp(i * (phi_i(t) - phi_j(t))) ) |

    where ``phi_i(t)`` is the instantaneous phase of cell i at time t,
    and T is the number of timepoints. PLV measures the **consistency**
    of the phase difference over time, not its absolute value.

    Parameters
    ----------
    phases : ndarray, shape (n_cells, n_timepoints)
        Instantaneous phase in radians for each cell at each timepoint.
        Obtain via ``get_phase`` after bandpass filtering.

    Returns
    -------
    PLV : ndarray, shape (n_cells, n_cells)
        Symmetric PLV matrix. Diagonal entries equal 1 (each cell is
        perfectly locked with itself). Off-diagonal entries are in [0, 1],
        where 1 indicates perfect phase locking and 0 indicates no locking.

    Notes
    -----
    PLV = 1 does **not** imply the two cells are in phase (phase difference
    equal to 0). It only means the phase difference is **constant** over time.
    Two cells locked at 180° (antiphase) also produce PLV = 1. Use the mean
    phase difference separately to determine the nature of the coupling.

    The einsum notation ``'it,jt->ij'`` computes the inner product over the
    time axis t for every pair (i, j) simultaneously, avoiding explicit loops
    over cell pairs.

    Examples
    --------
    >>> phases = np.array([get_phase(sig) for sig in filtered_signals])
    >>> PLV = plv_einsum(phases)
    """
    # Convert to complex exponentials
    exp_phases = np.exp(1j * phases)  # (n_cells, n_timepoints)

    # 'it,jt->ij': for each pair (i,j), multiply across time and sum
    PLV = np.abs(
        np.einsum('it,jt->ij', exp_phases, np.conj(exp_phases)) / phases.shape[1]
    )
    return PLV


# =============================================================================
# Surrogate generators
# =============================================================================

def circular_shift_surrogate(signals, min_interval, max_interval):
    """
    Generate one surrogate dataset by circularly shifting each cell's signal.

    Each cell is independently shifted by a random lag drawn uniformly from
    ``[min_interval, max_interval)``. This breaks cross-cell phase
    relationships while preserving each cell's autocorrelation structure
    and power spectrum.

    Parameters
    ----------
    signals : ndarray, shape (n_cells, n_timepoints)
        Bandpass-filtered signals (real-valued).
    min_interval : int
        Minimum shift in samples. Should be at least one full oscillation
        cycle (i.e. ``int(fs / dominant_freq)``) to ensure the phase
        relationship is genuinely disrupted.
    max_interval : int
        Maximum shift in samples. Typically set to
        ``len(signal) - min_interval``.

    Returns
    -------
    surrogate_signals : ndarray, shape (n_cells, n_timepoints)
        Circularly shifted surrogate signals, same shape as input.

    Notes
    -----
    Shifting the signal is equivalent to shifting the extracted phase because
    the Hilbert transform is linear and time-invariant. Operating in signal
    space keeps both surrogate generators consistent — both accept signals
    as input and return signals as output, so ``permutation_test`` remains
    agnostic to the method.

    **Limitation**: if the shift amount happens to be close to a multiple of
    the dominant oscillation period, the phase relationship may be partially
    preserved by chance. This is the main motivation for preferring
    ``phase_randomisation_surrogate`` for slow oscillations such as TIDA
    neurons (~0.05-0.1 Hz).

    See Also
    --------
    phase_randomisation_surrogate : Stronger null for slow oscillations.

    Examples
    --------
    >>> from functools import partial
    >>> surrogate_fn = partial(circular_shift_surrogate,
    ...                        min_interval=100, max_interval=500)
    >>> surrogate = surrogate_fn(signals)
    """
    surrogate = np.empty_like(signals)
    for i in range(signals.shape[0]):
        shift = np.random.randint(min_interval, max_interval)
        surrogate[i] = np.roll(signals[i], shift)
    return surrogate


def phase_randomisation_surrogate(signals):
    """
    Generate one surrogate dataset by randomising FFT phases.

    For each cell, the FFT is computed and the phase of every frequency
    bin is replaced by an independent uniform random value in ``[0, 2*pi)``,
    while the amplitude spectrum is left exactly unchanged. The surrogate
    signal is recovered by inverse FFT.

    This produces surrogates that are statistically indistinguishable from
    the originals in terms of power spectrum and autocorrelation structure,
    but with all cross-cell phase relationships completely destroyed.

    Parameters
    ----------
    signals : ndarray, shape (n_cells, n_timepoints)
        Bandpass-filtered signals (real-valued). Each row is one cell.

    Returns
    -------
    surrogate_signals : ndarray, shape (n_cells, n_timepoints)
        Phase-randomised surrogate signals, real-valued, same shape as input.

    Notes
    -----
    Two frequency bins must remain real-valued to preserve the conjugate
    symmetry required for a real-valued output signal:

    - **DC bin** (index 0, 0 Hz): always fixed at phase = 0.
    - **Nyquist bin** (index -1): fixed at phase = 0 for even-length signals
      only.

    All other bins are independently randomised. Conjugate symmetry of the
    upper half of the spectrum is enforced automatically by
    ``numpy.fft.irfft``.

    **Advantage over circular shift**: because every frequency bin is
    independently randomised, no shift amount can accidentally preserve the
    phase relationship at the dominant frequency. This makes it a stronger
    and more principled null hypothesis, particularly for slow oscillations
    where the oscillation period is a significant fraction of the recording
    length.

    **Assumption**: the signal is approximately stationary. Strong
    non-stationarities (bursts, drifts) may cause surrogate amplitudes to
    differ visually from the original despite identical power spectra. Use
    the ``plot_surrogates`` option of ``permutation_test`` to visually verify
    plausibility before trusting the p-values.

    See Also
    --------
    circular_shift_surrogate : Simpler alternative surrogate method.

    Examples
    --------
    >>> surrogate = phase_randomisation_surrogate(signals)
    """
    n_cells, n_timepoints = signals.shape
    surrogate = np.empty_like(signals, dtype=float)

    for i in range(n_cells):
        fft_vals   = np.fft.rfft(signals[i])
        amplitudes = np.abs(fft_vals)

        random_phases    = np.random.uniform(0, 2 * np.pi, size=len(fft_vals))
        random_phases[0] = 0  # DC bin must stay real
        if n_timepoints % 2 == 0:
            random_phases[-1] = 0  # Nyquist bin must stay real (even-length only)

        surrogate[i] = np.fft.irfft(
            amplitudes * np.exp(1j * random_phases), n=n_timepoints
        )
    return surrogate


# =============================================================================
# Significance testing
# =============================================================================

def permutation_test(
    signals,
    empirical_plv,
    surrogate_fn,
    n_permutations=1000,
    plot_surrogates=0,
):
    """
    Surrogate permutation test for PLV significance.

    For each permutation, a surrogate dataset is generated by ``surrogate_fn``,
    phases are extracted internally, and a surrogate PLV matrix is computed.
    The empirical PLV is compared against this null distribution to produce
    p-values.

    This function is agnostic to the surrogate method — any callable that
    accepts signals and returns signals of the same shape can be passed.

    Parameters
    ----------
    signals : ndarray, shape (n_cells, n_timepoints)
        Bandpass-filtered signals (real-valued). Phases are extracted
        internally so that both surrogate methods operate consistently
        in signal space.
    empirical_plv : ndarray, shape (n_cells, n_cells)
        Observed PLV matrix computed from the real data via ``plv_einsum``.
    surrogate_fn : callable
        Function with signature ``surrogate_fn(signals) -> surrogate_signals``
        where ``surrogate_signals`` has the same shape as ``signals``.
        Use ``circular_shift_surrogate`` or ``phase_randomisation_surrogate``,
        wrapping with ``functools.partial`` if extra arguments are needed::

            from functools import partial
            surrogate_fn = partial(circular_shift_surrogate,
                                   min_interval=100, max_interval=500)

        For phase randomisation (no extra arguments needed)::

            surrogate_fn = phase_randomisation_surrogate

    n_permutations : int, optional
        Number of surrogate datasets to generate. Default is 1000.
        Higher values give more precise p-values but increase runtime linearly.
    plot_surrogates : int, optional
        Number of surrogate datasets to plot for visual sanity checking.
        The empirical signals are plotted first in red, followed by the
        first ``plot_surrogates`` surrogates in blue, one figure each.
        Set to 0 to disable all plotting. Default is 0.

    Returns
    -------
    surrogate_plv_matrix : ndarray, shape (n_permutations, n_cells, n_cells)
        PLV matrix computed for each surrogate dataset. Can be used to
        inspect the null distribution directly.
    p_values : ndarray, shape (n_cells, n_cells)
        Proportion of surrogates with PLV >= empirical PLV, with continuity
        correction: ``p = (count + 1) / (n_permutations + 1)``.
        Values are symmetric (``p[i,j] == p[j,i]``). These are uncorrected —
        apply FDR correction to the upper triangle before thresholding.
    surrogate_phases_all : ndarray, shape (n_permutations, n_cells, n_timepoints)
        Instantaneous phase extracted from each surrogate dataset. Stored for
        downstream reuse (e.g. phase consistency analysis) without recomputing.

    Notes
    -----
    **FDR correction**: the returned p-values are uncorrected. Apply
    Benjamini-Hochberg correction to the upper triangle only to avoid
    inflating the number of tests due to the symmetric matrix::

        from scipy import stats
        n = p_values.shape[0]
        upper = np.triu_indices(n, k=1)
        p_corrected = stats.false_discovery_control(p_values[upper], method='bh')

    **Random seed**: set ``np.random.seed(seed)`` before calling this function
    to ensure reproducible results across runs.

    **Runtime**: scales as O(n_permutations x n_cells^2 x n_timepoints).
    ``phase_randomisation_surrogate`` is slightly slower per permutation than
    ``circular_shift_surrogate`` due to FFT overhead on each cell.

    See Also
    --------
    circular_shift_surrogate : Circular-shift surrogate generator.
    phase_randomisation_surrogate : FFT phase-randomisation surrogate generator.
    plv_einsum : PLV matrix computation.

    Examples
    --------
    >>> from functools import partial
    >>> surrogate_fn = partial(circular_shift_surrogate,
    ...                        min_interval=100, max_interval=500)
    >>> surr_plv, p_vals, surr_phases = permutation_test(
    ...     signals, PLV, surrogate_fn, n_permutations=1000, plot_surrogates=3
    ... )
    """
    surrogate_plv_list    = []
    surrogate_phases_list = []

    # Plot empirical signals once before any surrogates
    if plot_surrogates > 0:
        n_cells = len(signals)
        fig, axes = plt.subplots(n_cells, 1,
                                 figsize=(12, 2 * n_cells),
                                 sharex=True)
        axes = np.atleast_1d(axes)
        for ax, sig in zip(axes, signals):
            ax.plot(sig, color='red', linewidth=0.8)
        axes[0].set_title('Empirical signals', fontweight='bold')
        axes[-1].set_xlabel('Samples')
        plt.tight_layout()
        plt.show()

    for perm_idx in range(n_permutations):
        surrogate_signals = surrogate_fn(signals)
        surrogate_phases  = np.array([get_phase(s) for s in surrogate_signals])

        # Plot only the first `plot_surrogates` surrogates
        if perm_idx < plot_surrogates:
            n_cells = len(surrogate_signals)
            fig, axes = plt.subplots(n_cells, 1,
                                     figsize=(12, 2 * n_cells),
                                     sharex=True)
            axes = np.atleast_1d(axes)
            for ax, sig in zip(axes, surrogate_signals):
                ax.plot(sig, linewidth=0.8)
            axes[0].set_title(f'Surrogate #{perm_idx + 1}', fontweight='bold')
            axes[-1].set_xlabel('Samples')
            plt.tight_layout()
            plt.show()

        surrogate_phases_list.append(surrogate_phases)
        surrogate_plv_list.append(plv_einsum(surrogate_phases))

    surrogate_plv_matrix = np.array(surrogate_plv_list)    # (n_perms, n_cells, n_cells)
    surrogate_phases_all = np.array(surrogate_phases_list) # (n_perms, n_cells, n_timepoints)

    # Vectorised p-value: fraction of surrogates with PLV >= empirical
    # surrogate_plv_matrix broadcasts over empirical_plv along axis 0
    count    = np.sum(surrogate_plv_matrix >= empirical_plv[np.newaxis, :, :], axis=0)
    p_values = (count + 1) / (n_permutations + 1)  # continuity correction

    return surrogate_plv_matrix, p_values, surrogate_phases_all

# =============================================================================
# Plotting traces
# =============================================================================

def plot_traces(time, data, title='Signals', figsize=(14, 6), status=None):
    """
    Plot multiple time series traces in a single figure.

    Parameters
    ----------
    time : array_like, shape (n_timepoints,)
        Time axis values corresponding to the data points.
    data : DataFrame, shape (n_timepoints, n_traces)
        Data to plot. Each column is one trace.
    title : str, optional
        Title of the plot. Default is 'Signals'.
    figsize : tuple, optional
        Figure size in inches (width, height). Default is (14, 6).
    status : array_like of str, optional
        Status labels for each trace (must match number of columns).
        Traces with the same status get the same color. If None, each trace
        gets a unique color from the default matplotlib cycle. Default is None.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure object.
    ax : matplotlib.axes.Axes
        The created axes object.

    Examples
    --------
    >>> # Color by status
    >>> status = [' accepted', ' accepted', ' undecided']
    >>> fig, ax = plot_traces(time, data, status=status)
    
    >>> # Each trace different color
    >>> fig, ax = plot_traces(time, data)
    """

    fig, ax = plt.subplots(len(data.columns), 1, figsize=figsize, sharex=True)
    ax = np.atleast_1d(ax)

    if status is not None:
        # Map unique labels to colors
        unique_labels = list(dict.fromkeys(status))
        color_map = {label: f'C{i}' for i, label in enumerate(unique_labels)}
        
        # Plot with colors by label
        plotted_labels = set()
        for i, col in enumerate(data.columns):
            label = status.iloc[i]
            color = color_map[label]
            legend_label = label if label not in plotted_labels else None
            ax[i].plot(time, data[col], color=color, alpha=0.7, linewidth=0.8, label=legend_label)
            plotted_labels.add(label)
    else:
        for i, col in enumerate(data.columns):
            ax[i].plot(time, data[col], alpha=0.7, linewidth=0.8)
        ax[0].lines[0].set_label('traces')  # Label only the first line to avoid duplicates

    fig.legend(loc='upper right')
    fig.suptitle(title, fontweight='bold')
    fig.supxlabel('Time (s)')
    fig.supylabel('Signal')    
    return fig, ax

# =============================================================================
# Finding discontinuities in the time axis
# =============================================================================

def detect_time_jumps(time, factor=1.5):
    """
    Detect time jumps in the time axis.

    Parameters
    ----------
    time : array_like, shape (n_timepoints,)
        Time axis values corresponding to the data points.
    factor : float, optional
        Multiplier for the median inter-sample interval to define a jump.
        Default is 1.5, meaning any gap larger than 1.5 times the median
        interval is considered a jump.

    Returns
    -------
    jump_time : np array of shape (n_jumps,)
        List of time values where a time jump is detected. A time jump is defined
        as a gap between consecutive time points that exceeds 1.5 times the
        median inter-sample interval.

    Examples
    --------
    >>> jump_indices = detect_time_jumps(time)
    >>> print(f"Time jumps detected at indices: {jump_indices}")
    """
    dt = np.diff(time)
    median_dt = np.median(dt)
    jump_threshold = factor * median_dt
    jump_indices = np.where(dt > jump_threshold)[0] + 1  # +1 because diff reduces length by 1

    # recalculating linear time if jumps are detected
    if len(jump_indices) > 0:
        print(f"Time jumps detected at: {time[jump_indices]}s reconstructing linear time axis")
        time = np.linspace(0, median_dt * (len(time) - 1), len(time))
                
    jump_time = time[jump_indices]

    return time, jump_time