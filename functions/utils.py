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
    compute_spectra               : Apply dominant_frequency across all cells in a DataFrame.
    compute_spectral_correlation  : Vectorised pairwise Pearson r and p-values from power spectra.

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

import warnings

import math

from scipy import stats
from scipy.signal import hilbert, butter, filtfilt
from scipy.stats import t as t_dist
from scipy.spatial.distance import jensenshannon


import numpy as np
import pandas as pd 

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
        signal = poly_detrend(signal, order=poly_order)  # Convert back to ndarray if input was a Series

    # FFT
    fft_vals = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(len(signal), d=1 / fs)

    # Power spectrum
    power = np.abs(fft_vals) ** 2

    # Ignore DC bin (index 0)
    idx = np.argmax(power[1:]) + 1

    if plot:
        plt.figure(figsize=(14, 6))
        plt.plot(freqs, power, alpha=0.7)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power')
        plt.xlim(0, 1)
        plt.title('Power Spectrum', fontweight='bold')

    max_freq = freqs[idx]
    return max_freq, freqs, power, signal

def compute_spectra(filtered_df, fs, hann=False, poly_order=4):
    """
    Apply ``dominant_frequency`` to every cell in a DataFrame and collect results.

    Iterates over columns of ``filtered_df``, calls ``dominant_frequency`` on
    each, and assembles the outputs into arrays suitable for downstream spectral
    similarity analysis. The frequency axis is computed once and reused, since
    it is identical for all cells given a fixed sampling rate and signal length.

    Parameters
    ----------
    filtered_df : DataFrame, shape (n_timepoints, n_cells)
        Bandpass-filtered calcium traces. Each column is one cell.
        Should be the clustering-sorted DataFrame (``filtered_sorted``) so
        that the returned arrays are already in the correct cell order for
        comparison with the PLV matrix.
    fs : float
        Sampling frequency in Hz.
    hann : bool, optional
        If True, apply a Hann window before computing the FFT inside
        ``dominant_frequency``. Default is False.
    poly_order : int or None, optional
        Polynomial detrending order passed to ``dominant_frequency``.
        Set to None to skip detrending. Default is 4.

    Returns
    -------
    dominant_freqs : Series, shape (n_cells,)
        Peak FFT frequency in Hz for each cell, indexed by column name.
    freq_axis : ndarray, shape (n_freqs,)
        Frequency axis in Hz, shared across all cells. Length is
        ``n_timepoints // 2 + 1`` (one-sided spectrum).
    psds : ndarray, shape (n_cells, n_freqs)
        Power spectrum for each cell. Row order matches the column order
        of ``filtered_df``.

    Notes
    -----
    This function is a thin organisational wrapper around ``dominant_frequency``
    and adds no signal processing logic of its own. If you need per-cell
    frequency estimates only (not the spectra), use ``dominant_frequency``
    directly or call ``filtered_df.apply``.

    The returned ``psds`` array is the direct input to spectral similarity
    analysis via ``np.corrcoef(psds)``, which computes the pairwise Pearson
    correlation between spectral profiles.

    See Also
    --------
    dominant_frequency : Single-cell FFT-based dominant frequency extraction.

    Examples
    --------
    >>> dominant_freqs, freq_axis, psds = compute_spectra(filtered_sorted, sampling_rate)
    >>> print(f"Slowest cell: {dominant_freqs.idxmin()} @ {dominant_freqs.min():.4f} Hz")
    >>> spec_corr = np.corrcoef(psds)   # pairwise spectral similarity matrix
    """
    dominant_freqs_dict, psds, freq_axis = {}, [], None

    for col in filtered_df.columns:
        max_freq, f, power, _ = dominant_frequency(
            filtered_df[col], fs, hann=hann, poly_order=poly_order
        )
        dominant_freqs_dict[col] = max_freq
        psds.append(power)
        if freq_axis is None:
            freq_axis = f  # frequency axis is identical for all cells given fixed fs and length

    return pd.Series(dominant_freqs_dict), freq_axis, np.array(psds)


# =============================================================================
# Phase Locking Value
# =============================================================================

def plv_einsum(phases):

    """
    Compute the full pairwise PLV matrix and mean phase differences using
    einsum (vectorised).

    The Phase Locking Value between two signals i and j is defined as::

        PLV(i, j) = | (1/T) * sum_t( exp(i * (phi_i(t) - phi_j(t))) ) |

    where ``phi_i(t)`` is the instantaneous phase of cell i at time t,
    and T is the number of timepoints. PLV measures the **consistency**
    of the phase difference over time, not its absolute value.

    The mean phase difference is extracted from the same intermediate
    complex quantity before taking the absolute value, so both outputs
    are computed in a single pass.

    Parameters
    ----------
    phases : ndarray, shape (n_cells, n_timepoints)
        Instantaneous phase in radians for each cell at each timepoint.
        Obtain via ``get_phase`` after bandpass filtering.

    Returns
    -------
    PLV : ndarray, shape (n_cells, n_cells)
        Symmetric PLV matrix. Diagonal entries are exactly 1. Off-diagonal
        entries are in [0, 1], where 1 indicates perfect phase locking and
        0 indicates no locking.
    phase_diff : ndarray, shape (n_cells, n_cells)
        Mean phase difference in radians between each pair, in ``[-pi, pi]``.
        Accounts for the circular nature of phases. Diagonal entries are 0.
        Positive values indicate cell i leads cell j, negative values
        indicate cell i lags cell j.

    Notes
    -----
    PLV = 1 does **not** imply the two cells are in phase (phase difference
    equal to 0). It only means the phase difference is **constant** over time.
    Two cells locked at 180° (antiphase) also produce PLV = 1. Use
    ``phase_diff`` to determine the nature of the coupling.

    The einsum notation ``'it,jt->ij'`` computes the inner product over the
    time axis t for every pair (i, j) simultaneously, avoiding explicit loops
    over cell pairs.

    Diagonal entries of ``PLV`` are forced to exactly 1.0 via
    ``np.fill_diagonal`` to correct for floating point errors introduced
    by the Hilbert transform edge effects.

    Examples
    --------
    >>> phases = np.array([get_phase(sig) for sig in filtered_signals])
    >>> PLV, phase_diff = compute_plv(phases)
    """

    # Convert to complex exponentials
    exp_phases = np.exp(1j * phases)  # (n_cells, n_timepoints)

    # 'it,jt->ij': for each pair (i,j), multiply across time and sum
    mean_complex =  np.einsum('it,jt->ij', exp_phases, np.conj(exp_phases)) / phases.shape[1]
    PLV = np.abs(mean_complex)

    phase_diff = np.angle(mean_complex) # Mean phase difference for each pair (i,j)

    np.fill_diagonal(PLV, 1)  # Ensure diagonal is exactly 1
    np.fill_diagonal(phase_diff, 0)  # Phase difference with self is 0

    return PLV, phase_diff


def circular_shift_surrogate(signals, min_interval=None, max_interval=None):
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
    min_interval : int or None, optional
        Minimum shift in samples. Should be at least one full oscillation
        cycle (i.e. ``int(fs / dominant_freq)``) to ensure the phase
        relationship is genuinely disrupted. If None, defaults to 0.
        Default is None.

        .. warning::
            A minimum shift of 0 allows trivial surrogates that preserve
            the original phase relationship. For meaningful null distributions
            set this to at least one full oscillation cycle.

    max_interval : int or None, optional
        Maximum shift in samples. If None, defaults to the number of
        timepoints in the signal. Default is None.

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
    >>> # Default: shift between 0 and full trace length
    >>> surrogate = circular_shift_surrogate(signals)

    >>> # Recommended: shift by at least one oscillation cycle
    >>> surrogate = circular_shift_surrogate(signals, min_interval=100, max_interval=500)
    """
    n_timepoints = signals.shape[1]

    if min_interval is None:
        min_interval = 0
        warnings.warn(
            "min_interval not set, defaulting to 0. "
            "A shift of 0 samples produces a surrogate identical to the original signal. ",
            UserWarning
        )
    if max_interval is None:
        max_interval = n_timepoints
        warnings.warn(
            f"max_interval not set, defaulting to {n_timepoints} samples (full trace length). "
            "A shift equal to the trace length is equivalent to a shift of 0 and produces "
            "a surrogate identical to the original signal. ",
            UserWarning
        )

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


def permutation_test(
    signals,
    empirical_plv,
    empirical_phase_diff,
    surrogate_fn,
    n_permutations=1000,
    plot_surrogates=0,
    **surrogate_kwargs
):
    """
    Surrogate permutation test for PLV and phase difference significance.

    For each permutation, a surrogate dataset is generated by ``surrogate_fn``,
    phases are extracted internally, and surrogate PLV and phase difference
    matrices are computed. The empirical values are compared against these
    null distributions to produce two sets of p-values.

    This function is agnostic to the surrogate method — any callable that
    accepts signals and returns signals of the same shape can be passed.
    Additional keyword arguments are forwarded directly to ``surrogate_fn``.

    Parameters
    ----------
    signals : ndarray, shape (n_cells, n_timepoints)
        Bandpass-filtered signals (real-valued). Phases are extracted
        internally so that both surrogate methods operate consistently
        in signal space.
    empirical_plv : ndarray, shape (n_cells, n_cells)
        Observed PLV matrix computed from the real data via ``compute_plv``.
    empirical_phase_diff : ndarray, shape (n_cells, n_cells)
        Observed mean phase difference matrix in radians, computed from
        the real data via ``compute_plv``.
    surrogate_fn : callable
        Function with signature ``surrogate_fn(signals, **kwargs) -> surrogate_signals``
        where ``surrogate_signals`` has the same shape as ``signals``.
        Built-in options are ``circular_shift_surrogate`` and
        ``phase_randomisation_surrogate``. Any additional keyword arguments
        required by ``surrogate_fn`` should be passed directly to
        ``permutation_test`` as keyword arguments (see ``**surrogate_kwargs``).
    n_permutations : int, optional
        Number of surrogate datasets to generate. Default is 1000.
        Higher values give more precise p-values but increase runtime linearly.
    plot_surrogates : int, optional
        Number of surrogate datasets to plot for visual sanity checking.
        The empirical signals are plotted first in red, followed by the
        first ``plot_surrogates`` surrogates in blue, one figure each.
        Set to 0 to disable all plotting. Default is 0.
    **surrogate_kwargs
        Additional keyword arguments forwarded to ``surrogate_fn``.

        For ``circular_shift_surrogate``:

        - ``min_interval`` (int) – minimum shift in samples. Should be at
          least one full oscillation cycle (``int(fs / dominant_freq)``).
          Defaults to 0 if not provided.
        - ``max_interval`` (int) – maximum shift in samples. Defaults to
          the full trace length if not provided.

        For ``phase_randomisation_surrogate``:
            No additional arguments required.

    Returns
    -------
    surrogate_plv_matrix : ndarray, shape (n_permutations, n_cells, n_cells)
        PLV matrix computed for each surrogate dataset. Can be used to
        inspect the null distribution directly.
    p_values_plv : ndarray, shape (n_cells, n_cells)
        Proportion of surrogates with PLV >= empirical PLV, with continuity
        correction: ``p = (count + 1) / (n_permutations + 1)``.
        One-tailed test — low p-value indicates the empirical PLV is
        unusually high compared to chance, i.e. significant phase locking.
        Values are symmetric. Uncorrected — pass to ``correct_p_values``
        before thresholding.
    surrogate_phase_diff_matrix : ndarray, shape (n_permutations, n_cells, n_cells)
        Mean phase difference matrix in radians computed for each surrogate
        dataset. Can be used to inspect the null distribution directly.
    p_values_phase_diff : ndarray, shape (n_cells, n_cells)
        Two-tailed test on ``cos(phase_diff)``, with continuity correction:
        ``p = (count + 1) / (n_permutations + 1)``.
        Low p-value indicates the empirical phase difference is unusually
        close to in-phase (0) or antiphase (π) compared to chance.
        Values are symmetric. Uncorrected — pass to ``correct_p_values``
        before thresholding.

    Notes
    -----
    **Two-stage significance filter**: the two p-value matrices are designed
    to be used sequentially. First threshold ``p_values_plv`` to identify
    significantly phase-locked pairs, then among those check
    ``p_values_phase_diff`` to identify pairs whose phase relationship is
    significantly close to in-phase or antiphase::

        sig_plv        = p_vals_plv_corrected < 0.05
        sig_phase_diff = p_vals_phase_diff_corrected < 0.05
        sig_both       = sig_plv & sig_phase_diff

    **Surrogate method and phase difference test**: the phase difference
    permutation test is most meaningful when using ``circular_shift_surrogate``,
    which preserves each cell's individual oscillatory structure (frequency,
    amplitude, autocorrelation) while destroying cross-cell phase relationships
    — simulating a disconnected network of otherwise identical neurons.
    ``phase_randomisation_surrogate`` also destroys the individual oscillatory
    structure of each cell, making the null distribution less biologically
    meaningful for phase difference testing, although it remains the stronger
    null for the PLV test.

    **Phase difference test**: uses ``cos(phase_diff)`` to linearise the
    circular phase difference before comparison. ``cos(0)`` = 1 (in-phase),
    ``cos(π)`` = -1 (antiphase), ``cos(π/2)`` = 0 (null centre). Taking
    ``abs(cos(...))`` makes the test symmetric around 0, detecting both
    in-phase and antiphase relationships.

    **FDR correction**: both p-value matrices are uncorrected. Pass each
    independently to ``correct_p_values``::

        p_plv_matrix, p_plv_pairs     = correct_p_values(p_values_plv, cell_labels=cell_names)
        p_phase_matrix, p_phase_pairs = correct_p_values(p_values_phase_diff, cell_labels=cell_names)

    **Random seed**: set ``np.random.seed(seed)`` before calling this function
    to ensure reproducible results across runs.

    **Runtime**: scales as O(n_permutations x n_cells^2 x n_timepoints).
    ``phase_randomisation_surrogate`` is slightly slower per permutation than
    ``circular_shift_surrogate`` due to FFT overhead on each cell.

    See Also
    --------
    compute_plv : PLV and phase difference matrix computation.
    circular_shift_surrogate : Circular-shift surrogate generator.
    phase_randomisation_surrogate : FFT phase-randomisation surrogate generator.
    correct_p_values : FDR correction for the returned p-values.

    Examples
    --------
    >>> PLV, phase_diff = compute_plv(phases)

    >>> # Circular shift — recommended for phase difference test
    >>> surr_plv, p_vals_plv, surr_phase_diff, p_vals_phase_diff = permutation_test(
    ...     signals, PLV, phase_diff, circular_shift_surrogate,
    ...     n_permutations=1000,
    ...     min_interval=100, max_interval=500
    ... )

    >>> # Phase randomisation — stronger null for PLV test
    >>> surr_plv, p_vals_plv, surr_phase_diff, p_vals_phase_diff = permutation_test(
    ...     signals, PLV, phase_diff, phase_randomisation_surrogate,
    ...     n_permutations=1000
    ... )
    """

    surrogate_plv_list    = []
    surrogate_phase_diff_list = []

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
        surrogate_signals = surrogate_fn(signals, **surrogate_kwargs)
        surrogate_phases  = np.array([get_phase(s) for s in surrogate_signals])

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

        surrogate_plv, surrogate_phase_diff = plv_einsum(surrogate_phases)
        surrogate_plv_list.append(surrogate_plv)
        surrogate_phase_diff_list.append(surrogate_phase_diff)

    surrogate_plv_matrix = np.array(surrogate_plv_list)
    surrogate_phase_diff_matrix = np.array(surrogate_phase_diff_list)

    count_plv    = np.sum(surrogate_plv_matrix >= empirical_plv[np.newaxis, :, :], axis=0)
    p_values_plv = (count_plv + 1) / (n_permutations + 1)

  # Single_tailed: tests whether phase difference is significantly close to 0 than chance
    count_phase_diff = np.sum(
    np.abs((surrogate_phase_diff_matrix)) <= np.abs((empirical_phase_diff[np.newaxis, :, :])),
    axis=0) 

    p_values_phase_diff = (count_phase_diff + 1) / (n_permutations + 1)

    return surrogate_plv_matrix, p_values_plv, surrogate_phase_diff_matrix, p_values_phase_diff

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

# =============================================================================
# Correcting and flattening p-value matrix
# =============================================================================

def correct_p_values(p_values, FDR_correction=True, cell_labels=None, method='bh'):
    
    """
    Apply multiple comparison correction to a symmetric p-value matrix.

    Extracts the upper triangle of the p-value matrix, applies optional
    FDR correction, and returns both a symmetric corrected matrix and a
    flat DataFrame of pairwise results.

    Parameters
    ----------
    p_values : ndarray, shape (n_cells, n_cells)
        Symmetric matrix of uncorrected p-values, as returned by
        ``permutation_test``. Diagonal is ignored.
    FDR_correction : bool, optional
        If True, apply FDR correction using ``scipy.stats.false_discovery_control``.
        If False, return uncorrected p-values. Default is True.
    cell_labels : list of str or None, optional
        Labels for each cell, used as row/column names in the output DataFrames.
        If None, cells are labelled ``'Cell 0'``, ``'Cell 1'``, etc.
        Default is None.
    method : {'bh', 'by'}, optional
        FDR correction method, passed to ``scipy.stats.false_discovery_control``.
        Only used when ``FDR_correction=True``.

        - ``'bh'`` : Benjamini-Hochberg (controls FDR under independence).
        - ``'by'`` : Benjamini-Yekutieli (controls FDR under arbitrary dependence).

        Default is ``'bh'``.

    Returns
    -------
    p_vals_matrix : DataFrame, shape (n_cells, n_cells)
        Symmetric matrix of corrected (or uncorrected) p-values, with
        ``cell_labels`` as both row and column names. Diagonal entries are 1.
    p_vals_pairs : DataFrame, shape (n_pairs, 1)
        Flat table of corrected (or uncorrected) p-values for each unique
        cell pair (upper triangle only). Index contains pair labels in the
        format ``'Cell i vs. Cell j'``. Column is ``'p_value'``.

    Raises
    ------
    ValueError
        If ``FDR_correction=True`` and ``method`` is not ``'bh'`` or ``'by'``.

    Examples
    --------
    >>> p_matrix, p_pairs = correct_p_values(p_values, cell_labels=cell_names)

    >>> # Benjamini-Yekutieli correction (conservative, for dependent tests)
    >>> p_matrix, p_pairs = correct_p_values(p_values, method='by', cell_labels=cell_names)

    >>> # No correction
    >>> p_matrix, p_pairs = correct_p_values(p_values, FDR_correction=False, cell_labels=cell_names)
    """
    
    if FDR_correction and method not in ['bh', 'by']:
        raise ValueError(
            f"Invalid method '{method}'. "
            "Use 'bh' for Benjamini-Hochberg or 'by' for Benjamini-Yekutieli."
        )

    n = p_values.shape[0]

    if cell_labels is None:
        cell_labels = [f"Cell {i}" for i in range(n)]

    # Extract upper triangle (excluding diagonal)
    upper_triangle_indices = np.triu_indices(n, k=1)
    p_vals_upper = p_values[upper_triangle_indices]

    # Apply correction or pass through
    if FDR_correction:
        p_vals_upper_corrected = stats.false_discovery_control(p_vals_upper, method=method)
    else:
        p_vals_upper_corrected = p_vals_upper

    # Reconstruct symmetric matrix
    p_vals_matrix = np.ones((n, n))
    p_vals_matrix[upper_triangle_indices] = p_vals_upper_corrected
    lower_triangle_indices = np.tril_indices(n, k=-1)
    p_vals_matrix[lower_triangle_indices] = p_vals_matrix.T[lower_triangle_indices]

    # Build output DataFrames
    pair_labels = [
        f"{cell_labels[i]} vs. {cell_labels[j]}"
        for i, j in zip(*upper_triangle_indices)
    ]

    p_vals_matrix_df = pd.DataFrame(p_vals_matrix, index=cell_labels, columns=cell_labels)
    p_vals_pairs_df  = pd.DataFrame(p_vals_upper_corrected, columns=['p_value'], index=pair_labels)

    return p_vals_matrix_df, p_vals_pairs_df

# =============================================================================
# Computes pairwise distance
# =============================================================================

def compute_pairwise_distances(props, pair_labels):
    """
    Compute Euclidean distances between cell pairs from their centroid coordinates.

    Parameters
    ----------
    props : DataFrame
        Cell properties table containing at least ``Name``, ``CentroidX``
        and ``CentroidY`` columns, as returned by ``load_props``.
    pair_labels : Index or list of str
        Pair labels in the format ``'Cell i vs. Cell j'``, as returned by
        the index of ``correct_p_values`` output. Must match cell names
        in the ``Name`` column of ``props``.

    Returns
    -------
    distances : DataFrame, shape (n_pairs, 1)
        Euclidean distances in pixels between each cell pair, indexed by
        ``pair_labels``. Column is ``'distance_px'``.

    Raises
    ------
    ValueError
        If a cell name from ``pair_labels`` is not found in ``props['Name']``.

    Notes
    -----
    Distances are in pixel units and reflect the coordinate system of the
    original cell map image. To convert to physical units (e.g. micrometres),
    multiply by the image pixel size.

    Examples
    --------
    >>> distances = compute_pairwise_distances(recording.props, plv_pairs.index)
    >>> from scipy.stats import pearsonr
    >>> r, p = pearsonr(distances['distance_px'], plv_pairs['p_value'])
    """
    distances = []

    for name in pair_labels:
        parts = name.split('vs.')
        c1 = parts[0].strip()
        c2 = parts[1].strip()

        row1 = props[props['Name'] == c1]
        row2 = props[props['Name'] == c2]

        if row1.empty:
            raise ValueError(f"Cell '{c1}' not found in props.")
        if row2.empty:
            raise ValueError(f"Cell '{c2}' not found in props.")

        x1, y1 = row1[['CentroidX', 'CentroidY']].values[0]
        x2, y2 = row2[['CentroidX', 'CentroidY']].values[0]

        distances.append(math.dist([x1, y1], [x2, y2]))

    return pd.DataFrame(distances, index=pair_labels, columns=['distance_px'])

# =============================================================================
# Pairwise spectral correlation
# =============================================================================

def compute_spectral_correlation(psds, cell_labels=None):
    """
    Compute the pairwise Pearson correlation matrix of power spectra and
    the associated two-tailed p-values, fully vectorised.

    Correlation values are computed in a single call to ``numpy.corrcoef``.
    P-values are derived analytically from the t-distribution, avoiding
    O(n_cells^2) individual ``pearsonr`` calls::

        t = r * sqrt((n_freqs - 2) / (1 - r^2))
        p = 2 * t.sf(|t|, df = n_freqs - 2)

    Parameters
    ----------
    psds : ndarray, shape (n_cells, n_freqs)
        Power spectra, one row per cell. Typically the third return value
        of ``compute_spectra``. Row order must match the clustering order
        of the PLV matrix for cross-matrix consistency.
    cell_labels : array-like of str or None, optional
        Labels for each cell, used as row and column names in the output
        DataFrames. If None, cells are labelled ``'Cell 0'``, ``'Cell 1'``,
        etc. Default is None.

    Returns
    -------
    corr_matrix : DataFrame, shape (n_cells, n_cells)
        Symmetric matrix of Pearson r values, indexed by ``cell_labels``.
        Diagonal is exactly 1. Values are in [-1, 1]; in practice near
        [0, 1] for smooth unimodal calcium imaging power spectra.
    p_matrix : DataFrame, shape (n_cells, n_cells)
        Symmetric matrix of two-tailed p-values corresponding to each r,
        indexed by ``cell_labels``. Diagonal is 0 by definition
        (self-correlation is exact). Pass to ``correct_p_values`` for
        FDR correction before thresholding.

    Notes
    -----
    The t-statistic is undefined when ``|r| = 1`` exactly (division by zero
    in ``sqrt(1 - r^2)``). Diagonal entries are clipped to avoid this:
    the diagonal of ``r`` is set to ``1 - eps`` before computing ``t``,
    then the diagonal of ``p_matrix`` is forced back to 0 afterwards.

    The p-values assume that the frequency bins are independent observations.
    In practice, adjacent FFT bins are correlated (especially after windowing),
    so the effective degrees of freedom are lower than ``n_freqs - 2`` and
    p-values should be interpreted with appropriate caution.

    See Also
    --------
    compute_spectra    : Produces the ``psds`` array used as input here.
    correct_p_values   : FDR correction for the returned ``p_matrix``.
    numpy.corrcoef     : Underlying function for the correlation computation.

    Examples
    --------
    >>> dominant_freqs, freq_axis, psds = compute_spectra(filtered_sorted, sampling_rate)
    >>> corr_matrix, p_matrix = compute_spectral_correlation(psds, cell_labels=filtered_sorted.columns)
    >>> corr_matrix_fdr, corr_vector = correct_p_values(p_matrix.to_numpy(), cell_labels=filtered_sorted.columns)
    """
    n_cells, n_freqs = psds.shape

    if cell_labels is None:
        cell_labels = [f"Cell {i}" for i in range(n_cells)]

    # Pearson r matrix — operates on rows, shape (n_cells, n_freqs) is correct
    r = np.corrcoef(psds)

    # Derive p-values from the t-distribution.
    # Clip diagonal to avoid division by zero at r = 1 exactly.
    r_clipped = r.copy()
    np.fill_diagonal(r_clipped, 1 - 1e-10)
    t_stat = r_clipped * np.sqrt((n_freqs - 2) / (1 - r_clipped ** 2))
    p      = 2 * t_dist.sf(np.abs(t_stat), df=n_freqs - 2)

    # Restore diagonal: self-correlation is exact, p = 0 by definition
    np.fill_diagonal(p, 0.0)

    corr_matrix = pd.DataFrame(r, index=cell_labels, columns=cell_labels)
    p_matrix    = pd.DataFrame(p, index=cell_labels, columns=cell_labels)

    return corr_matrix, p_matrix

def compute_spectral_jsd(psds, cell_labels=None):
    """
    Compute pairwise Jensen-Shannon Divergence (JSD) between power spectra.

    Each power spectrum is normalised to a probability distribution over
    frequencies before comparison, so only spectral *shape* is compared —
    cells with identical profiles but different overall power levels score 0.
    High-power bins (the oscillatory peaks) contribute more to the divergence
    than the flat noise floor, directly addressing the noise-dominance problem
    of plain Pearson correlation on raw PSDs.

    The Jensen-Shannon Divergence between two distributions P and Q is::

        JSD(P || Q) = 0.5 * KL(P || M) + 0.5 * KL(Q || M),  M = (P + Q) / 2

    where KL is the Kullback-Leibler divergence. JSD is symmetric and bounded
    in [0, log(2)]. ``scipy.spatial.distance.jensenshannon`` returns the
    square root, which is a proper metric in [0, 1].

    Parameters
    ----------
    psds : ndarray, shape (n_cells, n_freqs)
        Power spectra, one row per cell. Typically the third return value
        of ``compute_spectra``. Row order must match the clustering order
        of the PLV matrix for cross-matrix consistency. Values must be
        non-negative (satisfied by construction as squared FFT magnitudes).
    cell_labels : array-like of str or None, optional
        Labels for each cell, used as row and column names in the output
        DataFrames. If None, cells are labelled ``'Cell 0'``, ``'Cell 1'``,
        etc. Default is None.

    Returns
    -------
    jsd_matrix : DataFrame, shape (n_cells, n_cells)
        Symmetric matrix of sqrt(JSD) dissimilarity values, indexed by
        ``cell_labels``. Values in [0, 1]: 0 means identical spectral
        shapes, 1 means maximally different. Diagonal is exactly 0.
    jsd_similarity_matrix : DataFrame, shape (n_cells, n_cells)
        ``1 - jsd_matrix``, provided for direct visual and numerical
        comparison with PLV and spectral Pearson r matrices, both of which
        are similarity measures. Diagonal is exactly 1.

    Notes
    -----
    Frequency bins with zero power in both spectra of a pair are handled
    gracefully by ``scipy.spatial.distance.jensenshannon`` (the KL term
    contributes 0 there). Rows that sum to zero (flat-zero spectra) are
    detected and raise a ValueError before any computation.

    Unlike Pearson r, JSD has no associated parametric p-value. To assess
    significance, pass ``jsd_similarity_matrix`` to a permutation test or
    use it descriptively alongside the PLV significance results.

    See Also
    --------
    compute_spectra              : Produces the ``psds`` array used as input.
    compute_spectral_correlation : Pearson r alternative for spectral similarity.
    scipy.spatial.distance.jensenshannon : Underlying implementation.

    Examples
    --------
    >>> dominant_freqs, freq_axis, psds = compute_spectra(filtered_sorted, sampling_rate)
    >>> jsd_matrix, jsd_sim_matrix = compute_spectral_jsd(psds, cell_labels=filtered_sorted.columns)
    >>> # Compare with PLV
    >>> scatter: plt.scatter(PLV_vals_vector.values, jsd_sim_vector.values)
    """
    n_cells = psds.shape[0]

    if cell_labels is None:
        cell_labels = [f"Cell {i}" for i in range(n_cells)]

    # Guard against zero-sum rows — jensenshannon would produce NaN silently
    row_sums = psds.sum(axis=1)
    if np.any(row_sums == 0):
        bad = np.where(row_sums == 0)[0]
        raise ValueError(
            f"Cells at indices {list(bad)} have a flat-zero power spectrum. "
            "Check bandpass filtering and interval selection."
        )

    # Normalise each row to a probability distribution over frequencies
    psds_norm = psds / row_sums[:, np.newaxis]

    # Compute upper triangle, fill both sides — jensenshannon is symmetric
    jsd = np.zeros((n_cells, n_cells))
    for i in range(n_cells):
        for j in range(i + 1, n_cells):
            d = jensenshannon(psds_norm[i], psds_norm[j])
            jsd[i, j] = jsd[j, i] = d  # already sqrt(JSD) ∈ [0, 1]

    # Diagonal is 0 by definition — jensenshannon(x, x) = 0
    jsd_matrix            = pd.DataFrame(jsd,       index=cell_labels, columns=cell_labels)
    jsd_similarity_matrix = pd.DataFrame(1 - jsd,   index=cell_labels, columns=cell_labels)

    return jsd_matrix, jsd_similarity_matrix