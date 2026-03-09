"""
plots.py
========
Visualization utilities for Phase Locking Value (PLV) analysis
of neuronal calcium imaging traces.

This submodule contains plotting functions for inspecting
time-series activity, preprocessing steps, connectivity
structure, and statistical testing results.

Routine Listings
----------------
Time-series visualization
    plot_traces       : Plot multiple calcium traces in stacked format,
                        optionally color-coded by status labels.

Notes
-----
The typical visualization workflow is::

    raw signal
        → plot_traces
        → (bandpass, phase extraction, PLV computation)
        → connectivity and statistical plots

The function returns matplotlib Figure and Axes objects to allow
further customization (titles, annotations, saving, styling).

Design principles
-----------------
- One subplot per trace (stacked vertically)
- Shared time axis
- Optional categorical color mapping via `status`
- Minimal visual clutter (thin lines, transparency)

See Also
--------
matplotlib.pyplot
"""

from pydoc import text

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import warnings

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

    if len(data.shape) == 1: 
        data = data.to_frame() # Convert Series to DataFrame if input is 1D

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
# Plotting PLV matrices
# =============================================================================

def plot_plv_matrix(plv_matrix, p_val_matrix=None, plot_stat=True, only_significant=True, 
                    title='Phase Locking Value (PLV) Matrix', figsize=(12, 12), **kwargs):
    
    """
    Plot a Phase Locking Value (PLV) matrix as a heatmap with optional
    significance annotations.

    Parameters
    ----------
    plv_matrix : DataFrame, shape (n_cells, n_cells)
        Symmetric matrix of PLV values between all cell pairs. Values are
        in [0, 1], where 1 indicates perfect phase locking and 0 indicates
        no locking. Row and column labels are used as axis tick labels.
    p_val_matrix : DataFrame, shape (n_cells, n_cells) or None, optional
        Symmetric matrix of BH-adjusted p-values, as returned by
        ``correct_p_values``. Must have the same shape as ``plv_matrix``.
        If None, no significance annotations are added. Default is None.
    plot_stat : bool, optional
        If True and ``p_val_matrix`` is provided, annotate each cell with
        significance stars and the adjusted p-value. Default is True.
    only_significant : bool, optional
        If True, only annotate cells that are significant (p < 0.05).
        If False, annotate all cells with their p-value. Default is True.
    title : str, optional
        Title of the plot. Default is ``'Phase Locking Value (PLV) Matrix'``.
    figsize : tuple of float, optional
        Figure size in inches (width, height). Default is (12, 12).
    **kwargs
        Additional keyword arguments forwarded to ``seaborn.heatmap``
        (e.g. ``cmap``, ``vmin``, ``vmax``, ``xticklabels``, ``yticklabels``).

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure object.
    ax : matplotlib.axes.Axes
        The created axes object.

    Raises
    ------
    ValueError
        If ``p_val_matrix`` is provided but has a different shape than
        ``plv_matrix``.

    Notes
    -----
    Significance stars follow the conventional thresholds:

    - ``****`` : p < 0.0001
    - ``***``  : p < 0.001
    - ``**``   : p < 0.01
    - ``*``    : p < 0.05

    Stars are shown in red; non-significant p-values (when ``only_significant=False``)
    are shown in black. The adjusted p-value is displayed below the stars in
    scientific notation.

    Examples
    --------
    >>> # PLV matrix only
    >>> fig, ax = plot_plv_matrix(plv_matrix, cmap='viridis')

    >>> # With significance annotations (significant pairs only)
    >>> fig, ax = plot_plv_matrix(plv_matrix, p_val_matrix=p_matrix,
    ...                           cmap='viridis')

    >>> # Annotate all pairs including non-significant
    >>> fig, ax = plot_plv_matrix(plv_matrix, p_val_matrix=p_matrix,
    ...                           only_significant=False, cmap='viridis')
    """

    if p_val_matrix is not None and plv_matrix.shape != p_val_matrix.shape:
        raise ValueError(
            f'plv_matrix shape is incompatible with p_val_matrix shape: '
            f'{plv_matrix.shape} vs. {p_val_matrix.shape}'
        )

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(plv_matrix, ax=ax, **kwargs)

    if p_val_matrix is not None and plot_stat:
        p_vals_np = p_val_matrix.to_numpy()  # convert once outside the loop
        for i in range(p_vals_np.shape[0]):
            for j in range(p_vals_np.shape[1]):
                p_val = p_vals_np[i, j]

                if p_val < 0.0001:
                    marker, text_color = f'****\n{p_val:.4g}', 'red'
                elif p_val < 0.001:
                    marker, text_color = f'***\n{p_val:.4g}', 'red'
                elif p_val < 0.01:
                    marker, text_color = f'**\n{p_val:.4g}', 'red'
                elif p_val < 0.05:
                    marker, text_color = f'*\n{p_val:.4g}', 'red'
                elif not only_significant:
                    marker, text_color = f'{p_val:.4g}', 'black'
                else:
                    marker, text_color = '', 'black'

                if marker:
                    ax.text(j + 0.5, i + 0.5, marker, ha='center', va='center',
                            color=text_color, fontsize=8)

    ax.set_title(f'{title}\n(* p<0.05, ** p<0.01, *** p<0.001, **** p<0.0001)')
    plt.tight_layout()
    plt.show()

    return fig, ax

# =============================================================================
# Plotting PLV map
# =============================================================================

def plot_plv_map(props, plv_vector, p_val_vector=None, map=None, figsize=(12, 12), alpha=0.05,
                 cmap='Reds', title='Significant Phase-Locking Connections',
                 vmin=None, vmax=None):
    """
    Plot PLV connections between cells overlaid on an optional cell map image.

    Each cell pair is drawn as a line between their centroids, coloured by
    PLV value. If a p-value vector is provided, only significant pairs are
    plotted. If no p-value vector is provided, all connections are plotted.

    Parameters
    ----------
    props : DataFrame
        Cell properties table containing at least ``Name``, ``CentroidX``
        and ``CentroidY`` columns, as returned by ``load_props``.
    plv_vector : DataFrame, shape (n_pairs, 1)
        PLV values for each unique cell pair, as returned by
        ``correct_p_values``. Index must contain pair labels in the
        format ``'Cell i vs. Cell j'``.
    p_val_vector : DataFrame, shape (n_pairs, 1) or None, optional
        BH-adjusted p-values for each cell pair, as returned by
        ``correct_p_values``. If None, all connections are plotted
        regardless of significance. Default is None.
    map : ndarray or None, optional
        Cell map image array, as returned by ``load_map``. If None,
        connections are plotted on a blank axes. Default is None.
    figsize : tuple of float, optional
        Figure size in inches (width, height). Default is (12, 12).
    alpha : float, optional
        Significance threshold. Only pairs with ``p_value < alpha`` are
        plotted when ``p_val_vector`` is provided. Default is 0.05.
    cmap : str, optional
        Colormap name for colouring connections by PLV value.
        Default is ``'Reds'``.
    title : str, optional
        Title of the plot. Default is
        ``'Significant Phase-Locking Connections'``.
    vmin : float or None, optional
        Minimum PLV value for colormap normalisation. If None, defaults
        to the minimum value in ``plv_vector``. Default is None.
    vmax : float or None, optional
        Maximum PLV value for colormap normalisation. If None, defaults
        to the maximum value in ``plv_vector``. Default is None.

    Returns
    -------
    fig : matplotlib.figure.Figure or None
        The created figure, or None if ``props`` is None.
    ax : matplotlib.axes.Axes or None
        The created axes, or None if ``props`` is None.

    Raises
    ------
    ValueError
        If ``p_val_vector`` is provided but has a different shape than
        ``plv_vector``.

    Notes
    -----
    Cell centroids are plotted as white dots with gray outlines on top of
    the connection lines (``zorder=5``). Fixing ``vmin`` and ``vmax`` to
    the same values across recordings is useful for visual comparison of
    PLV strength between experiments.

    Examples
    --------
    >>> # All connections, no map
    >>> fig, ax = plot_PLV_map(recording.props, plv_pairs)

    >>> # Significant connections only, with map background
    >>> fig, ax = plot_PLV_map(recording.props, plv_pairs,
    ...                        p_val_vector=p_val_pairs,
    ...                        map=recording.map_img)

    >>> # Fix colormap range for comparison across recordings
    >>> fig, ax = plot_PLV_map(recording.props, plv_pairs,
    ...                        p_val_vector=p_val_pairs,
    ...                        map=recording.map_img,
    ...                        vmin=0.5, vmax=1.0)
    """

    if props is None:
        return None

    if p_val_vector is not None and plv_vector.shape != p_val_vector.shape:
        raise ValueError(
            f'plv_vector shape is incompatible with p_val_vector shape: '
            f'{plv_vector.shape} vs. {p_val_vector.shape}'
        )

    if p_val_vector is not None and all(plv_vector.index != p_val_vector.index):
        warnings.warn(
            f'plv_vector index does not match p_val_vector index: '
            f'{plv_vector.index} vs. {p_val_vector.index}. '
            'Using plv_vector index instead.', UserWarning
        )

    names = plv_vector.index.to_list()

    fig, ax = plt.subplots(figsize=figsize)
    if map is not None:
        ax.imshow(map, aspect='auto', cmap='gray')

    # Normalisation: use provided vmin/vmax or fall back to data range
    norm     = plt.Normalize(
                    vmin=vmin if vmin is not None else np.min(plv_vector.values),
                    vmax=vmax if vmax is not None else np.max(plv_vector.values)
                )
    colormap = plt.colormaps[cmap]

    # Draw connections
    for i, name in enumerate(names):
        parts = name.split('vs.')
        c1 = parts[0].strip()
        c2 = parts[1].strip()

        x1, y1 = props[props['Name'] == c1][['CentroidX', 'CentroidY']].values[0]
        x2, y2 = props[props['Name'] == c2][['CentroidX', 'CentroidY']].values[0]

        plv_value = float(plv_vector.iloc[i].iloc[0])
        color     = colormap(norm(plv_value))
        p_value   = float(p_val_vector.iloc[i].iloc[0]) if p_val_vector is not None else None

        if p_value is None or p_value < alpha:
            ax.plot([x1, x2], [y1, y2], linewidth=2, color=color, zorder=3)

    # Draw cell positions on top of connections
    ax.scatter(props['CentroidX'], props['CentroidY'],
               s=50, color='white', edgecolors='gray', linewidth=1, zorder=5)

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label='PLV Value')

    ax.set_title(title)
    ax.axis('off')
    plt.tight_layout()
    plt.show()

    return fig, ax

# =============================================================================
# Plot suroogate vs empiric distribution
# =============================================================================

def plot_surrogate_vs_empirical(empirical_matrix, surrogate_matrix, n_bins=100,
                                 figsize=(24, 24), n_cols=6, xlabel='Value',
                                 title='Surrogate vs Empirical'):
    """
    Plot surrogate distributions against empirical values for all unique cell pairs.

    For each unique cell pair (upper triangle), plots a histogram of the surrogate
    distribution with the empirical value marked as a vertical line. Works with
    any pairwise matrix such as PLV or phase difference.

    Parameters
    ----------
    empirical_matrix : DataFrame, shape (n_cells, n_cells)
        Empirical pairwise matrix, such as PLV or phase difference, as returned
        by ``correct_p_values`` or ``compute_plv``. Row and column labels are
        used as pair names in subplot titles.
    surrogate_matrix : ndarray, shape (n_permutations, n_cells, n_cells)
        Surrogate pairwise matrix, as returned by ``permutation_test``.
        Must have the same number of cells as ``empirical_matrix``.
    n_bins : int, optional
        Number of bins for the histogram. Shared across all pairs for
        comparability. Default is 100.
    figsize : tuple of float, optional
        Figure size in inches (width, height). Default is (24, 24).
    n_cols : int, optional
        Number of columns in the subplot grid. Rows are computed automatically
        from the number of pairs. Default is 6.
    xlabel : str, optional
        Label for the x-axis of each subplot. Default is ``'Value'``.
    title : str, optional
        Overall figure title. Default is ``'Surrogate vs Empirical'``.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure object.
    axes : ndarray of matplotlib.axes.Axes
        The created axes objects.

    Notes
    -----
    Bin edges are computed from all surrogate values across all pairs to
    ensure comparability between subplots. No transformation is applied
    to the values — pass pre-transformed values if needed (e.g. absolute
    values). Cell pair names are taken from the row and column labels of
    ``empirical_matrix``.

    Examples
    --------
    >>> # PLV
    >>> fig, axes = plot_surrogate_vs_empirical(PLV_sorted, surrogate_plv_matrix,
    ...                                          xlabel='PLV',
    ...                                          title='PLV Surrogate vs Empirical')

    >>> # Phase difference
    >>> fig, axes = plot_surrogate_vs_empirical(phase_diff_sorted, phase_diff_surrogate,
    ...                                          xlabel='Phase diff (rad)',
    ...                                          title='Phase Difference Surrogate vs Empirical')
    """
    n = empirical_matrix.shape[0]
    upper_triangle_indices = np.triu_indices(n, k=1)
    n_pairs = len(upper_triangle_indices[0])

    # Compute shared bins from all surrogate values across all pairs
    all_surrogate_values = surrogate_matrix[
        :, upper_triangle_indices[0], upper_triangle_indices[1]
    ].flatten()
    bins = np.histogram_bin_edges(all_surrogate_values, bins=n_bins)

    n_rows = int(np.ceil(n_pairs / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()

    for idx in range(n_pairs):
        i = upper_triangle_indices[0][idx]
        j = upper_triangle_indices[1][idx]

        # Get cell names from DataFrame index and columns
        name_i = empirical_matrix.index[i]
        name_j = empirical_matrix.columns[j]

        surrogate_vals = surrogate_matrix[:, i, j]
        counts, _      = np.histogram(surrogate_vals, bins=bins)
        avg_counts     = counts / len(surrogate_vals)
        bin_centres    = (bins[:-1] + bins[1:]) / 2

        axes[idx].bar(bin_centres, avg_counts, width=np.diff(bins), alpha=0.7)
        axes[idx].axvline(empirical_matrix.iloc[i, j], color='red',
                          linestyle='--', label='empirical')
        axes[idx].set_xlabel(xlabel)
        axes[idx].set_title(f'{name_i} vs {name_j}')

    # Hide unused axes
    for idx in range(n_pairs, len(axes)):
        axes[idx].set_visible(False)

    axes[0].legend()
    fig.suptitle(title, fontweight='bold', fontsize=14)
    plt.tight_layout()
    plt.show()

    return fig, axes


# =============================================================================
# Plotting power spectra overview
# =============================================================================

def plot_spectra(freq_axis, psds, dominant_freqs, highpass_cutoff=None,
                 figsize=(14, 5), title='Spectral Overview'):
    """
    Plot a 2-panel spectral overview: PSD per cell and dominant frequency
    distribution.

    Parameters
    ----------
    freq_axis : ndarray, shape (n_freqs,)
        Frequency axis in Hz, as returned by ``compute_spectra``.
    psds : ndarray, shape (n_cells, n_freqs)
        Power spectra, one row per cell, as returned by ``compute_spectra``.
    dominant_freqs : Series, shape (n_cells,)
        Dominant frequency per cell in Hz, as returned by ``compute_spectra``.
    highpass_cutoff : float, optional
        Upper bound of the bandpass filter in Hz. Used to limit the x-axis
        of the PSD panel to the frequency range of interest. Default is None.
        The spectrum will be plotted from 0 to fs/2. 
    figsize : tuple of float, optional
        Figure size in inches (width, height). Default is (14, 5).
    title : str, optional
        Overall figure title. Default is ``'Spectral Overview'``.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure object.
    axes : ndarray of matplotlib.axes.Axes, shape (2,)
        The two created axes: ``axes[0]`` is the PSD panel,
        ``axes[1]`` is the dominant frequency histogram.

    Notes
    -----
    The median dominant frequency is marked on the PSD panel (red dashed
    line) while the mean is marked on the histogram panel. Median is
    preferred on the PSD panel because it is more robust to outlier cells
    with anomalous dominant frequencies.

    A coefficient of variation (CV) summary is printed to stdout as a
    scalar measure of spectral heterogeneity across the population:
    CV ≈ 0 indicates a spectrally homogeneous population; larger values
    indicate heterogeneous dominant frequency distributions.

    See Also
    --------
    compute_spectra : Produces all three array inputs to this function.

    Examples
    --------
    >>> dominant_freqs, freq_axis, psds = compute_spectra(filtered_sorted, sampling_rate)
    >>> fig, axes = plot_spectra(freq_axis, psds, dominant_freqs, highpass_cutoff=1.0)
    """
    n_cells = psds.shape[0]
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # ── Panel 1: PSD per cell ──────────────────────────────────────────────────
    ax = axes[0]
    for psd in psds:
        ax.plot(freq_axis, psd, alpha=0.5, linewidth=0.9)
    ax.axvline(
        np.median(dominant_freqs),
        color='red', linestyle='--', linewidth=1.8,
        label=f'Median dominant freq ({np.median(dominant_freqs):.3f} Hz)',
    )
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Power')
    ax.set_title('PSD per Cell')
    if highpass_cutoff is not None:
        ax.set_xlim([0, highpass_cutoff])
    ax.legend(fontsize=8)

    # ── Panel 2: dominant frequency histogram ─────────────────────────────────
    ax = axes[1]
    ax.hist(dominant_freqs, bins=min(30, n_cells), color='steelblue', edgecolor='white')
    ax.axvline(
        dominant_freqs.mean(),
        color='red', linestyle='--', linewidth=1.8,
        label=f'Mean: {dominant_freqs.mean():.3f} Hz',
    )
    ax.set_xlabel('Dominant Frequency (Hz)')
    ax.set_ylabel('Cell count')
    ax.set_title('Dominant Frequency Distribution')
    ax.legend(fontsize=8)

    fig.suptitle(title, fontweight='bold')
    plt.tight_layout()
    plt.show()

    # Scalar summary of spectral heterogeneity
    cv = dominant_freqs.std() / dominant_freqs.mean()
    print(f"Dominant frequency CV: {cv:.3f}")
    print(f"  (CV \u2248 0 = spectrally homogeneous population, CV >> 0 = heterogeneous)")

    return fig, axes

# =============================================================================
# PLV vs. JS Distance scatter plot
# =============================================================================

def plot_plv_vs_jsd(plv_vector, jsd_vector, p_val_vector=None,
                    r=None, p=None,
                    alpha_threshold=0.05,
                    colors=None,
                    figsize=(7, 6),
                    title='PLV vs. JS Distance',
                    **kwargs):
    """
    Scatter plot of PLV against Jensen-Shannon spectral distance for all
    unique cell pairs, with optional colour coding by PLV significance.

    Parameters
    ----------
    plv_vector : DataFrame, shape (n_pairs, 1)
        PLV values for each unique cell pair, as returned by
        ``correct_p_values``.
    jsd_vector : DataFrame, shape (n_pairs, 1)
        JS Distance values for each unique cell pair, as returned by
        linearising ``compute_spectral_jsd`` output via ``correct_p_values``.
    p_val_vector : DataFrame, shape (n_pairs, 1) or None, optional
        BH-corrected p-values for PLV, as returned by ``correct_p_values``.
        If provided, pairs are colour-coded by significance. If None, all
        pairs are plotted in a single colour. Default is None.
    r : float or None, optional
        Pearson correlation coefficient between JSD and PLV, computed
        externally. If provided alongside ``p``, annotated in the title.
        Default is None.
    p : float or None, optional
        P-value for the Pearson correlation, computed externally.
        Default is None.
    alpha_threshold : float, optional
        Significance threshold applied to ``p_val_vector``. Default is 0.05.
    colors : tuple of 2 str or None, optional
        Colours for (non-significant, significant) groups respectively.
        If None, the first two colours of the active matplotlib cycle are
        used. Default is None.
    figsize : tuple of float, optional
        Figure size in inches (width, height). Default is (7, 6).
    title : str, optional
        Title of the plot. Default is ``'PLV vs. JS Distance'``.
    **kwargs
        Additional keyword arguments forwarded to ``ax.scatter``
        (e.g. ``alpha``, ``s``, ``marker``). Applied equally to both
        groups when ``p_val_vector`` is provided.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure object.
    ax : matplotlib.axes.Axes
        The created axes object.

    Notes
    -----
    Pearson r and p-value are computed externally and passed in rather
    than computed inside this function, keeping statistical computation
    separate from visualisation::

        from scipy.stats import pearsonr
        r, p = pearsonr(jsd_vector.iloc[:, 0], plv_vector.iloc[:, 0])
        fig, ax = plot_plv_vs_jsd(plv_vector, jsd_vector,
                                   p_val_vector=p_vals_vector, r=r, p=p)

    A negative r is expected if significantly phase-locked pairs also
    tend to have similar spectral profiles — high PLV coincides with
    low JS Distance.

    See Also
    --------
    compute_spectral_jsd : Produces the JS Distance matrix.
    correct_p_values     : Linearises matrices into pair vectors.

    Examples
    --------
    >>> # Default matplotlib colours
    >>> fig, ax = plot_plv_vs_jsd(PLV_vals_vector, jsd_vector,
    ...                            p_val_vector=p_vals_vector, r=r, p=p)

    >>> # Custom colours and marker style
    >>> fig, ax = plot_plv_vs_jsd(PLV_vals_vector, jsd_vector,
    ...                            p_val_vector=p_vals_vector, r=r, p=p,
    ...                            colors=('gray', 'crimson'),
    ...                            alpha=0.6, s=40)
    """
    # Fall back to first two colours of the active matplotlib cycle
    cycle      = plt.rcParams['axes.prop_cycle'].by_key()['color']
    c_nonsig, c_sig = colors if colors is not None else (cycle[0], cycle[1])

    jsd_col = jsd_vector.columns[0]
    plv_col = plv_vector.columns[0]

    fig, ax = plt.subplots(figsize=figsize)

    if p_val_vector is not None:
        p_col    = p_val_vector.columns[0]
        sig_mask = p_val_vector[p_col] < alpha_threshold

        ax.scatter(
            jsd_vector.loc[~sig_mask, jsd_col],
            plv_vector.loc[~sig_mask, plv_col],
            color=c_nonsig,
            label='PLV not significant',
            **kwargs,
        )
        ax.scatter(
            jsd_vector.loc[sig_mask, jsd_col],
            plv_vector.loc[sig_mask, plv_col],
            color=c_sig,
            label=f'PLV significant (p<{alpha_threshold})',
            **kwargs,
        )
        ax.legend()
    else:
        ax.scatter(
            jsd_vector[jsd_col], plv_vector[plv_col],
            color=c_nonsig, **kwargs,
        )

    # Annotate title with Pearson r if provided
    if r is not None and p is not None:
        ax.set_title(
            f'{title}\nPearson r = {r:.3f}  (p = {p:.3g}, all pairs)',
            fontweight='bold',
        )
    else:
        ax.set_title(title, fontweight='bold')

    ax.set_xlabel('JS Distance (\u221aJSD)', fontsize=12)
    ax.set_ylabel('PLV', fontsize=12)
    plt.tight_layout()
    plt.show()

    return fig, ax