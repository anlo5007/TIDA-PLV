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