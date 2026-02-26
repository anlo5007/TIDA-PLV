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