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

def plot_plv_matrix(plv_matrix, title='PLV Matrix', figsize=(10, 10), cmap='viridis', labels=None):
    """
    Plot a Phase Locking Value (PLV) matrix as a heatmap.

    Parameters
    ----------
    plv_matrix : array_like, shape (n_traces, n_traces)
        Square matrix of PLV values between pairs of traces.
    title : str, optional
        Title of the plot. Default is 'PLV Matrix'.
    figsize : tuple, optional
        Figure size in inches (width, height). Default is (10, 10).
    cmap : str or Colormap, optional
        Colormap to use for the heatmap. Default is 'viridis'.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure object.
    ax : matplotlib.axes.Axes
        The created axes object.

    Examples
    --------
    >>> fig, ax = plot_plv_matrix(plv_matrix)
    """

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(plv_matrix, ax=ax, cmap=cmap, square=True,)
    ax.set_title(title, fontweight='bold')
    ax.set_xlabel('Trace Index')
    ax.set_ylabel('Trace Index')
    return fig, ax