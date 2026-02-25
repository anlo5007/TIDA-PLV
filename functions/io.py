"""
io.py
=====
Input/output utilities for loading calcium imaging recordings and
associated metadata files.

Routine Listings
----------------
File selection
    select_file     : Open a tkinter file dialog to select any file.

Recording loading
    load_recording  : Parse a traces CSV into time, data, labels and sampling rate.

Metadata location
    locate_props    : Infer or interactively locate the cell properties CSV.
    locate_map      : Infer or interactively locate the cell map image.

Metadata loading
    load_props      : Load cell properties CSV into a DataFrame.
    load_map        : Load cell map image into a numpy array.

Notes
-----
All file dialogs open with ``initialdir`` set to the recording folder so
the user is never more than one click away from the correct file.

Typical usage::

    from io import select_file, load_recording, locate_props, locate_map
    from io import load_props, load_map

    recording_path = select_file()
    time, data, labels, sampling_rate = load_recording(recording_path)

    props_path = locate_props(recording_path)
    map_path   = locate_map(recording_path)

    props   = load_props(props_path)
    map_img = load_map(map_path)
"""

from logging.handlers import WatchedFileHandler
from pyclbr import Class
import tkinter as tk
from tkinter import filedialog
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from functions.utils import detect_time_jumps
from functions.plots import plot_traces

import warnings

# =============================================================================
# Recording is a class that facilitate importing files
# =============================================================================

class Recording:
    """
    Container for a calcium imaging recording and its associated metadata.

    Loads traces, cell properties, and map image from a recording CSV file,
    and provides interactive interval selection methods.

    Parameters
    ----------
    path : str or Path
        Path to the traces CSV file.
    keep_status : str, tuple of str, or None, optional
        Cell status labels to retain (e.g. ``'accepted'`` or
        ``(' accepted', ' undecided')``). None keeps all cells.
        Default is None.

    Attributes
    ----------
    path : Path
        Path to the traces CSV file.
    keep_status : tuple of str or None
        Normalised status filter used when loading traces and props.
    time : ndarray, shape (n_timepoints,)
        Time axis in seconds. Updated in-place by ``select_interval``.
    data : DataFrame, shape (n_timepoints, n_cells)
        Fluorescence traces. Updated in-place by ``select_interval``.
    status : Series
        Status label for each retained cell.
    cell_names : Index
        Original column names from the CSV for the retained cells.
    sampling_rate : float
        Sampling rate in Hz.
    props : DataFrame or None
        Cell properties table (CentroidX, CentroidY, …), or None if no
        props file is found.
    map_img : ndarray or None
        Cell map image array, or None if no PNG is found.

    Examples
    --------
    >>> # Load from a known path
    >>> rec = Recording("recording.csv", keep_status="accepted").load()

    >>> # Load interactively via file dialog
    >>> rec = Recording.from_dialog(keep_status="accepted")
    >>> rec.select_interval(method='single_click', window=100)
    """

    def __init__ (self, path, keep_status=None):

        self.path = Path(path)

        if keep_status is None:
            self.keep_status = None
        elif isinstance(keep_status, tuple):
            self.keep_status = keep_status
        elif isinstance(keep_status, str):
            self.keep_status = (keep_status,) #wrap single strings in a tuple 
        else:
            self.keep_status = tuple(keep_status)

    @classmethod
    def from_dialog(cls, keep_status=None):
        """
        Create a Recording by selecting a CSV file via a file dialog.

        Opens a tkinter dialog for the user to choose the recording CSV,
        then calls ``load`` automatically. Returns None if the dialog is
        cancelled.

        Parameters
        ----------
        keep_status : str, tuple of str, or None, optional
            Passed directly to ``__init__``. Default is None (keep all cells).

        Returns
        -------
        rec : Recording or None
            Fully loaded Recording instance, or None if no file was selected.
        """

        path = select_file(title='Select recording CSV')
        if path is None:
            return None
        return cls(path, keep_status=keep_status).load()

    def load(self):
        """
        Load the recording, props, and map image from disk.

        Populates ``time``, ``data``, ``status``, ``cell_names``,
        ``sampling_rate``, ``props``, and ``map_img``. Props and map are
        located automatically and set to None if not found.

        Returns
        -------
        self : Recording
            Returns self to allow method chaining::

                rec = Recording(path).load()
        """
        self.time, self.data, self.status, self.cell_names, self.sampling_rate = \
            load_recording(self.path, keep_status=self.keep_status)

        props_path   = locate_props(self.path, require=False)
        self.props   = load_props(props_path, keep_status=self.keep_status) if props_path else None

        map_path     = locate_map(self.path, require=False)
        self.map_img = load_map(map_path) if map_path else None

        return self
    
    def select_interval(self, method='single_click', **kwargs):
        """
        Interactively trim the recording to a time interval.

        Updates ``self.time`` and ``self.data`` in-place. The time axis
        is reset to start at 0 after selection.

        Parameters
        ----------
        method : {'single_click', 'manual', 'double_click', 'block'}, optional
            Interval selection method. Default is ``'single_click'``.

            - ``'single_click'``  : click once to set interval start; duration
              controlled by ``window`` kwarg (default 100 s).
            - ``'manual'``        : pass ``start_time`` and ``end_time`` kwargs
              directly, no interaction required.
            - ``'double_click'``  : click twice to set start and end.
            - ``'block'``         : click within a contiguous block defined by
              detected time jumps.

        **kwargs
            Forwarded to the underlying selection function. Common kwargs:

            - ``window`` (float) - interval duration in seconds (single_click only).
            - ``start_time``, ``end_time`` (float) - interval bounds (manual only).
            - ``plot_selection`` (bool) - whether to plot the selected interval.
            - ``plot_original`` (bool) - whether to plot the full trace first (manual only).
            - ``reset_index`` (bool) - whether to reset the index of the returned interval_data DataFrame. Default is True.

        Returns
        -------
        self : Recording
            Returns self to allow method chaining::

                rec.select_interval(method='manual', start_time=30, end_time=90)

        """

        if method == 'single_click':
            self.time, self.data = select_click_interval(
                self.time, self.data, status=self.status, **kwargs)
        elif method == 'manual':
            self.time, self.data = select_interval_manual(
                self.time, self.data, status=self.status, **kwargs)
        elif method == 'double_click':
            self.time, self.data = select_double_click(
                self.time, self.data, status=self.status, **kwargs)
        elif method == 'block':
            self.time, self.data = select_block(
                self.time, self.data, status=self.status, **kwargs)
        else:
            raise ValueError(f"Unknown method '{method}'. "
                            f"Choose from: 'click', 'manual', 'double_click', 'block'")
        return self

# =============================================================================
# File selection
# =============================================================================

def select_file(title="Select a file", filetypes=None, initialdir=None):
    """
    Open a tkinter file dialog and return the selected file path.

    Parameters
    ----------
    title : str, optional
        Title displayed on the dialog window. Default is 'Select a file'.
    filetypes : list of tuple, optional
        List of (label, pattern) pairs to filter files shown in the dialog.
        Default is ``[("CSV files", "*.csv")]``.
    initialdir : str or Path, optional
        Directory the dialog opens in. Default is None (system default).

    Returns
    -------
    path : Path or None
        Path to the selected file, or None if the user cancelled the dialog.

    Examples
    --------
    >>> path = select_file(title="Select recording", filetypes=[("CSV", "*.csv")])
    >>> if path is not None:
    ...     print(f"Selected: {path}")
    """
    if filetypes is None:
        filetypes = [("CSV files", "*.csv")]

    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)   # bring dialog to front
    root.update()                        # ensure it renders before dialog opens

    selected = filedialog.askopenfilename(
        parent=root,                     # bind dialog to root window
        title=title,
        filetypes=filetypes,
        initialdir=str(initialdir) if initialdir is not None else None,
    )
    root.destroy()

    if selected:
        print(f"Selected file: {selected}")
        return Path(selected)

    print("No file selected.")
    return None


# =============================================================================
# Recording loading
# =============================================================================

def load_recording(path, keep_status=[' accepted', ' undecided']):
    """
    Parse a calcium imaging traces CSV into its component arrays.

    Expected CSV format
    -------------------
    - Row 0  : header row — first column is ignored, remaining columns contain
               cell status strings (e.g. ' accepted', ' undecided', ' rejected').
    - Row 1+ : data rows — first column is time in seconds, remaining columns
               are fluorescence traces (one column per cell).

    Parameters
    ----------
    path : str or Path
        Path to the traces CSV file.
    keep_status : tuple of str, optional
        Cell status labels to retain. Must match the strings in the CSV
        header row exactly, including leading spaces.
        Default is ``(' accepted', ' undecided')``.
        
        Common values:
        
        - ``(' accepted',)``             — accepted cells only
        - ``(' accepted', ' undecided')`` — accepted and undecided (default)
        - ``(' accepted', ' undecided', ' rejected')`` — all cells

    Returns
    -------
    time : ndarray, shape (n_timepoints,)
        Time axis in seconds.
    data : DataFrame, shape (n_timepoints, n_cells)
        Fluorescence traces for cells matching ``keep_status``.
        Columns are the original column indices from the CSV.
    status : Series, shape (n_cells,)
        Status labels for the retained cells, indexed by the same column indices as `data`.
    cell_names : Index
        Original column names from the CSV for the retained cells.
    sampling_rate : float
        Sampling rate in Hz, inferred as ``1 / (time[1] - time[0])``.

    Raises
    ------
    FileNotFoundError
        If the file at ``path`` does not exist.
    ValueError
        If no cells matching ``keep_status`` are found, or if
        ``keep_status`` contains labels not present in the file
        (warns but does not raise).

    Notes
    -----
    Status strings in the CSV typically have a leading space
    (e.g. ' accepted' not 'accepted'). If no cells are found, print
    the unique status values present in the file to help diagnose
    label mismatches.

    Sampling rate is inferred from the first two timepoints and assumes
    uniform sampling throughout the recording.

    Examples
    --------
    >>> # Default: accepted + undecided
    >>> time, data, labels, fs = load_recording("recording.csv")

    >>> # Accepted only
    >>> time, data, labels, fs = load_recording("recording.csv",
    ...                                          keep_status=(' accepted',))

    >>> # All cells regardless of status
    >>> time, data, labels, fs = load_recording("recording.csv",
    ...                                          keep_status=None)
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Recording file not found: {path}")

    traces       = pd.read_csv(path)
    status_cells = traces.iloc[0, 1:].str.strip()  # assuming status is in the first row
    data         = traces.iloc[1:, 1:].astype(float)
    time         = traces.iloc[1:, 0].astype(float).values

    # keep_status=None means keep everything
    if keep_status is not None:
            if isinstance(keep_status, str):
                keep_status = (keep_status,)
            else:
                keep_status = tuple(keep_status)

            keep_mask = status_cells.isin(keep_status)

            # Warn about any requested labels not present in the file
            found_labels   = set(status_cells.unique())
            missing_labels = set(keep_status) - found_labels
            if missing_labels:
                print(f"Warning: requested status labels not found in file: {missing_labels}")
                print(f"  Available labels: {found_labels}")
    else:
        keep_mask = pd.Series([True] * len(status_cells), index=status_cells.index)

    data   = data.loc[:, keep_mask]
    all_status = status_cells[keep_mask] 
    cell_names = data.columns.astype(str)

    if data.shape[1] == 0:
        unique_status = set(status_cells.unique())
        raise ValueError(
            f"No cells found matching keep_status={keep_status}.\n"
            f"Status labels present in file: {unique_status}\n"
            f"Check for leading/trailing spaces in label strings."
        )

    sampling_rate = 1.0 / (time[1] - time[0])

    # Summary per status
    for status in (keep_status if keep_status is not None else status_cells.unique()):
        count = int((status_cells == status).sum())
        if count > 0:
            print(f"  {status.strip()}: {count} cells")
    print(f"Loaded {data.shape[1]} cells total at {sampling_rate:.4f} Hz")

    return time, data, all_status, cell_names, sampling_rate

# =============================================================================
# Metadata location
# =============================================================================

def locate_props(recording_path, require=True):
    """
    Infer the cell properties CSV path from the recording path.

    Tries automatically first, assuming the convention::

        <recording_stem>-props.csv

    in the same folder as the recording. Opens a file dialog if the file
    is not found at the expected location.

    Parameters
    ----------
    recording_path : str or Path
        Path to the traces CSV file.
    require : bool, optional
        If True and the user cancels the dialog, raises FileNotFoundError.
        If False, returns None silently. Default is True.

    Returns
    -------
    props_path : Path or None
        Path to the props CSV, or None if not found and ``require=False``.

    Raises
    ------
    FileNotFoundError
        If ``require=True`` and no file is found or selected.

    Notes
    -----
    The file dialog opens in the recording folder (``initialdir``), so the
    user only needs to click the correct file rather than navigate to it.

    Examples
    --------
    >>> props_path = locate_props("8Nov2024_sl#4_v1_20x_fp1.csv")
    >>> # tries: 8Nov2024_sl#4_v1_20x_fp1-props.csv automatically
    """
    recording_path = Path(recording_path)
    props_path     = recording_path.parent / (recording_path.stem + '-props.csv')

    if not props_path.exists():
        print(f"Props file not found at expected location:\n  {props_path}")
        print("Please select the props file manually.")
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)   # bring dialog to front
        root.update()                        # ensure it renders before dialog opens

        selected = filedialog.askopenfilename(
            title="Select props CSV file",
            initialdir=str(recording_path.parent),
            filetypes=[("CSV files", "*.csv")],
        )
        root.destroy()
        props_path = Path(selected) if selected else None

    if require and props_path is None:
        raise FileNotFoundError(
            "No props file found or selected. "
            "Expected: " + str(recording_path.parent / (recording_path.stem + '-props.csv'))
        )

    if props_path is not None:
        print(f"Props file: {props_path.name}")

    return props_path


def locate_map(recording_path, require=False):
    """
    Locate the cell map image for a recording.

    Looks for PNG files in the recording folder:

    - If exactly one PNG is found, use it automatically.
    - If more than one PNG is found, ask the user to choose via dialog.
    - If none are found, print a warning and return None.

    Parameters
    ----------
    recording_path : str or Path
        Path to the traces CSV file.
    require : bool, optional
        If True, raises FileNotFoundError if no image is found or selected.
        Default is False.

    Returns
    -------
    map_path : Path or None
        Path to the map image file, or None if not found and ``require=False``.

    Raises
    ------
    FileNotFoundError
        If ``require=True`` and no PNG is found or selected.

    Examples
    --------
    >>> map_path = locate_map("8Nov2024_sl#4_v1_20x_fp1.csv")
    """
    recording_path = Path(recording_path)
    png_files = sorted(recording_path.parent.glob('*.png'))

    if len(png_files) == 0:
        print(f"Warning: no PNG files found in {recording_path.parent}")
        if require:
            raise FileNotFoundError(
                f"No map image found for recording: {recording_path.stem}"
            )
        return None

    if len(png_files) == 1:
        print(f"Map image found: {png_files[0].name}")
        return png_files[0]

    # More than one PNG — ask the user
    print(f"Multiple PNG files found in {recording_path.parent}:")
    for i, f in enumerate(png_files):
        print(f"  [{i}] {f.name}")

    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)   # bring dialog to front
    root.update()                        # ensure it renders before dialog opens
    
    selected = filedialog.askopenfilename(
        title="Multiple PNG files found — select the map image",
        initialdir=str(recording_path.parent),
        filetypes=[("PNG files", "*.png")],
    )
    root.destroy()
    map_path = Path(selected) if selected else None

    if require and map_path is None:
        raise FileNotFoundError(
            f"No map image selected for recording: {recording_path.stem}"
        )

    if map_path is not None:
        print(f"Map image selected: {map_path.name}")

    return map_path

# =============================================================================
# Metadata loading
# =============================================================================

def load_props(props_path, keep_status=[' accepted', ' undecided']):
    """
    Load cell properties from a CSV file into a DataFrame.

    Parameters
    ----------
    props_path : str or Path
        Path to the props CSV file, typically located via ``locate_props``.

    Returns
    -------
    props : DataFrame
        Cell properties table. Expected to contain at least the columns
        ``CentroidX`` and ``CentroidY`` for spatial analysis. Row index
        corresponds to cell ID.

    Raises
    ------
    FileNotFoundError
        If the file at ``props_path`` does not exist.
    ValueError
        If the required columns ``CentroidX`` or ``CentroidY`` are missing.

    Examples
    --------
    >>> props = load_props(locate_props(recording_path))
    >>> x, y = props['CentroidX'].values, props['CentroidY'].values
    """
    props_path = Path(props_path)
    if not props_path.exists():
        raise FileNotFoundError(f"Props file not found: {props_path}")

    props = pd.read_csv(props_path)

    status_cells = props.iloc[:, 1].str.strip() # assuming status is in the second column

    # keep_status=None means keep everything
    if keep_status is not None:
            if isinstance(keep_status, str):
                keep_status = (keep_status,)
            else:
                keep_status = tuple(keep_status)

            keep_mask = status_cells.isin(keep_status)

            # Warn about any requested labels not present in the file
            found_labels   = set(status_cells.unique())
            missing_labels = set(keep_status) - found_labels
            if missing_labels:
                print(f"Warning: requested status labels not found in file: {missing_labels}")
                print(f"  Available labels: {found_labels}")
    else:
        keep_mask = pd.Series([True] * len(status_cells), index=status_cells.index)

    props   = props.loc[keep_mask, :]

    required_cols = {'CentroidX', 'CentroidY'}
    missing = required_cols - set(props.columns)
    if missing:
        raise ValueError(
            f"Props file is missing required columns: {missing}. "
            f"Found columns: {list(props.columns)}"
        )

    print(f"Loaded props: {len(props)} cells")
    return props


def load_map(map_path):
    """
    Load a cell map image into a numpy array.

    Parameters
    ----------
    map_path : str or Path
        Path to the map image file, typically located via ``locate_map``.
        Supported formats: PNG, TIFF, JPEG.

    Returns
    -------
    map_img : ndarray, shape (height, width) or (height, width, channels)
        Image array as returned by ``matplotlib.pyplot.imread``.
        Grayscale images have shape ``(H, W)``, RGB images ``(H, W, 3)``,
        RGBA images ``(H, W, 4)``.

    Raises
    ------
    FileNotFoundError
        If the file at ``map_path`` does not exist.

    Notes
    -----
    Uses ``matplotlib.pyplot.imread`` which normalises pixel values to
    ``[0, 1]`` for PNG files and ``[0, 255]`` for JPEG/TIFF files.
    Both are handled correctly by ``ax.imshow``.

    Examples
    --------
    >>> map_img = load_map(locate_map(recording_path))
    >>> plt.imshow(map_img, cmap='gray')
    """
    map_path = Path(map_path)
    if not map_path.exists():
        raise FileNotFoundError(f"Map image not found: {map_path}")

    map_img = plt.imread(str(map_path))
    print(f"Loaded map image: {map_path.name} {map_img.shape}")
    return map_img

# =============================================================================
# Interval trace selection 
# =============================================================================

def select_interval_manual(time, data, start_time=0, end_time=None, status=None, plot_original=True, plot_selection=True, mark_jumps=True, reset_index=True):
    """
    Extract a time interval from the recording based on manual start/end times.

    Parameters
    ----------
    time : ndarray, shape (n_timepoints,)
        Time axis in seconds from the original recording.
    data : DataFrame, shape (n_timepoints, n_cells)
        Fluorescence traces for each cell.
    start_time : float, optional
        Start time of the interval in seconds. Default is 0 (beginning of recording).
    end_time : float or None, optional
        End time of the interval in seconds. If None, uses the end of the recording.
        Default is None.
    status : Series or list of str, optional
        Cell status labels for color-coding traces in plots. If None, all traces
        use the default matplotlib color cycle. Default is None.
    plot_original : bool, optional
        If True, plot the full recording with the selected interval highlighted
        by red dashed vertical lines. Default is True.
    plot_selection : bool, optional
        If True, plot only the selected interval. Default is True.
    mark_jumps : bool, optional
        If True, detect and mark time jumps on the plot with vertical dashed lines.
        Default is True.
    reset_index : bool, optional
        If True, reset the index of the returned interval_data DataFrame. Default is True.

    Returns
    -------
    interval_time : ndarray, shape (n_interval_timepoints,)
        Time axis for the selected interval, reset to start at 0.
    interval_data : DataFrame, shape (n_interval_timepoints, n_cells)
        Fluorescence traces for the selected interval.

    Notes
    -----

    The interval time axis is reset to start at 0 for convenience in downstream
    analysis.

    Examples
    --------
    >>> # Select 30-60s interval
    >>> t_int, data_int = select_interval_manual(
    ...     time, data, start_time=30, end_time=60, status=status
    ... )
    >>> print(f"Selected: {t_int[0]:.2f}s to {t_int[-1]:.2f}s")
    >>> print(f"Interval duration: {t_int[-1]:.2f}s")

    >>> # Select from 10s to end, no plots
    >>> t_int, data_int = select_interval_manual(
    ...     time, data, start_time=10, plot_original=False, plot_selection=False
    ... )
    """

    if mark_jumps:
        time, jump_times = detect_time_jumps(time)

    indx_start = np.searchsorted(time, start_time)

    if end_time is not None:
        indx_end = np.searchsorted(time, end_time)
    else:
        indx_end = len(time)
        end_time = time[-1]

    if plot_original:   
        fig, axes = plot_traces(time, data, status=status, title="Full traces")
        for ax in axes:
            ax.axvline(start_time, color='red', linestyle='--')
            ax.axvline(end_time, color='red', linestyle='--')
            if mark_jumps:
                for jt in jump_times:
                    ax.axvline(jt, color='black', linestyle='--')

        axes[-1].lines[1].set_label('selected interval')
        if mark_jumps and len(jump_times) > 0:
            axes[-1].lines[-1].set_label('section boundary')

        fig.legend(loc='upper right')

        plt.show()

    if plot_selection:
        fig, axes = plot_traces(time[indx_start:indx_end], data.iloc[indx_start:indx_end, :], status=status, title="Selected interval")
        plt.show()

    interval_data = data.iloc[indx_start:indx_end, :]
    
    if reset_index:
        interval_data = interval_data.reset_index(drop=True)

    interval_time = time[indx_start:indx_end] - time[indx_start]  # reset time to start at 0 for the interval

    return interval_time, interval_data

def select_click_interval (time, data, window=100, status=None, plot_selection=True, mark_jumps=True, reset_index=True):
    """
    Extract a time interval from the recording by clicking on a plot.

    The user clicks once to select a center point. The function automatically
    creates an interval of specified width starting from that point.

    Parameters
    ----------
    time : ndarray, shape (n_timepoints,)
        Time axis in seconds from the original recording.
    data : DataFrame, shape (n_timepoints, n_cells)
        Fluorescence traces for each cell.
    window : float, optional
        Duration of the interval in seconds, starting from the clicked time.
        Default is 100 seconds.
    status : Series or list of str, optional
        Cell status labels for color-coding traces in plots. If None, all traces
        use the default matplotlib color cycle. Default is None.
    plot_selection : bool, optional
        If True, plot the selected interval after extraction. Default is True.
    mark_jumps : bool, optional
        If True, detect and mark time jumps on the plot with vertical dashed lines.
        Default is True.
    reset_index : bool, optional
        If True, reset the index of the returned interval_data DataFrame. Default is True.

    Returns
    -------
    interval_time : ndarray, shape (n_interval_timepoints,)
        Time axis for the selected interval, reset to start at 0.
    interval_data : DataFrame, shape (n_interval_timepoints, n_cells)
        Fluorescence traces for the selected interval.

    Notes
    -----
    Requires an interactive matplotlib backend (e.g., `%matplotlib qt` or
    `%matplotlib notebook` in Jupyter). The default inline backend will not work.

    The interval extends from the clicked time to `clicked_time + window`.
    If the interval would extend beyond the end of the recording, a warning
    is issued and the interval is truncated to fit the available data.

    The returned time axis is reset to start at 0 for convenience in downstream
    analysis.

    Examples
    --------
    >>> # Select a 100s interval starting from a clicked point
    >>> t_int, data_int = select_click_interval(time, data, window=100, status=status)

    >>> # Select a 30s interval without plotting the result
    >>> t_int, data_int = select_click_interval(time, data, window=30, plot_selection=False)
    """

    if mark_jumps:
        time, jump_times = detect_time_jumps(time)

    fig, axes = plot_traces(time, data, status=status, title="Click to select interval")
    if mark_jumps and len(jump_times) > 0:
        for jt in jump_times:
            for ax in axes:
                ax.axvline(jt, color='black', linestyle='--')
        axes[-1].lines[-1].set_label('section boundary')
        fig.legend(loc='upper right')

    print("Please click one point on the plot to select the interval.")
    clicks = plt.ginput(1)  # This displays the plot AND waits for input

    if len(clicks) < 1:
        print("No clicks detected.")
        plt.close(fig)
        return time, data

    click_time = clicks[0][0]
    print(f"Clicked at time: {click_time:.2f}s")

    # Update the plot with the selection
    for ax in axes:
        ax.axvline(click_time, color='red', linestyle='--')
    fig.canvas.draw()

    plt.show()  # Show the updated plot with the red line
    plt.pause(0.5)  # Pause to ensure the plot updates before proceeding
    plt.close(fig) # Close the plot after selection

    if click_time + window > time[-1]:
        warnings.warn("Selected interval extends beyond the end of the recording. Adjusting to fit within available data.")

    # Define the interval around the clicked time
    start_indx = np.searchsorted(time, click_time)
    end_indx = np.searchsorted(time, click_time + window)

    interval_data = data.iloc[start_indx:end_indx, :]

    if reset_index:
        interval_data = interval_data.reset_index(drop=True)

    interval_time = time[start_indx:end_indx] - time[start_indx]  # reset time to start at 0 for the interval

    if plot_selection:
        fig, axes = plot_traces(interval_time, interval_data, status=status, title="Selected interval")
        plt.show()

    return interval_time, interval_data

def select_double_click(time, data, status=None, plot_selection=True, mark_jumps=True, reset_index=True):
    """
    Extract a time interval from the recording by double-clicking on a plot.

    The user double-clicks to select a center point. The function automatically
    creates an interval of specified width starting from that point.

    Parameters
    ----------
    time : ndarray, shape (n_timepoints,)
        Time axis in seconds from the original recording.
    data : DataFrame, shape (n_timepoints, n_cells)
        Fluorescence traces for each cell.
    status : Series or list of str, optional
        Cell status labels for color-coding traces in plots. If None, all traces
        use the default matplotlib color cycle. Default is None.
    plot_selection : bool, optional
        If True, plot the selected interval after extraction. Default is True.  
    mark_jumps : bool, optional
        If True, detect and mark time jumps on the plot with vertical dashed lines.
        Default is True.
    reset_index : bool, optional
        If True, reset the index of the returned interval_data DataFrame. Default is True.

    Returns
    -------
    interval_time : ndarray, shape (n_interval_timepoints,)
        Time axis for the selected interval, reset to start at 0.   
    interval_data : DataFrame, shape (n_interval_timepoints, n_cells)
        Fluorescence traces for the selected interval.

    Notes
    -----   
    Requires an interactive matplotlib backend (e.g., `%matplotlib qt` or `%matplotlib notebook` in Jupyter). The default inline backend will not work.
        The user must double-click twice to define the start and end of the interval.
        The returned time axis is reset to start at 0 for convenience in downstream
        analysis.

    Examples
    --------
    >>> # Select an interval by double-clicking start and end points
    >>> t_int, data_int = select_double_click(time, data, status=status)
    >>> # Select an interval without plotting the result
    >>> t_int, data_int = select_double_click(time, data, status=status, plot_selection=False)
    """
    if mark_jumps:
        time, jump_times = detect_time_jumps(time)

    fig, axes = plot_traces(time, data, status=status, title="Double-click to select interval")
    if mark_jumps and len(jump_times) > 0:
        for jt in jump_times:
            for ax in axes:
                ax.axvline(jt, color='black', linestyle='--')
        axes[-1].lines[-1].set_label('section boundary')
        fig.legend(loc='upper right')

    print("Please click two times on the plot to select the interval.")
    clicks = plt.ginput(2)  # Wait indefinitely for a double-click

    if len(clicks) < 2:
        print("Less than two clicks detected. Selecting the full recording.")
        plt.close(fig)
        return time, data
   
    click_left = clicks[0][0]
    click_right = clicks[1][0]

    if click_left > click_right:
        click_left, click_right = click_right, click_left  # Swap to ensure left < right
        warnings.warn("Clicks were in reverse order. Swapping to create a valid interval.")
    
    print(f"Clicked at times: {click_left:.2f}s and {click_right:.2f}s")

    # Update the plot with the selection
    for ax in axes:
        ax.axvline(click_left, color='red', linestyle='--')
        ax.axvline(click_right, color='red', linestyle='--')
    fig.canvas.draw()

    plt.show()  # Show the updated plot with the red line
    plt.pause(0.5)  # Pause to ensure the plot updates before proceeding
    plt.close(fig) # Close the plot after selection

    start_indx = np.searchsorted(time, click_left)
    end_indx = np.searchsorted(time, click_right)

    interval_data = data.iloc[start_indx:end_indx, :]

    if reset_index:
        interval_data = interval_data.reset_index(drop=True) 

    interval_time = time[start_indx:end_indx] - time[start_indx]  # reset time to start at 0 for the interval

    if plot_selection:
        fig, axes = plot_traces(interval_time, interval_data, status=status, title="Selected interval")
        plt.show()

    return interval_time, interval_data

def select_block(time, data, status=None, plot_selection=True, reset_index=True):
    """
    Extract a contiguous time block from a concatenated recording by clicking.

    Detects time jumps (concatenation points) in the recording and allows the
    user to select one contiguous block by clicking on it. Useful for analyzing
    individual recording segments from concatenated datasets.

    Parameters
    ----------
    time : ndarray, shape (n_timepoints,)
        Time axis in seconds from the original recording.
    data : DataFrame, shape (n_timepoints, n_cells)
        Fluorescence traces for each cell.
    status : Series or list of str, optional
        Cell status labels for color-coding traces in plots. If None, all traces
        use the default matplotlib color cycle. Default is None.
    plot_selection : bool, optional
        If True, plot the selected block after extraction. Default is True.
    reset_index : bool, optional
        If True, reset the index of the returned block_data DataFrame. Default is True.

    Returns
    -------
    block_time : ndarray, shape (n_block_timepoints,)
        Time axis for the selected block, reset to start at 0.
    block_data : DataFrame, shape (n_block_timepoints, n_cells)
        Fluorescence traces for the selected block.

    Notes
    -----
    Requires an interactive matplotlib backend (e.g., `%matplotlib qt` or
    `%matplotlib notebook` in Jupyter). The default inline backend will not work.

    Time jumps are detected automatically using `detect_time_jumps`. If no jumps
    are found, the full recording is returned.

    Block boundaries are marked with black dashed vertical lines on the plot.
    Click anywhere within a block to select it.

    The returned time axis is reset to start at 0 for convenience in downstream
    analysis.

    Examples
    --------
    >>> # Select a block interactively
    >>> block_time, block_data = select_block(time, data, status=status)
    Selected block 2: 120.50s → 180.30s

    >>> # Select without plotting the result
    >>> block_time, block_data = select_block(time, data, plot_selection=False)
    """
    # Detect time jumps
    time, jump_times = detect_time_jumps(time)

    if len(jump_times) == 0:
        print("No time jumps detected. Returning full recording.")
        plot_traces(time, data, status=status, title="Full recording (no jumps detected)")
        plt.show()
        return time, data

    # Plot with jump markers
    fig, axes = plot_traces(time, data, status=status, title="Click to select block")
    for jt in jump_times:
        for ax in axes:
            ax.axvline(jt, color='black', linestyle='--', linewidth=1.5, alpha=0.7)

    axes[-1].lines[-1].set_label('Section boundary')
    axes[-1].legend(loc='upper right')

    print(f"Detected {len(jump_times)} concatenation point(s).")
    print("Click anywhere within a block to select it.")

    # Wait for user click
    clicks = plt.ginput(1)
    plt.close(fig)

    if len(clicks) < 1:
        print("No click detected. Returning full recording.")
        return time, data

    click_time = clicks[0][0]

    # Define block boundaries
    boundaries = np.concatenate(([time[0]], jump_times, [time[-1]]))

    # Find which block contains the click
    block_idx = np.searchsorted(boundaries, click_time, side='right') - 1

    if block_idx < 0 or block_idx >= len(boundaries) - 1:
        print("Click outside valid range. Returning full recording.")
        return time, data

    t_start = boundaries[block_idx]
    t_end = boundaries[block_idx + 1]

    print(f"Selected block {block_idx + 1}: {t_start:.2f}s → {t_end:.2f}s")

    # Extract block
    mask = (time >= t_start) & (time < t_end)
    block_time = time[mask] - time[mask][0]  # Reset to 0
    block_data = data.loc[mask, :]

    if reset_index:
        block_data = block_data.reset_index(drop=True)

    # Plot selected block
    if plot_selection:
        plot_traces(block_time, block_data, status=status,
                   title=f"Selected block {block_idx + 1}")
        plt.show()

    return block_time, block_data


        
        