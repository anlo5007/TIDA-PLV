# %%
import pandas as pd
import numpy as np

from functools import partial

import matplotlib.pyplot as plt
import seaborn as sns

from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform

from functions.utils import bandpass, get_phase, dominant_frequency, plv_einsum, permutation_test, circular_shift_surrogate, phase_randomisation_surrogate, correct_p_values, compute_pairwise_distances
from functions.io import Recording, save_open_figs
from functions.plots import plot_plv_matrix, plot_plv_map

# %%
# ===================================================
# Define parameters
# ===================================================

# Define which cells labels to keep
keep_cells = ("accepted","undecided") # change at will, if you don´t pass keep_status or you leave it empty the analysis will proceed on all the cells

# Define filtering parameters (in Hz)
lowpass_cutoff = 0.01
highpass_cutoff = 1 

# %%
# Select a .csv file to analyze
recording = Recording.from_dialog(keep_status=keep_cells)

# if you want to manually select a .csv here is a pipeline

# recording_path = "...path to my .csv"
# recording = Recording(recording_path, keep_status=keep_cells)

# %%
# Setting Matplotlib to interactive mode for interval selection
%matplotlib qt
# Using slect_interval method for interval selection
recording.select_interval(method='single_click', window=300)  # available methods: 'single_click', 'manual', 'double_click', 'block_selection', select the duration appropriately

# %%
# Break down the recording into traces and apply bandpass filter
data = recording.data
time = recording.time
sampling_rate = recording.sampling_rate

# %%
# Apply bandpass filter to each trace in the recording
filtered = data.apply(lambda x : bandpass(x, sampling_rate,lowpass_cutoff, highpass_cutoff))

# %%
# Obtaining the phase for each trace and calculating the PLV matrix
filtered_phase = filtered.apply(get_phase)

# %%
# Calculate PLV matrix using the optimized einsum function
PLV, phase_diff = plv_einsum(filtered_phase.T.to_numpy()) #Transpose to have shape (n_traces, n_timepoints) for plv_einsum
PLV_df = pd.DataFrame(PLV, index=recording.cell_names, columns=recording.cell_names) #convert to DataFrame for better visualization and handling of labels
phase_diff_df = pd.DataFrame(phase_diff, index=recording.cell_names, columns=recording.cell_names) 

# %%
# Convert similarity (PLV) to distance
dist = 1-PLV_df
# Hierarchical clustering (Ward works well)
Z = linkage(squareform(dist), method='ward')
# Get order of leaves
idx = leaves_list(Z)

PLV_sorted = PLV_df.iloc[idx, idx]  #sort the PLV matrix according to the hierarchical clustering order
phase_diff_sorted = phase_diff_df.iloc[idx,idx]

# %%
# Extract dominant frequencies for each trace
dominant_freqs = filtered.apply(lambda x: dominant_frequency(x, sampling_rate, hann=False, poly_order=4)[0])

# Find the slowest dominant frequency and its index
slowest_dominant_freq = dominant_freqs.min()
slowest_index = dominant_freqs.idxmin()
print(f"Slowest dominant frequency: {slowest_dominant_freq} Hz")

# %%
min_interval = int((1/slowest_dominant_freq)*sampling_rate) # At least one full cycle
max_interval = int(filtered_phase.shape[0]-min_interval) # Use full length of the signal

# %%
filtered_sorted = filtered.iloc[:,idx]  

_, p_vals, _, _ = permutation_test  (
                                    filtered_sorted.T.to_numpy(), 
                                    PLV_sorted.to_numpy(),
                                    phase_diff_sorted.to_numpy(),
                                    n_permutations=1000, 
                                    plot_surrogates=0, 
                                    surrogate_fn=circular_shift_surrogate, 
                                    min_interval=min_interval, 
                                    max_interval=max_interval
                                    )

# %%
# Apply FDR correction to the p-values and generate a matrix and a sorted array of p-values
p_val_matrix, p_vals_vector = correct_p_values(p_vals, FDR_correction=True, cell_labels=filtered_sorted.columns.values, method='bh')
_, p_vals_uncorrected = correct_p_values(p_vals, FDR_correction=False, cell_labels=filtered_sorted.columns.values) #This is just to get the vector of uncorrected p_values, FDR is set to False
p_vals_uncorrected = p_vals_uncorrected.rename(columns={'p_value' : 'p_value_uncorrected'})#rename the column to "PLV Value"

# %%
# Creates a DF with PLV values but only for the upper triangle of the matrix (since it's symmetric) and with the same labels as the original PLV matrix sorted 
_ , PLV_vals_vector = correct_p_values(PLV_sorted.to_numpy(), FDR_correction=False, cell_labels=filtered_sorted.columns.values) #this is used just for linearization
PLV_vals_vector = PLV_vals_vector.rename(columns={'p_value':'PLV Value'}) #rename the column to "PLV Value"
_, phase_diff_vector = correct_p_values(phase_diff_sorted.to_numpy(), FDR_correction=False, cell_labels=filtered_sorted.columns.values)#this is used just for linearization
phase_diff_vector = phase_diff_vector.rename(columns={'p_value' : 'Phase Diff (radians)'})#rename the column to "PLV Value"

# %%
# Plot sorted PLV matrix with superimposed significance values
plot_plv_matrix(PLV_sorted, p_val_matrix, square='True', only_significant=True, cmap='viridis', cbar_kws={'label': 'PLV'}) # If you want to plot all p-values, set only_significant to False

# %%
plot_plv_map(recording.props, PLV_vals_vector, map=recording.map_img, p_val_vector=p_vals_vector)

# %%
# Computes distances
distance_vector = compute_pairwise_distances(recording.props, PLV_vals_vector.index)

# %%
results_df = pd.concat([PLV_vals_vector, p_vals_vector, p_vals_uncorrected, phase_diff_vector, distance_vector] ,axis=1)

# %%
save_folder = recording.path.parent
results_df.to_csv(save_folder / 'results_PLV.csv', index=True, index_label='Cell(i) vs. Cell(j)')
save_open_figs(save_folder=save_folder, save_name='Results_PLV_figures.pdf')


