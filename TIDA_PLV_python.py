# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from functools import partial

from functions.utils import bandpass, get_phase, dominant_frequency, poly_detrend, plv_einsum, permutation_test, circular_shift_surrogate, phase_randomisation_surrogate, plot_traces
from functions.io import select_file, load_recording, locate_props, locate_map, load_props, load_map,select_interval_manual, select_click_interval, select_double_click, select_block

# %%
# ===================================================
# Define parameters
# ===================================================

# Define which cells labels to keep
keep_cells = ["accepted"]

# Define filtering parameters (in Hz)
lowpass_cutoff = 0.01
highpass_cutoff = 1 

# %%
# Select a .csv file to analyze
traces_path = select_file()

# %%
time, data, status, cell_names, sampling_rate= load_recording(traces_path, keep_status=keep_cells)
labels = status + cell_names.astype(str)

props_path = locate_props(traces_path)
props = load_props(props_path, keep_status=keep_cells)

map_path = locate_map(traces_path)
map = load_map(map_path)

# %%
%matplotlib qt
time, data = select_click_interval(time, data, status=status, plot_selection=True, window=100)

# %%
filtered = data.apply(bandpass, args=(sampling_rate, lowpass_cutoff, highpass_cutoff))
filtered_phase = filtered.apply(get_phase)

# %%
dominat_freq = [dominant_frequency(filtered[neuron], sampling_rate, hann=False, poly_order=4, plot=True)[0] for neuron in filtered]
freq_domain = dominant_frequency(filtered.iloc[0], sampling_rate, hann=False, poly_order=4)[1]
power_spec = [dominant_frequency(filtered[neuron], sampling_rate, hann=False, poly_order=4)[2] for neuron in filtered]
processed_signals = [dominant_frequency(filtered[neuron], sampling_rate, hann=False, poly_order=4)[3] for neuron in filtered]   
slowest_dominant_freq = min(dominat_freq)
indx_slowest = dominat_freq.index(slowest_dominant_freq)
print(f"Slowest dominant frequency: {slowest_dominant_freq} Hz")

# %%
for dom_freq in dominat_freq:
    print(f"{dom_freq} Hz")

# %%
#Sanity  check to visualize the processed signal and the power spectrum of the slowest neuron
fig, axs = plt.subplots(2, 1, figsize=(10, 8))
fig.suptitle('Power Spectrum of slowest neuron')
axs[0].plot(np.linspace(0, len(processed_signals[indx_slowest])/sampling_rate, len(processed_signals[indx_slowest])),processed_signals[indx_slowest])
axs[0].set_title(f"Processed Signal - Neuron {indx_slowest+1} with period of {1/slowest_dominant_freq} s")
axs[0].set_xlabel('Time (samples)')
axs[0].set_ylabel('Amplitude')
axs[1].plot(freq_domain, power_spec[indx_slowest])
axs[1].set_xlim(0, 1)
axs[1].set_xlabel('Frequency (Hz)')
axs[1].set_ylabel('Power')


# %%
fig = plt.figure()
for i, neuron in enumerate(filtered):
    ax = fig.add_subplot(len(filtered), 1,i+1)
    ax.plot(time, filtered[i])
plt.savefig(traces_path.parent / 'Filtered_Traces.svg')

# %%
fig = plt.figure()
for i, neuron in enumerate(filtered_phase):
    ax = fig.add_subplot(len(filtered_phase), 1,i+1)
    ax.plot(time, neuron)
    ax.set_xlim(0,30)
plt.savefig(traces_path.parent / 'Filtered_Phases.svg')

# %%
filtered_phase = np.array(filtered_phase)
n = filtered_phase.shape[0]
PLV = plv_einsum(filtered_phase)

# %%
from seaborn import heatmap
plt.figure(figsize=(10, 8))
heatmap(PLV, cmap='viridis')
plt.title('Phase Locking Value (PLV) Matrix')

# %%
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform

# Convert similarity (PLV) to distance
dist = 1 - PLV
# Hierarchical clustering (Ward works well)
Z = linkage(squareform(dist), method='ward')
# Get order of leaves
idx = leaves_list(Z)

PLV_sorted = PLV[idx][:, idx]
labels_sorted = [labels.iloc[i] for i in idx]

# %%
from seaborn import heatmap
plt.figure(figsize=(10, 8))
heatmap(PLV_sorted, xticklabels=labels_sorted, yticklabels=labels_sorted, cmap='viridis')
plt.title('Phase Locking Value (PLV) Matrix')

# %%
avg_PLV_per_cell = []
min_PLV_per_cell = []
max_PLV_per_cell = []
for i, cell in enumerate(PLV_sorted):
    others = np.delete(cell, i)
    avg_PLV_per_cell.append(np.mean(others))
    min_PLV_per_cell.append(np.min(others))
    max_PLV_per_cell.append(np.max(others))

# %%
min_interval = int((1/slowest_dominant_freq)*sampling_rate) # At least one full cycle
max_interval = int(len(filtered_phase[0])-min_interval) # Use full length of the signal
surrogate_fn = partial(circular_shift_surrogate, min_interval=min_interval, max_interval=max_interval)

# %%
test, p_vals, surrogate_phases_all = permutation_test(np.array(filtered), PLV, surrogate_fn=surrogate_fn, n_permutations=1000, plot_surrogates=0)
p_vals_sorted = p_vals[idx][:, idx]

# %%
from scipy import stats
alpha = 0.05
p_vals_sorted_corrected = stats.false_discovery_control(p_vals_sorted.flatten(), method='bh').reshape(p_vals_sorted.shape)

# %%
fig, ax = plt.subplots(figsize=(10, 8))
heatmap(p_vals_sorted_corrected, xticklabels=labels_sorted, yticklabels=labels_sorted, cmap='viridis', ax=ax)

# Highlight cells with p-value < 0.05 in red
for i in range(p_vals_sorted_corrected.shape[0]):
    for j in range(p_vals_sorted_corrected.shape[1]):
        if p_vals_sorted_corrected[i, j] < 0.05:
            ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, edgecolor='red', linewidth=2))

# %%
fig, ax = plt.subplots(figsize=(10, 8))
heatmap(PLV_sorted, xticklabels=labels_sorted, yticklabels=labels_sorted, cmap='viridis', cbar_kws={'label': 'PLV Value'}, ax=ax)

# Highlight cells with p-value < 0.05 in red
for i in range(p_vals_sorted_corrected.shape[0]):
    for j in range(p_vals_sorted_corrected.shape[1]):
        if p_vals_sorted_corrected[i, j] < 0.05:
            ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, edgecolor='red', linewidth=2))

ax.set_title('Phase Locking Value (PLV) Matrix with Significant Connections (p < 0.05)', fontsize=12)
plt.tight_layout()
plt.savefig(traces_path.parent / 'PLV_Matrix_with_Significant_Connections.svg')
plt.show()

# %%
# Count the significant p-values (p < 0.05)
# Extract only the unique pairs (upper triangle excluding diagonal)
n = p_vals_sorted_corrected.shape[0]
upper_triangle_indices = np.triu_indices(n, k=1)
unique_p_values = p_vals_sorted_corrected[upper_triangle_indices]

significant = np.sum(unique_p_values < 0.05)
total_tests = len(unique_p_values)

print(f"Number of significant connections (p < 0.05): {significant}")
print(f"Total unique connections tested: {total_tests}")
print(f"Min p-value: {np.min(unique_p_values):.6f}")
print(f"Max p-value: {np.max(unique_p_values):.6f}")

if significant > 0:
    print(f"\nYes! Found {significant} significant connections (p < 0.05)")
else:
    print("\nNo significant connections found at p < 0.05 threshold")

# %%

# For each cell ID in sorted order, get its props data
props_coords = []
for _, row in props.iterrows():
    props_coords.append(row[['CentroidX', 'CentroidY']].values)
props_coords = np.array(props_coords)

x_coords = props_coords[:, 0]
y_coords = props_coords[:, 1]

# Get indices of significant connections (upper triangle, excluding diagonal)
n = p_vals_sorted_corrected.shape[0]
upper_triangle_indices = np.triu_indices(n, k=1)
sig_mask = p_vals_sorted_corrected[upper_triangle_indices] < 0.05
sig_i = upper_triangle_indices[0][sig_mask]
sig_j = upper_triangle_indices[1][sig_mask]

# Create figure with map background
fig, ax = plt.subplots(figsize=(14, 10))
ax.imshow(map, aspect='auto', cmap='gray')

# Normalize colormap for PLV values
significant_plv = PLV_sorted[sig_i, sig_j]
norm = plt.Normalize(vmin=np.min(significant_plv), vmax=np.max(significant_plv))
cmap = plt.cm.Reds

# Draw lines between significant pairs
for i_idx, j_idx in zip(sig_i, sig_j):
    x1, y1 = x_coords[i_idx], y_coords[i_idx]
    x2, y2 = x_coords[j_idx], y_coords[j_idx]
    
    plv_val = PLV_sorted[i_idx, j_idx]
    color = cmap(norm(plv_val))
    
    ax.plot([x1, x2], [y1, y2], color=color, linewidth=2, alpha=0.7)

# Draw cell positions as dots
ax.scatter(x_coords, y_coords, s=100, c='blue', edgecolors='white', linewidth=1.5, zorder=5, alpha=0.8)


# Add colorbar
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, label='PLV Value')

ax.set_title('Significant Phase-Locking Connections (p < 0.05)', fontsize=14, fontweight='bold')
ax.axis('off')
plt.tight_layout()
plt.savefig(traces_path.parent / 'Significant_Phase_Locking_Connections.svg')
plt.show()

# %%
# Analyze correlation between euclidean distance and PLV for significant connections
from scipy.stats import pearsonr, spearmanr


props_sorted_coords = props_coords[idx]  # Reorder props coordinates to match sorted labels and PLV

x_coords_sorted = props_sorted_coords[:, 0]
y_coords_sorted = props_sorted_coords[:, 1]

# Calculate euclidean distances for significant pairs
distances = []
plv_values = []

for i_idx, j_idx in zip(sig_i, sig_j):
    x1, y1 = x_coords_sorted[i_idx], y_coords_sorted[i_idx]
    x2, y2 = x_coords_sorted[j_idx], y_coords_sorted[j_idx]
    
    # Euclidean distance
    euclidean_dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    distances.append(euclidean_dist)
    
    # PLV value
    plv_val = PLV_sorted[i_idx, j_idx]
    plv_values.append(plv_val)

distances = np.array(distances)
plv_values = np.array(plv_values)

# Calculate correlations
pearson_corr, pearson_pval = pearsonr(distances, plv_values)
spearman_corr, spearman_pval = spearmanr(distances, plv_values)

print("Correlation between Euclidean Distance and PLV (Significant Connections)")
print("=" * 70)
print(f"Pearson correlation: r = {pearson_corr:.4f}, p-value = {pearson_pval:.4f}")
print(f"Spearman correlation: rho = {spearman_corr:.4f}, p-value = {spearman_pval:.4f}")
print(f"\nNumber of significant connections analyzed: {len(distances)}")
print(f"Distance range: {np.min(distances):.2f} - {np.max(distances):.2f}")
print(f"PLV range: {np.min(plv_values):.4f} - {np.max(plv_values):.4f}")

# Create scatter plot
fig, ax = plt.subplots(figsize=(10, 7))
ax.scatter(distances, plv_values, alpha=0.6, s=100, edgecolors='black', linewidth=1)

# Add trend line
z = np.polyfit(distances, plv_values, 1)
p = np.poly1d(z)
ax.plot(distances, p(distances), "r--", linewidth=2, label=f'Linear fit')

ax.set_xlabel('Euclidean Distance (pixels)', fontsize=12)
ax.set_ylabel('PLV Value', fontsize=12)
ax.set_title(f'Correlation: Distance vs PLV\nPearson r={pearson_corr:.4f} (p={pearson_pval:.4f})', fontsize=14)
ax.grid(True, alpha=0.3)
ax.legend()
plt.tight_layout()
plt.show()

# %%
# Analyze phase relationships: are neurons in-phase or antiphase?
# Calculate the mean phase difference for each neuron pair

# Get indices of ALL unique pairs (upper triangle excluding diagonal)
n = p_vals_sorted_corrected.shape[0]
upper_triangle_indices = np.triu_indices(n, k=1)

# Calculate mean phase difference for all pairs
phase_differences = []
phase_difference_labels = []

for i_idx, j_idx in zip(upper_triangle_indices[0], upper_triangle_indices[1]):
    # Calculate phase difference
    phase_diff = filtered_phase[idx[i_idx]] - filtered_phase[idx[j_idx]]
    
    # Wrap phase difference to [-π, π]
    phase_diff_wrapped = np.angle(np.exp(1j * phase_diff))
    
    # Mean phase difference across time
    mean_phase_diff = np.mean(phase_diff_wrapped)
    phase_differences.append(mean_phase_diff)
    phase_difference_labels.append((i_idx, j_idx))

phase_differences = np.array(phase_differences) * 180 / np.pi  # Convert to degrees

# Classify pairs as in-phase or antiphase
in_phase_threshold = 45  # Within 45° of 0 = in-phase
antiphase_threshold = 135  # Within 45° of π = antiphase

abs_phase_diff = np.abs(phase_differences)
in_phase_mask = abs_phase_diff < in_phase_threshold
antiphase_mask = abs_phase_diff > antiphase_threshold
intermediate_mask = ~(in_phase_mask | antiphase_mask)

in_phase_count = np.sum(in_phase_mask)
antiphase_count = np.sum(antiphase_mask)
intermediate_count = np.sum(intermediate_mask)

print("Phase Relationship Analysis: Are Neurons In-Phase or Antiphase?")
print("=" * 70)
print(f"\nTotal neuron pairs analyzed: {len(phase_differences)}")
print(f"\nPhase Relationship Classification:")
print(f"  In-phase (|phase diff| < 45°):        {in_phase_count:3d} pairs ({in_phase_count/len(phase_differences)*100:5.1f}%)")
print(f"  Intermediate (45° < |phase diff| < 135°): {intermediate_count:3d} pairs ({intermediate_count/len(phase_differences)*100:5.1f}%)")
print(f"  Antiphase (|phase diff| > 135°):      {antiphase_count:3d} pairs ({antiphase_count/len(phase_differences)*100:5.1f}%)")

print(f"\nPhase Difference Statistics:")
print(f"  Mean phase difference: {np.mean(phase_differences):.1f}°")
print(f"  Median phase difference: {np.median(phase_differences):.1f}°")
print(f"  Std phase difference: {np.std(phase_differences):.1f}°")
print(f"  Min phase difference: {np.min(phase_differences):.1f}°")
print(f"  Max phase difference: {np.max(phase_differences):.1f}°")

# Visualize phase differences
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Histogram of phase differences (in degrees)
axes[0].hist(phase_differences, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
axes[0].axvline(0, color='green', linestyle='--', linewidth=2.5, label='In-phase (0°)')
axes[0].axvline(180, color='red', linestyle='--', linewidth=2.5, label='Antiphase (180°)')
axes[0].axvline(-180, color='red', linestyle='--', linewidth=2.5)
axes[0].axvline(45, color='gray', linestyle=':', linewidth=1.5, alpha=0.7)
axes[0].axvline(-45, color='gray', linestyle=':', linewidth=1.5, alpha=0.7)
axes[0].axvline(135, color='gray', linestyle=':', linewidth=1.5, alpha=0.7)
axes[0].axvline(-135, color='gray', linestyle=':', linewidth=1.5, alpha=0.7)
axes[0].set_xlabel('Phase Difference (degrees)', fontsize=12)
axes[0].set_ylabel('Number of Pairs', fontsize=12)
axes[0].set_title('Distribution of Phase Differences', fontsize=13, fontweight='bold')
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

# Plot 2: Pie chart of classification
colors = ['green', 'gray', 'red']
sizes = [in_phase_count, intermediate_count, antiphase_count]
labels_pie = [f'In-phase\n({in_phase_count})', f'Intermediate\n({intermediate_count})', f'Antiphase\n({antiphase_count})']
axes[1].pie(sizes, labels=labels_pie, colors=colors, autopct='%1.1f%%', startangle=90, textprops={'fontsize': 11})
axes[1].set_title('Phase Relationship Classification', fontsize=13, fontweight='bold')

plt.tight_layout()
plt.show()

print("\n" + "=" * 70)
if in_phase_count > antiphase_count:
    print(f"CONCLUSION: Most neurons are IN-PHASE synchronized")
    print(f"({in_phase_count} in-phase pairs vs {antiphase_count} antiphase pairs)")
elif antiphase_count > in_phase_count:
    print(f"CONCLUSION: Most neurons are ANTIPHASE synchronized")
    print(f"({antiphase_count} antiphase pairs vs {in_phase_count} in-phase pairs)")
else:
    print(f"CONCLUSION: Equal number of in-phase and antiphase pairs")


# %%
# Compare phase distributions: significant vs non-significant connections
# Do significant connections have different phase relationships than non-significant ones?

from scipy.stats import ttest_ind, mannwhitneyu

# Get indices of ALL unique pairs (upper triangle excluding diagonal)
n = p_vals_sorted_corrected.shape[0]
upper_triangle_indices = np.triu_indices(n, k=1)

# Get p-values for all pairs
p_values_all_pairs = p_vals_sorted_corrected[upper_triangle_indices]

# Separate phase differences into significant and non-significant
sig_mask = p_values_all_pairs < 0.05
phase_diffs_sig = np.abs(phase_differences[sig_mask])
phase_diffs_nonsig = np.abs(phase_differences[~sig_mask])

print("Phase Distribution: Significant vs Non-Significant Connections")
print("=" * 70)
print(f"\nSignificant Connections (p < 0.05):")
print(f"  Count: {len(phase_diffs_sig)}")
print(f"  Mean phase difference: {np.mean(phase_diffs_sig):.1f}°")
print(f"  Median phase difference: {np.median(phase_diffs_sig):.1f}°")
print(f"  Std: {np.std(phase_diffs_sig):.1f}°")
print(f"  Min: {np.min(phase_diffs_sig):.1f}°")
print(f"  Max: {np.max(phase_diffs_sig):.1f}°")

print(f"\nNon-Significant Connections (p >= 0.05):")
print(f"  Count: {len(phase_diffs_nonsig)}")
print(f"  Mean phase difference: {np.mean(phase_diffs_nonsig):.1f}°")
print(f"  Median phase difference: {np.median(phase_diffs_nonsig):.1f}°")
print(f"  Std: {np.std(phase_diffs_nonsig):.1f}°")
print(f"  Min: {np.min(phase_diffs_nonsig):.1f}°")
print(f"  Max: {np.max(phase_diffs_nonsig):.1f}°")

# Statistical tests
t_stat, t_pval = ttest_ind(phase_diffs_sig, phase_diffs_nonsig)
u_stat, u_pval = mannwhitneyu(phase_diffs_sig, phase_diffs_nonsig)

print(f"\nStatistical Comparison:")
print(f"  t-test: t = {t_stat:.4f}, p = {t_pval:.6f}")
print(f"  Mann-Whitney U test: U = {u_stat:.4f}, p = {u_pval:.6f}")

# Classify each group by phase relationship type
sig_in_phase = np.sum(phase_diffs_sig < in_phase_threshold)
sig_antiphase = np.sum(phase_diffs_sig > antiphase_threshold)
sig_intermediate = np.sum((phase_diffs_sig >= in_phase_threshold) & (phase_diffs_sig <= antiphase_threshold))

nonsig_in_phase = np.sum(phase_diffs_nonsig < in_phase_threshold)
nonsig_antiphase = np.sum(phase_diffs_nonsig > antiphase_threshold)
nonsig_intermediate = np.sum((phase_diffs_nonsig >= in_phase_threshold) & (phase_diffs_nonsig <= antiphase_threshold))

print(f"\nPhase Relationship Breakdown:")
print(f"  Significant Connections:")
print(f"    In-phase: {sig_in_phase} ({sig_in_phase/len(phase_diffs_sig)*100:.1f}%)")
print(f"    Intermediate: {sig_intermediate} ({sig_intermediate/len(phase_diffs_sig)*100:.1f}%)")
print(f"    Antiphase: {sig_antiphase} ({sig_antiphase/len(phase_diffs_sig)*100:.1f}%)")
print(f"  Non-Significant Connections:")
print(f"    In-phase: {nonsig_in_phase} ({nonsig_in_phase/len(phase_diffs_nonsig)*100:.1f}%)")
print(f"    Intermediate: {nonsig_intermediate} ({nonsig_intermediate/len(phase_diffs_nonsig)*100:.1f}%)")
print(f"    Antiphase: {nonsig_antiphase} ({nonsig_antiphase/len(phase_diffs_nonsig)*100:.1f}%)")

# Create visualizations
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Plot 1: Overlaid histograms (counts)
axes[0, 0].hist(phase_diffs_sig, bins=25, alpha=0.6, label='Significant (p < 0.05)', 
                color='blue', edgecolor='darkblue', linewidth=0.5)
axes[0, 0].hist(phase_diffs_nonsig, bins=25, alpha=0.6, label='Non-Significant (p ≥ 0.05)', 
                color='gray', edgecolor='black', linewidth=0.5)
axes[0, 0].set_xlabel('Absolute Phase Difference (degrees)', fontsize=11)
axes[0, 0].set_ylabel('Number of Pairs', fontsize=11)
axes[0, 0].set_title('Phase Distribution (Counts)', fontsize=12, fontweight='bold')
axes[0, 0].legend(fontsize=10)
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Normalized histograms (probability density)
axes[0, 1].hist(phase_diffs_sig, bins=25, alpha=0.6, label='Significant (p < 0.05)', 
                color='blue', edgecolor='darkblue', linewidth=0.5, density=True)
axes[0, 1].hist(phase_diffs_nonsig, bins=25, alpha=0.6, label='Non-Significant (p ≥ 0.05)', 
                color='gray', edgecolor='black', linewidth=0.5, density=True)
axes[0, 1].set_xlabel('Absolute Phase Difference (degrees)', fontsize=11)
axes[0, 1].set_ylabel('Probability Density', fontsize=11)
axes[0, 1].set_title('Phase Distribution (Normalized)', fontsize=12, fontweight='bold')
axes[0, 1].legend(fontsize=10)
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Box plot comparison
box_data = [phase_diffs_sig, phase_diffs_nonsig]
bp = axes[0, 2].boxplot(box_data, labels=['Significant', 'Non-Significant'], patch_artist=True)
bp['boxes'][0].set_facecolor('blue')
bp['boxes'][1].set_facecolor('gray')
axes[0, 2].set_ylabel('Absolute Phase Difference (degrees)', fontsize=11)
axes[0, 2].set_title('Distribution Quartiles', fontsize=12, fontweight='bold')
axes[0, 2].grid(True, alpha=0.3, axis='y')

# Plot 4: Pie chart for significant connections
colors_sig = ['green', 'gray', 'blue']
sizes_sig = [sig_in_phase, sig_intermediate, sig_antiphase]
labels_sig_pie = [f'In-phase\n({sig_in_phase})', f'Intermediate\n({sig_intermediate})', f'Antiphase\n({sig_antiphase})']
axes[1, 0].pie(sizes_sig, labels=labels_sig_pie, colors=colors_sig, autopct='%1.1f%%', 
               startangle=90, textprops={'fontsize': 10})
axes[1, 0].set_title('Significant Connections (p < 0.05)', fontsize=12, fontweight='bold')

# Plot 5: Pie chart for non-significant connections
colors_nonsig = ['green', 'gray', 'red']
sizes_nonsig = [nonsig_in_phase, nonsig_intermediate, nonsig_antiphase]
labels_nonsig_pie = [f'In-phase\n({nonsig_in_phase})', f'Intermediate\n({nonsig_intermediate})', f'Antiphase\n({nonsig_antiphase})']
axes[1, 1].pie(sizes_nonsig, labels=labels_nonsig_pie, colors=colors_nonsig, autopct='%1.1f%%', 
               startangle=90, textprops={'fontsize': 10})
axes[1, 1].set_title('Non-Significant Connections (p ≥ 0.05)', fontsize=12, fontweight='bold')

# Plot 6: Cumulative distribution comparison
axes[1, 2].hist(phase_diffs_sig, bins=25, alpha=0.6, label='Significant (p < 0.05)', 
                color='blue', edgecolor='darkblue', linewidth=0.5, cumulative=True, density=True)
axes[1, 2].hist(phase_diffs_nonsig, bins=25, alpha=0.6, label='Non-Significant (p ≥ 0.05)', 
                color='gray', edgecolor='black', linewidth=0.5, cumulative=True, density=True)
axes[1, 2].set_xlabel('Absolute Phase Difference (degrees)', fontsize=11)
axes[1, 2].set_ylabel('Cumulative Probability', fontsize=11)
axes[1, 2].set_title('Cumulative Distribution', fontsize=12, fontweight='bold')
axes[1, 2].legend(fontsize=10)
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(traces_path.parent / 'Phase_Distribution_Significant_vs_NonSignificant.svg')
plt.show()

print("\n" + "=" * 70)
print("CONCLUSION:")
if t_pval < 0.05:
    if np.mean(phase_diffs_sig) > np.mean(phase_diffs_nonsig):
        print(f"YES! Significant connections have LARGER phase differences")
        print(f"(mean diff: {np.mean(phase_diffs_sig):.1f}° vs {np.mean(phase_diffs_nonsig):.1f}°, p={t_pval:.6f})")
    else:
        print(f"YES! Significant connections have SMALLER phase differences")
        print(f"(mean diff: {np.mean(phase_diffs_sig):.1f}° vs {np.mean(phase_diffs_nonsig):.1f}°, p={t_pval:.6f})")
else:
    print(f"NO significant difference in phase distributions")
    print(f"(mean diff sig: {np.mean(phase_diffs_sig):.1f}° vs nonsig: {np.mean(phase_diffs_nonsig):.1f}°, p={t_pval:.6f})")


# %%
# Single-pair analysis: Is phase difference at each pair more consistent than chance?
# Reuse surrogates from the original permutation test - no need to regenerate!

print("Single-Pair Phase Consistency Analysis")
print("=" * 70)
print("\nTesting if individual pairs have phase relationships more consistent than chance...")
print("(Using the same 1000 surrogate datasets from the PLV permutation test)")

# Calculate phase differences from the SAME surrogate data already computed
print("Extracting phase differences from existing surrogates...")
surrogate_phase_diffs_all = []

for perm in range(len(surrogate_phases_all)):
    # Use the shifted phases from the original permutation test
    shifted_phases_temp = surrogate_phases_all[perm]
    
    # Calculate phase differences for this permutation
    phase_diffs_perm = []
    for i_idx, j_idx in zip(upper_triangle_indices[0], upper_triangle_indices[1]):
        phase_diff = shifted_phases_temp[idx[i_idx]] - shifted_phases_temp[idx[j_idx]]
        phase_diff_wrapped = np.angle(np.exp(1j * phase_diff))
        mean_phase_diff = np.abs(np.mean(phase_diff_wrapped))
        phase_diffs_perm.append(mean_phase_diff)
    
    surrogate_phase_diffs_all.append(phase_diffs_perm)

surrogate_phase_diffs_all = np.array(surrogate_phase_diffs_all) * 180 / np.pi  # Convert to degrees to match empirical phase_differences

# For each pair, calculate p-value: how often is surrogate phase diff <= empirical phase diff?
p_values_phase = np.zeros(len(phase_differences))
z_scores_phase = np.zeros(len(phase_differences))

for pair_idx in range(len(phase_differences)):
    empirical_phase = np.abs(phase_differences[pair_idx])
    surrogate_phases = surrogate_phase_diffs_all[:, pair_idx]
    
    # Count how many surrogates have smaller phase difference
    count = np.sum(surrogate_phases <= empirical_phase)
    p_val = (count + 1) / (len(surrogate_phases) + 1)
    p_values_phase[pair_idx] = p_val
    
    # Also calculate z-score
    mean_surr = np.mean(surrogate_phases)
    std_surr = np.std(surrogate_phases)
    z_score = (empirical_phase - mean_surr) / (std_surr + 1e-10)
    z_scores_phase[pair_idx] = z_score

# Count significant pairs (p < 0.05)
sig_phase_pairs = np.sum(p_values_phase < 0.05)
print(f"\nSignificant phase consistency (p < 0.05): {sig_phase_pairs} pairs")
print(f"Total pairs tested: {len(p_values_phase)}")
print(f"Percentage: {sig_phase_pairs/len(p_values_phase)*100:.1f}%")

print(f"\nPhase Consistency Statistics:")
print(f"  Mean z-score: {np.mean(z_scores_phase):.4f}")
print(f"  Std z-score: {np.std(z_scores_phase):.4f}")
print(f"  Pairs with z > 1: {np.sum(z_scores_phase > 1)} ({np.sum(z_scores_phase > 1)/len(z_scores_phase)*100:.1f}%)")
print(f"  Pairs with z > 2: {np.sum(z_scores_phase > 2)} ({np.sum(z_scores_phase > 2)/len(z_scores_phase)*100:.1f}%)")
print(f"  Min p-value: {np.min(p_values_phase):.6f}")
print(f"  Max p-value: {np.max(p_values_phase):.6f}")

# Visualize results
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Histogram of p-values
axes[0, 0].hist(p_values_phase, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
axes[0, 0].axvline(0.05, color='red', linestyle='--', linewidth=2, label='p = 0.05 threshold')
axes[0, 0].set_xlabel('p-value', fontsize=11)
axes[0, 0].set_ylabel('Number of Pairs', fontsize=11)
axes[0, 0].set_title('Distribution of Phase Consistency p-values', fontsize=12, fontweight='bold')
axes[0, 0].legend(fontsize=10)
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Histogram of z-scores
axes[0, 1].hist(z_scores_phase, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
axes[0, 1].axvline(0, color='gray', linestyle='--', linewidth=2, label='z = 0 (no effect)')
axes[0, 1].axvline(1.96, color='orange', linestyle='--', linewidth=2, label='z = 1.96 (p≈0.05)')
axes[0, 1].set_xlabel('z-score', fontsize=11)
axes[0, 1].set_ylabel('Number of Pairs', fontsize=11)
axes[0, 1].set_title('Distribution of Phase Consistency z-scores', fontsize=12, fontweight='bold')
axes[0, 1].legend(fontsize=10)
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Empirical vs surrogate phase differences (significant pairs only)
sig_mask_phase = p_values_phase < 0.05
empirical_sig = np.abs(phase_differences[sig_mask_phase])
surrogate_mean_sig = np.mean(surrogate_phase_diffs_all[:, sig_mask_phase], axis=0)

axes[1, 0].scatter(surrogate_mean_sig, empirical_sig, alpha=0.6, s=100, edgecolors='red', linewidth=1)
lims = [min(axes[1, 0].get_xlim()[0], axes[1, 0].get_ylim()[0]),
        max(axes[1, 0].get_xlim()[1], axes[1, 0].get_ylim()[1])]
axes[1, 0].plot(lims, lims, 'k--', alpha=0.3, linewidth=2)
axes[1, 0].set_xlabel('Mean Surrogate Phase Difference (degrees)', fontsize=11)
axes[1, 0].set_ylabel('Empirical Phase Difference (degrees)', fontsize=11)
axes[1, 0].set_title('Significant Pairs: Empirical vs Chance', fontsize=12, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: Count of consistent pairs by PLV significance
plv_sig_mask = p_vals_sorted_corrected[upper_triangle_indices] < 0.05
phase_and_plv_sig = sig_mask_phase & plv_sig_mask
phase_only_sig = sig_mask_phase & ~plv_sig_mask
plv_only_sig = ~sig_mask_phase & plv_sig_mask
neither_sig = ~sig_mask_phase & ~plv_sig_mask

categories = ['Phase & PLV\nSignificant', 'Phase Only\nSignificant', 'PLV Only\nSignificant', 'Neither\nSignificant']
counts = [np.sum(phase_and_plv_sig), np.sum(phase_only_sig), np.sum(plv_only_sig), np.sum(neither_sig)]
colors_bar = ['darkgreen', 'lightgreen', 'orange', 'gray']

bars = axes[1, 1].bar(categories, counts, color=colors_bar, edgecolor='black', linewidth=1.5)
axes[1, 1].set_ylabel('Number of Pairs', fontsize=11)
axes[1, 1].set_title('Phase & PLV Significance Overlap', fontsize=12, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.show()

print("\n" + "=" * 70)
print("Phase Consistency Summary:")
print(f"  Pairs with consistent phase (p < 0.05): {sig_phase_pairs}")
print(f"  Both phase & PLV significant: {np.sum(phase_and_plv_sig)}")
print(f"  Phase consistent but PLV not significant: {np.sum(phase_only_sig)}")
print(f"  PLV significant but phase not consistent: {np.sum(plv_only_sig)}")



