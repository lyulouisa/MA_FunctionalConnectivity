import numpy as np
import joblib
import matplotlib.pyplot as plt
import os

# Load edges_used_array
edges_used_array = joblib.load('../3_results/stats/edges_used_rest.pkl')
print(f"Loaded edges_used_array with shape: {edges_used_array.shape}")

# Compute mean coefficients for positive and negative edges separately
mean_coefficients = np.mean(edges_used_array, axis=0)
positive_edges = mean_coefficients[mean_coefficients > 0]
negative_edges = mean_coefficients[mean_coefficients < 0]

# Thresholds for the top 2.5% positive and negative edges
percentile = 2.5
positive_threshold = np.percentile(positive_edges, 100 - percentile)
negative_threshold = np.percentile(negative_edges, percentile)

print(f"Positive threshold for top {percentile}% edges: {positive_threshold}")
print(f"Negative threshold for top {percentile}% edges: {negative_threshold}")

# Create masks for significant positive and negative edges
positive_significant_edges_mask = mean_coefficients >= positive_threshold
negative_significant_edges_mask = mean_coefficients <= negative_threshold

print(f"Number of significant positive edges: {np.sum(positive_significant_edges_mask)}")
print(f"Number of significant negative edges: {np.sum(negative_significant_edges_mask)}")

# Map features back to the 268x268 matrix
num_regions = 268
tril_indices = np.tril_indices(num_regions, k=-1)

# Positive edge matrix
positive_edge_matrix = np.zeros((num_regions, num_regions))
positive_edge_matrix[tril_indices] = positive_significant_edges_mask.astype(int)
positive_edge_matrix = positive_edge_matrix + positive_edge_matrix.T

# Negative edge matrix
negative_edge_matrix = np.zeros((num_regions, num_regions))
negative_edge_matrix[tril_indices] = negative_significant_edges_mask.astype(int)
negative_edge_matrix = negative_edge_matrix + negative_edge_matrix.T

# Visualize the positive and negative edge matrices
plt.figure(figsize=(8, 8))
plt.imshow(positive_edge_matrix, cmap='hot', interpolation='nearest')
plt.title('Top 2.5% Positive Edges Used in Classification')
plt.colorbar(label='Edge Used (1) or Not (0)')
plt.xlabel('Brain Regions')
plt.ylabel('Brain Regions')
plt.show()

plt.figure(figsize=(8, 8))
plt.imshow(negative_edge_matrix, cmap='cool', interpolation='nearest')
plt.title('Top 2.5% Negative Edges Used in Classification')
plt.colorbar(label='Edge Used (1) or Not (0)')
plt.xlabel('Brain Regions')
plt.ylabel('Brain Regions')
plt.show()

# Save matrices in ASCII format
output_directory = '../3_results/stats/'
os.makedirs(output_directory, exist_ok=True)
np.savetxt(os.path.join(output_directory, 'network_diff_positive_rest.txt'), positive_edge_matrix, fmt='%d')
np.savetxt(os.path.join(output_directory, 'network_diff_negative_rest.txt'), negative_edge_matrix, fmt='%d')

# check overlapping edges with high attention and low attention network
import numpy as np
import os
from scipy.stats import hypergeom

data_directory = '../3_results/stats/'
network_diff_positive = np.loadtxt(os.path.join(data_directory, 'network_diff_positive_rest.txt'), dtype=int)
network_diff_negative = np.loadtxt(os.path.join(data_directory, 'network_diff_negative_rest.txt'), dtype=int)

# Combine positive and negative edges into a single matrix
network_diff_combined = (network_diff_positive + network_diff_negative) > 0
network_diff_combined = network_diff_combined.astype(int)

# Load attention network masks
high_attention_mask = np.loadtxt(os.path.join(data_directory, 'high_attention_mask.txt'), dtype=int)
low_attention_mask = np.loadtxt(os.path.join(data_directory, 'low_attention_mask.txt'), dtype=int)

# Ensure matrices have the same dimensions
assert network_diff_combined.shape == high_attention_mask.shape == low_attention_mask.shape, "Matrices must have the same dimensions."

# Extract the Lower Triangular Part
lower_triangle_network_diff = np.tril(network_diff_combined, k=-1)
lower_triangle_high_attention = np.tril(high_attention_mask, k=-1)
lower_triangle_low_attention = np.tril(low_attention_mask, k=-1)

# Define function to compute overlap and p-value
def compute_overlap_and_pvalue(matrix1, matrix2, num_regions):
    """
    Computes the number of edges in each matrix, the number of overlapping edges,
    and the p-value using the hypergeometric cumulative distribution function.
    """
    overlap_matrix = (matrix1 == 1) & (matrix2 == 1)
    num_overlapping_edges = np.sum(overlap_matrix)
    num_edges_matrix1 = np.sum(matrix1 == 1)
    num_edges_matrix2 = np.sum(matrix2 == 1)
    
    total_num_edges = num_regions * (num_regions - 1) // 2
    print(f"total_num_edges:{total_num_edges}")
    M = total_num_edges
    K = num_edges_matrix1
    N = num_edges_matrix2
    x = num_overlapping_edges

    # p_value = 1 - hypergeom.cdf(x - 1, M, K, N)
    p_value = 1 - hypergeom.cdf(x , M, K, N)
    return num_edges_matrix1, num_edges_matrix2, num_overlapping_edges, p_value

# Number of brain regions
num_regions = network_diff_combined.shape[0]

# Overlap with high attention network
num_edges_network_diff, num_edges_high_attention, num_overlapping_edges_high, p_value_high = compute_overlap_and_pvalue(
    lower_triangle_network_diff, lower_triangle_high_attention, num_regions)
print("High Attention vs Combined Network Diff:")
print(f"Number of edges in network_diff_combined: {num_edges_network_diff}")
print(f"Number of edges in high_attention_mask: {num_edges_high_attention}")
print(f"Number of overlapping edges: {num_overlapping_edges_high}")
print(f"P-value: {p_value_high:.6f}")

# Overlap with low attention network
num_edges_network_diff, num_edges_low_attention, num_overlapping_edges_low, p_value_low = compute_overlap_and_pvalue(
    lower_triangle_network_diff, lower_triangle_low_attention, num_regions)
print("\nLow Attention vs Combined Network Diff:")
print(f"Number of edges in network_diff_combined: {num_edges_network_diff}")
print(f"Number of edges in low_attention_mask: {num_edges_low_attention}")
print(f"Number of overlapping edges: {num_overlapping_edges_low}")
print(f"P-value: {p_value_low:.6f}")

# Combined high and low attention mask
combined_attention_mask = (high_attention_mask == 1) | (low_attention_mask == 1)
lower_triangle_combined_attention = np.tril(combined_attention_mask, k=-1)

# Overlap with combined high and low attention network
num_edges_network_diff, num_edges_combined_attention, num_overlapping_edges_combined, p_value_combined = compute_overlap_and_pvalue(
    lower_triangle_network_diff, lower_triangle_combined_attention, num_regions)
print("\nCombined High and Low Attention vs Combined Network Diff:")
print(f"Number of edges in network_diff_combined: {num_edges_network_diff}")
print(f"Number of edges in combined_attention_mask: {num_edges_combined_attention}")
print(f"Number of overlapping edges: {num_overlapping_edges_combined}")
print(f"P-value: {p_value_combined:.6f}")


## PLOT CONFUCION MATRIX
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba

# --------------------------- data --------------------------------- #
cm = np.array([[72, 20],
               [13, 69]])

accuracy = (cm[0, 0] + cm[1, 1]) / cm.sum()

# Color bases
green_hex = "#759116"
red_hex   = "#bf1029"

# Maximum values for scaling
max_green = cm[[0, 1], [0, 1]].max()
max_red   = cm[[0, 1], [1, 0]].max()

# Pre‑compute RGBA colors with alpha scaling
color_img = np.zeros((2, 2, 4))  # include alpha channel

for (i, j) in [(0, 0), (1, 1)]:               # correct predictions
    alpha = cm[i, j] / max_green              # 0–1
    color_img[i, j] = to_rgba(green_hex, alpha)

for (i, j) in [(0, 1), (1, 0)]:               # errors
    alpha = cm[i, j] / max_red                # 0–1
    color_img[i, j] = to_rgba(red_hex, alpha)

# ------------------------- plotting ------------------------------- #
fig, ax = plt.subplots(figsize=(4, 4))
ax.imshow(color_img, interpolation='nearest')

# Text labels
classes = ['Placebo', 'Amphetamine']
for i in range(2):
    for j in range(2):
        # choose text color based on background opacity
        text_color = 'white' if color_img[i, j, 3] > 0.5 else 'black'
        ax.text(j, i, f'{cm[i, j]}',
                va='center', ha='center',
                color=text_color, fontsize=14, fontweight='bold')

ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(classes)
ax.set_yticklabels(classes)
ax.set_xlabel('Predicted label')
ax.set_ylabel('True label')
ax.set_title(f'Confusion Matrix (Accuracy = {accuracy:.3f})', pad=15)

# Light grid for readability
ax.set_xticks(np.arange(-0.5, 2, 1), minor=True)
ax.set_yticks(np.arange(-0.5, 2, 1), minor=True)
ax.grid(which='minor', color='gray', linestyle='-', linewidth=1)
ax.tick_params(which='minor', bottom=False, left=False)

plt.tight_layout()
plt.savefig('confusion_matrix.jpg', dpi=300)
plt.close()

print('Saved confusion_matrix.jpg')