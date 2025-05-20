import numpy as np
import joblib
import matplotlib.pyplot as plt
import os

edges_used_array = joblib.load('../3_results/edges_used_rest.pkl')
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

output_directory = '../3_results/stats/'
os.makedirs(output_directory, exist_ok=True)
np.savetxt(os.path.join(output_directory, 'network_diff_positive_rest.txt'), positive_edge_matrix, fmt='%d')
np.savetxt(os.path.join(output_directory, 'network_diff_negative_rest.txt'), negative_edge_matrix, fmt='%d')


## PLOT CONFUCION MATRIX
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba


cm = np.array([[75, 17],
               [11, 71]])

accuracy = (cm[0, 0] + cm[1, 1]) / cm.sum()
green_hex = "#759116"
red_hex   = "#bf1029"
max_green = cm[[0, 1], [0, 1]].max()
max_red   = cm[[0, 1], [1, 0]].max()
color_img = np.zeros((2, 2, 4))

for (i, j) in [(0, 0), (1, 1)]:
    alpha = cm[i, j] / max_green
    color_img[i, j] = to_rgba(green_hex, alpha)

for (i, j) in [(0, 1), (1, 0)]:
    alpha = cm[i, j] / max_red
    color_img[i, j] = to_rgba(red_hex, alpha)

# ------------------------- plotting ------------------------------- #
fig, ax = plt.subplots(figsize=(5, 4))
ax.imshow(color_img, interpolation='nearest')

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
plt.savefig('../3_results/confusion_matrix.jpg', dpi=300)
plt.close()

print('Saved confusion_matrix.jpg')
