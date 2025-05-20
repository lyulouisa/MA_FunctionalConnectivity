import os
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Define the target names (labels)
target_names = ['Placebo', 'MA']

results_directory = '../3_results'

# Load the accuracy
with open(os.path.join(results_directory, 'rest_accuracy.txt'), 'r') as f:
    accuracy = float(f.read())

# 2. Null Distribution Plot
permutation_accuracies_path = '../3_results/permutation_accuracies.npy'
if not os.path.exists(permutation_accuracies_path):
    raise FileNotFoundError(f"File not found: {permutation_accuracies_path}")

permutation_accuracies = np.load(permutation_accuracies_path)


plt.figure(figsize=(8, 6))
sns.set(style='whitegrid')
palette = sns.color_palette(["#A63333", "#2365B0"])
sns.histplot(permutation_accuracies, bins=30, kde=False, color=palette[1], edgecolor='black', alpha=0.7)

plt.axvline(x=accuracy, color=palette[0], linestyle='--', linewidth=2, label=f'Actual Accuracy ({accuracy:.2f})')
plt.title('Null Distribution of Classification Accuracies', fontsize=18)
plt.xlabel('Accuracy', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.legend(fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(False)
plt.tight_layout()
plt.show()
