import os
import numpy as np
from sklearn.model_selection import LeaveOneOut
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import scipy.io
import joblib

print("Loading functional connectivity matrices......")
data_folder = "../1_data/fc"
fc_amph_rest_path = os.path.join(data_folder, 'fc_amph_rest.mat')
fc_PL_rest_path = os.path.join(data_folder, 'fc_PL_rest.mat')
if not os.path.exists(fc_amph_rest_path):
    raise FileNotFoundError(f"File not found: {fc_amph_rest_path}")

if not os.path.exists(fc_PL_rest_path):
    raise FileNotFoundError(f"File not found: {fc_PL_rest_path}")

# Extract data
fc_amph_rest_mat = scipy.io.loadmat(fc_amph_rest_path)
fc_PL_rest_mat = scipy.io.loadmat(fc_PL_rest_path)
print("Loaded .mat")

fc_amph_rest = fc_amph_rest_mat['fc_amph_s']
fc_PL_rest = fc_PL_rest_mat['fc_PL_s']
print(f"fc_amph_rest shape: {fc_amph_rest.shape}")
print(f"fc_PL_rest shape: {fc_PL_rest.shape}")

# prepare labels
labels_amph = np.ones(fc_amph_rest.shape[2], dtype=int)
labels_pl = np.zeros(fc_PL_rest.shape[2], dtype=int)
print(f"fc_amph_rest.shape[2]: {fc_amph_rest.shape[2]}; labels_amph.shape: {labels_amph.shape}")
print(f"fc_PL_rest.shape[2]: {fc_PL_rest.shape[2]}; labels_pl.shape: {labels_pl.shape}")

fc_matrices = np.concatenate((fc_amph_rest, fc_PL_rest), axis=2)
labels = np.concatenate((labels_amph, labels_pl), axis=0)
print(f"fc_matrices shape: {fc_matrices.shape}")
print(f"labels shape: {labels.shape}")

num_samples = fc_matrices.shape[2]
num_regions = fc_matrices.shape[0]
num_features = num_regions * (num_regions - 1) // 2
num_total_features = num_features
print(f"Number of samples: {num_samples}")
print(f"Number of regions: {num_regions}")
print(f"Number of features: {num_features}")

X = np.zeros((num_samples, num_features))

for i in range(num_samples):
    fc_matrix = fc_matrices[:, :, i]
    lower_triangular = fc_matrix[np.tril_indices(num_regions, k=-1)]
    X[i, :] = lower_triangular
if np.isnan(X).any():
    print("NaN values detected.")

# LOO CV
loo = LeaveOneOut()
predictions = []
true_labels = []
edges_used_full = []

for train_index, test_index in loo.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = labels[train_index], labels[test_index]
    print(f"\nIteration {len(predictions) + 1}")

    nan_mask_train = ~np.isnan(X_train).any(axis=0)
    nan_mask_test = ~np.isnan(X_test).any(axis=0)
    nan_mask = nan_mask_train & nan_mask_test
    X_train = X_train[:, nan_mask]
    X_test = X_test[:, nan_mask]

    # standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    print(f"X_train shape after scaling: {X_train.shape}")
    print(f"X_test shape after scaling: {X_test.shape}")

    classifier = SVC(kernel='linear', random_state=123)
    classifier.fit(X_train, y_train)

    # make pred
    y_pred = classifier.predict(X_test)
    predictions.append(y_pred[0])
    true_labels.append(y_test[0])
    coefficients = classifier.coef_.flatten()
    edges_full = np.zeros(num_total_features)
    edges_full[nan_mask] = coefficients
    edges_used_full.append(edges_full)
    print(f"Test sample index: {test_index}")
    print(f"Prediction: {y_pred[0]}, True label: {y_test[0]}")
    print(f"Classifier coefficients shape: {classifier.coef_.shape}")

# accuracy
predictions = np.array(predictions)
true_labels = np.array(true_labels)

conf_matrix = confusion_matrix(true_labels, predictions)
print("Confusion Matrix:")
print(conf_matrix)
TN = conf_matrix[0, 0]
FP = conf_matrix[0, 1]
FN = conf_matrix[1, 0]
TP = conf_matrix[1, 1]
print(f"True Positives (amphetamine classified as amphetamine): {TP}")
print(f"False Negatives (amphetamine classified as placebo): {FN}")
print(f"False Positives (placebo classified as amphetamine): {FP}")
print(f"True Negatives (placebo classified as placebo): {TN}")

accuracy = accuracy_score(true_labels, predictions)
print(f"Overall Accuracy: {accuracy:.4f}")
print("Classification Report:")
target_names = ['Placebo', 'Amphetamine']
print(classification_report(true_labels, predictions, target_names=target_names))

# save edges for classification
edges_used_array = np.array(edges_used_full)
joblib.dump(edges_used_array, 'edges_used_MID.pkl')


# permutation test on signficance
import random
from tqdm import tqdm

num_permutations = 1000
permutation_accuracies = []

print("\nStarting permutation testing...")
for i in tqdm(range(num_permutations), desc="Permutations"):

    print(f"iteration:{i}")
    shuffled_labels = np.random.permutation(labels)
    perm_predictions = []
    perm_true_labels = []
    
    # LOO
    for train_index, test_index in loo.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = shuffled_labels[train_index], shuffled_labels[test_index]

        nan_mask_train = ~np.isnan(X_train).any(axis=0)
        nan_mask_test = ~np.isnan(X_test).any(axis=0)
        nan_mask = nan_mask_train & nan_mask_test
        X_train_perm = X_train[:, nan_mask]
        X_test_perm = X_test[:, nan_mask]

        scaler = StandardScaler()
        X_train_perm = scaler.fit_transform(X_train_perm)
        X_test_perm = scaler.transform(X_test_perm)

        classifier_perm = SVC(kernel='linear', random_state=123)
        classifier_perm.fit(X_train_perm, y_train)

        y_pred_perm = classifier_perm.predict(X_test_perm)
        perm_predictions.append(y_pred_perm[0])
        perm_true_labels.append(y_test[0])

    # calculate accuracy
    perm_accuracy = accuracy_score(perm_true_labels, perm_predictions)
    permutation_accuracies.append(perm_accuracy)
    print(f"accuracy:{perm_accuracy}")

# Compute p-value
actual_accuracy = accuracy
permutation_accuracies = np.array(permutation_accuracies)
p_value = np.mean(permutation_accuracies >= actual_accuracy)

print(f"\nPermutation test completed with {num_permutations} permutations.")
print(f"Actual Accuracy: {actual_accuracy:.4f}")
print(f"Mean Permutation Accuracy: {np.mean(permutation_accuracies):.4f}")
print(f"Standard Deviation of Permutation Accuracies: {np.std(permutation_accuracies):.4f}")
print(f"P-value: {p_value:.4f}")

permutation_accuracies = np.array(permutation_accuracies)
np.save('permutation_accuracies.npy', permutation_accuracies)
print("Permutation accuracies saved to 'permutation_accuracies.npy'.")