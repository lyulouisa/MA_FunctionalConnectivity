import os
import numpy as np
import scipy.io
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
from tqdm import tqdm

print("Loading functional connectivity matrices......")
data_folder = "../1_data/fc"

fc_amph_rest   = scipy.io.loadmat(os.path.join(data_folder, 'fc_amph_rest.mat'))['fc_amph_s']
fc_PL_rest     = scipy.io.loadmat(os.path.join(data_folder, 'fc_PL_rest.mat'))['fc_PL_s']
subj_amph_rest = np.squeeze(scipy.io.loadmat(os.path.join(data_folder, 'subj_num_amph_rest.mat'))['subj_num_amph'])
subj_PL_rest   = np.squeeze(scipy.io.loadmat(os.path.join(data_folder, 'subj_num_PL_rest.mat'))['subj_num_PL'])

print(f"fc_amph_rest shape: {fc_amph_rest.shape}")
print(f"fc_PL_rest shape:   {fc_PL_rest.shape}")

# prepare labels
fc_matrices = np.concatenate((fc_amph_rest, fc_PL_rest), axis=2)
labels      = np.concatenate((np.ones(fc_amph_rest.shape[2], dtype=int),
                              np.zeros(fc_PL_rest.shape[2],  dtype=int)))
subject_ids = np.concatenate((subj_amph_rest, subj_PL_rest))

n_samples, n_regions = fc_matrices.shape[2], fc_matrices.shape[0]
n_features = n_regions * (n_regions - 1) // 2
X = np.zeros((n_samples, n_features))
for i in range(n_samples):
    X[i] = fc_matrices[:, :, i][np.tril_indices(n_regions, k=-1)]

if np.isnan(X).any():
    print("NaNs detected in feature matrix!")

unique_subjs = np.unique(subject_ids)
folds = []
for s in unique_subjs:
    idx = np.where(subject_ids == s)[0]
    folds.append(idx)     # idx has length 2 (both sessions) or 1 (single session)

preds, trues, edges_used_full = [], [], []
for k, test_idx in enumerate(folds, 1):
    train_idx = np.setdiff1d(np.arange(n_samples), test_idx)

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = labels[train_idx], labels[test_idx]

    nan_mask_train = ~np.isnan(X_train).any(axis=0)
    nan_mask_test  = ~np.isnan(X_test).any(axis=0)
    nan_mask = nan_mask_train & nan_mask_test

    X_train, X_test = X_train[:, nan_mask], X_test[:, nan_mask]

    scaler = StandardScaler().fit(X_train)
    X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)

    clf = SVC(kernel='linear', random_state=123).fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    preds.extend(y_pred.tolist())
    trues.extend(y_test.tolist())

    coef_full = np.zeros(n_features)
    coef_full[nan_mask] = clf.coef_.flatten()
    edges_used_full.extend([coef_full] * len(test_idx))

    print(f"Fold {k}: test idx {test_idx}, preds {y_pred}, trues {y_test}")

# ─────────────────── metrics ─────────────────── #
preds, trues = np.array(preds), np.array(trues)
cm = confusion_matrix(trues, preds)
TN, FP, FN, TP = cm.ravel()
acc = accuracy_score(trues, preds)

print("\nConfusion matrix\n", cm)
print(f"TP={TP}  FN={FN}  FP={FP}  TN={TN}")
print(f"Overall accuracy = {acc:.4f}")
with open('../3_results/rest_accuracy.txt', 'w') as f:
    f.write(f"{acc:.4f}\n")
print(classification_report(trues, preds, target_names=['Placebo','MA']))

joblib.dump(np.array(edges_used_full), '../3_results/edges_used_rest.pkl')

# ─────────────────── permutation test ─────────────────── #
n_perm = 1000
perm_path = 'permutation_accuracies.npy'

if os.path.exists(perm_path):
    perm_acc = np.load(perm_path)
    start_idx = np.sum(~np.isnan(perm_acc))
else:
    perm_acc = np.full(n_perm, np.nan)
    np.save(perm_path, perm_acc)
    start_idx = 0

print("\nStarting permutation test…")
for i in tqdm(range(n_perm), desc="Permutations"):
    perm_preds, perm_trues = [], []

    for test_idx in folds:
        train_idx = np.setdiff1d(np.arange(n_samples), test_idx)
        X_train, X_test = X[train_idx], X[test_idx]
        y_train = labels[train_idx]

        # randomize test labels
        if len(test_idx) == 2:
            y_test_rand = np.array([0, 1])
            np.random.shuffle(y_test_rand)
        else: # single session
            y_test_rand = np.random.randint(0, 2, size=1)

        nan_train = ~np.isnan(X_train).any(axis=0)
        nan_test  = ~np.isnan(X_test).any(axis=0)
        nan_mask  = nan_train & nan_test
        X_train, X_test = X_train[:, nan_mask], X_test[:, nan_mask]

        scaler = StandardScaler().fit(X_train)
        X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)

        clf = SVC(kernel='linear', random_state=123).fit(X_train, y_train)
        perm_preds.extend(clf.predict(X_test).tolist())
        perm_trues.extend(y_test_rand.tolist())

    perm_acc[i] = accuracy_score(perm_trues, perm_preds)
    np.save(perm_path, perm_acc)

valid = ~np.isnan(perm_acc)
p_val = (perm_acc[valid] >= acc).mean()
print(f"\nPermutation complete ({valid.sum()}/{n_perm} finished). p = {p_val:.4f}")