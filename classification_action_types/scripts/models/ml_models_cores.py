from torch.utils.data import Dataset
from scipy.io import loadmat
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from sklearn.svm import SVC
import numpy as np
import torch
from scripts.config import FC_DATA_PATH, OPTIMIZER_TRIALS, K_FOLDS, NUM_REPEATS_TRAINING, NUM_REPEATS_FINAL, NUM_CLASSES
class MatlabDataset(Dataset):
    """Handles loading MATLAB files into PyTorch Datasets"""
    def __init__(self, file_list, tensor=False, transform=None):
        self.file_list = file_list
        self.transform = transform
        self.tensor = tensor
        class_names = sorted({label for _, label in file_list})
        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(class_names)}

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path, class_name = self.file_list[idx]
        mat_data = loadmat(file_path)
        matrix = mat_data.get('out_data')  
        if matrix is None:
            raise ValueError(f"No 'out_data' key found in {file_path}")
        if self.tensor:
            matrix = torch.tensor(matrix, dtype=torch.float32)
        if self.transform:
            matrix = self.transform(matrix)
        label = self.class_to_idx[class_name]
        return matrix, label

# Define hyperparameter tuning functions
def objective_svc(trial, train_x, train_y):
    C = trial.suggest_float("svc_C", 1e-3, 10)
    gamma = trial.suggest_float("svc_gamma", 1e-3, 1)
    classifier = SVC(C=C, gamma=gamma, probability=True)
    return cross_validate_model(classifier, train_x, train_y)

def objective_logreg(trial, train_x, train_y):
    C = trial.suggest_float("logreg_C", 1e-3, 10)
    classifier = LogisticRegression(C=C, max_iter=1000)
    return cross_validate_model(classifier, train_x, train_y)

def objective_rfc(trial, train_x, train_y):
    n_estimators = trial.suggest_int("rfc_n_estimators", 10, 200)
    max_depth = trial.suggest_int("rfc_max_depth", 2, 20)
    classifier = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
    return cross_validate_model(classifier, train_x, train_y)

def cross_validate_model(classifier, train_x, train_y):
    """Performs cross-validation and returns AUC score."""
    cv = RepeatedStratifiedKFold(n_splits=K_FOLDS, n_repeats=NUM_REPEATS_TRAINING, random_state=42)
    auc_scores = []
    for train_idx, test_idx in cv.split(train_x, train_y):
        X_train, X_test = train_x[train_idx], train_x[test_idx]
        y_train, y_test = train_y[train_idx], train_y[test_idx]
        classifier.fit(X_train, y_train)
        y_scores = classifier.predict_proba(X_test)
        auc = roc_auc_score(y_test, y_scores, multi_class="ovr", average="macro")
        auc_scores.append(auc)
    return np.mean(auc_scores)