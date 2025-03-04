import os
import argparse
import optuna
import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, RepeatedStratifiedKFold, RepeatedKFold

from scripts.datasets_loader import load_datasets
from scripts.utils import get_files_by_class, split_datasets, get_accuracy_measures, get_accuracy_measures_errors
from scripts.config import (
    FC_DATA_PATH, OPTIMIZER_TRIALS, K_FOLDS, NUM_REPEATS_TRAINING, NUM_REPEATS_FINAL, NUM_CLASSES, DEVICE, NUM_EPOCH_TRAINING, NUM_EPOCH_FINAL, PATIENCE
)
from scripts.save_results import save_to_json
from scripts.models.dl_models_cores import train_model, evaluate_model, CNNClassifier
from scripts.models.ml_models_cores import MatlabDataset

# ✅ Enable CUDA optimizations
torch.backends.cudnn.benchmark = True

def cnn_objective(trial, full_dataset, num_classes, device, FC_name):
    """Objective function for Optuna hyperparameter tuning of CNN."""
    print(f"🔍 [FC: {FC_name}] Running Optuna Trial {trial.number}...")
    
    num_layers = trial.suggest_int("N-CNN", 1, 3)
    hidden_dim = trial.suggest_categorical("CNN-hidden", [32, 64, 128])
    kernel_size0 = trial.suggest_int("Kernel Size 0", 3, 5)
    kernel_size1 = trial.suggest_int("Kernel Size 1", 1, 2)
    padding = trial.suggest_int("Padding", 1, 2)
    lr = trial.suggest_categorical("Learning Rate", [0.001, 0.005, 0.01])
    batch_size = trial.suggest_categorical("Batch Size", [8, 16, 32])

    auc_scores = []
    kf = RepeatedKFold(n_splits=K_FOLDS, n_repeats=NUM_REPEATS_TRAINING, random_state=42)

    for fold, (train_idx, test_idx) in enumerate(kf.split(full_dataset)):
        train_subset = torch.utils.data.Subset(full_dataset, train_idx)
        test_subset = torch.utils.data.Subset(full_dataset, test_idx)

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)

        model = CNNClassifier(
            in_channels=1, hidden_dim=hidden_dim, num_classes=num_classes,
            kernel_size0=kernel_size0, kernel_size1=kernel_size1, padding=padding
        ).to(device)

        optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        criterion = torch.nn.CrossEntropyLoss()

        train_model(model, "CNN", NUM_EPOCH_TRAINING, criterion, optimizer, train_loader, device, PATIENCE, FC_name, trial.number)
        y_true, y_pred, y_scores = evaluate_model(model, test_loader, device)

        fold_auc = roc_auc_score(y_true, y_scores, multi_class="ovr", average="macro")
        auc_scores.append(fold_auc)

    return np.mean(auc_scores)

def train_cnn_model(FC_name, full_dataset):
    """Conducts hyperparameter tuning and final training for CNN."""
    print(f"🚀 [FC: {FC_name}] Starting Training...")
    
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: cnn_objective(trial, full_dataset, NUM_CLASSES, DEVICE, FC_name), n_trials=OPTIMIZER_TRIALS)
    
    print(f"🏆 [FC: {FC_name}] Best CNN Hyperparameters: {study.best_params}")
    best_params = study.best_params

    batch_size = best_params["Batch Size"]
    hidden_dim = best_params["CNN-hidden"]
    kernel_size0 = best_params["Kernel Size 0"]
    kernel_size1 = best_params["Kernel Size 1"]
    padding = best_params["Padding"]
    lr = best_params["Learning Rate"]

    kf = RepeatedKFold(n_splits=K_FOLDS, n_repeats=NUM_REPEATS_FINAL, random_state=42)

    y_true_all, y_pred_all, y_scores_all, y_true_all_compute_errors, y_pred_all_compute_errors, y_scores_all_compute_errors = [], [], [], [], [], []

    for fold, (train_idx, test_idx) in enumerate(kf.split(full_dataset)):
        train_subset = torch.utils.data.Subset(full_dataset, train_idx)
        test_subset = torch.utils.data.Subset(full_dataset, test_idx)

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)

        model = CNNClassifier(
            in_channels=1, hidden_dim=hidden_dim, num_classes=NUM_CLASSES,
            kernel_size0=kernel_size0, kernel_size1=kernel_size1, padding=padding
        ).to(DEVICE)

        optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        criterion = torch.nn.CrossEntropyLoss()

        train_model(model, "CNN", NUM_EPOCH_FINAL, criterion, optimizer, train_loader, DEVICE, PATIENCE, FC_name)
        y_true, y_pred, y_scores = evaluate_model(model, test_loader, DEVICE)

        y_true_all.extend(y_true)
        y_pred_all.extend(y_pred)
        y_scores_all.extend(y_scores)

        y_true_all_compute_errors.append(y_true)
        y_pred_all_compute_errors.append(y_pred)
        y_scores_all_compute_errors.append(y_scores)

    final_auc = roc_auc_score(y_true_all, y_scores_all, multi_class="ovr", average="macro")
    fpr, tpr, auc_dict, accuracy, specificity, sensitivity = get_accuracy_measures(y_true_all, y_pred_all, y_scores_all, NUM_CLASSES)
    
    auc_errors, accuracy_errors, specificity_errors, sensitivity_errors = get_accuracy_measures_errors(y_true_all_compute_errors, y_pred_all_compute_errors, y_scores_all_compute_errors, NUM_CLASSES)
    
    results = {
        "CNN": {
            "best_params": best_params,
            "FPR": fpr,
            "TPR": tpr,
            "AUC": {0: final_auc},
            "Acc": accuracy,
            "Spec": specificity,
            "Sens": sensitivity,
            "AUC_errors": auc_errors,
            "Acc_errors": accuracy_errors,
            "Spec_errors": specificity_errors,
            "Sens_errors": sensitivity_errors
        }
    }
    save_to_json(results, f"CNN_{FC_name}")
    print(f"✅ Results saved for {FC_name}")

def main():
    parser = argparse.ArgumentParser(description="Train CNN model with selected FC measure")
    parser.add_argument("--FC_name", type=str, required=True, help="Name of the FC measure to train with")
    args = parser.parse_args()
    
    FC_dataset, _ = load_datasets()
    FC_name = args.FC_name

    basepath = os.path.join(FC_DATA_PATH, FC_name)
    if not os.path.exists(basepath):
        print(f"❌ FC measure '{FC_name}' not found in {FC_DATA_PATH}")
        return

    dataset = get_files_by_class(basepath)
    splits = split_datasets(dataset)
    full_splits = splits["full"]

    full_dataset = MatlabDataset(file_list=full_splits, tensor=True)
    train_cnn_model(FC_name, full_dataset)

if __name__ == "__main__":
    main()
