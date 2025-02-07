import os
import optuna
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
import torch
from torch.optim import Adam
from torch_geometric.loader import DataLoader
from torch.utils.data import Dataset

from scripts.datasets_loader import load_datasets, FC_dataset
from scripts.utils import get_files_by_class, split_datasets, dataset_type_converter, get_accuracy_measures
from scripts.config import (
    FC_DATA_PATH, OPTIMIZER_TRIALS, K_FOLDS, NUM_REPEATS, NUM_CLASSES, DEVICE,
    NUM_EPOCH_TRAINING, NUM_EPOCH_FINAL
)
from scripts.save_results import save_to_json
from scripts.models.dl_models_cores import train_model, evaluate_model, MLPClassifier
from scripts.models.ml_models_cores import MatlabDataset

# Store output results
output_data = {}

def mlp_objective(trial, full_dataset, num_classes, device):
    """
    Objective function for Optuna hyperparameter tuning of MLP model.
    """
    # Define hyperparameter search space
    num_layers = trial.suggest_int("N-MLP", 1, 5)
    hidden_dim = trial.suggest_categorical("MLP-hidden", [32, 64, 128])
    lr = trial.suggest_categorical("Learning Rate", [0.001, 0.005, 0.01])
    batch_size = trial.suggest_categorical("Batch Size", [8, 16, 32])

    auc_scores = []
    kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=42)

    for fold, (train_idx, test_idx) in enumerate(kf.split(full_dataset)):
        print(f"🔍 Optuna Tuning - Fold {fold+1}/{K_FOLDS}")

        # Create training and testing subsets
        train_subset = torch.utils.data.Subset(full_dataset, train_idx)
        test_subset = torch.utils.data.Subset(full_dataset, test_idx)
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)

        # Initialize model with selected hyperparameters
        model = MLPClassifier(
            input_size=full_dataset[0][0].numel(),  # Dynamically set input size
            hidden_dim=hidden_dim, num_layers=num_layers, num_classes=num_classes
        ).to(device)

        optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        criterion = torch.nn.CrossEntropyLoss()

        # Train and evaluate
        train_model(model, "MLP", NUM_EPOCH_TRAINING, criterion, optimizer, train_loader, device)
        y_true, y_pred, y_scores = evaluate_model(model, test_loader, device)

        # Compute AUC for this fold
        fold_auc = roc_auc_score(y_true, y_scores, multi_class="ovr", average="macro")
        auc_scores.append(fold_auc)

    return np.mean(auc_scores)


def train_mlp_model(FC_name, full_dataset):
    """
    Conducts hyperparameter tuning and final training for MLP.
    """
    print(f"🚀 Starting Training for FC Metric: {FC_name}")
    output_data[FC_name] = {}

    # Perform hyperparameter tuning using Optuna
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: mlp_objective(trial, full_dataset, NUM_CLASSES, DEVICE), n_trials=OPTIMIZER_TRIALS)
    
    print("🏆 Best MLP Hyperparameters:", study.best_params)
    best_params = study.best_params

    # Extract best hyperparameters
    batch_size = best_params["Batch Size"]
    hidden_dim = best_params["MLP-hidden"]
    num_layers = best_params["N-MLP"]
    lr = best_params["Learning Rate"]

    kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
    y_true_all, y_pred_all, y_scores_all = [], [], []

    for fold, (train_idx, test_idx) in enumerate(kf.split(full_dataset)):
        print(f"📊 Final Training - Fold {fold+1}/{K_FOLDS}")

        # Train-test split
        train_subset = torch.utils.data.Subset(full_dataset, train_idx)
        test_subset = torch.utils.data.Subset(full_dataset, test_idx)
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)

        # Initialize final model
        model = MLPClassifier(
            input_size=full_dataset[0][0].numel(),
            hidden_dim=hidden_dim, num_layers=num_layers, num_classes=NUM_CLASSES
        ).to(DEVICE)

        optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        criterion = torch.nn.CrossEntropyLoss()

        train_model(model, "MLP", NUM_EPOCH_FINAL, criterion, optimizer, train_loader, DEVICE)
        y_true, y_pred, y_scores = evaluate_model(model, test_loader, DEVICE)

        y_true_all.extend(y_true)
        y_pred_all.extend(y_pred)
        y_scores_all.extend(y_scores)

    # Compute final AUC and accuracy measures
    final_auc = roc_auc_score(y_true_all, y_scores_all, multi_class="ovr", average="macro")
    fpr, tpr, auc_dict, accuracy, specificity, sensitivity = get_accuracy_measures(y_true_all, y_pred_all, y_scores_all, NUM_CLASSES)

    output_data[FC_name].update({
        "MLP": {
            "best_params": best_params,
            "FPR": fpr,
            "TPR": tpr,
            "AUC": {0: final_auc},
            "Acc": accuracy,
            "Spec": specificity,
            "Sens": sensitivity
        }
    })


def main():
    """
    Main execution function to process all Functional Connectivity (FC) metrics.
    """
    # Check CUDA availability and print CUDA info
    if torch.cuda.is_available():
        print(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
    else:
        print("CUDA is not available. Using CPU.")

    load_datasets()

    for FC_name in os.listdir(FC_DATA_PATH):
        basepath = os.path.join(FC_DATA_PATH, FC_name)
        dataset = get_files_by_class(basepath)
        splits = split_datasets(dataset)
        full_splits = splits["full"]

        # Convert dataset into PyTorch format
        full_dataset = MatlabDataset(file_list=full_splits, tensor=True)
        train_mlp_model(FC_name, full_dataset)

    # Save results to JSON
    json_filename = save_to_json(output_data)
    print(f"✅ Results saved: {json_filename}")


if __name__ == "__main__":
    main()
