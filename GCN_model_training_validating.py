import os
import optuna
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.model_selection import KFold
import torch
from torch.optim import Adam
from torch_geometric.loader import DataLoader

from scripts.datasets_loader import load_datasets
from scripts.utils import get_files_by_class, split_datasets, get_accuracy_measures
from scripts.config import (
    FC_DATA_PATH, OPTIMIZER_TRIALS, K_FOLDS, NUM_CLASSES, DEVICE,
    NUM_EPOCH_TRAINING, NUM_EPOCH_FINAL, NUM_FREQS
)
from scripts.save_results import save_to_json
from scripts.models.dl_models_cores import train_gnn, evaluate_gnn, GCN, GraphDataset

# ✅ Enable CUDA performance optimizations
torch.backends.cudnn.benchmark = True  # Optimizes CNNs on fixed input sizes
torch.backends.cuda.matmul.allow_tf32 = True  # Allows TF32 precision for better speed
torch.backends.cudnn.deterministic = False  # Enables non-deterministic algorithms

# Store output results
output_data = {}

def gcn_objective(trial, full_dataset, num_classes, device):
    """
    Objective function for Optuna hyperparameter tuning of GCN.
    """
    print(f"🔍 Running Optuna Trial {trial.number}...")

    # Hyperparameter search space
    num_layers = trial.suggest_int("N-GCN", 1, 3)
    hidden_dim = trial.suggest_categorical("GCN-hidden", [32, 64, 128])
    drop_edge = trial.suggest_float("DropEdge", 0.1, 0.5, step=0.1)
    dropout = trial.suggest_float("Dropout", 0.1, 0.5, step=0.1)
    lr = trial.suggest_categorical("Learning Rate", [0.001, 0.005, 0.01])
    batch_size = trial.suggest_categorical("Batch Size", [8, 16, 32])

    auc_scores = []
    kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=42)

    for fold, (train_idx, test_idx) in enumerate(kf.split(full_dataset)):
        print(f"📊 Hyperparameter Tuning - Fold {fold+1}/{K_FOLDS}")

        # Create training and testing subsets
        train_subset = torch.utils.data.Subset(full_dataset.graphs, train_idx)
        test_subset = torch.utils.data.Subset(full_dataset.graphs, test_idx)

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
        test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

        # Initialize model with trial-selected hyperparameters
        model = GCN(
            in_channels=NUM_FREQS,  
            hidden_dim=hidden_dim, num_classes=num_classes,
            num_layers=num_layers, dropout=dropout, drop_edge=drop_edge
        ).to(device)

        optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        criterion = torch.nn.CrossEntropyLoss()

        # Train and evaluate
        train_gnn(model, train_loader, optimizer, criterion, NUM_EPOCH_TRAINING, device)
        y_true, y_pred, y_scores = evaluate_gnn(model, test_loader, device)

        # Compute AUC for this fold
        fold_auc = roc_auc_score(y_true, y_scores, multi_class="ovr", average="macro")
        auc_scores.append(fold_auc)

    return np.mean(auc_scores)


def train_gcn_model(FC_name, full_dataset):
    """
    Conducts hyperparameter tuning and final training for GCN.
    """
    print(f"🚀 Starting Training for FC Metric: {FC_name}")
    output_data[FC_name] = {}

    # Perform hyperparameter tuning using Optuna
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: gcn_objective(trial, full_dataset, NUM_CLASSES, DEVICE), n_trials=OPTIMIZER_TRIALS)
    
    print("🏆 Best GCN Hyperparameters:", study.best_params)
    best_params = study.best_params

    # Extract best hyperparameters
    batch_size = best_params["Batch Size"]
    hidden_dim = best_params["GCN-hidden"]
    num_layers = best_params["N-GCN"]
    drop_edge = best_params["DropEdge"]
    dropout = best_params["Dropout"]
    lr = best_params["Learning Rate"]

    kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
    y_true_all, y_pred_all, y_scores_all = [], [], []

    for fold, (train_idx, test_idx) in enumerate(kf.split(full_dataset)):
        print(f"📊 Final Training - Fold {fold+1}/{K_FOLDS}")

        # Train-test split
        train_subset = torch.utils.data.Subset(full_dataset.graphs, train_idx)
        test_subset = torch.utils.data.Subset(full_dataset.graphs, test_idx)

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True)
        test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, persistent_workers=True)

        # Initialize final model
        model = GCN(
            in_channels=NUM_FREQS,  
            hidden_dim=hidden_dim, num_classes=NUM_CLASSES,
            num_layers=num_layers, dropout=dropout, drop_edge=drop_edge
        ).to(DEVICE)

        optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        criterion = torch.nn.CrossEntropyLoss()

        train_gnn(model, train_loader, optimizer, criterion, NUM_EPOCH_FINAL, DEVICE)
        y_true, y_pred, y_scores = evaluate_gnn(model, test_loader, DEVICE)

        y_true_all.extend(y_true)
        y_pred_all.extend(y_pred)
        y_scores_all.extend(y_scores)

    # Compute final AUC and accuracy measures
    final_auc = roc_auc_score(y_true_all, y_scores_all, multi_class="ovr", average="macro")
    fpr, tpr, auc_dict, accuracy, specificity, sensitivity = get_accuracy_measures(y_true_all, y_pred_all, y_scores_all, NUM_CLASSES)

    output_data[FC_name].update({
        "GCN": {
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
        print(f"🚀 CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️ CUDA is not available. Using CPU.")

    FC_dataset, PSD_dataset = load_datasets()

    for FC_name in os.listdir(FC_DATA_PATH):
        basepath = os.path.join(FC_DATA_PATH, FC_name)
        dataset = get_files_by_class(basepath)
        splits = split_datasets(dataset)
        full_splits = splits["full"]

        # Convert dataset into PyTorch format
        full_dataset = GraphDataset(file_list=full_splits, features=PSD_dataset)
        train_gcn_model(FC_name, full_dataset)

    # Save results to JSON
    json_filename = save_to_json(output_data)
    print(f"✅ Results saved: {json_filename}")


if __name__ == "__main__":
    main()
