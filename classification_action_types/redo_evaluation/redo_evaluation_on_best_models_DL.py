import os
import sys
import json
import torch
import numpy as np

from torch.utils.data import DataLoader, Subset
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support,
    balanced_accuracy_score,
    roc_auc_score
)
from sklearn.model_selection import RepeatedKFold
from torch.optim import Adam

# Determine project root and add to path
this_dir     = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(this_dir, os.pardir))
sys.path.insert(0, project_root)

# Import project modules
from scripts.config import (
    MODEL_NAMES,
    FC_METRICS,
    NUM_CLASSES,
    NUM_CHANNELS,
    NUM_FREQS,
    DEVICE,
    K_FOLDS,
    NUM_REPEATS_FINAL,
    NUM_EPOCH_FINAL,
    PATIENCE,
    FC_DATA_PATH
)
from scripts.datasets_loader import load_datasets
from scripts.utils import get_files_by_class, split_datasets
from scripts.models.dl_models_cores import (
    CNNClassifier,
    MLPClassifier,
    GCN,
    GraphDataset,
    train_model,
    evaluate_model,
    train_gnn,
    evaluate_gnn
)
from scripts.models.ml_models_cores import MatlabDataset

# Models to re-evaluate
model_names = ['CNN', 'MLP', 'GCN']

# Build full dataset for final evaluation
def build_full_dataset(FC_name, model_name):
    FC_dataset, PSD_dataset = load_datasets()
    basepath = os.path.join(FC_DATA_PATH, FC_name)
    dataset = get_files_by_class(basepath)
    splits = split_datasets(dataset)
    full_splits = splits["full"]
    if model_name == 'GCN':
        return GraphDataset(file_list=full_splits, features=PSD_dataset)
    else:
        return MatlabDataset(file_list=full_splits, tensor=True)

if __name__ == '__main__':
    for model_name in model_names:
        json_path = os.path.join(this_dir, f"output_data-merged-{model_name}.json")
        results   = json.load(open(json_path, 'r'))

        for FC_name, metrics in results.items():
            print(f"\n=== {model_name} on FC = {FC_name} ===")
            best_params = metrics[model_name]['best_params']
            print('Best hyperparameters:', json.dumps(best_params, indent=2))

            full_ds = build_full_dataset(FC_name, model_name)
            kf      = RepeatedKFold(n_splits=K_FOLDS,
                                   n_repeats=NUM_REPEATS_FINAL,
                                   random_state=42)

            y_true_all, y_pred_all, y_scores_all = [], [], []

            for fold, (train_idx, test_idx) in enumerate(kf.split(full_ds)):
                print(f"Fold {fold+1}/{K_FOLDS * NUM_REPEATS_FINAL}")
                train_ds = Subset(full_ds, train_idx)
                test_ds  = Subset(full_ds, test_idx)

                # DataLoaders
                bs = best_params.get('Batch Size', 64)
                train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True)
                test_loader  = DataLoader(test_ds,  batch_size=bs, shuffle=False)

                # Instantiate model
                if model_name == 'GCN':
                    model = GCN(
                        in_channels=NUM_FREQS,
                        hidden_dim=best_params['GCN-hidden'],
                        num_layers=best_params['N-GCN'],
                        dropout=best_params['Dropout'],
                        drop_edge=best_params['DropEdge'],
                        num_classes=NUM_CLASSES
                    ).to(DEVICE)
                elif model_name == 'CNN':
                    model = CNNClassifier(
                        in_channels=1,
                        hidden_dim=best_params['CNN-hidden'],
                        num_classes=NUM_CLASSES,
                        kernel_size0=best_params['Kernel Size 0'],
                        kernel_size1=best_params['Kernel Size 1'],
                        padding=best_params['Padding']
                    ).to(DEVICE)
                else:
                    model = MLPClassifier(
                        input_size=NUM_CHANNELS,
                        hidden_dim=best_params['MLP-hidden'],
                        num_layers=best_params['N-MLP'],
                        num_classes=NUM_CLASSES
                    ).to(DEVICE)

                # Optimizer and loss
                optimizer = Adam(model.parameters(),
                                 lr=best_params['Learning Rate'],
                                 weight_decay=1e-4)
                criterion = torch.nn.CrossEntropyLoss()

                # Train and evaluate per model type
                if model_name == 'GCN':
                    train_gnn(
                        model, train_loader, optimizer,
                        criterion, NUM_EPOCH_FINAL,
                        DEVICE, PATIENCE, FC_name
                    )
                    y_true, y_pred, y_scores = evaluate_gnn(model, test_loader, DEVICE)
                else:
                    train_model(
                        model, model_name,
                        NUM_EPOCH_FINAL, criterion,
                        optimizer, train_loader,
                        DEVICE, PATIENCE, FC_name
                    )
                    y_true, y_pred, y_scores = evaluate_model(model, test_loader, DEVICE)

                y_true_all.extend(y_true)
                y_pred_all.extend(y_pred)
                y_scores_all.extend(y_scores)

            # Compute metrics
            y_true_arr   = np.array(y_true_all)
            y_pred_arr   = np.array(y_pred_all)
            y_scores_arr = np.vstack(y_scores_all)

            acc      = accuracy_score(y_true_arr, y_pred_arr)
            bal_acc  = balanced_accuracy_score(y_true_arr, y_pred_arr)
            prec_m, rec_m, f1_m, _ = precision_recall_fscore_support(
                y_true_arr, y_pred_arr, average='macro'
            )
            cm       = confusion_matrix(y_true_arr, y_pred_arr)
            auc      = roc_auc_score(y_true_arr, y_scores_arr,
                                     multi_class='ovr', average='macro')

            # Display
            print(f"AUC (macro OVR):     {auc:.3f}")
            print(f"Accuracy:            {acc:.3f}")
            print(f"Balanced Accuracy:   {bal_acc:.3f}")
            print(f"Macro Prec/Rec/F1:   {prec_m:.3f} / {rec_m:.3f} / {f1_m:.3f}")
            print("Confusion Matrix:")
            print(cm)