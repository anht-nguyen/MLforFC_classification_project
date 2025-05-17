import os
import sys
import json
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    jaccard_score
)
from sklearn.model_selection import RepeatedKFold
from torch.optim import Adam

# Project setup
this_dir     = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(this_dir, os.pardir))
sys.path.insert(0, project_root)

# Imports from project
from scripts.config import (
    FC_DATA_PATH,
    NUM_CLASSES,
    NUM_CHANNELS,
    NUM_FREQS,
    DEVICE,
    K_FOLDS,
    NUM_REPEATS_FINAL,
    NUM_EPOCH_FINAL,
    PATIENCE
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


def build_full_dataset(FC_name, model_name):
    # Load datasets keyed by FC metric
    FC_dataset, PSD_dataset = load_datasets()
    files = FC_dataset[FC_name]          # list of (path, label)
    splits = split_datasets(files)
    full_list = splits['full']
    if model_name == 'GCN':
        return GraphDataset(file_list=full_list, features=PSD_dataset)
    return MatlabDataset(file_list=full_list, tensor=True)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Re-evaluate DL models with full metrics')
    parser.add_argument('model', choices=['CNN','MLP','GCN'], help='Model to evaluate')
    args = parser.parse_args()
    selected_model = args.model

    # Ensure datasets loaded
    load_datasets()

    # Prepare checkpoint directory for predictions
    ckpt_dir = os.path.join(this_dir, 'dl_checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)

    # Load saved best hyperparameters
    meta_json = os.path.join(this_dir, f"output_data-2025-{selected_model}.json")
    meta = json.load(open(meta_json, 'r'))

    all_evals = {}

    for FC_name, content in meta.items():
        print(f"\n=== {selected_model} on FC = {FC_name} ===")
        best_params = content[selected_model]['best_params']
        print('Best params:', best_params)

        # Build dataset
        full_ds = build_full_dataset(FC_name, selected_model)

        # Cross-validator
        rkf = RepeatedKFold(n_splits=K_FOLDS,
                            n_repeats=NUM_REPEATS_FINAL,
                            random_state=42)

        y_true_all, y_pred_all, y_score_all = [], [], []
        train_splits, test_splits = [], []

        # Loop folds
        for fold, (train_idx, test_idx) in enumerate(rkf.split(full_ds), start=1):
            print(f"Fold {fold}/{K_FOLDS*NUM_REPEATS_FINAL}")
            train_splits.append(train_idx)
            test_splits.append(test_idx)

            # Create loaders
            bs = best_params.get('Batch Size', 64)
            if selected_model == 'GCN':
                from torch_geometric.loader import DataLoader as GeoLoader
                ds_tr = Subset(full_ds.graphs, train_idx)
                ds_te = Subset(full_ds.graphs, test_idx)
                loader_tr = GeoLoader(ds_tr, batch_size=bs, shuffle=True)
                loader_te = GeoLoader(ds_te, batch_size=bs, shuffle=False)
            else:
                ds_tr = Subset(full_ds, train_idx)
                ds_te = Subset(full_ds, test_idx)
                loader_tr = DataLoader(ds_tr, batch_size=bs, shuffle=True)
                loader_te = DataLoader(ds_te, batch_size=bs, shuffle=False)

            # Instantiate model
            if selected_model == 'CNN':
                model = CNNClassifier(
                    in_channels=1,
                    hidden_dim=best_params['CNN-hidden'],
                    num_classes=NUM_CLASSES,
                    kernel_size0=best_params['Kernel Size 0'],
                    kernel_size1=best_params['Kernel Size 1'],
                    padding=best_params['Padding']
                )
            elif selected_model == 'MLP':
                model = MLPClassifier(
                    input_size=NUM_CHANNELS,
                    hidden_dim=best_params['MLP-hidden'],
                    num_layers=best_params['N-MLP'],
                    num_classes=NUM_CLASSES
                )
            else:
                model = GCN(
                    in_channels=NUM_FREQS,
                    hidden_dim=best_params['GCN-hidden'],
                    num_layers=best_params['N-GCN'],
                    dropout=best_params['Dropout'],
                    drop_edge=best_params['DropEdge'],
                    num_classes=NUM_CLASSES
                )
            model.to(DEVICE)

            # Optimizer & loss
            optimizer = Adam(model.parameters(), lr=best_params['Learning Rate'], weight_decay=1e-4)
            criterion = torch.nn.CrossEntropyLoss()

            # Train
            if selected_model == 'GCN':
                train_gnn(model, loader_tr, optimizer, criterion, NUM_EPOCH_FINAL, DEVICE, PATIENCE)
                y_t, y_p, y_s = evaluate_gnn(model, loader_te, DEVICE)
            else:
                train_model(model, selected_model, NUM_EPOCH_FINAL, criterion,
                            optimizer, loader_tr, DEVICE, PATIENCE)
                y_t, y_p, y_s = evaluate_model(model, loader_te, DEVICE)

            y_true_all.extend(y_t)
            y_pred_all.extend(y_p)
            y_score_all.extend(y_s)

        # Aggregate
        y_true = np.array(y_true_all)
        y_pred = np.array(y_pred_all)
        y_score = np.vstack(y_score_all)

        # Save predictions
        ckpt_file = os.path.join(ckpt_dir, f"{selected_model}_{FC_name}_checkpoint.npz")
        np.savez_compressed(ckpt_file,
                             y_true=y_true,
                             y_pred=y_pred,
                             y_score=y_score,
                             train_splits=train_splits,
                             test_splits=test_splits)

        # Compute metrics
        metrics = {}
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)
        # Precision/Recall/F1
        prec_mac, rec_mac, f1_mac, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
        prec_mic, rec_mic, f1_mic, _ = precision_recall_fscore_support(y_true, y_pred, average='micro')
        metrics.update({
            'precision_macro': prec_mac,
            'recall_macro': rec_mac,
            'f1_macro': f1_mac,
            'precision_micro': prec_mic,
            'recall_micro': rec_mic,
            'f1_micro': f1_mic
        })
        # Jaccard index
        metrics['jaccard_macro'] = jaccard_score(y_true, y_pred, average='macro')
        metrics['jaccard_micro'] = jaccard_score(y_true, y_pred, average='micro')
        # ROC AUC
        metrics['auc_macro'] = roc_auc_score(y_true, y_score, multi_class='ovr', average='macro')
        metrics['auc_micro'] = roc_auc_score(y_true, y_score, multi_class='ovr', average='micro')
        # Confusion matrix and classification report
        metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred).tolist()
        metrics['classification_report'] = classification_report(y_true, y_pred, output_dict=True)

        all_evals[FC_name] = {
            'best_params': best_params,
            'metrics': metrics,
            'checkpoint': ckpt_file
        }

        # Display summary
        print(json.dumps(metrics, indent=2))

    # Save summary
    out_json = os.path.join(this_dir, f"re_evaluation_{selected_model}_full_metrics.json")
    with open(out_json, 'w') as f:
        json.dump({selected_model: all_evals}, f, indent=2)
    print(f"\n✅ Saved full metrics to {out_json}")

if __name__ == '__main__':
    main()