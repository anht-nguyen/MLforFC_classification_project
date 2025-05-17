import os
import sys
import json
import torch
import numpy as np
import torch_geometric

# Add project root to path
this_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(this_dir, os.pardir))
sys.path.insert(0, project_root)

# Project imports
def main():
    import argparse
    from scripts.config import (
        FC_DATA_PATH, NUM_CLASSES, NUM_CHANNELS, NUM_FREQS,
        DEVICE, K_FOLDS, NUM_REPEATS_FINAL,
        NUM_EPOCH_FINAL, PATIENCE
    )
    from scripts.datasets_loader import load_datasets
    from scripts.utils import get_files_by_class, split_datasets
    from scripts.models.dl_models_cores import (
        CNNClassifier, MLPClassifier, GCN,
        GraphDataset, train_model, evaluate_model,
        train_gnn, evaluate_gnn
    )
    from scripts.models.ml_models_cores import MatlabDataset
    from sklearn.metrics import (
        confusion_matrix, accuracy_score,
        precision_recall_fscore_support,
        balanced_accuracy_score, roc_auc_score
    )
    from torch.utils.data import DataLoader, Subset
    from sklearn.model_selection import RepeatedKFold
    from torch.optim import Adam

    # Argument parser
    parser = argparse.ArgumentParser(
        description="Re-evaluate best DL models with checkpointing"
    )
    parser.add_argument(
        "model", choices=["GCN","CNN","MLP"],
        help="Which model to re-evaluate"
    )
    args = parser.parse_args()
    selected_model = args.model
    print(f"Selected model: {selected_model}")

    # Ensure datasets loaded
    load_datasets()

    # Prepare checkpoint dir
    ckpt_dir = os.path.join(this_dir, 'dl_checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)

    # Load prior metrics/hyperparams
    meta_json = os.path.join(this_dir, f"output_data-merged-{selected_model}.json")
    meta = json.load(open(meta_json, 'r'))

    all_evals = {}
    # Iterate FC measures
    for FC_name, content in meta.items():
        print(f"\n=== {selected_model} on FC={FC_name} ===")
        best_params = content[selected_model]['best_params']
        print("Params:", best_params)
        all_evals[FC_name] = {}

        # Build dataset
        FC_dataset, PSD_dataset = load_datasets()
        base = os.path.join(FC_DATA_PATH, FC_name)
        files = get_files_by_class(base)
        splits = split_datasets(files)['full']
        if selected_model == 'GCN':
            full_ds = GraphDataset(file_list=splits, features=PSD_dataset)
        else:
            full_ds = MatlabDataset(file_list=splits, tensor=True)

        rkf = RepeatedKFold(n_splits=K_FOLDS, n_repeats=NUM_REPEATS_FINAL, random_state=42)

        y_true_all, y_pred_all, y_score_all = [], [], []
        train_list, test_list = [], []

        for fold, (train_idx, test_idx) in enumerate(rkf.split(full_ds), start=1):
            print(f"Fold {fold}/{K_FOLDS*NUM_REPEATS_FINAL}")
            train_list.append(train_idx)
            test_list.append(test_idx)

            # Create splits and loaders
            if selected_model == 'GCN':
                from torch_geometric.loader import DataLoader as GeoLoader
                ds_tr = Subset(full_ds.graphs, train_idx)
                ds_te = Subset(full_ds.graphs, test_idx)
                loader_tr = GeoLoader(ds_tr, batch_size=best_params['Batch Size'], shuffle=True)
                loader_te = GeoLoader(ds_te, batch_size=best_params['Batch Size'], shuffle=False)
            else:
                ds_tr = Subset(full_ds, train_idx)
                ds_te = Subset(full_ds, test_idx)
                loader_tr = DataLoader(ds_tr, batch_size=best_params.get('Batch Size',64), shuffle=True)
                loader_te = DataLoader(ds_te, batch_size=best_params.get('Batch Size',64), shuffle=False)

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
            else:  # GCN
                model = GCN(
                    in_channels=NUM_FREQS,
                    hidden_dim=best_params['GCN-hidden'],
                    num_layers=best_params['N-GCN'],
                    dropout=best_params['Dropout'],
                    drop_edge=best_params['DropEdge'],
                    num_classes=NUM_CLASSES
                )
            model.to(DEVICE)

            optimizer = Adam(model.parameters(), lr=best_params['Learning Rate'], weight_decay=1e-4)
            criterion = torch.nn.CrossEntropyLoss()

            # Train
            if selected_model == 'GCN':
                train_gnn(model, loader_tr, optimizer, criterion, NUM_EPOCH_FINAL, DEVICE, PATIENCE, FC_name)
            else:
                train_model(model, selected_model, NUM_EPOCH_FINAL, criterion, optimizer, loader_tr, DEVICE, PATIENCE, FC_name)

            # Evaluate
            if selected_model == 'GCN':
                y_t, y_p, y_s = evaluate_gnn(model, loader_te, DEVICE)
            else:
                y_t, y_p, y_s = evaluate_model(model, loader_te, DEVICE)

            y_true_all.extend(y_t)
            y_pred_all.extend(y_p)
            y_score_all.extend(y_s)

        # Convert
        y_true = np.array(y_true_all)
        y_pred = np.array(y_pred_all)
        y_score = np.vstack(y_score_all)

        # Save checkpoint
        ckpt_file = os.path.join(ckpt_dir, f"{selected_model}_{FC_name}_checkpoint.npz")
        np.savez_compressed(
            ckpt_file,
            y_true=y_true,
            y_pred=y_pred,
            y_score=y_score,
            train_idx=train_list,
            test_idx=test_list
        )

        # Metrics
        results = {
            'Accuracy': float(accuracy_score(y_true, y_pred)),
            'AUC': float(roc_auc_score(y_true, y_score, multi_class='ovr', average='macro')),
            'Balanced Accuracy': float(balanced_accuracy_score(y_true, y_pred)),
        }
        all_evals[FC_name] = {
            'best_params': best_params,
            'metrics': results,
            'checkpoint': ckpt_file
        }
        for k, v in results.items(): print(f"{k}: {v}")

    # Write summary
    out_json = os.path.join(this_dir, f"re_evaluation_{selected_model}_results.json")
    with open(out_json, 'w') as f:
        json.dump({selected_model: all_evals}, f, indent=2)
    print(f"\n✔ Saved metrics at {out_json} and checkpoints in {ckpt_dir}")

if __name__ == '__main__':
    main()
