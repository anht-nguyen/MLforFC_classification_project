import argparse
import os
import sys
import json
import torch
import numpy as np
import torch_geometric

def main():
    # 1. Parse command-line argument
    parser = argparse.ArgumentParser(
        description="Re-evaluate best models: choose one of GCN, CNN, or MLP"
    )
    parser.add_argument(
        "model",
        choices=["GCN", "CNN", "MLP"],
        help="The model architecture to re-evaluate"
    )
    args = parser.parse_args()
    selected_model = args.model
    print(f"Selected model for re-evaluation: {selected_model}")

    # 2. Determine project root and add to path
    this_dir     = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(this_dir, os.pardir))
    sys.path.insert(0, project_root)

    # 3. Imports
    from scripts.config import (
        FC_DATA_PATH,
        NUM_CLASSES,
        NUM_CHANNELS,
        NUM_FREQS,
        DEVICE,
        K_FOLDS,
        NUM_REPEATS_FINAL,
        NUM_EPOCH_FINAL,
        PATIENCE,
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
        evaluate_gnn,
    )
    from scripts.models.ml_models_cores import MatlabDataset
    from sklearn.metrics import (
        confusion_matrix,
        accuracy_score,
        precision_recall_fscore_support,
        balanced_accuracy_score,
        roc_auc_score
    )
    from torch.utils.data import DataLoader, Subset
    from sklearn.model_selection import RepeatedKFold
    from torch.optim import Adam

    # 4. Use the parsed argument
    model_names = [selected_model]


    # Build full dataset for final evaluation
    def build_full_dataset(FC_name, model_name):
        FC_dataset, PSD_dataset = load_datasets()

        basepath = os.path.join(FC_DATA_PATH, FC_name)
        if not os.path.exists(basepath):
            print(f"❌ FC measure '{FC_name}' not found in {FC_DATA_PATH}")
            return
        
        dataset = get_files_by_class(basepath)
        splits = split_datasets(dataset) 
        full_splits = splits['full']
        if model_name == 'GCN':
            return GraphDataset(file_list=full_splits, features=PSD_dataset)
        else:
            return MatlabDataset(file_list=full_splits, tensor=True)

    if __name__ == '__main__':
        # Container to hold all metrics
        all_evals = {}

        for model_name in model_names:
            json_path = os.path.join(this_dir, f"output_data-merged-{model_name}.json")
            metrics_store = json.load(open(json_path, 'r'))
            all_evals[model_name] = {}

            for FC_name, metrics in metrics_store.items():
                print(f"\n=== {model_name} on FC = {FC_name} ===")
                best_params = metrics[model_name]['best_params']
                print('Best hyperparameters:', json.dumps(best_params, indent=2))

                # Prepare dataset and cross-validator
                full_ds = build_full_dataset(FC_name, model_name)
                rkf     = RepeatedKFold(
                    n_splits=K_FOLDS,
                    n_repeats=NUM_REPEATS_FINAL,
                    random_state=42
                )

                # Accumulate across folds
                y_true_all, y_pred_all, y_scores_all = [], [], []

                for fold, (train_idx, test_idx) in enumerate(rkf.split(full_ds)):
                    print(f"Fold {fold}/{K_FOLDS * NUM_REPEATS_FINAL}")

                    train_ds = Subset(full_ds, train_idx)
                    test_ds  = Subset(full_ds, test_idx)

                    bs = best_params.get('Batch Size', 64)
                    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True)
                    test_loader  = DataLoader(test_ds,  batch_size=bs, shuffle=False)

                    # Instantiate model
                    if model_name == 'CNN':
                        model = CNNClassifier(
                            in_channels=1,
                            hidden_dim=best_params['CNN-hidden'],
                            num_classes=NUM_CLASSES,
                            kernel_size0=best_params['Kernel Size 0'],
                            kernel_size1=best_params['Kernel Size 1'],
                            padding=best_params['Padding'],
                        )
                    elif model_name == 'MLP':
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

                    optimizer = Adam(
                        model.parameters(),
                        lr=best_params['Learning Rate'],
                        weight_decay=1e-4
                    )
                    criterion = torch.nn.CrossEntropyLoss()

                    # Train and evaluate
                    if model_name == 'GCN':
                        from torch_geometric.loader import DataLoader
                        train_ds = Subset(full_ds.graphs, train_idx)
                        test_ds  = Subset(full_ds.graphs, test_idx)  
                        train_loader = DataLoader(train_ds, batch_size=best_params['Batch Size'], shuffle=True, num_workers=8, pin_memory=True)
                        test_loader = DataLoader(test_ds, batch_size=best_params['Batch Size'], shuffle=False, num_workers=8, pin_memory=True)

                        train_gnn(model, train_loader, optimizer,
                                criterion, NUM_EPOCH_FINAL,
                                DEVICE, PATIENCE, FC_name)
                        y_t, y_p, y_s = evaluate_gnn(model, test_loader, DEVICE)
                    else:
                        train_model(model, model_name,
                                    NUM_EPOCH_FINAL, criterion,
                                    optimizer, train_loader,
                                    DEVICE, PATIENCE, FC_name)
                        y_t, y_p, y_s = evaluate_model(model, test_loader, DEVICE)

                    y_true_all.extend(y_t)
                    y_pred_all.extend(y_p)
                    y_scores_all.extend(y_s)

                # Compute aggregated metrics
                y_true_arr   = np.array(y_true_all)
                y_pred_arr   = np.array(y_pred_all)
                y_scores_arr = np.vstack(y_scores_all)

                eval_results = {
                    'AUC': float(roc_auc_score(y_true_arr, y_scores_arr,
                                                multi_class='ovr', average='macro')),
                    'Accuracy': float(accuracy_score(y_true_arr, y_pred_arr)),
                    'Balanced Accuracy': float(balanced_accuracy_score(y_true_arr, y_pred_arr)),
                    'Macro Precision': float(precision_recall_fscore_support(
                        y_true_arr, y_pred_arr, average='macro')[0]),
                    'Macro Recall': float(precision_recall_fscore_support(
                        y_true_arr, y_pred_arr, average='macro')[1]),
                    'Macro F1': float(precision_recall_fscore_support(
                        y_true_arr, y_pred_arr, average='macro')[2]),
                    'Confusion Matrix': confusion_matrix(y_true_arr, y_pred_arr).tolist()
                }

                # Save to container
                all_evals[model_name][FC_name] = {
                    'best_params': best_params,
                    'metrics': eval_results
                }

                # Display
                for k, v in eval_results.items():
                    print(f"{k}: {v}")

        # Write all evaluation results to JSON
        out_path = os.path.join(this_dir, 're_evaluation_results.json')
        with open(out_path, 'w') as f:
            json.dump(all_evals, f, indent=2)
        print(f"\n✔ All evaluation results saved to {out_path}")


if __name__ == "__main__":
    main()