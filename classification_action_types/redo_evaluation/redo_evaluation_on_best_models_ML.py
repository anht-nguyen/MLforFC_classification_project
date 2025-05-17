import os
import sys
import json
import numpy as np

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.model_selection import RepeatedStratifiedKFold

# Determine project root and add to path
this_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(this_dir, os.pardir))
sys.path.insert(0, project_root)

# Imports from project
from scripts.datasets_loader import load_datasets
from scripts.utils import get_files_by_class, split_datasets, flatten_transform, dataset_type_converter
from scripts.models.ml_models_cores import MatlabDataset
from scripts.config import (
    FC_DATA_PATH,
    K_FOLDS,
    NUM_REPEATS_FINAL,
    NUM_CLASSES
)

model_names = ["SVC", "LogReg", "RFC"]

if __name__ == '__main__':
    # Ensure data is loaded
    load_datasets()

    # Load best hyperparameters from previous tuning
    json_path = os.path.join(this_dir, f"output_data-merged-SK.json")
    metrics_store = json.load(open(json_path, 'r'))
    
    for model_name in model_names:


        all_evals = {}

        for FC_name, metrics in metrics_store.items():

            print(f"\n=== {model_name} FC = {FC_name} ===")
            best_params = metrics[model_name]['best_params']
            print('Best hyperparameters:', json.dumps(best_params, indent=2))

            all_evals[FC_name] = {}

            # Prepare full dataset
            basepath = os.path.join(FC_DATA_PATH, FC_name)
            files = get_files_by_class(basepath)
            splits = split_datasets(files)
            matlab_ds = MatlabDataset(file_list=splits['full'], tensor=False, transform=flatten_transform)
            data = dataset_type_converter(matlab_ds)
            X_full = np.array(data['x'])
            y_full = np.array(data['y'])

            # Cross-validator
            cv = RepeatedStratifiedKFold(
                n_splits=K_FOLDS,
                n_repeats=NUM_REPEATS_FINAL,
                random_state=42
            )

        # Loop over each classifier
        # for clf_name, params in classifiers.items():
        #     print(f"-> {clf_name} with params: {params}")

            # Instantiate classifier
            if model_name == 'SVC':
                model = SVC(
                    C=best_params['svc_C'],
                    gamma=best_params['svc_gamma'],
                    probability=True
                )
            elif model_name == 'LogReg':
                model = LogisticRegression(
                    C=best_params['logreg_C'],
                    max_iter=1000
                )
            else:  # RFC
                model = RandomForestClassifier(
                    n_estimators=best_params['rfc_n_estimators'],
                    max_depth=best_params['rfc_max_depth'],
                    random_state=42
                )

            # Storage for predictions
            y_true_all, y_pred_all = [], []
            y_scores_all = []

            # Perform cross-validated evaluation
            for fold, (train_idx, test_idx) in enumerate(cv.split(X_full, y_full), start=1):
                print(f"  Fold {fold}/{cv.get_n_splits()}")
                X_train, X_test = X_full[train_idx], X_full[test_idx]
                y_train, y_test = y_full[train_idx], y_full[test_idx]

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_score = model.predict_proba(X_test)

                y_true_all.extend(y_test)
                y_pred_all.extend(y_pred)
                y_scores_all.extend(y_score)

            y_true_arr = np.array(y_true_all)
            y_pred_arr = np.array(y_pred_all)
            y_scores_arr = np.vstack(y_scores_all)

            # Aggregate metrics
            metrics = {
                'Accuracy': float(accuracy_score(y_true_arr, y_pred_arr)),
                'AUC': float(roc_auc_score(y_true_arr, y_scores_arr, multi_class='ovr', average='macro')),
                'Confusion Matrix': confusion_matrix(y_true_arr, y_pred_arr).tolist()
            }

            # Save per-classifier results
            all_evals[FC_name][model_name] = {
                'best_params': best_params,
                'metrics': metrics
            }

            # Display
            for m, v in metrics.items():
                print(f"    {m}: {v}")

    # Write all evaluation results
    out_path = os.path.join(this_dir, 're_evaluation_ML_results.json')
    with open(out_path, 'w') as f:
        json.dump(all_evals, f, indent=2)
    print(f"\n✔ All ML evaluation results saved to {out_path}")
