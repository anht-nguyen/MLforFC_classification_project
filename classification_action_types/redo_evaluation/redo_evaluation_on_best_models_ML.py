import os
import sys
import json
import numpy as np

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import RepeatedStratifiedKFold

# Add project root to path
this_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(this_dir, os.pardir))
sys.path.insert(0, project_root)

# Project imports
from scripts.datasets_loader import load_datasets
from scripts.utils import get_files_by_class, split_datasets, flatten_transform, dataset_type_converter
from scripts.models.ml_models_cores import MatlabDataset
from scripts.config import FC_DATA_PATH, K_FOLDS, NUM_REPEATS_FINAL

# Define which classifiers to evaluate
model_names = ["SVC", "LogReg", "RFC"]
# Path to tuned hyperparameters JSON
HP_JSON = os.path.join(this_dir, 'output_data-merged-SK.json')
# Output results file
OUT_JSON = os.path.join(this_dir, 're_evaluation_ML_results.json')


def build_model(name, params):
    """Instantiate a classifier given its name and parameter dict."""
    if name == 'SVC':
        return SVC(C=params['svc_C'], gamma=params['svc_gamma'], probability=True)
    elif name == 'LogReg':
        return LogisticRegression(C=params['logreg_C'], max_iter=1000)
    elif name == 'RFC':
        return RandomForestClassifier(n_estimators=params['rfc_n_estimators'],
                                      max_depth=params['rfc_max_depth'], random_state=42)
    else:
        raise ValueError(f"Unknown model name: {name}")


def load_all_datasets(fc_names):
    """Load and preprocess datasets for each FC name."""
    data_store = {}
    for fc in fc_names:
        base = os.path.join(FC_DATA_PATH, fc)
        files = get_files_by_class(base)
        splits = split_datasets(files)
        ds = MatlabDataset(file_list=splits['full'], tensor=False, transform=flatten_transform)
        arr = dataset_type_converter(ds)
        X = np.array(arr['x'])
        y = np.array(arr['y'])
        data_store[fc] = (X, y)
    return data_store


if __name__ == '__main__':
    # Ensure datasets are available
    load_datasets()

    # Load hyperparameters
    with open(HP_JSON, 'r') as f:
        hp_store = json.load(f)

    # Preload data for each FC
    fc_names = list(hp_store.keys())
    datasets = load_all_datasets(fc_names)

    # Prepare cross-validator
    cv = RepeatedStratifiedKFold(n_splits=K_FOLDS, n_repeats=NUM_REPEATS_FINAL, random_state=42)

    results = {}

    # Evaluate each model across all FC datasets
    for model_name in model_names:
        print(f"\n*** Evaluating {model_name} ***")
        results[model_name] = {}
        for fc, hp_entry in hp_store.items():
            print(f"\n-> Dataset: {fc}")
            params = hp_entry[model_name]['best_params']
            print("  Params:", params)

            model = build_model(model_name, params)
            X, y = datasets[fc]

            # Collect predictions
            y_true, y_pred, y_score = [], [], []
            for fold, (train_idx, test_idx) in enumerate(cv.split(X, y), start=1):
                print(f"  Fold {fold}/{cv.get_n_splits()}")
                X_tr, X_te = X[train_idx], X[test_idx]
                y_tr, y_te = y[train_idx], y[test_idx]

                model.fit(X_tr, y_tr)
                y_pred_fold = model.predict(X_te)
                y_score_fold = model.predict_proba(X_te)

                y_true.extend(y_te)
                y_pred.extend(y_pred_fold)
                y_score.extend(y_score_fold)

            y_true = np.array(y_true)
            y_pred = np.array(y_pred)
            y_score = np.vstack(y_score)

            # Compute metrics
            acc = accuracy_score(y_true, y_pred)
            auc = roc_auc_score(y_true, y_score, multi_class='ovr', average='macro')
            cm = confusion_matrix(y_true, y_pred).tolist()

            results[model_name][fc] = {
                'best_params': params,
                'metrics': {
                    'accuracy': float(acc),
                    'auc_macro': float(auc),
                    'confusion_matrix': cm,
                }
            }

            print(f"  Accuracy: {acc:.4f}, AUC: {auc:.4f}")

    # Save all results
    with open(OUT_JSON, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✔ Results written to {OUT_JSON}")
