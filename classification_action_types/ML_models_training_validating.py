import os
import optuna
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import RepeatedStratifiedKFold
from functools import partial

from scripts.datasets_loader import load_datasets, FC_dataset
from scripts.utils import get_files_by_class, split_datasets, flatten_transform, dataset_type_converter, get_accuracy_measures 
from scripts.config import FC_DATA_PATH, OPTIMIZER_TRIALS, K_FOLDS, NUM_REPEATS, NUM_CLASSES
from scripts.save_results import save_to_json
from scripts.models.ml_models_cores import objective_svc, objective_logreg, objective_rfc, MatlabDataset

# Initialize output dictionary
output_data = {}

# Load datasets
load_datasets()

# Process each functional connectivity metric
for FC_name in os.listdir(FC_DATA_PATH):
    print("************************")
    print("FC metric: " + str(FC_name))
    basepath = os.path.join(FC_DATA_PATH, FC_name)
    dataset = get_files_by_class(basepath)
    splits = split_datasets(dataset)

    # Create datasets
    full_matlabdataset = MatlabDataset(file_list=splits['full'], tensor=False, transform=flatten_transform)
    full_dataset = dataset_type_converter(full_matlabdataset)
    full_x, full_y = np.array(full_dataset['x']), np.array(full_dataset['y'])

    # Tune hyperparameters separately for each classifier
    classifiers = {
        "SVC": objective_svc,
        "LogReg": objective_logreg,
        "RFC": objective_rfc
    }

    best_models = {}

    for classifier_name, objective_func in classifiers.items():
        print(f"Optimizing {classifier_name}...")
        study = optuna.create_study(direction="maximize")
        study.optimize(partial(objective_func, train_x=full_x, train_y=full_y), n_trials=OPTIMIZER_TRIALS)

        best_models[classifier_name] = study.best_params
        print(f"Best hyperparameters for {classifier_name}: {study.best_params}")

    # Evaluate best models with cross-validation
    for classifier_name, best_params in best_models.items():
        print(f"Final model training for {classifier_name}...")

        if classifier_name == "SVC":
            best_model = SVC(C=best_params["svc_C"], gamma=best_params["svc_gamma"], probability=True)
        elif classifier_name == "LogReg":
            best_model = LogisticRegression(C=best_params["logreg_C"], max_iter=1000)
        else:
            best_model = RandomForestClassifier(n_estimators=best_params["rfc_n_estimators"], max_depth=best_params["rfc_max_depth"])

        cv = RepeatedStratifiedKFold(n_splits=K_FOLDS, n_repeats=NUM_REPEATS, random_state=42)
        y_true_all, y_pred_all, y_scores_all = [], [], []

        for fold, (train_idx, test_idx) in enumerate(cv.split(full_x, full_y)):
            print(f"Final Model Training - Fold {fold+1}/{cv.get_n_splits()}")

            X_train, X_test = full_x[train_idx], full_x[test_idx]
            y_train, y_test = full_y[train_idx], full_y[test_idx]

            best_model.fit(X_train, y_train)
            y_pred = best_model.predict(X_test)
            y_scores = best_model.predict_proba(X_test)

            y_true_all.extend(y_test)
            y_pred_all.extend(y_pred)
            y_scores_all.extend(y_scores)

        final_accuracy = accuracy_score(y_true_all, y_pred_all)
        final_auc = roc_auc_score(np.array(y_true_all), np.array(y_scores_all), multi_class="ovr", average="macro")

        print(f"Final Model Accuracy for {classifier_name} (Cross-Validation): {final_accuracy:.4f}")
        print(f"Final Model AUC for {classifier_name} (Cross-Validation): {final_auc:.4f}")

        cm_df = pd.DataFrame(confusion_matrix(y_true_all, y_pred_all))
        # plot_confusion_matrix(cm_df, title=f'{classifier_name} Confusion Matrix')
        fpr, tpr, auc_dict, accuracy, specificity, sensitivity =  get_accuracy_measures(y_true_all, y_pred_all, y_scores_all, NUM_CLASSES)
        output_data[FC_name].update({classifier_name: {"best_params":best_params, "FPR": fpr, "TPR": tpr, "AUC": {0: final_auc}, "Acc": accuracy, "Spec": specificity, "Sens": sensitivity}})

# Save results
json_filename = save_to_json(output_data, "ML_models")
print("Saved file:", json_filename)