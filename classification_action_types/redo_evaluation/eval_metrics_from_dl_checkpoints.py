#!/usr/bin/env python3
import os
import sys
import json
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    jaccard_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# Determine project paths
this_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(this_dir, os.pardir))

# Import project configuration
from scripts.config import FC_DATA_PATH, FC_METRICS

# Models to process
model_names = ['CNN', 'MLP', 'GCN']

# Base folders
dl_ckpt_dir = os.path.join(this_dir, 'dl_checkpoints')
meta_dir = this_dir  # meta JSONs are stored alongside this script



for model_name in model_names:
    # Load hyperparameter metadata
    meta_path = os.path.join(meta_dir, f"output_data-merged-{model_name}.json")
    if not os.path.isfile(meta_path):
        print(f"Warning: meta file not found for model {model_name}: {meta_path}")
        continue
    meta = json.load(open(meta_path, 'r'))

    # Prepare per-model container
    all_evals = {}
    all_evals[model_name] = {}

    for FC_name in meta:
        entry = meta[FC_name].get(model_name, {})
        best_params = entry.get('best_params', {})

        # Load checkpoint of predictions
        ckpt_path = os.path.join(dl_ckpt_dir, f"{model_name}_{FC_name}_checkpoint.npz")
        if not os.path.isfile(ckpt_path):
            print(f"  Checkpoint not found: {ckpt_path}")
            continue
        data = np.load(ckpt_path, allow_pickle=True)
        y_true = data['y_true']
        y_pred = data['y_pred']
        y_score = data['y_score']

        # Compute a full suite of metrics
        metrics = {}
        # Basic accuracy
        metrics['accuracy'] = float(accuracy_score(y_true, y_pred))
        metrics['balanced_accuracy'] = float(balanced_accuracy_score(y_true, y_pred))
        # Precision/Recall/F1 (macro and micro)
        p_mac, r_mac, f1_mac, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
        p_mic, r_mic, f1_mic, _ = precision_recall_fscore_support(y_true, y_pred, average='micro')
        metrics.update({
            'precision_macro': float(p_mac),
            'recall_macro': float(r_mac),
            'f1_macro': float(f1_mac),
            'precision_micro': float(p_mic),
            'recall_micro': float(r_mic),
            'f1_micro': float(f1_mic)
        })
        # Jaccard index
        metrics['jaccard_macro'] = float(jaccard_score(y_true, y_pred, average='macro'))
        metrics['jaccard_micro'] = float(jaccard_score(y_true, y_pred, average='micro'))
        # ROC AUC (macro and micro)
        metrics['auc_macro'] = float(roc_auc_score(y_true, y_score, multi_class='ovr', average='macro'))
        metrics['auc_micro'] = float(roc_auc_score(y_true, y_score, multi_class='ovr', average='micro'))
        # Confusion matrix
        metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred).tolist()
        # Classification report (per class precision/recall/f1)
        metrics['classification_report'] = classification_report(y_true, y_pred, output_dict=True)

        # Assemble entry
        all_evals[model_name][FC_name] = {
            'best_params': best_params,
            'metrics': metrics,
            'checkpoint': ckpt_path
        }

        print(f"Processed {model_name} - {FC_name}: accuracy={metrics['accuracy']:.3f}, auc_macro={metrics['auc_macro']:.3f}")

    # Save results to JSON
    out_path = os.path.join(this_dir, f"re_evaluation_from_checkpoints_{model_name}.json")
    with open(out_path, 'w') as f:
        json.dump(all_evals, f, indent=2)

    print(f"Full evaluation results written to {out_path}")
