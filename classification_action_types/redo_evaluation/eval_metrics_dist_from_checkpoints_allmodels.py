#!/usr/bin/env python3
"""
Compute mean and 95 % CI (5th–95th percentile) for each metric from
saved DL checkpoints that contain *per‑fold* predictions.

A “CI” here is reported as a single conservative error bar:
    err = max(mean – q05 , q95 – mean)

Author: <your name>, 2025‑05‑18
"""
import os
import json
import numpy as np
from collections import defaultdict
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_recall_fscore_support,
    jaccard_score, roc_auc_score, confusion_matrix, classification_report
)

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
this_dir      = os.path.dirname(os.path.abspath(__file__))
dl_ckpt_dir   = os.path.join(this_dir, "dl_checkpoints")   # <- adjust if needed
ml_ckpt_dir   = os.path.join(this_dir, "ml_checkpoints")   # <- adjust if needed
meta_dir      = this_dir                                   # HP/meta JSONs live here
model_names   = ["CNN", "MLP", "GCN", "SVC", "LogReg", "RFC"]                      #  models of interest
# --------------------------------------------------------------------------- #


# ---------- helpers -------------------------------------------------------- #
def _fold_indices(z, model_name=None):
    """
    Yield the test indices of each fold stored in the checkpoint.

    Accepts either:
      * 'test_idx' : object array of index arrays
      * 'fold_ids' : same length as y_true, integer fold label
    Falls back to a single fold containing every sample.
    """
    
    if model_name in ["CNN", "MLP", "GCN"]:
        keys = ["train_splits", "test_splits"]
    else:
        keys = ["train_idx", "test_idx"]
    
    if keys[1] in z:
        for idx in z[keys[1]]:
            yield idx
    elif "fold_ids" in z:
        fid = z["fold_ids"]
        for f in np.unique(fid):
            yield np.where(fid == f)[0]
    else:                         # unknown layout -> treat as one fold
        n = len(z["y_true"])
        yield np.arange(n)


def _compute_metrics(y_true, y_pred, y_score):
    """
    Return a dict of scalar metrics for a single fold.
    """
    metrics = {}
    metrics["accuracy"]           = accuracy_score(y_true, y_pred)
    metrics["balanced_accuracy"]  = balanced_accuracy_score(y_true, y_pred)

    p_ma, r_ma, f1_ma, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0)
    p_mi, r_mi, f1_mi, _ = precision_recall_fscore_support(
        y_true, y_pred, average="micro", zero_division=0)

    metrics.update({
        "precision_macro": p_ma, "recall_macro": r_ma, "f1_macro": f1_ma,
        "precision_micro": p_mi, "recall_micro": r_mi, "f1_micro": f1_mi,
        "jaccard_macro":   jaccard_score(y_true, y_pred, average="macro"),
        "jaccard_micro":   jaccard_score(y_true, y_pred, average="micro"),
        "auc_macro":       roc_auc_score(y_true, y_score, multi_class="ovr",
                                         average="macro"),
    })
    return metrics


def _agg_distribution(values):
    """
    Given a list/array of numbers, return mean and conservative 95 % CI error.
    """
    arr = np.asarray(values, dtype=float)
    mean = float(arr.mean())
    q05, q95 = np.percentile(arr, [5, 95])
    err = float(max(mean - q05, q95 - mean))
    return {"mean": mean, "err": err}


# ---------------------------- main loop ----------------------------------- #
for model_name in model_names:

    if model_name in ["CNN", "MLP", "GCN"]:
        ckpt_dir = dl_ckpt_dir
        meta_path = os.path.join(meta_dir, f"output_data-merged-{model_name}.json")
        if not os.path.isfile(meta_path):
            print(f"[WARN] meta JSON not found for {model_name}: {meta_path}")
            continue
    else:
        ckpt_dir = ml_ckpt_dir
        meta_path = os.path.join(meta_dir, f"output_data-merged-SK.json")
        if not os.path.isfile(meta_path):
            print(f"[WARN] meta JSON not found for {model_name}: {meta_path}")
            continue

    meta = json.load(open(meta_path, "r"))

    all_results = defaultdict(dict)

    for fc_name in meta:
        ckpt_path = os.path.join(ckpt_dir,
                                 f"{model_name}_{fc_name}_checkpoint.npz")
        if not os.path.isfile(ckpt_path):
            print(f"  · checkpoint missing: {ckpt_path}")
            continue

        z = np.load(ckpt_path, allow_pickle=True)
        y_true_all, y_pred_all, y_score_all = z["y_true"], z["y_pred"], z["y_score"]

        # --- evaluate each fold separately -------------------------------- #
        perfold_metrics = defaultdict(list)
        for idx in _fold_indices(z, model_name):
            m = _compute_metrics(y_true_all[idx],
                                 y_pred_all[idx],
                                 y_score_all[idx])
            for k, v in m.items():
                perfold_metrics[k].append(v)

        # --- aggregate across 50×CV reps --------------------------------- #
        summary = {m: _agg_distribution(vals)
                   for m, vals in perfold_metrics.items()}
        
        summary['confusion_matrix'] = confusion_matrix(y_true_all, y_pred_all).tolist()
        # Classification report (per class precision/recall/f1)
        summary['classification_report'] = classification_report(y_true_all, y_pred_all, output_dict=True)

        # still store per‑fold raw values for future plots
        raw_per_fold = {m: [float(x) for x in vals]
                        for m, vals in perfold_metrics.items()}

        all_results[model_name][fc_name] = {
            "best_params" : meta[fc_name][model_name]["best_params"],
            "metrics"     : summary,
            "raw_per_fold": raw_per_fold,
            "checkpoint"  : ckpt_path,
        }

        acc = summary["accuracy"]
        print(f"{model_name:>4} | {fc_name:<10} : "
              f"μ={acc['mean']:.3f}  ±{acc['err']:.3f} "
              f"(n={len(raw_per_fold['accuracy'])})")

    # ---- save per‑model JSON ------------------------------------------- #
    out_json = os.path.join(
        this_dir, f"re_evaluation_from_checkpoints_{model_name}_dist.json")
    with open(out_json, "w") as fp:
        json.dump(all_results, fp, indent=2)
    print(f"✔ Results for {model_name} written → {out_json}\n")
