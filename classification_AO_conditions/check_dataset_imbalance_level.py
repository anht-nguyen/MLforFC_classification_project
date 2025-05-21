import os, sys
import numpy as np
import json
# Imports from project
from scripts.utils import compute_class_imbalance
from scripts.config import LABEL_NAMES, FC_METRICS

this_dir      = os.path.dirname(os.path.abspath(__file__))
dl_ckpt_dir   = os.path.join(this_dir, "redo_evaluation/dl_checkpoints")   # <- adjust if needed
ml_ckpt_dir   = os.path.join(this_dir, "redo_evaluation/ml_checkpoints")   # <- adjust if needed
model_names   = ["CNN", "MLP", "GCN", "SVC", "LogReg", "RFC"] 

for model_name in model_names:

    if model_name in ["CNN", "MLP", "GCN"]:
        ckpt_dir = dl_ckpt_dir
    else:
        ckpt_dir = ml_ckpt_dir

    for fc_name in FC_METRICS:
        ckpt_path = os.path.join(ckpt_dir,
                                 f"{model_name}_{fc_name}_checkpoint.npz")
        if os.path.isfile(ckpt_path):
            print(f"  · checkpoint found: {ckpt_path}")
            break
    break
z = np.load(ckpt_path, allow_pickle=True)
y_true_all = z["y_true"]

# Compute imbalance level measures
ir_dict, mean_ir, max_ir, cvir, entropy, normalized_entropy  = compute_class_imbalance(y_true_all)

print("Imbalance Ratio per class:")
for cls, ir in ir_dict.items():
    print(f"  Class {cls:>2} : IR = {ir:.3f}")

print(f"\nMeanIR = {mean_ir:.3f}")
print(f"MaxIR  = {max_ir:.3f}")
print(f"CVIR   = {cvir:.3f}")
print(f"Entropy = {entropy:.3f}")
print(f"Normalized Entropy = {normalized_entropy:.3f}")

# save imbalance level measures to json
imbalance_json = os.path.join(this_dir, f"dataset_imbalance_level.json")
with open(imbalance_json, 'w') as f:
    json.dump({
        "ir_dict": ir_dict,
        "mean_ir": mean_ir,
        "max_ir": max_ir,
        "cvir": cvir,
        "entropy": entropy,
        "normalized_entropy": normalized_entropy
    }, f, indent=4)
print(f"Saved imbalance level measures to {imbalance_json}")
