import os, sys
import numpy as np
import json
import torch

this_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(this_dir, os.pardir))        # …/classification_action_types
sys.path.insert(0, parent_dir)

from scripts.config import (MODEL_NAMES, FC_METRICS, NUM_CLASSES, NUM_CHANNELS, NUM_FREQS)
from scripts.models.dl_models_cores import CNNClassifier, MLPClassifier, GCN
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support,
    balanced_accuracy_score,
)

# model_names = MODEL_NAMES
model_names = [ "CNN", "MLP", "GCN"]

for model_name in model_names:
    # Load json files for best models
    json_file = f"output_data-merged-{model_name}.json"
    script_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(script_dir, json_file)
    with open(json_path, "r") as f:
        loaded_data = json.load(f)

        for FC_name, models in loaded_data.items():
            print(f"Model: {model_name}, FC Metric: {FC_name}")
            # Load the best model for the current combination
            best_params = loaded_data[FC_name][model_name]["best_params"]
            print(f"Best parameters: {best_params}")

            match model_name:
                case "GCN":
                    model = GCN(
                        in_channels=NUM_FREQS,
                        hidden_dim=best_params['GCN-hidden'], num_classes=NUM_CLASSES,
                        num_layers=best_params['N-GCN'], dropout=best_params['Dropout'], drop_edge=best_params['DropEdge']
                    )
                case "CNN":
                    model = CNNClassifier(
                        in_channels=best_params['N-CNN'],
                        hidden_dim=best_params['CNN-hidden'],
                        num_classes=NUM_CLASSES,
                        kernel_size0=best_params['Kernel Size 0'],
                        kernel_size1=best_params['Kernel Size 1'],
                        padding=best_params['Padding'],

                    )
                case "MLP":
                    model = MLPClassifier(input_size=NUM_CHANNELS, hidden_dim=best_params['MLP-hidden'], num_layers=best_params['N-MLP'], num_classes=NUM_CLASSES)
                case _:
                    raise ValueError(f"Unknown model name: {model_name}")


