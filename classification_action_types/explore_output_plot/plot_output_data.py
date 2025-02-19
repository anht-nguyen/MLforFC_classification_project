# === Import Necessary Libraries ===
import os
import numpy as np
import json
import matplotlib.pyplot as pyplt
import matplotlib as plt
from scipy.interpolate import interp1d  # Import interpolation function

# === Utility Functions ===

def average_fpr(fpr_dict):
    """
    Calculate the averaged False Positive Rate (FPR) across different classes.

    Args:
        fpr_dict: Dictionary with class indices as keys and FPR arrays as values.

    Returns:
        mean_fpr: Averaged FPR across classes as a NumPy array.
    """
    all_fpr = [fpr.tolist() for fpr in fpr_dict.values()]
    max_len = max(len(fpr) for fpr in all_fpr)

    # Interpolate to the same length for averaging
    interpolated_fpr = [
        interp1d(np.linspace(0, 1, len(fpr)), fpr, kind='linear', fill_value="extrapolate")(np.linspace(0, 1, max_len))
        for fpr in all_fpr
    ]
    return np.mean(interpolated_fpr, axis=0)


def assign_model_colors(model_names):
    """Assign unique colors to models for visualization."""
    colors = pyplt.cm.get_cmap("Paired", len(model_names))
    return {model: colors(i) for i, model in enumerate(model_names)}


def assign_fc_colors(fc_names):
    """Assign unique colors to Feature Connectivity (FC) metrics for visualization."""
    colors = pyplt.cm.get_cmap("Paired", len(fc_names))
    return {fc: colors(i) for i, fc in enumerate(fc_names)}


def convert_to_numpy(obj):
    """
    Recursively convert lists in JSON data to NumPy arrays.

    Args:
        obj: JSON object (list/dictionary/other)

    Returns:
        Converted NumPy array or the object as-is.
    """
    if isinstance(obj, list):
        return np.array(obj) if all(isinstance(i, (int, float, list)) for i in obj) else [convert_to_numpy(i) for i in obj]
    elif isinstance(obj, dict):
        return {key: convert_to_numpy(value) for key, value in obj.items()}
    return obj  # Return as-is if not a list or dictionary


def load_json_files(json_files, script_dir):
    """
    Load and merge multiple JSON files into a single dictionary.

    Args:
        json_files: List of JSON file paths.
        script_dir: Directory where JSON files are located.

    Returns:
        combined_data: Dictionary containing merged data.
    """
    combined_data = {}

    for file in json_files:
        file_path = os.path.join(script_dir, file)  # Construct the full file path

        # Check if the file exists before reading
        if not os.path.exists(file_path):
            print(f"Warning: File {file} not found. Skipping.")
            continue

        with open(file_path, "r") as f:
            loaded_data = json.load(f)
            loaded_data = convert_to_numpy(loaded_data)  # Convert lists to NumPy arrays

        # Merge data into the combined dictionary
        for FC_name, models in loaded_data.items():
            combined_data.setdefault(FC_name, {})
            for model_name, metrics in models.items():
                combined_data[FC_name].setdefault(model_name, {})
                for metric_name, values in metrics.items():
                    combined_data[FC_name][model_name].setdefault(metric_name, {}).update(values)
    return combined_data


def plot_mean_roc_fc(data, script_dir):
    """Plot and save mean ROC curves for different Feature Connectivity (FC) metrics."""
    
    plt.rcParams.update({
        "pgf.texsystem": "pdflatex",
        'font.family': 'serif',
        'font.size': 20,
        'text.usetex': True,
        'pgf.rcfonts': False,
    })
    
    fc_metricses = list(data.keys())  
    model_nameses = list(next(iter(data.values())).keys())  

    fig, axes = pyplt.subplots(2, 3, figsize=(15, 10))  
    fig.patch.set_facecolor('white')  
    pyplt.subplots_adjust(hspace=0.5)
    fig.delaxes(axes[1,2])  # Remove empty subplot

    for idx, fc_name in enumerate(fc_metricses):
        ax = axes[idx // 3, idx % 3]
        ax.set_title(fc_name)
        model_colors = assign_model_colors(model_nameses)

        for model_name in model_nameses:
            fpr  = average_fpr(data[fc_name][model_name]["FPR"])
            tpr  = average_fpr(data[fc_name][model_name]["TPR"])
            auc_data = next(iter(data[fc_name][model_name]["AUC"].values()))

            if auc_data >= 0.6:  # Only plot if AUC is significant
                ax.plot(fpr, tpr, label=f'{model_name}', linewidth=4, color=model_colors[model_name])

        ax.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=2)  # Diagonal line
        ax.set_xlabel('1-Specificity')
        ax.set_ylabel('Sensitivity')
        ax.legend(frameon=False)
        ax.grid(True)

    pyplt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save figures
    pyplt.savefig(os.path.join(script_dir, 'mean_roc_by_fc.pgf'))
    pyplt.savefig(os.path.join(script_dir, 'mean_roc_by_fc.png'), dpi=300)
    pyplt.show()


def plot_mean_roc_model(data, script_dir):
    """Plot and save mean ROC curves grouped by model types."""
    
    plt.rcParams.update({
        "pgf.texsystem": "pdflatex",
        'font.family': 'serif',
        'font.size': 20,
        'text.usetex': True,
        'pgf.rcfonts': False,
    })

    fc_metricses = list(data.keys())
    model_nameses = list(next(iter(data.values())).keys())

    fig, axes = pyplt.subplots(2, 3, figsize=(15, 10))
    fig.patch.set_facecolor('white')  
    pyplt.subplots_adjust(hspace=0.5)

    for idx, model_name in enumerate(model_nameses):
        ax = axes[idx // 3, idx % 3]
        ax.set_title(model_name)
        fc_colors = assign_fc_colors(fc_metricses)

        for fc_name in fc_metricses:
            fpr  = average_fpr(data[fc_name][model_name]["FPR"])
            tpr  = average_fpr(data[fc_name][model_name]["TPR"])
            auc_data = next(iter(data[fc_name][model_name]["AUC"].values()))

            if auc_data >= 0.6:  # Only plot if AUC is significant
                ax.plot(fpr, tpr, label=f'{fc_name}', linewidth=4, color=fc_colors[fc_name])

        ax.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=2)  # Diagonal line
        ax.set_xlabel('1-Specificity')
        ax.set_ylabel('Sensitivity')
        ax.legend()
        ax.grid(True)

    pyplt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save figures
    pyplt.savefig(os.path.join(script_dir, 'mean_roc_by_model.pgf'))
    pyplt.savefig(os.path.join(script_dir, 'mean_roc_by_model.png'), dpi=300)
    pyplt.show()


# === Main Function ===
def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    model_names = ["GCN", "CNN", "MLP"]

    for model in model_names:
        merged_data = {}
        for root, dirs, files in os.walk(script_dir):
            for file in files:
                if file.endswith(".json") and 'output_data-'+ model in file:
                    print(file)
                    with open(os.path.join(root, file), "r") as f:
                        loaded_data = json.load(f)
                        key = os.path.basename(file).split("-"+model+"_")[1].split("-")[0]  # Extract FC type (COH, iCOH, PDC, PLV)
                        merged_data[key] = loaded_data[model]  # Store under the extracted key
        output_file_path = os.path.join(script_dir, f"output_data-merged-{model}.json")

        # Save merged data
        with open(output_file_path, "w") as output_file:
            json.dump(merged_data, output_file, indent=4)

        print(f"Merged JSON saved as: {output_file_path}")
                        


    merged_json_files = [
        # "output_data-merged-GCN.json",
        "output_data-merged-CNN.json",
        # "output_data-merged-MLP.json",
        "output_data-merged-SK.json"
    ]

    combined_data = load_json_files(merged_json_files, script_dir)

    plot_mean_roc_fc(combined_data, script_dir)
    plot_mean_roc_model(combined_data, script_dir)


# Execute main function
if __name__ == "__main__":
    main()
