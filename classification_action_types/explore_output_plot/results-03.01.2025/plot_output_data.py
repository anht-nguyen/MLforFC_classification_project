# === Import Necessary Libraries ===
import os
import numpy as np
import json
import matplotlib.pyplot as pyplt
import matplotlib as plt
from scipy.interpolate import interp1d  # Import interpolation function
import scipy.stats as stats
import pandas as pd
import math

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
                print(model_name)
                combined_data[FC_name].setdefault(model_name, {})
                for metric_name, values in metrics.items():
                    if isinstance(values, dict):
                        combined_data[FC_name][model_name].setdefault(metric_name, {}).update(values)
                    else:
                        combined_data[FC_name][model_name][metric_name] = values
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


def compute_performance_errors(data):
    fc_metricses = list(data.keys())
    model_nameses = list(next(iter(data.values())).keys())

    all_measures_errors = {}
    for idx, model_name in enumerate(model_nameses):
        all_measures_errors[model_name] = {}
        print(f"\n=== Model: {model_name} ===")
        for fc_name in fc_metricses:
            print(f"\n--- Feature Category: {fc_name} ---")
            all_measures_errors[model_name][fc_name] = {}

            for measure in ["AUC_errors", "Acc_errors", "Spec_errors", "Sens_errors"]:
                all_measures_errors[measure] = {}
                print(f"\n### Measure: {measure} ###")

                values_list = []
                for trial, value in data[fc_name][model_name][measure].items():
                    if not math.isnan(value["0"]):
                        values_list.append(value["0"])  # Add the AUC value to the list
                mean_value = np.mean(np.array(values_list))
                # print(f"Mean: {mean_value}")
                    
                # Check if standard error can be computed
                if len(values_list) > 1:
                    ci_95 = stats.t.interval(0.95, len(values_list) - 1, loc=mean_value, scale=stats.sem(values_list))
                else:
                    ci_95 = (mean_value, mean_value)  # If only one value, CI is just the mean itself
                # print(f"95% CI: {ci_95}")

                all_measures_errors[model_name][fc_name][measure] = {"Mean": mean_value, "95% CI": ci_95}

                # Convert to DataFrame for display
                df_results = pd.DataFrame.from_dict(all_measures_errors[model_name][fc_name][measure], orient='index')
                print(df_results.to_string())  # Print results in table format

def export_results_to_csv(df, filename):
    df.to_csv(filename, index=False)
    print(f"Results exported to {filename}")

# Re-import necessary libraries after execution state reset
import numpy as np
import scipy.stats as stats
import pandas as pd
import math

def compute_performance_errors(data):
    """
    Computes the mean and 95% confidence interval (CI) error for performance metrics 
    across different models and feature categories.

    Parameters:
    data (dict): A nested dictionary where:
        - The first level contains feature categories.
        - The second level contains model names.
        - The third level contains performance metrics (e.g., "AUC_errors", "Acc_errors").
        - The fourth level contains trials with multiple recorded values.

    Returns:
    pd.DataFrame: A DataFrame containing the mean and the computed 95% CI error (difference between mean and CI bound)
                  for each model, feature category, and performance measure.
    """
    fc_metricses = list(data.keys())
    model_nameses = list(next(iter(data.values())).keys())

    results_list = []  # List to store results for DataFrame

    for model_name in model_nameses:
        for fc_name in fc_metricses:
            for measure in ["AUC_errors", "Acc_errors", "Spec_errors", "Sens_errors"]:
                values_list = []
                for trial, value in data[fc_name][model_name][measure].items():
                    if not math.isnan(value["0"]):
                        values_list.append(value["0"])  # Extract the first value

                mean_value = round(np.mean(np.array(values_list)), 3) if values_list else np.nan

                # Compute 95% CI error (difference between mean and one of the CI bounds)
                if len(values_list) > 1 and not any(math.isnan(x) for x in values_list):
                    ci_bounds = stats.t.interval(0.95, len(values_list) - 1, loc=mean_value, scale=stats.sem(values_list))
                    ci_error = round(mean_value - ci_bounds[0], 3)  # Compute error as the difference from mean to lower bound
                else:
                    ci_error = 0  # If only one value or contains NaNs, there's no confidence interval

                # Append results to list
                results_list.append({
                    "Model": model_name,
                    "FC_metrics": fc_name,
                    "Measure": measure,
                    "Mean": mean_value,
                    "95% CI Error": ci_error  # Only exporting the error instead of CI bounds
                })

    # Convert list to DataFrame
    df_results = pd.DataFrame(results_list)
    return df_results


def curate_merge_json(script_dir):
    model_names = ["GCN", "CNN", "MLP", "SK"]

    for model in model_names:
        if model == "SK":
            # Iterate through files in the directory
            for root, dirs, files in os.walk(script_dir):
                for file in files:
                    if "output_data-ML_models" in file and file.endswith(".json"):
                        old_path = os.path.join(root, file)
                        new_path = os.path.join(root, "output_data-merged-SK.json")
                        
                        # Rename the file
                        os.rename(old_path, new_path)
                        print(f"Renamed: {file} -> output_data-merged-SK.json")
        else:        
            merged_data = {}
            for root, dirs, files in os.walk(script_dir):
                for file in files:
                    if file.endswith(".json") and 'output_data-'+ model in file:
                        print(file)
                        with open(os.path.join(root, file), "r") as f:
                            if model == "GCN":
                                loaded_data = json.load(f)
                                loaded_data = loaded_data[list(loaded_data.keys())[0]]
                                if len(list(loaded_data.keys())) > 1:
                                    # Move the data under 'GCN'
                                    for key in ['AUC_errors', 'Acc_errors', 'Spec_errors', 'Sens_errors']:
                                        if key in loaded_data:
                                            loaded_data['GCN'][key] = loaded_data.pop(key)
                                print('List keys: ' ,list(loaded_data.keys()))

                            else:
                                loaded_data = json.load(f)
                            key = os.path.basename(file).split("-"+model+"_")[1].split("-")[0]  # Extract FC type (COH, iCOH, PDC, PLV)
                            print(key)
                            merged_data[key] = loaded_data  # Store under the extracted key
            output_file_path = os.path.join(script_dir, f"output_data-merged-{model}.json")

            # Save merged data
            with open(output_file_path, "w") as output_file:
                json.dump(merged_data, output_file, indent=4)

            print(f"Merged JSON saved as: {output_file_path}")


# === Main Function ===
def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # curate_merge_json(script_dir)

    merged_json_files = [
        "output_data-merged-GCN.json",
        "output_data-merged-CNN.json",
        "output_data-merged-MLP.json",
        "output_data-merged-SK.json"
    ]

    combined_data = load_json_files(merged_json_files, script_dir)

    # plot_mean_roc_fc(combined_data, script_dir)
    # plot_mean_roc_model(combined_data, script_dir)

    df_results = compute_performance_errors(combined_data)
    export_results_to_csv(df_results, os.path.join(script_dir, "performance_errors.csv"))

# Execute main function
if __name__ == "__main__":
    main()
