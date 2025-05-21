import json
import pandas as pd
import matplotlib.pyplot as plt
import os


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
this_dir = os.path.dirname(os.path.abspath(__file__))
meta_dir = this_dir  # JSONs live here
model_names = ["CNN", "MLP", "GCN", "SVC", "LogReg", "RFC"]  # models of interest
out_dir = 'analysis_outputs'
os.makedirs(out_dir, exist_ok=True) 



for model_name in model_names:
    meta_json = os.path.join(meta_dir, f"re_evaluation_from_checkpoints_{model_name}_dist.json")
    if not os.path.exists(meta_json):
        raise FileNotFoundError(f"Meta JSON not found: {meta_json}")

    # Load full-metrics results
    def load_results(path):
        with open(path, 'r') as f:
            return json.load(f)

    data = load_results(meta_json)

    # Flatten into DataFrame: one row per Classifier × FC Metric
    rows = []
    for clf, fc_dict in data.items():
        for fc, entry in fc_dict.items():
            metrics = entry['metrics']
            row = {'Classifier': clf, 'FC Metric': fc}
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    row[k] = v
                elif isinstance(v, dict) and 'mean' in v and 'err' in v:
                    row[f'{k}_mean'] = v['mean']
                    row[f'{k}_err'] = v['err']
            rows.append(row)
    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(os.path.join(out_dir, 'all_metrics_summary.csv'), index=False)
    print('Saved all_metrics_summary.csv')

    # Define metrics to pivot and plot
    base_metrics = [
        'accuracy', 'balanced_accuracy',
        'precision_macro', 'recall_macro', 'f1_macro',
        'precision_micro', 'recall_micro', 'f1_micro',
        'jaccard_macro', 'jaccard_micro',
        'auc_macro', 'auc_micro', 'hamming_loss',
    ]

    # Generate pivot tables (with mean & err) and bar charts with error bars
    for m in base_metrics:
        mean_col = f'{m}_mean'
        err_col = f'{m}_err'
        if mean_col not in metrics_df.columns:
            continue

        # Pivot means and errors separately
        pivot_mean = metrics_df.pivot(index='FC Metric', columns='Classifier', values=mean_col)
        pivot_err  = metrics_df.pivot(index='FC Metric', columns='Classifier', values=err_col)

        # Combine for CSV output
        combined = pd.concat([pivot_mean, pivot_err], axis=1, keys=['mean', 'err'])
        csv_path = os.path.join(out_dir, f'{model_name}_{m}_summary.csv')
        combined.to_csv(csv_path)
        print(f'Saved {csv_path}')

        # Plot with error bars
        ax = pivot_mean.plot(kind='bar', yerr=pivot_err, figsize=(10, 6),
                            title=m.replace('_', ' ').title())
        ax.set_xlabel('FC Metric')
        ax.set_ylabel(f"{m.replace('_', ' ').title()} (mean ± error)")
        plt.tight_layout()
        png_path = os.path.join(out_dir, f'{model_name}_{m}_bar.png')
        plt.savefig(png_path)
        plt.close()
        print(f'Saved plot {png_path}')

    # Export classification reports and confusion matrices for all
    for clf, fc_dict in data.items():
        for fc, entry in fc_dict.items():
            metrics = entry['metrics']
            # Classification report
            report = metrics.get('classification_report')
            if report:
                rep_df = pd.DataFrame(report).transpose()
                rep_path = os.path.join(out_dir, f'{clf}_{fc}_classification_report.csv')
                rep_df.to_csv(rep_path)
                print(f'Saved classification report {rep_path}')
            # Confusion matrix
            cm = metrics.get('confusion_matrix')
            if cm:
                import seaborn as sns
                cm_df = pd.DataFrame(
                    cm,
                    index=[f'True_{i}' for i in range(len(cm))],
                    columns=[f'Pred_{i}' for i in range(len(cm))]
                )
                plt.figure(figsize=(8, 6))
                sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
                plt.title(f'Confusion Matrix: {clf} on {fc}')
                plt.xlabel('Predicted')
                plt.ylabel('True')
                plt.tight_layout()
                cm_path = os.path.join(out_dir, f'{clf}_{fc}_confusion_matrix.png')
                plt.savefig(cm_path)
                plt.close()
                print(f'Saved confusion matrix heatmap {cm_path}')



# --------------------------------------------------------------------------- #
# Load and aggregate metrics across all models
# --------------------------------------------------------------------------- #
rows = []
for model_name in model_names:
    path = os.path.join(meta_dir, f"re_evaluation_from_checkpoints_{model_name}_dist.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Meta JSON not found: {path}")
    data = json.load(open(path, 'r'))
    for clf, fc_dict in data.items():
        for fc_metric, entry in fc_dict.items():
            metrics = entry.get('metrics', {})
            row = {'Model': clf, 'FC Metric': fc_metric}
            for key, val in metrics.items():
                mean_col = f"{key}_mean"
                err_col = f"{key}_err"
                if isinstance(val, dict) and 'mean' in val and 'err' in val:
                    row[mean_col] = val['mean']
                    row[err_col] = val['err']
                elif isinstance(val, (int, float)):
                    row[mean_col] = val
                    row[err_col] = 0.0
            rows.append(row)

metrics_df = pd.DataFrame(rows)
metrics_df.to_csv(os.path.join(out_dir, 'all_metrics_summary.csv'), index=False)
print('Saved all_metrics_summary.csv')

# --------------------------------------------------------------------------- #
# Plot bar plots for each evaluation metric
# --------------------------------------------------------------------------- #

for metric in base_metrics:
    mean_col = f"{metric}_mean"
    err_col = f"{metric}_err"
    if mean_col not in metrics_df.columns:
        continue

    pivot_mean = metrics_df.pivot(index='FC Metric', columns='Model', values=mean_col)
    pivot_err = metrics_df.pivot(index='FC Metric', columns='Model', values=err_col)

    # Save CSV summary
    summary = pd.concat([pivot_mean, pivot_err], axis=1, keys=['mean', 'err'])
    summary_path = os.path.join(out_dir, f"{metric}_summary.csv")
    summary.to_csv(summary_path)
    print(f"Saved {summary_path}")

    # Plot all models side by side with error bars
    ax = pivot_mean.plot(kind='bar', yerr=pivot_err, figsize=(10, 6),
                         title=metric.replace('_', ' ').title())
    ax.set_xlabel('FC Metric')
    ax.set_ylabel(f"{metric.replace('_', ' ').title()} (mean ± error)")
    plt.tight_layout()
    img_path = os.path.join(out_dir, f"{metric}_all_models_bar.png")
    plt.savefig(img_path)
    plt.close()
    print(f"Saved plot {img_path}")