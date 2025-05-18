import json
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns

# Paths
DATA_JSON = 're_evaluation_ML_results_full_metrics.json'
OUT_DIR = 'analysis_outputs'
os.makedirs(OUT_DIR, exist_ok=True)

# Load full-metrics results
def load_results(path):
    with open(path, 'r') as f:
        return json.load(f)

data = load_results(DATA_JSON)

# Flatten into DataFrame: one row per Classifier×FC Metric
rows = []
for clf, fc_dict in data.items():
    for fc, entry in fc_dict.items():
        metrics = entry['metrics']
        row = {'Classifier': clf, 'FC Metric': fc}
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                row[k] = v
        rows.append(row)
metrics_df = pd.DataFrame(rows)
metrics_df.to_csv(os.path.join(OUT_DIR, 'all_metrics_summary.csv'), index=False)
print('Saved all_metrics_summary.csv')

# Define metrics to pivot and plot
target_metrics = [
    'accuracy', 'balanced_accuracy',
    'precision_macro', 'recall_macro', 'f1_macro',
    'precision_micro', 'recall_micro', 'f1_micro',
    'jaccard_macro', 'jaccard_micro',
    'auc_macro'#, 'auc_micro'
]

# Generate pivot tables and bar charts
for m in target_metrics:
    pivot = metrics_df.pivot(index='FC Metric', columns='Classifier', values=m)
    csv_path = os.path.join(OUT_DIR, f'{m}_summary.csv')
    pivot.to_csv(csv_path)
    print(f'Saved {csv_path}')

    # Plot
    ax = pivot.plot(kind='bar', figsize=(10, 6), title=m.replace('_', ' ').title())
    ax.set_xlabel('FC Metric')
    ax.set_ylabel(m.replace('_', ' ').title())
    plt.tight_layout()
    png_path = os.path.join(OUT_DIR, f'{m}_bar.png')
    plt.savefig(png_path)
    plt.close()
    print(f'Saved plot {png_path}')

# Export classification reports and confusion matrix heatmaps for all
for clf, fc_dict in data.items():
    for fc, entry in fc_dict.items():
        metrics = entry['metrics']
        # Classification report
        report = metrics.get('classification_report')
        if report:
            rep_df = pd.DataFrame(report).transpose()
            rep_path = os.path.join(OUT_DIR, f'{clf}_{fc}_classification_report.csv')
            rep_df.to_csv(rep_path)
            print(f'Saved classification report {rep_path}')
        # Confusion matrix
        cm = metrics.get('confusion_matrix')
        if cm:
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
            cm_path = os.path.join(OUT_DIR, f'{clf}_{fc}_confusion_matrix.png')
            plt.savefig(cm_path)
            plt.close()
            print(f'Saved confusion matrix heatmap {cm_path}')
