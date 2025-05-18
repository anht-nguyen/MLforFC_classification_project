import json
import pandas as pd
import matplotlib.pyplot as plt
import os

# Load the JSON results
json_path = 're_evaluation_ML_results_v0.json'
with open(json_path, 'r') as f:
    data = json.load(f)

# Build a DataFrame for accuracy and AUC
rows = []
for clf, fc_dict in data.items():
    for fc, vals in fc_dict.items():
        metrics = vals['metrics']
        rows.append({
            'Classifier': clf,
            'FC Metric': fc,
            'Accuracy': metrics['accuracy'],
            'AUC': metrics['auc_macro']
        })

metrics_df = pd.DataFrame(rows)

# Display the DataFrame to the user
# import ace_tools as tools
# tools.display_dataframe_to_user('ML Evaluation Metrics', metrics_df)

# Create pivot tables for reporting or export
accuracy_pivot = metrics_df.pivot(index='FC Metric', columns='Classifier', values='Accuracy')
auc_pivot      = metrics_df.pivot(index='FC Metric', columns='Classifier', values='AUC')

# Save pivot tables to CSV
accuracy_pivot.to_csv('accuracy_summary.csv')
auc_pivot.to_csv('auc_summary.csv')

# Plot grouped bar charts
# Accuracy plot
accuracy_pivot.plot(kind='bar', figsize=(8, 5), title='Accuracy by FC Metric and Classifier')
plt.xlabel('FC Metric')
plt.ylabel('Accuracy')
plt.tight_layout()
plt.show()

# AUC plot
auc_pivot.plot(kind='bar', figsize=(8, 5), title='AUC (Macro) by FC Metric and Classifier')
plt.xlabel('FC Metric')
plt.ylabel('AUC (Macro)')
plt.tight_layout()
plt.show()

# Example: Display confusion matrix for a chosen classifier and FC metric
# Let's pick SVC with iCOH
conf_mat = data['SVC']['iCOH']['metrics']['confusion_matrix']
conf_df = pd.DataFrame(conf_mat, 
                       index=[f'True_{i}' for i in range(len(conf_mat))],
                       columns=[f'Pred_{i}' for i in range(len(conf_mat))])
# tools.display_dataframe_to_user('SVC iCOH Confusion Matrix', conf_df)

# Save confusion matrix to CSV
conf_df.to_csv('svc_icoh_confusion_matrix.csv')

print("Saved summary CSV files: accuracy_summary.csv, auc_summary.csv, svc_icoh_confusion_matrix.csv")
