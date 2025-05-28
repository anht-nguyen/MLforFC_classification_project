import json
import pandas as pd
import matplotlib.pyplot as plt
import os


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
this_dir = os.path.dirname(os.path.abspath(__file__))
meta_dir = this_dir  # JSONs live here
model_names = ["CNN", "GCN", "MLP", "LogReg", "RFC", "SVC"]  # models of interest
model_names_save = ["CNN", "GNN", "MLP", "LogReg", "RFC", "SVM"]  # models to save in LaTeX table]
model_names_best_params = ["CNN", "GCN", "MLP", "logreg", "RFC", "svc"]  
FC_metrics = ["COH", "iCOH", "PDC", "PLV", "Spectral Granger"]
FC_metrics_save = [ "COH", "iCOH", "PDC", "PLV", "SpcG"]
out_dir = 'analysis_outputs'
os.makedirs(out_dir, exist_ok=True) 
dl_ckpt_dir   = os.path.join(this_dir, "dl_checkpoints")   # <- adjust if needed
ml_ckpt_dir   = os.path.join(this_dir, "ml_checkpoints")   # <- adjust if needed


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
            row = {'Classifier': model_names_save[model_names.index(clf)], 'FC Metric': FC_metrics_save[FC_metrics.index(fc)]}
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
        ax = pivot_mean.plot(kind='bar', yerr=pivot_err, figsize=(10, 4),
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
        for fc, entry in fc_dict.items():
            metrics = entry.get('metrics', {})
            row = {'Model': model_names_save[model_names.index(clf)], 'FC Metric': FC_metrics_save[FC_metrics.index(fc)]}
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
    ax = pivot_mean.plot(kind='bar', yerr=pivot_err, figsize=(10, 4),
                         title=metric.replace('_', ' ').title() )
    ax.set_xlabel('FC Metric')
    ax.set_ylabel(f"{metric.replace('_', ' ').title()} (mean ± error)")
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    img_path = os.path.join(out_dir, f"{metric}_all_models_bar_action_type.png")
    plt.savefig(img_path)
    plt.close()
    print(f"Saved plot {img_path}")


# --------------------------------------------------------------------------- #
# Generate LaTeX table of best models (auc_macro_mean > 0.9)
# --------------------------------------------------------------------------- #

# metrics we want to report (key names in JSON → (latex header, var name suffix))
REPORT_METRICS = {
    'auc_macro':       ('AUC',             'auc'),
    'balanced_accuracy': ('Accuracy',       'acc'),
    'precision_macro': ('Precision',       'prec'),
    'recall_macro':    ('Recall',          'rec'),
    'f1_macro':        ('F1-Score',        'f1'),
}

# collect rows
best_models = []
for model_name in model_names:
    path = os.path.join(meta_dir, f"re_evaluation_from_checkpoints_{model_name}_dist.json")
    data = json.load(open(path, 'r'))
    for clf, fc_dict in data.items():
        for fc, entry in fc_dict.items():
            if fc == 'Spectral Granger':
                fc = 'SpcG'
            if clf == 'GCN':
                clf = 'GNN'
            if clf == 'SVC':
                clf = 'SVM'
            m = entry['metrics']
            auc_m = m['auc_macro']['mean']
            if auc_m > 0.9:
                row = {
                    'Model': clf,
                    'FC':    fc,
                }
                # pull mean±err for each metric
                for key,(hdr,suffix) in REPORT_METRICS.items():
                    val = m[key]
                    # handle both dict and raw float
                    if isinstance(val, dict):
                        row[f'{suffix}_mean'] = val['mean']
                        row[f'{suffix}_err']  = val['err']
                    else:
                        row[f'{suffix}_mean'] = val
                        row[f'{suffix}_err']  = 0.0
                best_models.append(row)

# sort however you like, e.g. by Model then FC
best_models = sorted(best_models, key=lambda r: (r['Model'], r['FC']))

# build LaTeX
lines = []

# header row
hdrs = ["Model","FC"] + [hdr for hdr,_ in REPORT_METRICS.values()]
lines.append(" & ".join(hdrs) + r" \\")
lines.append(r"\hline")
# (optional) insert section breaks here if you have multiple tasks
lines.append(r"\multicolumn{6}{c}{\textbf{Action Type Classification}}\\")
lines.append(r"\hline")

# data rows
for r in best_models:
    parts = [r['Model'], r['FC']]
    for _,suffix in REPORT_METRICS.values():
        mean = r[f'{suffix}_mean']
        err  = r[f'{suffix}_err']
        parts.append(f"{mean:.3f} $\\pm$ {err:.3f}")
    lines.append(" & ".join(parts) + r" \\")
# (optional) repeat section break for second task here


# write out
out_path = os.path.join(this_dir, 'best_models_auc_table.txt')
with open(out_path, 'w') as f:
    f.write("\n".join(lines))
print(f"Exported LaTeX table to {out_path}")




# --------------------------------------------------------------------------- #
# Generate LaTeX table of optimized hyperparameters for each model & FC metric
# --------------------------------------------------------------------------- #

# Directory where your merged JSONs live; adjust if needed
meta_dir = this_dir  # already defined above in the script

# New output path for the LaTeX table
latex_out = os.path.join(this_dir, 'optimized_hyperparams_table.txt')

# Prepare table header
table_lines = []

# Loop through FC tasks (sections)
table_lines.append(r"\multicolumn{14}{c}{\textbf{Action Type Classification}}\\")
table_lines.append(r"\hline")

# Loop through each model JSON and extract best_params
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
    meta = json.load(open(meta_path, 'r'))

    # For each FC metric in this JSON
    for FC_name, entry_dict in meta.items():
        if model_name == 'SVC':
            model_name_best_param = 'svc'
        elif model_name == 'LogReg':
            model_name_best_param = 'logreg'
        else:
            model_name_best_param = model_name
        # get the sub-dictionary for this model (if names differ adjust key)
        entry = entry_dict.get(model_name, {})
        best_params = entry.get('best_params', {})

        # Fetch each hyperparam, defaulting to blank if missing
        batch   = best_params.get('Batch Size', '')
        nlay    = best_params.get(f'N-{model_name_best_param}', '')
        hidden  = best_params.get(f'{model_name_best_param}-hidden', '')
        lr      = best_params.get('Learning Rate', '')
        k0      = best_params.get('Kernel Size 0', '')
        k1      = best_params.get('Kernel Size 1', '')
        pad     = best_params.get('Padding', '')
        cost    = best_params.get(f'{model_name_best_param}_C', '')
        gamma   = best_params.get(f'{model_name_best_param}_gamma', '')
        nest    = best_params.get('rfc_n_estimators', '')
        mdepth  = best_params.get('rfc_max_depth', '')
        dropo   = best_params.get('Dropout', '')
        de      = best_params.get('DropEdge', '')

        # Build the LaTeX row
        if FC_name == 'Spectral Granger':
            FC_name_saved = 'SpcG'
        else:
            FC_name_saved = FC_name
        
        if model_name == 'GCN':
            model_name_saved = 'GNN'
        elif model_name == 'SVC':
            model_name_saved = 'SVM'
        else:
            model_name_saved = model_name

        row = [model_name_saved + f" ({FC_name_saved})",
               str(batch), str(nlay), str(hidden), str(lr), str(k0), str(k1),
               str(pad), str(cost)[:5], str(gamma)[:5], str(nest), str(mdepth),
               str(dropo)[:3], str(de)[:3]]
        table_lines.append("  & ".join(row) + r" \\")


# Write out the .tex file
with open(latex_out, 'w') as f:
    f.write("\n".join(table_lines))

print(f"Exported optimized hyperparams LaTeX table to {latex_out}")



# --------------------------------------------------------------------------- #
# Generate LaTeX table of macro-AUC for each model & FC metric
# --------------------------------------------------------------------------- #
# Path pattern for re-evaluation JSONs
auc_data = {}  # fc -> clf -> (mean, err)
for model_name in model_names:
    path = os.path.join(meta_dir, f"re_evaluation_from_checkpoints_{model_name}_dist.json")
    if not os.path.isfile(path):
        print(f"Warning: re-evaluation file not found for model {model_name}: {path}")
        continue
    data = json.load(open(path, 'r'))
    for clf, fc_dict in data.items():
        # Normalize classifier names
        if clf == 'GCN':      clf_name = 'GNN'
        elif clf == 'SVC':    clf_name = 'SVM'
        elif clf == 'RF':     clf_name = 'RFC'
        else:                 clf_name = clf
        for fc, entry in fc_dict.items():
            # Normalize FC names
            fc_name = 'SpcG' if fc == 'Spectral Granger' else fc

            m = entry.get('metrics', {})
            auc = m.get('auc_macro', {})
            if isinstance(auc, dict):
                mean = auc.get('mean', 0.0)
                err  = auc.get('err', 0.0)
            else:
                mean, err = auc, 0.0
            auc_data.setdefault(fc_name, {})[clf_name] = (mean, err)


# Build LaTeX table
auc_lines = []
auc_lines.append(" & " + " & ".join(model_names) + r" \\")
auc_lines.append(r"\hline")
# Data rows
for fc in FC_metrics_save:
    row = [fc]
    for clf in model_names_save:
        val = auc_data.get(fc, {}).get(clf, (None, None))
        if val[0] is not None:
            if val[0] > 0.9:
                row.append(r"\textbf{" f"{val[0]:.3f} $\\pm$ {val[1]:.3f}" r"}")
            else:
                row.append(f"{val[0]:.3f} $\\pm$ {val[1]:.3f}")
        else:
            row.append("")
    auc_lines.append(" & ".join(row) + r" \\")

# Write LaTeX to file
auc_out = os.path.join(this_dir, 'auc_all_models_fc.txt')
with open(auc_out, 'w') as f:
    f.write("\n".join(auc_lines))
print(f"Exported AUC table to {auc_out}")



# --------------------------------------------------------------------------- #
# Generate LaTeX table of best models (hyperparams + other metrics)
# --------------------------------------------------------------------------- #

# metrics to report (drop 'auc_macro')
REPORT_METRICS = {
    'balanced_accuracy': ('Accuracy',    'acc'),
    'precision_macro':   ('Precision',   'prec'),
    'recall_macro':      ('Recall',      'rec'),
    'f1_macro':          ('F1-Score',    'f1'),
}

best_models = []
for model_name in model_names:
    # 1) load re-evaluation JSON
    re_path = os.path.join(meta_dir, f"re_evaluation_from_checkpoints_{model_name}_dist.json")
    if not os.path.exists(re_path):
        print(f"[WARN] no re-eval file for {model_name}")
        continue
    data = json.load(open(re_path, 'r'))

    # 2) load the matching hyperparam JSON
    if model_name in ["CNN", "MLP", "GCN"]:
        meta_path = os.path.join(meta_dir, f"output_data-merged-{model_name}.json")
    else:
        # all sklearn‐style models share the same SK file
        meta_path = os.path.join(meta_dir, "output_data-merged-SK.json")
    if not os.path.exists(meta_path):
        print(f"[WARN] no meta JSON for {model_name}")
        continue
    meta = json.load(open(meta_path, 'r'))

    for clf, fc_dict in data.items():
        # normalize classifier name
        clf_name = {"GCN":"GNN", "SVC":"SVM", "RF":"RFC"}.get(clf, clf)

        for fc, entry in fc_dict.items():
            # normalize FC name
            fc_name = "SpcG" if fc=="Spectral Granger" else fc

            # skip models below AUC threshold
            auc_m = entry['metrics']['auc_macro']['mean']
            if auc_m <= 0.9:
                continue

            # --- correctly fetch best_params from meta JSON ---
            # If your JSON keys differ (upper/lower), tweak these names:
            params_dict = meta.get(fc, {}).get(model_name, {}).get('best_params', {})

            # pick the right prefix for naming consistency
            if model_name=="SVC":
                bp_model="svc"
            elif model_name=="LogReg":
                bp_model="logreg"
            else:
                bp_model=model_name

            batch  = params_dict.get("Batch Size","")
            nlay   = params_dict.get(f"N-{bp_model}","")
            hidden = params_dict.get(f"{bp_model}-hidden","")
            lr     = params_dict.get("Learning Rate","")
            k0     = params_dict.get("Kernel Size 0","")
            k1     = params_dict.get("Kernel Size 1","")
            pad    = params_dict.get("Padding","")
            cost   = params_dict.get(f"{bp_model}_C","")
            gamma  = params_dict.get(f"{bp_model}_gamma","")
            nest   = params_dict.get("rfc_n_estimators","")
            mdepth = params_dict.get("rfc_max_depth","")
            dropo  = params_dict.get("Dropout","")
            de     = params_dict.get("DropEdge","")

            # build a single “key=val” string
            parts = []
            if batch:  parts.append(f"batch\_sz={batch}")
            if nlay:   parts.append(f"n\_layers={nlay}")
            if hidden: parts.append(f"hidden={hidden}")
            if lr:     parts.append(f"lr={lr}")
            if k0:     parts.append(f"ksz\_0={k0}")
            if k1:     parts.append(f"ksz\_1={k1}")
            if pad:    parts.append(f"padding={pad}")
            if cost:   parts.append(f"cost={str(cost)[:5]}")
            if gamma:  parts.append(f"gamma={str(gamma)[:5]}")
            if nest:   parts.append(f"n\_est={nest}")
            if mdepth: parts.append(f"max\_dep={mdepth}")
            if dropo:  parts.append(f"dropout={str(dropo)[:3]}")
            if de:     parts.append(f"drop\_edge={str(de)[:3]}")
            hyper_str = ",\\; ".join(parts)

            # assemble row dict
            row = {'Model': clf_name,
                   'FC':    fc_name,
                   'hyper': hyper_str}

            # pull the other four metrics
            for key,(_,suffix) in REPORT_METRICS.items():
                val = entry['metrics'][key]
                if isinstance(val, dict):
                    row[f"{suffix}_mean"] = val['mean']
                    row[f"{suffix}_err"]  = val['err']
                else:
                    row[f"{suffix}_mean"] = val
                    row[f"{suffix}_err"]  = 0.0

            best_models.append(row)

# sort rows if desired
best_models = sorted(best_models, key=lambda r: (r['Model'], r['FC']))

# now build your LaTeX table exactly as before,
# using 'hyper' for the third column in place of AUC…


# sort as desired
best_models = sorted(best_models, key=lambda r: (r['Model'], r['FC']))

# build the LaTeX table
lines = []
# header    
hdrs = ['Model','FC','Best Hyper-parameters'] + [h for h,_ in REPORT_METRICS.values()]
lines.append(" & ".join(hdrs) + r" \\")
lines.append(r"\hline")
lines.append(r"\multicolumn{5}{c}{\textbf{Action Type Classification}}\\")
lines.append(r"\hline")

for r in best_models:
    parts = [r['Model'], r['FC'], r['hyper']]
    for _,suffix in REPORT_METRICS.values():
        m_ = r[f'{suffix}_mean']
        e_ = r[f'{suffix}_err']
        parts.append(f"{m_:.3f} $\\pm$ {e_:.3f}")
    lines.append(" & ".join(parts) + r" \\")

# write out
out_path = os.path.join(this_dir, 'best_models_hyperparams_table.txt')
with open(out_path, 'w') as f:
    f.write("\n".join(lines))

print(f"Exported best models table (with hyperparams) to {out_path}")
