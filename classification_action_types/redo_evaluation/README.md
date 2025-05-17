# Recompute Multiclass Evaluation Metrics

This directory contains scripts to correct our model evaluation. The original classification code applied binary‐only formulas (sensitivity, specificity, etc.) to a six‐class problem, leading to inconsistent results.  

Using the saved “best” checkpoints and output files from `explore_output_plot/results-03.01.2025`, we now:

1. Reload each model’s parameters in inference mode  
2. Run a clean forward pass over the test set  
3. Compute true multiclass metrics via scikit-learn:  
   - Overall accuracy  
   - Macro- and micro-averaged precision, recall, and F₁  
   - Balanced accuracy  
   - Per-class specificity  
4. Export corrected metrics for downstream analysis and plotting



---
Load the checkpoint via:

```
data = np.load('checkpoints/SVC_FOO_checkpoint.npz', allow_pickle=True)
y_true = data['y_true']
y_pred = data['y_pred']
y_score = data['y_score']
train_idx = data['train_idx']  # list of index arrays
test_idx = data['test_idx']
```


---
To use NOHUP: `cd` to `classification_action_types` folder and run commands like example below for DL and ML models
```
nohup python redo_evaluation\redo_evaluation_on_best_models_DL.py <1 model name (i.e., CNN, GCN, MLP)> > <unique name>.out &
```

```
nohup python redo_evaluation\redo_evaluation_on_best_models_ML.py > <unique name>.out &
```