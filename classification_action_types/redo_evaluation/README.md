# Recompute Multiclass Evaluation Metrics
This directory contains scripts to recompute multiclass evaluation metrics for our six-class classification models. The original evaluation mistakenly used binary metrics, resulting in inconsistencies.

We now:
1. Reload each model checkpoint from `explore_output_plot/results-03.01.2025`
2. Run inference on the test set
3. Compute true multiclass metrics using scikit-learn:
   - Overall, macro, and micro accuracy, precision, recall, F₁
   - Balanced accuracy
   - Per-class specificity
4. Export results for further analysis

**Checkpoint loading example:**
```python
data = np.load('checkpoints/SVC_FOO_checkpoint.npz', allow_pickle=True)
y_true, y_pred, y_score = data['y_true'], data['y_pred'], data['y_score']
train_idx, test_idx = data['train_idx'], data['test_idx']
```

---

## Running Evaluations in Parallel

You can run scripts in parallel with `screen` (recommended) or `nohup`.

**Using `screen`:**
```sh
sudo apt install screen
screen
# Start your evaluation script
# Create/switch windows: Ctrl+A, C/N/P
# Detach: Ctrl+A, D
screen -dr  # To resume
```

**Using `nohup`:**
```sh
nohup python redo_evaluation/redo_evaluation_on_best_models_DL.py <MODEL_NAME> > <name>.out &
# or for ML models:
nohup python redo_evaluation/redo_evaluation_on_best_models_ML.py > <name>.out &
```
Monitor output: `tail -f <name>.out`

**To kill nohup processes:**
1. Find PIDs with `ps -ef | grep redo_eval` (look for `?` in TTY column)
2. Kill with `kill -9 <PID1> <PID2> ...`
3. Verify with `ps -ef | grep redo_eval`
