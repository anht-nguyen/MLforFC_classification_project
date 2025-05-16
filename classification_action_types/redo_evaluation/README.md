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
