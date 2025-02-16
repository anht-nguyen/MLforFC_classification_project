# ML/DL Classification Project

This repository contains code and resources for a machine learning and deep learning classification project. The project involves various models and techniques for analyzing and classifying functional connectivity data.

## Project Structure
```
MLforFC_classification_project/
├── autorun_GCN_model_singleFC.zsh
├── CNN_model_training_validating.py
├── dataset_zip
│   ├── connectivity_epoched_data_healthy.zip
│   └── PSD_epoched_data_healthy.zip
├── exploratory
│   └── output_data_explore.ipynb
├── GCN_model_singleFC.py
├── GCN_model_training_validating.py
├── hierarchy
├── hierarchy.txt
├── ML_models_training_validating.py
├── MLP_model_training_validating.py
├── output_data
├── README.md
├── requirements.txt
└── scripts
    ├── config.py
    ├── datasets_loader.py
    ├── models
    │   ├── dl_models_cores.py
    │   ├── ml_models_cores.py
    │   └── __pycache__
    │       └── dl_models_cores.cpython-38.pyc
    ├── plotting.py
    ├── save_results.py
    └── utils.py
```

## Models and Techniques

The project includes the following models and techniques:

- **COH**: Coherence
- **iCOH**: Imaginary Coherence
- **PDC**: Partial Directed Coherence
- **PLV**: Phase Locking Value
- **Spectral Granger**: Spectral Granger Causality

## Scripts

- `GCN_model_singleFC.py`: Script for training and validating a Graph Convolutional Network (GCN) model on single functional connectivity data.
- `GCN_model_training_validating.py`: Script for training and validating GCN models. (deprecated)
- `ML_models_training_validating.py`: Script for training and validating machine learning models.
- `MLP_model_training_validating.py`: Script for training and validating a Multi-Layer Perceptron (MLP) model.
- `scripts/config.py`: Configuration settings for the project.
- `scripts/datasets_loader.py`: Functions for loading datasets.
- `scripts/models/dl_models_cores.py`: Core deep learning models.
- `scripts/models/ml_models_cores.py`: Core machine learning models.
- `scripts/plotting.py`: Functions for plotting results.
- `scripts/save_results.py`: Functions for saving results.
- `scripts/utils.py`: Utility functions.

## Installation

To install the required packages, run:

```sh
pip install -r requirements.txt
``` 

## Usage

To run the GCN model training and validation, use the following command:
```
./autorun_GCN_model_singleFC.zsh <session_name>
```