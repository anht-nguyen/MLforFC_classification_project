# Global Parameters for ML/DL Training and Evaluation
import os
import torch

# ✅ Paths to dataset directories
BASE_DIR = os.getcwd()
FC_DATA_PATH = os.path.join(BASE_DIR, "data/connectivity_epoched_data_healthy")
PSD_DATA_PATH = os.path.join(BASE_DIR, "data/PSD_epoched_data_healthy")

# Number of trials for hyperparameter optimization (Optuna)
OPTIMIZER_TRIALS = 20

# Number of folds for cross-validation
K_FOLDS = 15

# Number of epochs for initial training phase
NUM_EPOCH_TRAINING = 300

# Number of epochs for final training phase
NUM_EPOCH_FINAL = 300

# Number of epochs for early stopping
PATIENCE = 15

# Number of times to repeat cross-validation
NUM_REPEATS_TRAINING = 5
NUM_REPEATS_FINAL = 50

# Number of classes in the dataset: baseline, human-left, human-right, robot-left, robot-right, and landscape
NUM_CLASSES = 6 

# Number of frequencies in the PSD data
NUM_FREQS = 27

# number of EEG channels used to compute FC
NUM_CHANNELS = 10

# Device to use for training (CPU/GPU)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")