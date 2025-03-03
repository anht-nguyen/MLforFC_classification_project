# Params used for this run:


# Number of trials for hyperparameter optimization (Optuna)
OPTIMIZER_TRIALS = 50

# Number of folds for cross-validation
K_FOLDS = 20

# Number of epochs for initial training phase
NUM_EPOCH_TRAINING = 300

# Number of epochs for final training phase
NUM_EPOCH_FINAL = 300

# Number of epochs for early stopping
PATIENCE = 15

# Number of times to repeat cross-validation
NUM_REPEATS_TRAINING = 5
NUM_REPEATS_FINAL = 50