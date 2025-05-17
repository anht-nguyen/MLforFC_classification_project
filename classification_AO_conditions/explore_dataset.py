import os
import sys
import json
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from sklearn.model_selection import RepeatedKFold
from torch.optim import Adam

# Project setup
# Use current working directory for notebooks
this_dir = os.getcwd()
# Go up one directory to reach project root
project_root = os.path.abspath(os.path.join(this_dir))
sys.path.insert(0, project_root)
print(f"Project root: {project_root}")

from scripts.datasets_loader import load_dataset

load_dataset()

