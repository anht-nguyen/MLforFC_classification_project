import os
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score
import pandas as pd
import math
import numpy as np
from scipy.interpolate import interp1d # Import the interp1d function
import warnings

# Dictionary to store file paths for each class
def get_files_by_class(basepath):
  file_paths_by_class = {}

  for class_name in os.listdir(basepath):
      class_dir = os.path.join(basepath, class_name)
      if os.path.isdir(class_dir):
          file_paths_by_class[class_name] = [
              os.path.join(class_dir, file_name)
              for file_name in os.listdir(class_dir) if file_name.endswith(".mat")
          ]
  return file_paths_by_class

# Dictionary to store splits
def split_datasets(file_paths_by_class):
  splits = {'train': [], 'test': [], 'full' : []}

  for class_name, file_paths in file_paths_by_class.items():
      # Perform train-test split (80-20)
      train_files, test_files = train_test_split(file_paths, test_size=0.2, random_state=42)


      # Append splits to the respective lists
      splits['train'].extend([(file, class_name) for file in train_files])
      splits['test'].extend([(file, class_name) for file in test_files])
      splits['full'].extend([(file, class_name) for file in file_paths])
      #print(splits['train'])

  return splits

# Define a transform function to flatten the matrix
def flatten_transform(matrix):
    return matrix.flatten()

def dataset_type_converter(matlabdataset):
    # Initialize lists to collect matrices and labels
    matrices = []
    labels = []

    # Iterate over the dataset and collect matrices and labels
    for idx in range(len(matlabdataset)):
        matrix, label = matlabdataset[idx]
        matrices.append(matrix)  # Flatten the matrix to 1D
        labels.append(label)

    # Convert lists to numpy arrays
    data_dict = {
        'x': np.array(matrices),  # Convert list of matrices to 2D array
        'y': np.array(labels).reshape(len(labels),)  # Convert list of labels to 1D array
    }

    return data_dict

# def cm_generator(test_loader):
#   y_true, y_pred, _ = test_model_torch()
#   return pd.DataFrame(confusion_matrix(y_true, y_pred))

def construct_confusion_matrix(y_true, y_pred, class_names):
    # Compute the confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Convert to DataFrame for better visualization
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    
    return cm_df

def get_accuracy_measures(y_true, y_pred, y_scores, num_classes): #auc, accuracy, specificity, sensitiviity
    fpr = {}
    tpr = {}
    auc_dict = {}
    accuracy = {}
    specificity = {}
    sensitivity = {}
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(np.array(y_true) == i, np.array(y_scores)[:,i])
        auc_dict[i] = auc(fpr[i], tpr[i])
        # print(auc_dict[i])
        accuracy[i] = accuracy_score(y_true, y_pred)
        specificity[i] = confusion_matrix(y_true, y_pred)[0, 0] / (confusion_matrix(y_true, y_pred)[0, 0] + confusion_matrix(y_true, y_pred)[0, 1])
        sensitivity[i] = confusion_matrix(y_true, y_pred)[1, 1] / (confusion_matrix(y_true, y_pred)[1, 0] + confusion_matrix(y_true, y_pred)[1, 1])
    return fpr, tpr, auc_dict, accuracy, specificity, sensitivity

def feature_map_dim(input_dim, padding, kernel_size, stride):
  return math.floor((((input_dim) - kernel_size + 2*padding)/ stride) + 1)

def average_fpr(fpr_dict):
    """Calculates the averaged FPR across classes.

    Args:
        fpr_dict: A dictionary where keys are class indices
                  and values are FPR arrays of different lengths.

    Returns:
        mean_fpr: An array representing the averaged FPR across classes.
    """

    all_fpr = []
    for class_idx, fpr in fpr_dict.items():
      fpr = fpr.tolist()
      all_fpr.append(fpr)
        #print(all_fpr)
    #print(all_fpr)
    #print(fpr)
    # Find the maximum length among all FPR arrays
    max_len = max(len(fpr) for fpr in all_fpr)
    #print(max_len)
    # Interpolate FPR values to a common length
    interpolated_fpr = []
    for fpr in all_fpr:
        # Create interpolation function
        f = interp1d(np.linspace(0, 1, len(fpr)), fpr, kind='linear', fill_value="extrapolate")

        # Interpolate to the maximum length
        new_fpr = f(np.linspace(0, 1, max_len))
        interpolated_fpr.append(new_fpr)

    # Calculate the mean FPR across classes
    mean_fpr = np.mean(interpolated_fpr, axis=0)

    return mean_fpr



def compute_class_imbalance(y, classes=None):
    """
    Compute imbalance measures and normalized entropy for a multi‑class dataset.

    Parameters
    ----------
    y : array-like, shape (n_samples,)
        True class labels (e.g. integers 0,1,2,… or strings).
    classes : array-like, optional
        The set of all possible classes. If None, uses np.unique(y).

    Returns
    -------
    ir_per_class : dict
        {class_value: IR_class}, where
          IR_class = max_count / count_class,
          and classes with zero counts get IR = np.inf.
    mean_ir : float
        Mean of IR_class over all classes.
    max_ir : float
        Maximum IR_class (i.e., ratio between the most frequent and rarest classes).
    cvir : float
        Coefficient of variation of the IR_class values: std(IR)/mean(IR).
    entropy : float
        Shannon entropy of the class distribution (in nats).
    normalized_entropy : float
        Entropy normalized to [0,1] via dividing by log(K),
        where K is the number of classes.

    Example
    -------
    >>> y = [0, 2, 2, 1, 0, 2, 3, 3, 3, 3]
    >>> ir, mean_ir, max_ir, cvir, H, H_norm = compute_class_imbalance(y)
    >>> print(ir)        # e.g. {0:2.0, 1:4.0, 2:1.3333, 3:1.0}
    >>> print(H, H_norm)
    """
    # Convert to array and determine class set
    y = np.asarray(y)
    if classes is None:
        classes = np.unique(y)
    else:
        classes = np.asarray(classes)

    # Count occurrences per class
    counts = np.array([np.sum(y == c) for c in classes], dtype=float)
    max_count = counts.max()

    # Compute IRs (handle zero counts)
    ir = max_count / counts
    ir[counts == 0] = np.inf

    # Aggregate IR metrics
    mean_ir = ir.mean()
    max_ir = ir.max()
    std_ir = ir.std()
    cvir = std_ir / mean_ir if mean_ir != 0 else np.nan

    # Compute Shannon entropy
    total = counts.sum()
    probs = counts / total
    probs = probs[probs > 0]
    entropy = -np.sum(probs * np.log(probs))

    # Normalized entropy
    K = len(classes)
    normalized_entropy = entropy / np.log(K) if K > 1 else 0.0

    # Map back to class labels
    ir_per_class = {int(cls): float(val) for cls, val in zip(classes, ir)}

    return ir_per_class, mean_ir, max_ir, cvir, entropy, normalized_entropy

