import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np  
from sklearn.metrics import roc_curve, auc, RocCurveDisplay
from utils import average_fpr

def plot_heatmap(matrix, title):
    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
    plt.title(title)
    plt.show()

def plot_confusion_matrix(confusion_matrix_df, FC_name, title='Confusion Matrix'):
    fig, ax = plt.subplots()
    sns.set_theme(palette = 'husl')
    sns.set_theme(rc={"figure.figsize":(8, 4)})
    sns.heatmap(confusion_matrix_df, fmt='g',annot=True,cmap="GnBu").set_title(title + FC_name)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    plt.show()

def plot_training_accuracy(acc_LIST_CNN, epoch, FC_name):
  fig, ax = plt.subplots()
  sns.set_theme(palette = 'husl')
  sns.set_theme(rc={"figure.figsize":(5, 3)})
  ax.set_ylabel('Training Accuracy')
  ax.set_xlabel('Epochs')
  sns.lineplot(x=list(range(epoch)),y=acc_LIST_CNN).set_title('Training Accuracy vs Epochs for CNN' + FC_name)



def plot_roc_curve(num_classes, y_true, y_scores):
    fpr = {}
    tpr = {}
    roc_auc = {}
    plt.figure(figsize=(10, 8))
    # produces auc per class per FC metric per cmodel
    for i in range(num_classes):
        # print(i)
        fpr[i], tpr[i], _ = roc_curve(np.array(y_true) == i, np.array(y_scores)[:,i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot ROC curve for each class
    plt.figure(figsize=(10, 8))
    for i in range(num_classes):
        plt.plot(fpr[i], tpr[i], label=f"Class {i} (AUC = {roc_auc[i]:.2f})")

        # Plot the diagonal
    plt.plot([0, 1], [0, 1], 'k--', label='Random Chance (AUC = 0.50)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.show()

def plot_mean_roc_fc(data):
    fc_metricses = list(data.keys())  # FC metric names
    model_nameses = list(next(iter(data.values())).keys())  # Model names

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))  # 2 rows, 3 columns
    fig.suptitle('ROC Curves for Different FC Metrics', fontsize=16)

    for idx, fc_name in enumerate(fc_metricses):
        ax = axes[idx // 3, idx % 3]  # Determine subplot position
        ax.set_title(fc_name)

        for model_name in model_nameses:
         # print(list(data[fc_name][model_name][0].values()))
          #rint(type(data[fc_name][model_name][0]))
          fpr  = average_fpr(data[fc_name][model_name][0])
          # print('fpr')
          # print(fpr)
          tpr  = average_fpr(data[fc_name][model_name][1])
          auc_data = auc(fpr, tpr)

          ax.plot(fpr, tpr, label=f'{model_name} (Model={model_name}, AUC={auc_data:.2f})', linewidth=4)
        ax.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=2)  # Diagonal line

        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.legend()
        ax.grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

def plot_mean_roc_model(data):
    fc_metricses = list(data.keys())  # FC metric names
    model_nameses = list(next(iter(data.values())).keys())  # Model names

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))  # 2 rows, 3 columns
    fig.suptitle('ROC Curves for Different FC Metrics', fontsize=16)

    for idx, model_name in enumerate(model_nameses):
        ax = axes[idx // 3, idx % 3]  # Determine subplot position
        ax.set_title(model_name)

        for fc_name in fc_metricses:
          #print(data[fc_name][model_name][0])
          fpr  = average_fpr(data[fc_name][model_name][0])
          # print('fpr')
          # print(fpr)
          tpr  = average_fpr(data[fc_name][model_name][1])
          auc_data = auc(fpr, tpr)

          ax.plot(fpr, tpr, label=f'{fc_name} (FC Metric={fc_name}, AUC={auc_data:.2f})', linewidth=4)
          ax.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=2)  # Diagonal line

        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.legend()
        ax.grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()