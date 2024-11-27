import matplotlib.pyplot as plt
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score

def plot_loss_and_metric(epochs, train_loss, val_loss, train_metric, test_metric, model_name, save_path = None):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6)) 
    
    axes[0].plot([i+1 for i in range(epochs)], train_loss, label='training', linestyle='-', color='blue', marker='o')
    axes[0].plot([i+1 for i in range(epochs)], val_loss, label='validation', linestyle='--', color='red', marker='s')
    axes[0].set_xticks(range(1, epochs+1, 1))
    upper = min(1.1, max(max(train_loss), max(val_loss)))
    spacing = (upper - 0) / 10
    axes[0].set_ylim(0, upper)
    axes[0].set_yticks(np.arange(0, upper + spacing, spacing))
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('BCE Loss')
    axes[0].set_title(f'{model_name} loss')
    axes[0].legend()

    axes[1].plot([i+1 for i in range(epochs)], train_metric, label='training', linestyle='-', color='blue', marker='o')
    axes[1].plot([i+1 for i in range(epochs)], test_metric, label='test', linestyle='--', color='red', marker='s')
    axes[1].set_xticks(range(1, epochs+1, 1))
    axes[1].set_yticks(np.arange(0, 1.1, 0.1))
    axes[1].set_ylim(0, 1.1)
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Metric Value')
    axes[1].set_title(f'{model_name} metric')
    axes[1].legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)  
        print(f"Plot saved to {save_path}")

    plt.show()

def plot_roc_auc(models, dataloader, save_path=None):
    plt.figure(figsize=(8, 6))
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    
    text_shift = 0.02  
    for i, (model_name, model) in enumerate(models.items()):
        model.eval()
        y_true, y_pred = [], []

        with torch.no_grad():
            for inputs, labels in dataloader:
                outputs = torch.sigmoid(model(inputs))
                y_true.extend(labels.numpy())
                y_pred.extend(outputs.numpy())

        fpr, tpr, _ = roc_curve(y_true, y_pred)
        auc_score = roc_auc_score(y_true, y_pred)

        line, = plt.plot(fpr, tpr, label=model_name)

        plt.plot([0, 1], [0, 1], linestyle='--', color='gray')

        mid_point = len(fpr) // 2
        plt.text(
            fpr[mid_point], 
            tpr[mid_point] + (i * text_shift),  
            f'AUC = {auc_score:.2f}', 
            fontsize=10, color=line.get_color()  
        )

    plt.legend(loc='lower right')
    plt.grid()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Plot saved to {save_path}")
    plt.show()

def create_table(headers, data):
    df = pd.DataFrame(data, columns=headers)
    print(df)
