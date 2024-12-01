import matplotlib.pyplot as plt
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score

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

def plot_roc_auc(models, dataloader):
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

        line, = plt.plot(fpr, tpr, label=model_name)

        plt.plot([0, 1], [0, 1], linestyle='--', color='gray')

    plt.legend(loc='lower right')
    plt.grid()
    return plt

def get_metrics(model, test_loader):
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    with torch.no_grad():
        for inputs, targets in test_loader:
            y_true.extend(targets.numpy())
            y_prob.extend(torch.sigmoid(model(inputs)))
            y_pred.extend(torch.sigmoid(model(inputs)) >= 0.5)

    cm = confusion_matrix(y_true, y_pred)
    auc_score = roc_auc_score(y_true, y_prob)
    prec = cm[1,1]/(cm[1,1]+cm[0,1]) if (cm[1,1]+cm[0,1]) > 0 else 0
    rec = cm[1,1]/(cm[1,1]+cm[1,0]) if (cm[1,1]+cm[1,0]) > 0 else 0
    f1 = (2*cm[1,1])/(2*cm[1,1]+cm[0,1]+cm[1,0]) if (2*cm[1,1]+cm[0,1]+cm[1,0]) > 0 else 0

    print(f"Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f} | AUC: {auc_score:.4f}" )
    return {'precision': round(prec, 4), 'recall': round(rec, 4), 'f1': round(f1, 4), 'auc': round(auc_score, 4)}

def create_table(headers, data):
    df = pd.DataFrame(data, columns=headers)
    print(df)
