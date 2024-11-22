from evaluation import train_model
from evaluation import get_metrics
from explain import plot_loss_and_metric
from explain import vis_comparison
import torch

def train_and_run_model(model, train_loader, validation_loader, test_loader, sample_pos, sample_neg, model_name, vis = False, epochs = 10):
    model_trained, stats = train_model(model, train_loader, validation_loader, epochs)
    train_metrics = get_metrics(model_trained, train_loader)
    test_metrics = get_metrics(model_trained, test_loader)

    pred_pos = 'positive' if (torch.sigmoid(model_trained(sample_pos)) >= 0.5).item() else 'negative'
    pred_neg = 'positive' if (torch.sigmoid(model_trained(sample_neg)) >= 0.5).item() else 'negative'
    print(f'Predicted {pred_pos}, {pred_neg} for postive, negative')    
    
    plot_loss_and_metric(epochs, stats['train']['loss'], stats['val']['loss'], stats['train']['f1'], stats['val']['f1'], f'{model_name} loss', f'{model_name} f1')
    
    if vis:
        for param in model_trained.parameters():
            param.requires_grad = True
        
        vis_comparison(model_trained, sample_pos, sample_neg)

    return train_metrics, test_metrics