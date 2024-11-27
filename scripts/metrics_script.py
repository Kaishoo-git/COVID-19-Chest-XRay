import torch
import yaml
import json
import pandas as pd

from training_script import get_loaders
from modules.models import get_model
from modules.training import get_metrics
from modules.metrics import plot_loss_and_metric, plot_roc_auc, create_table

# [('convnet',), ('resnet', 'default'), ('densenet', 'nih')]
def get_models(resample, model_names, config):
    MODELS_PATH = config['path']['model_dir']['weights']
    models = {}

    for entry in model_names:
        model_name = entry[0]
        weights = entry[1] if len(entry) > 1 else None
        if len(entry) > 1:
            model_weight = f"{MODELS_PATH}{model_name}_{entry[1]}_{'resampled' if resample else 'unsampled'}.pth"
        else:
            model_weight = f"{MODELS_PATH}{model_name}_{'resampled' if resample else 'unsampled'}.pth"
        try:
            model = get_model(model_name, weights = weights)
            model.load_state_dict(torch.load(model_weight))  
            model.eval()  
            print(f"{model_name} weights loaded from {model_weight}")
            models[model_name] = model
        except Exception as e:
            print(f"Error loading {model_name}: {e}")
            raise

    return models

def get_model_stats(resample, config):
    LOSS_PATH = config['path']['model_save_dir']['stats']
    if resample:
        stats = f'{LOSS_PATH}models_resampled_stats.json'
    else:
        stats = f'{LOSS_PATH}models_resampled_stats.json'

    with open(stats, 'r') as f:
        model_stats = json.load(f)
    return model_stats

def save_table(data, resample, config):
    TABLES_PATH = config['path']['visualisations']['tables']
    headers = config['misc']['headers']
    rows = [
        [model_name, metrics['prec'], metrics['rec'], metrics['f1']]
        for model_name, metrics in data.items()
    ]
    df = pd.DataFrame(rows, columns=headers)
    df.to_csv(f"{TABLES_PATH}models_{'resampled' if resample else 'unsampled'}_metrics.csv", index=False)
    print("Table saved to models_metrics.csv")


def evaluate_workflow(resample):
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    BATCH_SIZE = config['training']['batch_size']
    NUM_WORKERS = config['training']['num_workers']
    PLOTS_PATH = config['path']['visualisations']['plots']

    train_loader, _, test_loader = get_loaders(resample, BATCH_SIZE, NUM_WORKERS, config)
    model_names = [('convnet',), ('resnet', 'default'), ('densenet', 'nih')]
    models = get_models(resample, model_names, config)
    model_metrics = {}

    for model_name, model in models.items():
        for data in ['train', 'test']:
            if data == 'train':
                metrics = get_metrics(model, train_loader)
            else:
                metrics = get_metrics(model, test_loader)
            model_metrics[f'{model_name} ({data})'] = metrics
    save_table(model_metrics, resample, config)
    
    # save_path = f'{PLOTS_PATH}roc_plot.png'
    # plot_roc_auc(models, test_loader, save_path)

    print(f'Saved model stats and metrics to {PLOTS_PATH}')

if __name__ == "__main__":
    resample_choice = input("Which model results would you like? (unsampled/resampled): ").strip().lower() == "resampled"
    evaluate_workflow(resample=resample_choice)
