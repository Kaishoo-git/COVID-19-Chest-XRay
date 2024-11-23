import torch
import yaml
import json
import pandas as pd

from training_script import get_loaders
from modules.models import LinearNet, ConvNet, MyResNet18, MyDenseNet
from modules.training import get_metrics
from modules.explain import plot_loss_and_metric

def get_models(resample, config):
    MODELS_PATH = config['path']['model_save_dir']

    file_suffix = "_resampled.pth" if resample else ".pth"
    model_files = {
        "linearnet": f"{MODELS_PATH}linearnet{file_suffix}",
        "convnet": f"{MODELS_PATH}convnet{file_suffix}",
        "resnet": f"{MODELS_PATH}resnet{file_suffix}",
        "densenet": f"{MODELS_PATH}densenet{file_suffix}"
    }

    models = {
        "linearnet": LinearNet(),
        "convnet": ConvNet(),
        "resnet": MyResNet18(),
        "densenet": MyDenseNet()
    }

    for model_name, model_path in model_files.items():
        try:
            models[model_name].load_state_dict(torch.load(model_path))
            models[model_name].eval()  
            print(f"{model_name} weights loaded from {model_path}")
        except FileNotFoundError:
            print(f"Warning: {model_name} weights file not found at {model_path}")

    return models

def get_model_stats(resample, config):
    STATS_PATH = config['path']['model_save_dir']
    if resample:
        stats = f'{STATS_PATH}models_resampled_stats.json'
    else:
        stats = f'{STATS_PATH}models_resampled_stats.json'

    with open(stats, 'r') as f:
        model_stats = json.load(f)
    return model_stats

def save_table(data, resample, config):
    headers = config['misc']['headers']
    path = config['path']['visualisations']
    rows = [
        [model_name, metrics['prec'], metrics['rec'], metrics['f1']]
        for model_name, metrics in data.items()
    ]
    df = pd.DataFrame(rows, columns=headers)
    df.to_csv(f"{path}models_{'resampled' if resample else 'unsampled'}_metrics.csv", index=False)
    print("Table saved to models_metrics.csv")


def evaluate_workflow(resample):
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    EPOCHS = config['training']['num_epochs']
    BATCH_SIZE = config['training']['batch_size']
    NUM_WORKERS = config['training']['num_workers']
    VISUALISATION_PATH = config['paths']['visualisation']

    train_loader, _, test_loader = get_loaders(resample, BATCH_SIZE, NUM_WORKERS, config)
    models = get_models(resample, config)
    model_metrics = {}
    for model_name, model in models.items():
        for data in ['train', 'test']:
            if data == 'train':
                metrics = get_metrics(model, train_loader)
            else:
                metrics = get_metrics(model, test_loader)
            model_metrics[f'{model_name} ({data})'] = metrics
    save_table(model_metrics, resample, config)
    
    model_stats = get_model_stats
    for model_name, stats in model_stats:
        plot_loss_and_metric(EPOCHS, stats['train']['loss'], stats['val']['loss'], stats['train']['f1'], stats['val']['f1'], model_name, VISUALISATION_PATH)

    print(f'Saved model stats and metrics to {VISUALISATION_PATH}')

if __name__ == "__main__":
    resample_choice = input("Which model results would you like? (unsampled/resampled): ").strip().lower() == "resampled"
    evaluate_workflow(resample=resample_choice)
