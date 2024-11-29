import torch
import yaml
import json
import pickle
import pandas as pd
from torch.utils.data import DataLoader

from modules.datasets import Covid19DataSet
from training_script import get_loaders
from modules.models import get_model
from modules.training import get_metrics
from modules.metrics import plot_roc_auc, create_table

def get_models(model_names, config):
    MODELS_PATH = config['path']['model_dir']['weights']
    models = {}

    for entry in model_names:
        model_name = entry[0]
        weights = entry[1] if len(entry) > 1 else None
        if len(entry) > 1:
            model_weight = f"{MODELS_PATH}{model_name}_{entry[1]}.pth"
        else:
            model_weight = f"{MODELS_PATH}{model_name}.pth"
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

def get_loaders(batch_size, num_workers, config):
    DATALOADER_PATH = config['path']['dataset']['preprocessed']

    with open(f"{DATALOADER_PATH}train.pkl", "rb") as f:
        train_data = pickle.load(f)
    with open(f"{DATALOADER_PATH}test.pkl", "rb") as f:
        test_data = pickle.load(f)
    
    train_dataset = Covid19DataSet(train_data['features'], train_data['labels'], 'augment')
    test_dataset = Covid19DataSet(test_data['features'], test_data['labels'])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader

def save_table(data, config):
    TABLES_PATH = config['path']['visualisations']['tables']
    headers = config['misc']['headers']
    rows = [
        [model_name, metrics['prec'], metrics['rec'], metrics['f1']]
        for model_name, metrics in data.items()
    ]
    df = pd.DataFrame(rows, columns=headers)
    df.to_csv(f"{TABLES_PATH}models_metrics.csv", index=False)
    print("Table saved to models_metrics.csv")

def evaluate_workflow(resample):
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    BATCH_SIZE = config['training']['batch_size']
    NUM_WORKERS = config['training']['num_workers']
    PLOTS_PATH = config['path']['visualisations']['plots']

    train_loader, test_loader = get_loaders(resample, BATCH_SIZE, NUM_WORKERS, config)

    # Modify this by looking at the models available at models/weights
    model_names = [('convnet',), ('resnet', 'default'), ('densenet', 'default')]

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
    
    save_path = f'{PLOTS_PATH}roc_plot.png'
    plot_roc_auc(models, test_loader, save_path)

    print(f'Saved model stats and metrics to {PLOTS_PATH}')

if __name__ == "__main__":
    resample_choice = input("Which model results would you like? (unsampled/resampled): ").strip().lower() == "resampled"
    evaluate_workflow(resample=resample_choice)
