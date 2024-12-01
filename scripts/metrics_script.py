import torch
import yaml
import json
import pickle
import pandas as pd
from torch.utils.data import DataLoader

from modules.datasets import Covid19DataSet
from modules.models import get_model
from modules.metrics import get_metrics, plot_roc_auc

def load_model(model_class, config):
    MODELS_PATH = config['path']['model_dir']['weights']
    model_weight = f"{MODELS_PATH}{model_class[0]}.pth"
    try:
        model = get_model(model_class[1], weights = model_class[2])
        model.load_state_dict(torch.load(model_weight, map_location=torch.device('cpu')))  
        model.eval()  
        print(f"{model_class[0]} weights loaded from {model_weight}")
    except Exception as e:
        print(f"Error loading {model_class[0]}: {e}")
        raise
    return model

def get_loaders(batch_size, num_workers, config):
    DATALOADER_PATH = config['path']['dataset']['preprocessed']
    try:
        with open(f"{DATALOADER_PATH}train.pkl", "rb") as f:
            train_data = pickle.load(f)
        with open(f"{DATALOADER_PATH}test.pkl", "rb") as f:
            test_data = pickle.load(f)
    except Exception as e:
        print(f'Error loading datasets: {e}')
        raise
    
    train_dataset = Covid19DataSet(train_data['features'], train_data['labels'])
    test_dataset = Covid19DataSet(test_data['features'], test_data['labels'])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader

def save_table(rows, config):
    TABLES_PATH = config['path']['visualisations']['tables']
    headers = config['misc']['headers']
    df = pd.DataFrame(rows, columns=headers)
    df.to_csv(f"{TABLES_PATH}models_metrics.csv", index=False)
    print("Table saved to models_metrics.csv")

def evaluate_workflow():
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    BATCH_SIZE = config['training']['batch_size']
    NUM_WORKERS = config['training']['num_workers']
    PLOTS_PATH = config['path']['visualisations']['plots']

    train_loader, test_loader = get_loaders(BATCH_SIZE, NUM_WORKERS, config)

    # Modify this by looking at the models available at models/weights
    model_classes = [('convnet', 'convnet', None), ('convnet_unsampled', 'convnet', None)]
    # model_classes = [('convnet', 'convnet', None), ('resnet', 'resnet', 'default'), ('densenet_default', 'densenet', 'default')]
    # model_classes = [('densenet_default', 'densenet', 'default'), ('densenet_nih', 'densenet', 'nih'), ('densenet_chexpert', 'densenet', 'chexpert'), ('densenet_pc', 'densenet', 'pc')]
    # model_classes = [('convnetkl', 'convnet_encoder', place_ae), ('convnetl21', 'convnet_encoder', place_ae)]
    place_ae = get_model('autoencoder', weights = None).encoder
    models = {}
    rows = []
    for model_class in model_classes:
        model = load_model(model_class, config)
        models[model_class[0]] = model
        for data in ['train', 'test']:
            if data == 'train':
                metrics = get_metrics(model, train_loader)
            else:
                metrics = get_metrics(model, test_loader)
            rows.append([f'{model_class[0]} ({data})', metrics['precision'], metrics['recall'], metrics['f1'], metrics['auc']])
    save_table(rows, config)

    save_path = f'{PLOTS_PATH}roc_plot.png'
    auc_curve = plot_roc_auc(models, test_loader)
    auc_curve.savefig(save_path, dpi = 300)
    print(f"Plot saved to {save_path}")


if __name__ == "__main__":
    evaluate_workflow()
