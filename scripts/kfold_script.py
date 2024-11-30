import torch
import numpy as np
import yaml
import pickle
import json

from modules.models import get_model, EnsembleModel
from modules.training import train_model, train_autoencoder

from metrics_script import get_models

def save_model_performance(model_name, model_performance, config):
    PERFORMANCE_PATH = config['path']['model_dir']['performance']
    saved_file = f'{PERFORMANCE_PATH}{model_name}.json'
    with open(saved_file, "w") as f:
        json.dump(model_performance, f, indent = 4)
    print(f"Model stats saved to {saved_file}")

def save_model(model_name, trained_model, config):
    model_save_dir = config['path']['model_dir']['weights']
    model_path = f"{model_save_dir}{model_name}.pth"
    torch.save(trained_model.state_dict(), model_path)
    print(f"{model_name} weights saved to {model_path}")

def load_convnet_autoencod(config):
    WEIGHTS_PATH = config['path']['model_dir']['weights']
    autoencoder = get_model('autoencoder', weights = None)

    autoencoder.load_state_dict(torch.load(f'{WEIGHTS_PATH}autoencoder.pth'))

    encoder = autoencoder.encoder
    model = get_model('convet_encoder', weights = encoder)
    return model

def kfold_workflow():
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    K_FOLDS = config['training']['k_folds']
    BATCH_SIZE = config['training']['batch_size']
    NUM_WORKERS = config['training']['num_workers']
    NUM_EPOCHS = config['training']['num_epochs']
    LEARNING_RATE = config['training']['learning_rate']
    RANDOM_STATE = config['misc']['random_seed']

    DATASET_PATH = config['path']['dataset']['preprocessed']
    
    with open(f"{DATASET_PATH}train.pkl", "rb") as f:
        train_dataset = pickle.load(f)

    # convnet = get_model('convnet', weights = None)
    # convnetu, convnetu_perf = train_model(convnet, train_dataset, K_FOLDS, BATCH_SIZE, NUM_EPOCHS, RANDOM_STATE, NUM_WORKERS, False, LEARNING_RATE)
    # convnetr, convnetr_perf = train_model(convnet, train_dataset, K_FOLDS, BATCH_SIZE, NUM_EPOCHS, RANDOM_STATE, NUM_WORKERS, True, LEARNING_RATE)
    # save_model('convnet_unsampled', convnetu, config)
    # save_model_performance('convnet_unsampled', convnetu_perf, config)
    # save_model('convnet', convnetr, config)
    # save_model_performance('convnet', convnetr_perf, config)
    
    # resnet = get_model('resnet', weights = 'default')
    # resnet, resnet_p = train_model(resnet, train_dataset, K_FOLDS, BATCH_SIZE, NUM_EPOCHS, RANDOM_STATE, NUM_WORKERS, False, LEARNING_RATE)
    # save_model('resenet', resnet, config)
    # save_model_performance('resnet', resnet_p, config)

    weights = ['default', 'nih', 'chexpert', 'pc']
    for weight in weights:
        model = get_model('densenet', weights = weight)
        model, model_p = train_model(model, train_dataset, K_FOLDS, BATCH_SIZE, NUM_EPOCHS, RANDOM_STATE, NUM_WORKERS, False, LEARNING_RATE)
        save_model(f'densenet_{weight}', model, config)
        save_model_performance(f'densenet_{weight}', model_p, config)
if __name__ == "__main__":
    kfold_workflow()

