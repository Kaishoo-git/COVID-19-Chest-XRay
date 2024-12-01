import torch
import numpy as np
import yaml
import pickle
import json

from modules.models import get_model, EnsembleModel
from modules.training import train_model, train_autoencoder, train_autoencoder_l21

from metrics_script import load_model

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

def load_convnet_autoencod(config, flag):
    WEIGHTS_PATH = config['path']['model_dir']['weights']
    autoencoder = get_model('autoencoder', weights = None)
    if flag:
        autoencoder.load_state_dict(torch.load(f'{WEIGHTS_PATH}autoencoder121.pth'))
    else:
        autoencoder.load_state_dict(torch.load(f'{WEIGHTS_PATH}autoencoderkl.pth'))
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
    ALPHA = config['training']['alpha']

    DATASET_PATH = config['path']['dataset']['preprocessed']
    
    with open(f"{DATASET_PATH}train.pkl", "rb") as f:
        train_dataset = pickle.load(f)

    # Train and save connvnet models
    # convnetu = get_model('convnet', weights = None)
    # convnetu, convnetu_perf = train_model(convnetu, train_dataset, K_FOLDS, BATCH_SIZE, NUM_EPOCHS, RANDOM_STATE, NUM_WORKERS, False, LEARNING_RATE)
    # save_model('convnet_unsampled', convnetu, config)
    # save_model_performance('convnet_unsampled', convnetu_perf, config)
    # convnetr = get_model('convnet', weights = None)
    # convnetr, convnetr_perf = train_model(convnetr, train_dataset, K_FOLDS, BATCH_SIZE, NUM_EPOCHS, RANDOM_STATE, NUM_WORKERS, True, LEARNING_RATE)
    # save_model('convnet', convnetr, config)
    # save_model_performance('convnet', convnetr_perf, config)
    
    # # Train and save resnet model
    # resnet = get_model('resnet', weights = 'default')
    # resnet, resnet_p = train_model(resnet, train_dataset, K_FOLDS, BATCH_SIZE, NUM_EPOCHS, RANDOM_STATE, NUM_WORKERS, True, LEARNING_RATE)
    # save_model('resnet', resnet, config)
    # save_model_performance('resnet', resnet_p, config)

    # Train and save densenet models
    # weights = ['default', 'nih', 'chexpert', 'pc']
    # for weight in weights:
    #     model = get_model('densenet', weights = weight)
    #     model, model_p = train_model(model, train_dataset, K_FOLDS, BATCH_SIZE, NUM_EPOCHS, RANDOM_STATE, NUM_WORKERS, True, LEARNING_RATE)
    #     save_model(f'densenet_{weight}', model, config)
    #     save_model_performance(f'densenet_{weight}', model_p, config)

    autoe = get_model('autoencoder', weights = None)
    autoekl, autoekl_p = train_autoencoder(autoe, train_dataset, K_FOLDS, BATCH_SIZE, NUM_EPOCHS, RANDOM_STATE, NUM_WORKERS, LEARNING_RATE)
    save_model('autoencoderkl', autoekl, config)
    save_model_performance('autoencoderkl', autoekl_p, config)
    autoel21, autoel21_p = train_autoencoder_l21(autoe, train_dataset, K_FOLDS, BATCH_SIZE, NUM_EPOCHS, RANDOM_STATE, NUM_WORKERS, LEARNING_RATE, ALPHA)
    save_model('autoencoderl21', autoel21, config)
    save_model_performance('autoencoderl21', autoel21_p, config)

    convnetl21 = load_convnet_autoencod(config, True)
    convnetl21, convnetl21_p = train_model(convnetl21, train_dataset, K_FOLDS, BATCH_SIZE, NUM_EPOCHS, RANDOM_STATE, NUM_WORKERS, True, LEARNING_RATE)
    save_model('convnetl21', convnetl21, config)
    save_model_performance('convnetl21', convnetl21_p, config)
    convnetkl = load_convnet_autoencod(config, False)
    convnetkl, convnetkl_p = train_model(convnetkl, train_dataset, K_FOLDS, BATCH_SIZE, NUM_EPOCHS, RANDOM_STATE, NUM_WORKERS, True, LEARNING_RATE)
    save_model('convnetkl', convnetkl, config)
    save_model_performance('convnetkl', convnetkl_p, config)

if __name__ == "__main__":
    kfold_workflow()

