import torch
import numpy as np
import yaml
import pickle
import json

from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
from imblearn.over_sampling import RandomOverSampler
from collections import Counter

from modules.datasets import Covid19DataSet
from modules.models import get_model
from modules.training import get_metrics

def cross_validate(model_class, dataset, k, batch_size, epochs, random_state, num_workers, resample):
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)

    features, labels = np.array(dataset['features'], dtype = np.float64), np.array(dataset['labels'], dtype = np.float64)
    all_metrics = {}

    for fold, (train_idx, test_idx) in enumerate(skf.split(features, labels)):

        print(f"Fold {fold + 1}/{k}")
        Xtrain, ytrain = features[train_idx], labels[train_idx]
        Xtest, ytest = features[test_idx], labels[test_idx]

        if resample:
            ros = RandomOverSampler(random_state=random_state)
            X_flattened = Xtrain.reshape(len(Xtrain), -1)
            X_resampled_flattened, y_resampled = ros.fit_resample(X_flattened, ytrain)
            X_resampled = X_resampled_flattened.reshape(-1, 224, 224)

            train_dataset = Covid19DataSet(X_resampled, y_resampled, transform='augment')
        else:
            train_dataset = Covid19DataSet(Xtrain, ytrain, transform='vanilla')

        test_dataset = Covid19DataSet(Xtest, ytest, transform='vanilla')

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers = num_workers)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers = num_workers)

        if len(model_class) > 1:
            model = get_model(model_class[0], weights = model_class[1])
        else:
            model = get_model(model_class[0], weights = None)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = torch.nn.CrossEntropyLoss()

        for epoch in range(epochs):
            model.train()
            for inputs, targets in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

        try:
            metrics = get_metrics(model, test_loader)
            all_metrics[f'Fold {fold+1}'] = metrics
        except Exception as e:
            print(f"Error during metrics computation for fold {fold}: {e}")

    return model, all_metrics

def save_model_performance(model_class, model_performance, config, resample):
    PERFORMANCE_PATH = config['path']['model_dir']['performance']
    if len(model_class) > 1:
        model_name = f'{model_class[0]}_{model_class[1]}'
    else:
        model_name = f'{model_class[0]}'
    saved_file = f'{PERFORMANCE_PATH}_{model_name}{'' if resample else '_unsampled'}.json'
    with open(saved_file, "w") as f:
        json.dump(model_performance, f, indent = 4)
    print(f"Model stats saved to {saved_file}")

def save_model(model_class, trained_model, config):
    if len(model_class) > 1:
        model_name = f'{model_class[0]}_{model_class[1]}'
    else:
        model_name = f'{model_class[0]}'
    model_save_dir = config['path']['model_dir']['weights']
    model_path = f"{model_save_dir}{model_name}.pth"
    torch.save(trained_model.state_dict(), model_path)
    print(f"{model_name} weights saved to {model_path}")

def kfold_workflow(resample):
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    K_FOLDS = config['training']['k_folds']
    BATCH_SIZE = config['training']['batch_size']
    NUM_WORKERS = config['training']['num_workers']
    NUM_EPOCHS = config['training']['num_epochs']
    RANDOM_STATE = config['misc']['random_seed']

    DATASET_PATH = config['path']['dataset']['preprocessed']
    
    model_class = ('convnet',)    # Change model architecture to test it's performance
    with open(f"{DATASET_PATH}train.pkl", "rb") as f:
        dataset = pickle.load(f)

    trained_model, model_performance = cross_validate(model_class, dataset, K_FOLDS, BATCH_SIZE, NUM_EPOCHS, RANDOM_STATE, NUM_WORKERS, resample)
    save_model(model_class, trained_model)
    save_model_performance(model_class, model_performance, config, resample)

if __name__ == "__main__":
    resample_choice = input("Would you like to do resampling? (yes/no): ").strip().lower() == "yes"
    print(f'Running script {'with' if resample_choice else 'without'} resampling')
    kfold_workflow(resample = resample_choice)