import torch
import numpy as np
import yaml
import pickle
import json

from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
from imblearn.over_sampling import RandomOverSampler

from modules.datasets import Covid19DataSet
from modules.models import get_model
from modules.training import get_metrics

def cross_validate(model_class, dataset, k, batch_size, epochs, random_state, num_workers):
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)

    features, labels = dataset.features, dataset.labels 

    all_metrics = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(features, labels)):
        print(f"Processing fold {fold + 1}/{k}...")

        train_data, test_data = dataset[train_idx], dataset[test_idx]

        ros = RandomOverSampler(random_state=random_state)
        X = [d['img'] for d in train_data]  
        y = [d['lab'] for d in train_data]  
        X_resampled, y_resampled = ros.fit_resample(np.array(X).reshape(len(X), -1), y)
        train_data_resampled = [{'img': img, 'lab': label} for img, label in zip(X_resampled, y_resampled)]

        train_dataset = Covid19DataSet(train_data_resampled, transform='vanilla')
        test_dataset = Covid19DataSet(test_data, transform='vanilla')

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers = num_workers)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers = num_workers)

        if len(model_class) > 1:
            model = get_model(model_class[0], model_class[1])
        else:
            model = get_model(model_class[0])

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

        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for inputs, targets in test_loader:
                outputs = model(inputs)
                y_true.extend(targets.numpy())
                y_pred.extend(torch.argmax(outputs, dim=1).cpu().numpy())

        try:
            metrics = get_metrics(y_true, y_pred)
            all_metrics.append(metrics)
        except Exception as e:
            print(f"Error during metrics computation for fold {fold}: {e}")

    mean_metrics = {key: np.mean([m[key] for m in all_metrics]) for key in all_metrics[0]}
    std_metrics = {key: np.std([m[key] for m in all_metrics]) for key in all_metrics[0]}

    print("Cross-validation results:")
    print(f"Mean metrics: {mean_metrics}")
    print(f"Std metrics: {std_metrics}")

    return mean_metrics, std_metrics

def save_model_performance(model_class, model_performance, config):
    PERFORMANCE_PATH = config['model_dir']['performance']
    if len(model_class) > 1:
        model_name = f'{model_class[0]}_{model_class[1]}'
    else:
        model_name = f'{model_class[1]}'
    saved_file = f'{PERFORMANCE_PATH}_{model_name}.json'
    with open(saved_file, "w") as f:
        json.dump(model_performance, f, indent = 4)
    print(f"Model stats saved to {saved_file}")

def kfold_workflow():
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    K_FOLDS = config['training']['k_folds']
    BATCH_SIZE = config['training']['batch_size']
    NUM_WORKERS = config['training']['num_workers']
    NUM_EPOCHS = config['training']['num_epochs']
    RANDOM_STATE = config['misc']['random_seed']

    DATASET_PATH = config['data']['preprocessed']
    
    model_class = ('convnet',)
    with open(f"{DATASET_PATH}dataset.pkl", "rb") as f:
        dataset = pickle.load(f)

    mean_metrics, std_metrics = cross_validate(model_class, dataset, K_FOLDS, BATCH_SIZE, NUM_EPOCHS, RANDOM_STATE, NUM_WORKERS)
    model_performance = {
        'mean_metrics': mean_metrics,
        'std_metrics': std_metrics
    }
    save_model_performance(model_class, model_performance, config)

if __name__ == "__main__":
    kfold_workflow()