import yaml
import json
import torch
import pickle

from torch.utils.data import DataLoader
from modules.dataset import Covid19DataSet
from modules.models import LinearNet, ConvNet, GenResNet18, GenDenseNet
from modules.training import train_model


def save_model_stats_and_weights(resample, models_stats, trained_models, config):
    model_save_dir = config['path']['model_save_dir']
    stats_path = f"{model_save_dir}models_{'resampled' if resample else 'unsampled'}_stats.json"
    with open(stats_path, "w") as f:
        json.dump(models_stats, f, indent = 4)
    print(f"Model stats saved to {stats_path}")

    for model_name, model in trained_models.items():
        model_path = f"{model_save_dir}{model_name}_{'resampled' if resample else 'unsampled'}.pth"
        torch.save(model.state_dict(), model_path)
        print(f"{model_name} weights saved to {model_path}")

def get_loaders(resample, batch_size, num_workers, config):
    DATALOADER_PATH = config['path']['dataset']['preprocessed']
    train_file = "train_resampled.pkl" if resample else "train_unsampled.pkl"
    val_file = "val.pkl"
    test_file = "test.pkl"

    with open(f"{DATALOADER_PATH}{train_file}", "rb") as f:
        train_data = pickle.load(f)
    with open(f"{DATALOADER_PATH}{val_file}", "rb") as f:
        val_data = pickle.load(f)
    with open(f"{DATALOADER_PATH}{test_file}", "rb") as f:
        test_data = pickle.load(f)

    train_loader = DataLoader(dataset=Covid19DataSet(train_data), batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(dataset=Covid19DataSet(val_data), batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(dataset=Covid19DataSet(test_data), batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader

def training_workflow(resample):
    print(f'Training on {'resampled' if resample else 'unsampled'} dataset')
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    EPOCHS = config['training']['num_epochs']
    LEARNING_RATE = config['training']['learning_rate']
    BATCH_SIZE = config['training']['batch_size']
    NUM_WORKERS = config['training']['num_workers']

    train_loader, val_loader, _ = get_loaders(resample, BATCH_SIZE, NUM_WORKERS, config)

    models = {
        # "linearnet": LinearNet(),
        # "convnet": ConvNet(),
        "gen_resnet": GenResNet18(weights = 'default')
        # "xray_resnet": GenResNet18(weights = 'xray'),
        # "gen_densenet": GenDenseNet(weights = 'default'),
        # "nih_densenet": GenDenseNet(weights = 'nih')
        # "chexpert_densenet": GenDenseNet(weights = 'chexpert'),
        # "pc_densenet": GenDenseNet(weights = 'pc'),
    }

    models_stats = {}
    trained_models = {}

    for model_name, model in models.items():
        print(f"Training {model_name}...")
        trained_model, model_stats = train_model(model, train_loader, val_loader, EPOCHS, LEARNING_RATE)
        models_stats[model_name] = model_stats
        trained_models[model_name] = trained_model

    save_model_stats_and_weights(resample, models_stats, trained_models, config)

    print(f'Models trained and saved in {config['path']['model_save_dir']}')

if __name__ == "__main__":
    resample_choice = input("Which dataset would you like to train on? (unsampled/resampled): ").strip().lower() == "resampled"
    training_workflow(resample=resample_choice)