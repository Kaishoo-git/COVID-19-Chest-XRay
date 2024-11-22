from preprocess_data import preprocess_data_workflow
from modules.models import LinearNet, ConvNet, MyResNet18, MyDenseNet
from modules.training import train_model

import yaml
import json
import torch

def save_model_stats_and_weights(models_stats, trained_models, config):
    model_save_dir = config['paths']['model_save_dir']
    
    stats_path = f"{model_save_dir}models_stats.json"
    with open(stats_path, "w") as f:
        json.dump(models_stats, f, indent=4)
    print(f"Model stats saved to {stats_path}")

    for model_name, model in trained_models.items():
        model_path = f"{model_save_dir}{model_name}.pth"
        torch.save(model.state_dict(), model_path)
        print(f"{model_name} weights saved to {model_path}")

def train_model_workflow(resample):
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    EPOCHS, LEARNING_RATE = config['training']['num_epochs'], config['training']['learning_rate']
    train, val, _ = preprocess_data_workflow(resample=resample)

    models = {
        "linearnet": LinearNet(),
        "convnet": ConvNet(),
        "resnet": MyResNet18(),
        "densenet": MyDenseNet()
    }

    models_stats = {}
    trained_models = {}

    for model_name, model in models.items():
        print(f"Training {model_name}...")
        trained_model, model_stats = train_model(model, train, val, EPOCHS, LEARNING_RATE)
        models_stats[model_name] = model_stats
        trained_models[model_name] = trained_model

    # Save model stats and weights
    save_model_stats_and_weights(models_stats, trained_models, config)

    return models_stats

if __name__ == "__main__":
    resample_choice = input("Do you want to resample the dataset? (yes/no): ").strip().lower() == "yes"
    train_model_workflow(resample=resample_choice)