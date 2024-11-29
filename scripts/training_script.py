import yaml
import json
import torch
import pickle


from modules.models import get_model
from modules.training import train_model, train_autoencoder


def save_model_stats_and_weights(model_name, trained_model, model_stats, config):

    model_save_dir = config['path']['model_dir']['weights']
    model_path = f"{model_save_dir}{model_name}.pth"
    torch.save(trained_model.state_dict(), model_path)
    print(f"{model_name} weights saved to {model_path}")

    stats_save_dir = config['path']['model_dir']['stats']
    stats_path = f"{stats_save_dir}{model_name}_stats.json"
    with open(stats_path, "w") as f:
        json.dump(model_stats, f, indent = 4)
    print(f"Model stats saved to {stats_path}")

def training_workflow():
    print('Running training script')
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    EPOCHS = config['training']['num_epochs']
    LEARNING_RATE = config['training']['learning_rate']
    BATCH_SIZE = config['training']['batch_size']
    NUM_WORKERS = config['training']['num_workers']

    train_loader, val_loader, _ = get_loaders(BATCH_SIZE, NUM_WORKERS, config)

    # Choose model as needed to save a pretrained model in models/weights/
    models = {
        # "linearnet": LinearNet(),
        # "convnet": ConvNet(),
        "resnet_default": get_model('resnet', weights = 'default'),
        # "densenet_default": get_model('densenet', 'default'),
        # "densenet_nih": get_model('densenet', 'nih'),
        # "densenet_chexpert": get_model('densenet', 'chexpert'),
        # "densenet_pc":get_model('densenet', 'pc'),
    }

    for model_name, model in models.items():
        print(f"Training {model_name}...")
        trained_model, model_stats = train_model(model, train_loader, val_loader, EPOCHS, LEARNING_RATE)
        save_model_stats_and_weights(model_name, trained_model, model_stats, config)

    print(f'Models trained and saved in {config['path']['model_dir']['weights']}')

if __name__ == "__main__":
    training_workflow()