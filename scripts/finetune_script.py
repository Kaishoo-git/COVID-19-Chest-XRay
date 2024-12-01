import yaml
import pickle
import re
import torch
import numpy as np

from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from imblearn.over_sampling import RandomOverSampler

from modules.models import get_model
from modules.datasets import Covid19DataSet

def finetune_workflow():
    """DO NOT TOUCH THIS"""
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
        dataset = pickle.load(f)

    model = get_model('resnet', weights = 'default')

    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    features, labels = np.array(dataset['features'], dtype = np.float64), np.array(dataset['labels'], dtype = np.float64)
    losses = []

    for epoch in range(NUM_EPOCHS):
        print(f'Epoch {epoch + 1}/{NUM_EPOCHS}')
        for name, params in model.resnet.named_parameters():
            layer_match = re.search(r'layer(\d+)', name)
            if layer_match:
                layer_idx = int(layer_match.group(1))  # Extract the layer index
                # Dynamically freeze/unfreeze layers based on epoch
                if layer_idx < epoch + 1:  # Unfreeze lower layers
                    params.requires_grad = True
                else:  # Freeze higher layers
                    params.requires_grad = False

        for fold, (train_idx, test_idx) in enumerate(skf.split(features, labels)):

            print(f"Fold {fold + 1}/{K_FOLDS}")
            Xtrain, ytrain = features[train_idx], labels[train_idx]
            Xtest, ytest = features[test_idx], labels[test_idx]

            ros = RandomOverSampler(random_state=RANDOM_STATE)
            X_flattened = Xtrain.reshape(len(Xtrain), -1)
            X_resampled_flattened, y_resampled = ros.fit_resample(X_flattened, ytrain)
            X_resampled = X_resampled_flattened.reshape(-1, 224, 224)

            train_dataset = Covid19DataSet(X_resampled, y_resampled, transform='augment')
            test_dataset = Covid19DataSet(Xtest, ytest, transform='vanilla')
            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers = NUM_WORKERS)
            test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers = NUM_WORKERS)

            optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
            grad_criterion = torch.nn.BCEWithLogitsLoss()
            scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode = 'min', factor = 0.5, patience = 2, threshold = 1e-4, min_lr = 1e-6)

            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()
                else:
                    model.eval()  
            
                target_loader = train_loader if phase == 'train' else test_loader
                epoch_loss = 0.0
            
                for i, (images, labs) in enumerate(target_loader):

                    with torch.set_grad_enabled(phase == 'train'):
                        # forward pass
                        y_preds = model(images)
                        loss = grad_criterion(y_preds, labs)
                        epoch_loss += loss.item()

                        if phase == 'train':
                            # backward prop
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()

                if phase == 'train':
                    scheduler.step(epoch_loss)    

                if phase == 'val':
                    losses.append(epoch_loss)

if __name__ == "__main__":
    finetune_workflow()