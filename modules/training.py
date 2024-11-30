import torch
import torch.nn as nn
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from imblearn.over_sampling import RandomOverSampler

from modules.datasets import Covid19DataSet

def train_model(model, dataset, k, batch_size, epochs, random_state, num_workers, resample, learning_rate):
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)
    features, labels = np.array(dataset['features'], dtype = np.float64), np.array(dataset['labels'], dtype = np.float64)
    losses = []

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

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        grad_criterion = nn.BCEWithLogitsLoss()
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode = 'min', factor = 0.5, patience = 2, threshold = 1e-4, min_lr = 1e-6)

        for epoch in range(epochs):

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

    return model, losses
    
def train_autoencoder(model, dataset, k, batch_size, epochs, random_state, num_workers):
    validation_losses = []  # To store validation loss for each fold
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)
    features = np.array(dataset['features'], dtype=np.float64)
    labels = np.array(dataset['labels'], dtype=np.float64)

    target_sparsity = 0.05
    beta = 1e-3  # Sparsity penalty weight

    for fold, (train_idx, test_idx) in enumerate(skf.split(features, labels)):
        print(f"Fold {fold + 1}/{k}")
        Xtrain, ytrain = features[train_idx], labels[train_idx]
        Xtest, ytest = features[test_idx], labels[test_idx]

        # Resampling
        ros = RandomOverSampler(random_state=random_state)
        X_flattened = Xtrain.reshape(len(Xtrain), -1)
        X_resampled_flattened, y_resampled = ros.fit_resample(X_flattened, ytrain)
        X_resampled = X_resampled_flattened.reshape(-1, 224, 224)

        # Datasets and DataLoaders
        train_dataset = Covid19DataSet(X_resampled, y_resampled, transform='augment')
        test_dataset = Covid19DataSet(Xtest, ytest, transform='vanilla')
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = torch.nn.MSELoss()

        for epoch in range(epochs):
            print(epoch)
            # Training phase
            model.train()
            for inputs, _ in train_loader:
                inputs = inputs 
                optimizer.zero_grad()

                latent, outputs = model(inputs)
                recon_loss = criterion(outputs, inputs)
                # Sparsity loss
                latent_mean = torch.mean(latent, dim=(0, 2, 3))
                sparsity_loss = kl_divergence(target_sparsity, latent_mean).mean()
                # Total loss
                loss = recon_loss + beta * sparsity_loss
                loss.backward()
                optimizer.step()

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, _ in test_loader:
                inputs = inputs 
                _, outputs = model(inputs)
                val_loss += criterion(outputs, inputs).item()

        val_loss /= len(test_loader)  
        validation_losses.append(val_loss)
        print(f"Validation Loss for Fold {fold + 1}: {val_loss:.4f}")

    return model, validation_losses

def get_metrics(model, test_loader):

    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for inputs, targets in test_loader:
            y_true.extend(targets.numpy())
            y_pred.extend(torch.sigmoid(model(inputs)) >= 0.5)

    cm = confusion_matrix(y_true, y_pred)
    prec = cm[1,1]/(cm[1,1]+cm[0,1]) if (cm[1,1]+cm[0,1]) > 0 else 0
    rec = cm[1,1]/(cm[1,1]+cm[1,0]) if (cm[1,1]+cm[1,0]) > 0 else 0
    f1 = (2*cm[1,1])/(2*cm[1,1]+cm[0,1]+cm[1,0]) if (2*cm[1,1]+cm[0,1]+cm[1,0]) > 0 else 0
    print(f"Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}" )
    return {'precision': prec, 'recall': rec, 'f1': f1}

def kl_divergence(p, p_hat):
    """
    p: Target sparsity (e.g., 0.05)
    p_hat: Mean activation of the latent space
    """
    return p * torch.log(p / (p_hat + 1e-10)) + (1 - p) * torch.log((1 - p) / (1 - p_hat + 1e-10))


