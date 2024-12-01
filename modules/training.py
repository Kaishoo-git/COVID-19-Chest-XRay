import torch
import numpy as np

from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from imblearn.over_sampling import RandomOverSampler

from modules.datasets import Covid19DataSet

def train_model(model, dataset, k, batch_size, epochs, random_state, num_workers, resample, learning_rate):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)  # Move model to the device
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)
    features, labels = np.array(dataset['features'], dtype=np.float64), np.array(dataset['labels'], dtype=np.float64)
    losses = []

    for epoch in range(epochs):
        print(f'Epoch {epoch + 1}/{epochs}')
        for fold, (train_idx, test_idx) in enumerate(skf.split(features, labels)):
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
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            grad_criterion = torch.nn.BCEWithLogitsLoss()
            scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, threshold=1e-4, min_lr=1e-6)

            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()
                else:
                    model.eval()

                target_loader = train_loader if phase == 'train' else test_loader
                epoch_loss = 0.0

                for i, (images, labs) in enumerate(target_loader):
                    # Move data to device
                    images, labs = images.to(device), labs.to(device)

                    with torch.set_grad_enabled(phase == 'train'):
                        # Forward pass
                        y_preds = model(images)
                        loss = grad_criterion(y_preds, labs)
                        epoch_loss += loss.item()

                        if phase == 'train':
                            # Backward pass
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()

                if phase == 'train':
                    scheduler.step(epoch_loss)

                if phase == 'val':
                    losses.append(epoch_loss)

    return model, losses
    
def train_autoencoder(model, dataset, k, batch_size, epochs, random_state, num_workers, learning_rate):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)  # Move model to the device
    validation_losses = []
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)
    features, labels = np.array(dataset['features'], dtype=np.float64), np.array(dataset['labels'], dtype=np.float64)

    target_sparsity = 0.05
    beta = 1e-3  # Sparsity penalty weight

    for epoch in range(epochs):
        print(f'Epoch {epoch + 1}/{epochs}')
        for fold, (train_idx, test_idx) in enumerate(skf.split(features, labels)):
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

            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            criterion = torch.nn.MSELoss()

            # Training phase
            model.train()
            for inputs, _ in train_loader:
                inputs = inputs.to(device)  # Move inputs to device
                optimizer.zero_grad()

                latent, outputs = model(inputs)
                recon_loss = criterion(outputs, inputs)

                # Sparsity loss
                latent_mean = torch.mean(latent, dim=(0, 2, 3))
                sparsity_loss = kl_divergence(target_sparsity, latent_mean).mean()
                loss = recon_loss + beta * sparsity_loss

                loss.backward()
                optimizer.step()

            # Validation phase
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inputs, _ in test_loader:
                    inputs = inputs.to(device)  # Move inputs to device
                    _, outputs = model(inputs)
                    val_loss += criterion(outputs, inputs).item()

            val_loss /= len(test_loader)
            validation_losses.append(val_loss)
            print(f"Validation Loss for Fold {fold + 1}: {val_loss:.4f}")

    return model, validation_losses

def train_autoencoder_l21(model, dataset, k, batch_size, epochs, random_state, num_workers, learning_rate, alpha):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)  # Move model to the device
    validation_losses = []  # To store validation loss for each fold
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)
    features = np.array(dataset['features'], dtype=np.float64)
    labels = np.array(dataset['labels'], dtype=np.float64)

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        for fold, (train_idx, test_idx) in enumerate(skf.split(features, labels)):
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

            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            criterion = torch.nn.MSELoss()

            # Training phase
            model.train()
            for inputs, _ in train_loader:
                inputs = inputs.to(device)
                optimizer.zero_grad()

                latent, outputs = model(inputs)
                # Reconstruction loss
                recon_loss = criterion(outputs, inputs)

                # Sparsity loss (L2,1 norm)
                l21_sparsity_loss = compute_l21_sparsity(model.encoder[0].weight)  

                # Total loss
                loss = recon_loss + alpha * l21_sparsity_loss
                loss.backward()
                optimizer.step()

            # Validation phase
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inputs, _ in test_loader:
                    inputs = inputs.to(device)
                    _, outputs = model(inputs)
                    val_loss += criterion(outputs, inputs).item()

            val_loss /= len(test_loader)
            validation_losses.append(val_loss)
            print(f"Validation Loss for Fold {fold + 1}: {val_loss:.4f}")

    return model, validation_losses

def kl_divergence(p, p_hat):
    """
    p: Target sparsity (e.g., 0.05)
    p_hat: Mean activation of the latent space
    """
    return p * torch.log(p / (p_hat + 1e-10)) + (1 - p) * torch.log((1 - p) / (1 - p_hat + 1e-10))

def compute_l21_sparsity(weight_matrix):
    """
    Compute the L2,1 norm for a weight matrix.

    Args:
        weight_matrix (torch.Tensor): The weight matrix to compute sparsity for.
    Returns:
        torch.Tensor: The L2,1 norm.
    """
    l21_norm = torch.sum(torch.sqrt(torch.sum(weight_matrix**2, dim=1) + 1e-10))  # Add small value for numerical stability
    return l21_norm
