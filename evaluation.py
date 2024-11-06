import torch
from torch.optim import lr_scheduler
import torch.nn as nn
import time
import copy

# Need to Re consider validation and stepping of gradients
def train_model(model, train_loader, validation_loader, epochs = 5, learning_rate = 0.1, grad_criterion = nn.BCEWithLogitsLoss()):
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode = 'min', factor = 0.8, patience = 1)
    best_model, best_loss = copy.deepcopy(model.state_dict()), 9999.9
    starttime = time.time()
    
    print("Training model...")

    for epoch in range(epochs):
        epoch_loss, t_loss, v_loss = 0.0, [], []

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()  
        
            target_loader = train_loader if phase == 'train' else validation_loader
        
            for i, (images, labels) in enumerate(target_loader):
                with torch.set_grad_enabled(phase == 'train'):
                    # forward pass
                    y_preds = model(images)
                    loss = grad_criterion(y_preds, labels)
                    epoch_loss += loss.item()

                    if phase == 'train':
                        # backward prop
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()    

            if phase == 'train':
                avg_loss = epoch_loss / i
                t_loss.append(avg_loss)
                # scheduler step
                scheduler.step(epoch_loss)    

            if phase == 'val':
                avg_loss = epoch_loss / i
                v_loss.append(avg_loss)
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    best_model = copy.deepcopy(model.state_dict())

        print(f"Epoch [{epoch + 1}/{epochs}] |Training Loss: {t_loss[-1]:.4f} | Validation Loss: {v_loss[-1]:.4f}")  

    
    elapsedtime = time.time() - starttime
    mins, sec = elapsedtime//60, elapsedtime%60
    print(f"Training completed in {mins:.0f}mins {sec:.0f}s")
    print(f"Best Loss: {best_loss}")

    model.load_state_dict(best_model)
    return best_model
    
def retrain_model(model, train_loader, validation_loader, epochs = 5, learning_rate = 0.1, grad_criterion = nn.BCEWithLogitsLoss()):
    with torch.no_grad():
        model.conv1.weight = nn.Parameter(model.conv1.weight.mean(dim=1, keepdim=True))
    for params in model.parameters():
        params.requires_grad = False
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 1)
    best_model = train_model(model, train_loader, validation_loader, epochs, learning_rate, grad_criterion)
    return best_model

def evaluate_model(model, test_loader, criteria):
    with torch.no_grad():
        tp = tn = fp = fn = t = f = 0
        for j, (images, labels) in enumerate(test_loader):
            outputs = model(images)
            y_preds = (torch.sigmoid(outputs) >= 0.5).float()
            n = len(y_preds)
            for i in range(n):
                pred_val, label_val = y_preds[i].item(), labels[i].item() 
                tp += (pred_val == label_val == 1)
                tn += (pred_val == label_val == 0)
                fp += (pred_val == 1 and label_val == 0)
                fn += (pred_val == 0 and label_val == 1)
                f += (pred_val == 0)
                t += (pred_val == 1)
    j += 1
    match criteria:
        case 'acc':
            return ((tp + tn) / j), t, f, j
        case 'rec':
            return tp / (tp + fn), t, f, j
        case 'prec':
            return tp / (tp + fp), t, f, j
        case 'f1':
            return (2 * tp) / (2 * tp + fp + fn), t, f, j
