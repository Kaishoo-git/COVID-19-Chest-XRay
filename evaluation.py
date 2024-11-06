import torch
from torch.optim import lr_scheduler
import torch.nn as nn
import time
import copy

# Need to Re consider validation and stepping of gradients
def train_model(model, train_loader, validation_loader, epochs = 5, learning_rate = 0.1, grad_criterion = nn.BCEWithLogitsLoss()):
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode = 'min', factor = 0.8, patience = 1)
    best_model, best_met = copy.deepcopy(model.state_dict()), 0.0
    starttime = time.time()
    
    print("Training model...")
    model.train()

    for epoch in range(epochs):
        for phase in ['train', 'val']:
            epoch_loss = tp = fp = fn = n = 0.0
            if phase == 'train':
                model.train()
            else:
                model.eval()  
            target_loader = train_loader if phase == 'train' else validation_loader
            for i, (images, labels) in enumerate(target_loader):

                with torch.set_grad_enabled(phase == 'train'):
                    # forward pass
                    y_preds = model(images)
                    pred_labels = (torch.sigmoid(y_preds) >= 0.5).float()
                    loss = grad_criterion(y_preds, labels)
                    epoch_loss += loss.item()

                    if phase == 'train':
                        # backward prop
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()    

                for i in range(len(labels)):
                    pred_val, label_val = pred_labels[i].item(), labels[i].item() 
                    tp += 1 if (pred_val == label_val == 1 ) else 0.0
                    fp += 1 if (pred_val == 1 and label_val == 0) else 0.0
                    fn += 1 if (pred_val == 0 and label_val == 1) else 0.0
                    n += 1
                    
            epoch_met = (2 * tp) / (2 * tp + fp + fn) 

            if phase == 'train':
                # scheduler step
                scheduler.step(epoch_loss)

            if phase == 'val' and epoch_met > best_met:            
                # validate model
                best_met = epoch_met
                best_model = copy.deepcopy(model.state_dict())
                
        print(f"Epoch [{epoch+1}/{epochs}] | Epoch Metric: {epoch_met:.4f} | Epoch Loss: {epoch_loss:.4f}")
    
    elapsedtime = time.time() - starttime
    mins, sec = elapsedtime//60, elapsedtime%60
    print(f"Training completed in {mins:.0f}mins {sec:.0f}s")
    print(f"Best validation: {best_met}")

    model.load_state_dict(best_model)
    return best_model
    
def retrain_model(model, train_loader, validation_loader, epochs = 5, learning_rate = 0.1, grad_criterion = nn.BCEWithLogitsLoss()):
    model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    for params in model.parameters():
        params.requires_grad = False
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 1)
    best_model = train_model(model, train_loader, validation_loader, epochs, learning_rate, grad_criterion)
    return best_model

def evaluate_model(model, test_loader, criteria):
    with torch.no_grad():
        tp = tn = fp = fn = total = t = f = 0
        for image, label in test_loader:
            output = model(image)

            y_pred = (torch.sigmoid(output) >= 0.5).float()
            n = len(y_pred)
            for i in range(n):
                pred_val, label_val = y_pred[i].item(), label[i].item() 
                tp += (pred_val == label_val == 1)
                tn += (pred_val == label_val == 0)
                fp += (pred_val == 1 and label_val == 0)
                fn += (pred_val == 0 and label_val == 1)
                f += (pred_val == 0)
                t += (pred_val == 1)
                total += 1
    match criteria:
        case 'acc':
            return ((tp + tn) / total), t, f, total
        case 'rec':
            return tp / (tp + fn), t, f, total
        case 'prec':
            return tp / (tp + fp), t, f, total
        case 'f1':
            return (2 * tp) / (2 * tp + fp + fn), t, f, total
