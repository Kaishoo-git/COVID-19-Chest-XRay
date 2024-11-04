import torch
from torch.optim import lr_scheduler
import torch.nn as nn
import time
import copy

# Need to Re consider validation and stepping of gradients
def train_model(model, train_loader, validation_loader, epochs = 5, \
                learning_rate = 0.01, grad_criterion = nn.BCEWithLogitsLoss(), validation_criterion = 'acc'):
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode = 'min', factor = 0.5, patience = 1)
    best_model, best_met = copy.deepcopy(model.state_dict()), 0.0
    starttime = time.time()
    
    print("Training model...")
    model.train()

    for epoch in range(epochs):
        epoch_loss = 0.0
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval  

        for i, (images, labels) in enumerate(train_loader):
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
            # scheduler step
            scheduler.step()

        if phase == 'val':            
            # validate model
            met = evaluate_model(model, validation_loader, validation_criterion)
            best_met = met
            best_model = copy.deepcopy(model.state_dict())

        print(f"Epoch [{epoch+1}/{epochs}] | Metric Val: {met:.4f} | Epoch Loss: {epoch_loss:.4f}")
    
    elapsedtime = time.time() - starttime
    mins, sec = elapsedtime//60, elapsedtime%60
    print(f"Training completed in {mins:.0f}mins {sec:.0f}s")
    print(f"Best validation: {best_met}")

    model.load_state_dict(best_model)
    return model
    

def evaluate_model(model, test_loader, criteria):
    with torch.no_grad():
        tp = tn = fp = fn = total = 0
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
                total += 1
    match criteria:
        case 'acc':
            return (tp + tn) / total
        case 'rec':
            return tp / (tp + fn)
        case 'prec':
            return tp / (tp + fp)
        case 'f1':
            return (2 * tp) / (2 * tp + fp + fn) 
