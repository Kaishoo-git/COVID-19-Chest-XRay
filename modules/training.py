import torch
import torch.nn as nn
import time
import copy
import numpy as np
from torch.optim import lr_scheduler
from sklearn.metrics import confusion_matrix

def train_model(model, train_loader, validation_loader, epochs, learning_rate, grad_criterion = nn.BCEWithLogitsLoss()):
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode = 'min', factor = 0.5, patience = 2, threshold = 1e-4, min_lr = 1e-6)
    best_model = copy.deepcopy(model.state_dict())
    t_loss, t_prec, t_rec, t_f1, v_loss, v_prec, v_rec, v_f1 = [], [], [], [], [], [], [], []
    best_loss, best_f1 = 1e10, 0.0
    starttime = time.time()

    for epoch in range(epochs):

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()  
        
            target_loader = train_loader if phase == 'train' else validation_loader
            n_total = r_tp = r_fp = r_fn = epoch_loss = 0.0
        
            for i, (images, labels) in enumerate(target_loader):
                n_total += images.size(dim = 0)

                with torch.set_grad_enabled(phase == 'train'):
                    # forward pass
                    y_preds = model(images)
                    pred_labels = (torch.sigmoid(y_preds) >= 0.5)
                    loss = grad_criterion(y_preds, labels)

                    epoch_loss += loss.item()
                    for j in range(pred_labels.size(dim = 0)):
                        actual, pred = labels[j].item(), pred_labels[j].item()
                        r_tp += 1 if (actual == 1 and pred == 1) else 0
                        r_fp += 1 if (actual == 0 and pred == 1) else 0
                        r_fn += 1 if (actual == 1 and pred == 0) else 0

                    if phase == 'train':
                        # backward prop
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

            avg_prec = r_tp / (r_tp + r_fp) if (r_tp + r_fp) > 0 else 0
            avg_rec = r_tp / (r_tp + r_fn) if (r_tp + r_fn) > 0 else 0
            avg_f1 = (2*r_tp) / (2*r_tp + r_fp + r_fn) if (2*r_tp + r_fp + r_fn) > 0 else 0

            if phase == 'train':
                t_loss.append(epoch_loss)
                t_prec.append(avg_prec)
                t_rec.append(avg_rec)
                t_f1.append(avg_f1)
                # scheduler step
                scheduler.step(epoch_loss)    

            if phase == 'val':
                v_loss.append(epoch_loss)
                v_prec.append(avg_prec)
                v_rec.append(avg_rec)
                v_f1.append(avg_f1)

                if (t_loss[-1] + v_loss[-1]) < best_loss:
                    best_loss, best_f1 = (t_loss[-1] + v_loss[-1]), t_f1[-1] 
                    best_model = copy.deepcopy(model.state_dict())
            
        print(f"Epoch [{epoch + 1}/{epochs}] |Training Loss: {t_loss[-1]:.4f} | Validation Loss: {v_loss[-1]:.4f} | Training F1: {t_f1[-1]:.4f}")
    
    elapsedtime = time.time() - starttime
    mins, sec = elapsedtime//60, elapsedtime%60

    print(f"Training completed in {mins:.0f}mins {sec:.2f}s")
    print(f"Best Loss: {best_loss:.4f} | Training F1: {best_f1:.4f}")

    model.load_state_dict(best_model)
    stats = {'train': {'loss': t_loss, 'prec': t_prec, 'rec': t_rec, 'f1': t_f1},
             'val': {'loss': v_loss, 'prec': v_prec, 'rec': v_rec, 'f1': v_f1},
             'time': elapsedtime}
    return model, stats
    
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

