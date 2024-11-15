import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import numpy as np
import matplotlib.pyplot as plt
import time
import copy

def train_model(model, train_loader, validation_loader, epochs = 5, learning_rate = 0.1, grad_criterion = nn.BCEWithLogitsLoss()):
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode = 'min', factor = 0.8, patience = 1)
    best_model, best_loss, best_acc = copy.deepcopy(model.state_dict()), 9999.9, 0.0
    starttime = time.time()
    
    print("Training model")

    for epoch in range(epochs):
        t_loss, v_loss, t_acc, v_acc = [], [], [], []

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()  
        
            target_loader = train_loader if phase == 'train' else validation_loader
            n_total = running_correct = epoch_loss = 0.0
        
            for i, (images, labels) in enumerate(target_loader):
                n_total += images.size(dim = 0)

                with torch.set_grad_enabled(phase == 'train'):
                    # forward pass
                    y_preds = model(images)
                    pred_labels = (torch.sigmoid(y_preds) >= 0.5)
                    loss = grad_criterion(y_preds, labels)
                    epoch_loss += loss.item()
                    running_correct += (pred_labels == labels).sum().item()

                    if phase == 'train':
                        # backward prop
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

            if phase == 'train':
                avg_loss, avg_correct = epoch_loss / n_total, running_correct / n_total
                t_loss.append(avg_loss)
                t_acc.append(avg_correct)
                # scheduler step
                scheduler.step(epoch_loss)    

            if phase == 'val':
                avg_loss, avg_correct = epoch_loss / n_total, running_correct / n_total
                v_loss.append(avg_loss)
                v_acc.append(avg_correct)
                if v_loss[-1] < best_loss:
                    best_loss, best_acc = v_loss[-1], v_acc[-1] 
                    best_model = copy.deepcopy(model.state_dict())
            
        print(f"Epoch [{epoch + 1}/{epochs}] |Training Loss: {t_loss[-1]:.4f} | Validation Loss: {v_loss[-1]:.4f} | Training Acc: {t_acc[-1]*100:.2f}%")
    
    elapsedtime = time.time() - starttime
    mins, sec = elapsedtime//60, elapsedtime%60

    print(f"Training completed in {mins:.0f}mins {sec:.2f}s")
    print(f"Best Loss: {best_loss:.4f} | Validation Accuracy: {(best_acc * 100):.0f}%")

    model.load_state_dict(best_model)
    stats = [t_loss, t_acc, v_loss, v_acc]
    return model, stats
    
def get_metrics(model, test_loader):

    model.eval()
    time_s = time.time()
    g_lab, preds = [], []

    with torch.no_grad():
        tp = tn = fp = fn = t = f = n = 0
        for j, (images, labels) in enumerate(test_loader):

            outputs = model(images)
            probs = torch.sigmoid(outputs)
            y_preds = (torch.sigmoid(outputs) >= 0.5).float()
            
            k = y_preds.size(dim = 0)
            n += k

            for i in range(k):
                pred_val, label_val = y_preds[i].item(), labels[i].item()
                tp += (pred_val == label_val == 1)
                tn += (pred_val == label_val == 0)
                fp += (pred_val == 1 and label_val == 0)
                fn += (pred_val == 0 and label_val == 1)
                f += (pred_val == 0)
                t += (pred_val == 1)

            preds.append(probs)
            g_lab.append(labels)

        preds = torch.cat(preds, dim = 0) if len(preds) > 1 else torch.tensor(preds[0]) # These are probabilities in order to match the pr curve
        g_lab = torch.cat(g_lab, dim = 0) if len(g_lab) > 1 else torch.tensor(labels[0])

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    run_time = time.time() - time_s
    print(f"Precision: {prec:.4f} | Recall: {recall:.4f} | Positives: {t} | Negatives: {f} | Total: {n}")
    # print(f"Time taken: {run_time:.2f}s")

    return recall, prec, t, f, n
