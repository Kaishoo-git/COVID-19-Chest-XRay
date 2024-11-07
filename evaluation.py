import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt
import time
import copy

# Creates a tensorboard scalar at the same time
def train_model(model, train_loader, validation_loader, writer1, writer2, epochs = 5, learning_rate = 0.1, grad_criterion = nn.BCEWithLogitsLoss()):
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode = 'min', factor = 0.8, patience = 1)
    best_model, best_loss, best_acc = copy.deepcopy(model.state_dict()), 9999.9, 0.0
    starttime = time.time()
    
    print("Training model")

    for epoch in range(epochs):
        epoch_loss, running_correct, t_loss, v_loss, t_acc, v_acc = 0.0, 0.0, [], [], [], []

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()  
        
            target_loader = train_loader if phase == 'train' else validation_loader
            # n_steps = len(target_loader)
        
            for i, (images, labels) in enumerate(target_loader):
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
                avg_loss, avg_correct = epoch_loss / i, running_correct / i
                t_loss.append(avg_loss)
                t_acc.append(avg_correct)
                # scheduler step
                scheduler.step(epoch_loss)    

            if phase == 'val':
                avg_loss, avg_correct = epoch_loss / i, running_correct / i
                v_loss.append(avg_loss)
                v_acc.append(avg_correct)
                if avg_loss < best_loss:
                    best_loss, best_acc = avg_loss, avg_correct
                    best_model = copy.deepcopy(model.state_dict())
            
        print(f"Epoch [{epoch + 1}/{epochs}] |Training Loss: {t_loss[-1]:.4f} | Validation Loss: {v_loss[-1]:.4f}")
        writer1.add_scalar('loss', t_loss[-1], epoch+1)
        writer1.add_scalar('acc', t_acc[-1], epoch+1)
        writer2.add_scalar('loss', v_loss[-1], epoch+1)
        writer2.add_scalar('acc', v_acc[-1], epoch+1)

    writer1.close()
    writer2.close()
    
    elapsedtime = time.time() - starttime
    mins, sec = elapsedtime//60, elapsedtime%60
    print(f"Training completed in {mins:.0f}mins {sec:.2f}s")
    print(f"Best Loss: {best_loss:.4f} | Best Accuracy: {(best_acc * 100):.0f}%")

    model.load_state_dict(best_model)
    return model
    
# Creates a tensorboard AUC curve at the same time
def get_metrics(model, test_loader, writer):
    time_s = time.time()
    g_lab, preds = [], []
    with torch.no_grad():
        tp = tn = fp = fn = t = f = 0
        for j, (images, labels) in enumerate(test_loader):
            outputs = model(images)
            probs = torch.sigmoid(outputs)
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

            preds.append(probs)
            g_lab.append(labels)

        j += 1

        preds = torch.cat(preds, dim = 0)
        labels = torch.cat(labels, dim = 0)

        writer.add_pr_curve('covid19', g_lab, preds, 0)
        writer.close()

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    print(f"Precision: {prec} | Recall: {recall} | Positives: {t} | Negatives: {f} | Total: {j}")
    run_time = time.time() - time_s
    print(f"Time taken: {run_time:.2f}s")
    return recall, prec, t, f, j

# Note that input_tensor can be a batch of images in a tensor
def infer(model, input_tensor):

    outputs = model(input_tensor)
    prob = torch.sigmoid(outputs)
    labels = (prob >= 0.5).float()

    for i in range(prob.size(dim=1)):
        print(f"Predicted img {i} {'positive' if labels[i].item() == 1 else 'negative'} with probability {prob[i].item():.4f}")

    return outputs
