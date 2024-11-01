import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ConvNet(nn.Module):

    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 5)
        self.pool = nn.MaxPool2d(4, 4)
        self.conv2 = nn.Conv2d(8, 16, 5)

        self.fc1 = nn.Linear(16 * 12 * 12, 576)
        self.fc2 = nn.Linear(576, 128)
        self.fc3 = nn.Linear(128, 1)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 12 * 12)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        return x

def train_model(model, train_loader, epochs, learning_rate = 0.001, criterion = nn.BCEWithLogitsLoss()):
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
    
    print("Training model...")
    model.train()

    for epoch in range(epochs):
        epoch_loss, count = 0, 0
        for i, (images, labels) in enumerate(train_loader):
            count += 1
            # forward pass
            y_preds = model(images)
            loss = criterion(y_preds, labels)
            total_loss += loss.item()

            # backward prop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        epoch_loss = total_loss / count 

        print(f"Epoch [{epoch+1}/{epochs}], Epoch Loss: {epoch_loss:.4f}")
        
    print("Done!")
    

def evaluate_model(model, test_loader):
    with torch.no_grad():
        tp = tn = fp = fn = 0
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
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    print(f"Precision: {precision} . Recall: {recall}")
