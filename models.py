import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler

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

class ConvNetGlobPooling(nn.Module):

    def __init__(self):
        super(ConvNetGlobPooling, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 5)
        self.pool = nn.MaxPool2d(4, 4)
        self.conv2 = nn.Conv2d(8, 16, 5)
        
        # Global Average Pooling (replaces flattening)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        
        self.fc1 = nn.Linear(16, 128)  # Adjusted to match the output from global pooling
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        
        # Apply global average pooling (or change to max pooling if needed)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)  # Flatten for fully connected layer
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

