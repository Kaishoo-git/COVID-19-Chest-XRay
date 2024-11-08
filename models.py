import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torchvision import models

class ConvNet(nn.Module):

    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, 5)
        self.pool = nn.MaxPool2d(4, 4)
        self.conv2 = nn.Conv2d(4, 16, 5)

        self.fc1 = nn.Linear(16 * 12 * 12, 576)
        self.fc2 = nn.Linear(576, 128)
        self.fc3 = nn.Linear(128, 1)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 12 * 12)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class ConvNetGlobPooling(nn.Module):

    def __init__(self):
        super(ConvNetGlobPooling, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, 5)
        self.pool = nn.MaxPool2d(4, 4)
        self.conv2 = nn.Conv2d(4, 16, 5)
        
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
        x = self.fc3(x)
        return x

class MyResNet18(nn.Module):
    def __init__(self):
        super(MyResNet18, self).__init__()
        self.resnet = models.resnet18(weights = 'DEFAULT')
        with torch.no_grad():
            self.resnet.conv1.weight = nn.Parameter(self.resnet.conv1.weight.mean(dim = 1, keepdim = True))
        for params in self.resnet.parameters():
            params.requires_grad = False
        num_features = self.resnet.fc.in_features
        self.features = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.relu,
            self.resnet.maxpool,
            self.resnet.layer1,
            self.resnet.layer2,
            self.resnet.layer3,
            self.resnet.layer4
        )
        self.avgpool = self.resnet.avgpool
        self.fc = nn.Linear(num_features, 1)
        self.gradients = None

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
    
    def activations_hook(self, grad):
        self.gradients = grad

    def get_activation_gradients(self):
        if self.gradients is None:
            print("Gradient is not set")
        else:
            return self.gradients
    
    def get_activations(self, x):
        return self.features(x)
    