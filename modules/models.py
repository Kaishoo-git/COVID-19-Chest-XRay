import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torchvision import models

class LinearNet(nn.Module):

    def __init__(self):
        super(LinearNet, self).__init__()
        self.fc1 = nn.Linear(224 * 224, 4096)
        self.fc2 = nn.Linear(4096, 1024)
        self.fc3 = nn.Linear(1024, 1)
    
    def forward(self, x):
        x = x.view(-1, 224 * 224)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class ConvNet(nn.Module):

    def __init__(self):
        super(ConvNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 4, 5),
            nn.MaxPool2d(4, 4),
            nn.Conv2d(4, 16, 5)
        )
        
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        
        self.fc1 = nn.Linear(16, 128) 
        self.fc2 = nn.Linear(128, 1)

        self.gradients = None

    def forward(self, x):
        x = self.features(x)
        if x.requires_grad == True:
            h = x.register_hook(self.activations_hook)
        
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)  
        
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def activations_hook(self, grad):
        self.gradients = grad

    def get_activation_gradients(self):
        if self.gradients is None:
            raise ValueError("Gradients were not captured by the hook. Check hook setup.")
        else:
            return self.gradients
    
    def get_activations(self, x):
        return self.features(x)

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
        self.classifier = nn.Linear(num_features, 1, bias = True)
        self.gradients = None

    def forward(self, x):
        x = self.features(x)
        if x.requires_grad:
            h = x.register_hook(self.activations_hook)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
    def activations_hook(self, grad):
        self.gradients = grad

    def get_activation_gradients(self):
        if self.gradients is None:
            raise ValueError("Gradients were not captured by the hook. Check hook setup.")
        else:
            return self.gradients
    
    def get_activations(self, x):
        return self.features(x)
    
class MyDenseNet(nn.Module):
    def __init__(self):
        super(MyDenseNet, self).__init__()
        self.densenet = models.densenet121(weights = 'DEFAULT')

        with torch.no_grad():
            self.densenet.features.conv0.weight = nn.Parameter(self.densenet.features.conv0.weight.mean(dim=1, keepdim = True))
        for params in self.densenet.parameters():
            params.requires_grad = False
        n_features = self.densenet.classifier.in_features
        
        self.features = self.densenet.features
        
        self.adpool = nn.AdaptiveAvgPool2d((1,1))
        self.classifier = nn.Linear(n_features, 1, bias = True)
        self.gradients = None

    def forward(self, x):
        x = self.features(x)
        if x.requires_grad:
            h = x.register_hook(self.activations_hook)
        x = self.adpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
    def activations_hook(self, grad):
        self.gradients = grad

    def get_activation_gradients(self):
        if self.gradients is None:
            raise ValueError("Gradients were not captured by the hook. Check hook setup.")
        else:
            return self.gradients
    
    def get_activations(self, x):
        return self.features(x)