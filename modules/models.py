import torch
import torchvision
import torchxrayvision

class LinearNet(torch.nn.Module):

    def __init__(self):
        super(LinearNet, self).__init__()
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(224 * 224, 4096),
            torch.nn.ReLU(),
            torch.nn.Linear(4096, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 1),
        )
    
    def forward(self, x):
        x = x.view(-1, 224 * 224)
        x = self.classifier(x)
        return x

class BaseModel(torch.nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        self.features = None
        self.avgpool = None
        self.classifier = None
        self.gradients = None

    def activations_hook(self, grad):
        """Save gradients for Grad-CAM."""
        self.gradients = grad

    def forward(self, x):
        """Forward pass."""
        x = self.features(x)
        if x.requires_grad:
            x.register_hook(self.activations_hook)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def get_activation_gradients(self):
        """Access captured gradients."""
        if self.gradients is None:
            raise ValueError("Gradients not captured. Perform backward pass first.")
        return self.gradients

    def get_activations(self, x):
        """Get activations for feature maps."""
        return self.features(x)

class ConvNet(BaseModel):

    def __init__(self):
        super(ConvNet, self).__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(1, 4, 3, stride=1, padding=1), 
            torch.nn.BatchNorm2d(4),  
            torch.nn.ReLU(),  
            torch.nn.MaxPool2d(2, 2),  
            
            torch.nn.Conv2d(4, 16, 3, stride=1, padding=1),
            torch.nn.BatchNorm2d(16), 
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),

            torch.nn.Conv2d(16, 64, 3, stride=1, padding=1),
            torch.nn.BatchNorm2d(64),  
            torch.nn.ReLU(),

            torch.nn.Conv2d(64, 256, 3, stride=1, padding=1),
            torch.nn.BatchNorm2d(256), 
            torch.nn.ReLU(),
        )

        self.avgpool = torch.nn.AdaptiveAvgPool2d(1)
        self.classifier = torch.nn.Linear(256, 1, bias=True)

class GenResNet(BaseModel):
    def __init__(self, weights='default'):
        super(GenResNet, self).__init__()
        valid_weights = {'default', 'xray'}
        if weights not in valid_weights:
            raise ValueError(f"Invalid weights: {weights}. Choose from {valid_weights}.")

        match weights:
            case 'default':
                self.resnet = torchvision.models.resnet18(weights='DEFAULT')
                with torch.no_grad():
                    self.resnet.conv1.weight = torch.nn.Parameter(self.resnet.conv1.weight.mean(dim=1, keepdim=True))

            case 'xray':
                self.resnet = torchxrayvision.models.ResNet(weights="resnet50-res512-all").model
                old_weights = self.resnet.conv1.weight.data
                new_weights = torch.nn.functional.interpolate(old_weights.permute(1, 0, 2, 3), size=(3, 3), mode='bilinear').permute(1, 0, 2, 3)
                self.resnet.conv1 = torch.nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
                self.resnet.conv1.weight = torch.nn.Parameter(new_weights)

        for name, params in self.resnet.named_parameters():
            if 'layer4' in name:
                params.requires_grad = True
            else:
                params.requires_grad = False

        self.features = torch.nn.Sequential(
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
        self.classifier = torch.nn.Linear(self.resnet.fc.in_features, 1)

class GenDenseNet(BaseModel):
    def __init__(self, weights='default'):
        super(GenDenseNet, self).__init__()
        valid_weights = {'default', 'nih', 'chexpert', 'pc'}
        if weights not in valid_weights:
            raise ValueError(f"Invalid weights: {weights}. Choose from {valid_weights}.")

        match weights:
            case 'default':
                self.densenet = torchvision.models.densenet121(weights='DEFAULT')
                with torch.no_grad():
                    self.densenet.features.conv0.weight = torch.nn.Parameter(
                        self.densenet.features.conv0.weight.mean(dim=1, keepdim=True)
                    )
            case 'nih':
                self.densenet = torchxrayvision.models.DenseNet(weights='densenet121-res224-nih')
            case 'chexpert':
                self.densenet = torchxrayvision.models.DenseNet(weights='densenet121-res224-chex')
            case 'pc':
                self.densenet = torchxrayvision.models.DenseNet(weights='densenet121-res224-rsna')

        for params in self.densenet.parameters():
            params.requires_grad = False

        self.features = self.densenet.features
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = torch.nn.Linear(self.densenet.classifier.in_features, 1)

class SparseAutoEncoder(torch.nn.Module):
    def __init__(self):
        super(SparseAutoEncoder, self).__init__()
        # Encoder
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),  
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), 
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1), 
            torch.nn.ReLU()
        )
        # Decoder
        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(256, 128, kernel_size=3, stride=1, padding=1, output_padding=1), 
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1, output_padding=1),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1, output_padding=1), 
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(32, 1, kernel_size=3, stride=1, padding=1, output_padding=1),  
            torch.nn.Sigmoid(),  # Use Sigmoid for normalized reconstruction
        )

    def forward(self, x):
        # Encode
        latent = self.encoder(x)
        # Decode
        reconstructed = self.decoder(latent)
        return latent, reconstructed


def get_model(model_name, **kwargs):
    """
    Returns an instance of a model based on the input string.

    Args:
        model_name (str): The name of the model to retrieve.
        **kwargs: Additional arguments for model initialization (e.g., weights).

    Returns:
        torch.nn.Module: The corresponding model instance.

    Raises:
        ValueError: If the model_name is invalid.
    """
    model_mapping = {
        "linear": LinearNet,
        "convnet": ConvNet,
        "resnet": GenResNet,
        "densenet": GenDenseNet,
    }

    if model_name not in model_mapping:
        raise ValueError(f"Invalid model name '{model_name}'. Choose from {list(model_mapping.keys())}.")
    if kwargs['weights']:
        return model_mapping[model_name](kwargs['weights'])
    else:
        return model_mapping[model_name]()
