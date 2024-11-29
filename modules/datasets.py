import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image

class Covid19DataSet(torch.utils.data.Dataset):

    def __init__(self, X, y, transform = 'vanilla'):
        data_transforms = {
            'augment': transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.3),
                transforms.RandomRotation(degrees=15),
                transforms.RandomResizedCrop(size=224, scale=(0.8, 1.0)),
                transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0)),
                transforms.ToTensor(),
            ]),
            'vanilla': transforms.Compose([
            transforms.ToTensor(),
            ]),
        }
        
        self.features = X 
        self.labels = y
        self.transform = data_transforms[transform]

    def __getitem__(self, index):

        img = self.features[index]
        img = Image.fromarray(normalize(img))

        if self.transform:
            img_tensor = self.transform(img)

        label = self.labels[index]
        label_tensor = torch.tensor([label], dtype = torch.float)
        
        return img_tensor, label_tensor

    def __len__(self):
        return len(self.features)

def normalize(img):
    return np.divide(img + 1024, 2048)
