import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image

class Covid19DataSet(torch.utils.data.Dataset):

    def __init__(self, d, transform = 'vanilla'):
        data_transforms = {
            'augment': transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.ToTensor(),
            ]),
            'vanilla': transforms.Compose([
            transforms.ToTensor(),
            ]),
        }
        
        high, low = get_images_max_min(d)
        self.data = d
        self.high = high
        self.low = low
        self.transform = data_transforms[transform]

    def __getitem__(self, index):

        img = self.data[index]['img']
        img = Image.fromarray(normalize(img, self.high, self.low))

        if self.transform:
            img_tensor = self.transform(img)

        label = self.data[index]['lab']
        label_tensor = torch.tensor([label], dtype = torch.float)
        
        return img_tensor, label_tensor

    def __len__(self):
        return len(self.data)

def get_images_max_min(data):
    high = low = 0.0
    for i in range(len(data)):
        img = data[i]['img']
        ch = np.max(img)
        cl = np.min(img)    
    high = max(high, ch)
    low = min(low, cl)
    return high, low    

def normalize(img, high, low):
    return np.divide(img - low, high - low)
