import torch
import torchvision.transforms as transforms
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from PIL import Image

class Covid19DataSet(Dataset):

    def __init__(self, type, d, transform = 'vanilla', random_state = 44):
        indices = [i for i in range(len(d))]
        train_idx, temp_idx = train_test_split(indices, train_size = 0.6, random_state = random_state)
        val_idx, test_idx = train_test_split(temp_idx, train_size = 0.5, random_state = random_state)
        
        match type:
            case 'train':
                idx = train_idx
            case 'val':
                idx = val_idx
            case 'test':
                idx = test_idx
        high, low = get_images_max_min(idx, d)

        data_transforms = {
            'augment': transforms.Compose([
            transforms.Resize((224,224)),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.ToTensor(),
            ]),
            'vanilla': transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            ]),
        }
        
        self.data = d
        self.high = high
        self.low = low
        self.indices = idx
        self.transform = data_transforms[transform]

    def __getitem__(self, index):

        img = self.data[self.indices[index]]['img']
        img = img - self.low
        img = np.divide(img, self.high-self.low)
        img = Image.fromarray(img)

        if self.transform:
            img_tensor = self.transform(img)

        label = self.data[self.indices[index]]['lab']
        label_tensor = torch.tensor([label], dtype = torch.float)
        
        return img_tensor, label_tensor

    def __len__(self):
        return len(self.indices)

def get_images_max_min(indices, data):
    high = low = 0.0
    for i in indices:
        img = data[i]['img']
        ch = np.max(img)
        cl = np.min(img)    
    high = max(high, ch)
    low = min(low, cl)
    return high, low    

def tensorize_image(img):
    img = img + 1024
    img = np.divide(img, 2048)
    pil = Image.fromarray(img)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    tensor = transform(pil)
    return tensor