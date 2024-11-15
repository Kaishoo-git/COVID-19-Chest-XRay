import torchxrayvision as xrv
import torch
import torchvision.transforms as transforms
import numpy as np
import cv2
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from PIL import Image

def get_images_max_min(indices, data):
    high = low = 0.0
    for i in indices:
        img = data[i]['img'][0]
        ch = np.max(img)
        cl = np.min(img)    
    high = max(high, ch)
    low = min(low, cl)
    return high, low

class Covid19DataSet(Dataset):

    def __init__(self, type, transform = 'vanilla', random_state = 44):
        d = xrv.datasets.COVID19_Dataset(imgpath = "data/images/", csvpath = "data/csv/metadata.csv")
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
            'vanilla': transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            ]),
            'augment': transforms.Compose([
            transforms.Resize((224,224)),   
            transforms.RandomHorizontalFlip(p = 0.3),
            transforms.ToTensor(), 
            ]),
        }

        self.data = d
        self.high = high
        self.low = low
        self.indices = idx
        if transform not in data_transforms:
            raise ValueError(f"Unknown transform type: {transform}. Choose from {list(data_transforms.keys())}")
        self.transform = data_transforms[transform]

    def __getitem__(self, index):

        img = self.data[index]['img'][0]
        img = img - self.low
        img = np.divide(img, self.high-self.low)
        img = Image.fromarray(img)

        label = self.data[index]['lab'][3]
        label_tensor = torch.tensor([label], dtype = torch.float)

        if self.transform:
            img_tensor = self.transform(img)

        return img_tensor, label_tensor
    

    def __len__(self):
        return len(self.indices)
    
def preprocess_image(img):
    img = img + 1024
    img = np.divide(img, 2048)
    pil = Image.fromarray(img)
    ld = transforms.ToTensor()
    tensor = ld(pil)
    return tensor