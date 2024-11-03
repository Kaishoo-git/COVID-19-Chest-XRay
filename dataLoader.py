import torchxrayvision as xrv
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import cv2
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

def get_images_max_min(indices, data):
    high = low = 0.0
    for i in indices:
        img = data[i]['img'][0]
        ch = np.max(img)
        cl = np.min(img)
    high = max(high, ch)
    low = max(low, cl)
    return high, low

def process_img(img, high, low, target_width, target_height):
    img = (img - low) / (high - low)
    img = cv2.resize(img, (target_width, target_height))
    return img

def covid_19_data_process(image_path, csv_path, train, random_state = 44, train_size = 0.6):

    d = xrv.datasets.COVID19_Dataset(imgpath = image_path, csvpath = csv_path)

    indices = [i for i in range(len(d))]
    train_idx, test_idx = train_test_split(indices, train_size = train_size, random_state = random_state)
    target_width, target_height = 224, 224

    if train:
        tr_h, tr_l = get_images_max_min(train_idx, d)
        X_train, y_train = [], []

        for i in train_idx:

            img = d[i]['img'][0]
            X_train.append(process_img(img, tr_h, tr_l, target_width, target_height))
            y_train.append(d[i]['lab'][3])

        X_train = np.array(X_train).reshape(-1, 1, target_width, target_height)
        y_train = np.array(y_train).reshape(-1, 1)

        print(f"Shape of X_train is {X_train.shape}. Shape of ytrain is {y_train.shape}")
        return X_train, y_train
    
    else:
        ts_h, ts_l = get_images_max_min(test_idx, d)
        X_test, y_test = [], []

        for i in test_idx:

            img = d[i]['img'][0]
            X_test.append(process_img(img, ts_h, ts_l, target_width, target_height))
            y_test.append(d[i]['lab'][3])      

        X_test = np.array(X_test).reshape(-1, 1, target_width, target_height)
        y_test = np.array(y_test).reshape(-1, 1)
        
        print(f"Shape of X_test is {X_test.shape}. Shape of ytest is {y_test.shape}")
        return X_test, y_test

class Covid19DataSet(Dataset):

    def __init__(self, train, transform = None):
        X, y = covid_19_data_process("data/images/", "data/csv/metadata.csv", train)

        self.x = X
        self.y = y
        self.size = X.shape[0]

        self.transform = transform

    def __getitem__(self, index):

        sample = self.x[index], self.y[index]

        if self.transform:
            sample = self.transform(sample)

        return sample
    

    def __len__(self):
        return self.size
    
class ToTensor:
    def __call__(self, sample):
        inputs, labels = sample
        return torch.from_numpy(inputs), torch.from_numpy(labels)
    
transform_random = transforms.Compose([
    transforms.RandomHorizontalFlip(p = 0.3),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
])