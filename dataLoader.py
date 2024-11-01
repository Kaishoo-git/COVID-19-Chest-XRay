import torchxrayvision as xrv
import torch
import torchvision
import numpy as np
import cv2
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

def covid_19_data_process(image_path, csv_path, train, train_size = 0.6):

    d = xrv.datasets.COVID19_Dataset(imgpath = image_path, csvpath = csv_path)
    X, y, low, high, target_width, target_height = [], [], -1024, 1024, 224, 224

    for i in range(len(d)):

        img = d[i]['img'][0]
        img  = 255 * (img - low) / (high - low)
        img = cv2.resize(img, (target_width, target_height))
        label = d[i]['lab'][3]

        X.append(img)
        y.append(label)

    X = np.array(X).reshape(-1, 1, target_width, target_height)
    y = np.array(y).reshape(-1, 1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1 - train_size, random_state = 42)
    print(f"Shape of X_train is {X_train.shape}. Shape of ytrain is {y_train.shape}")
    print(f"Shape of X_test is {X_test.shape}. Shape of ytest is {y_test.shape}")
    if train:
        return X_train, y_train
    else:
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