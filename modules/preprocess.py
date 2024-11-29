import numpy as np
import random
import cv2
from sklearn.model_selection import train_test_split

def get_data(neg_idx, pos_idx, data, random_state):
    res = {'features':[], 'labels':[]}
    random.seed(random_state)
    for i in neg_idx:
        img, lab = data[i]['img'][0], data[i]['lab'][3]
        img = cv2.resize(img, (224, 224))
        res['features'].append(img)
        res['labels'].append(lab)
    for i in pos_idx:
        img, lab = data[i]['img'][0], data[i]['lab'][3]
        img = cv2.resize(img, (224, 224))
        res['features'].append(img)
        res['labels'].append(lab)
    return res

def stratified_split(dataset, random_state):
    neg = [i for i, x in enumerate(dataset) if x['lab'][3] == 0]
    pos = [i for i, x in enumerate(dataset) if x['lab'][3] == 1]

    train_neg, test_neg = train_test_split(neg, train_size = 0.7, random_state = random_state)
    train_pos, test_pos = train_test_split(pos, train_size = 0.7, random_state = random_state)
    train = get_data(train_neg, train_pos, dataset, random_state)
    test = get_data(test_neg, test_pos, dataset, random_state)

    return train, test

def process_all(data):
    res = {'features':[], 'labels':[]}
    for i in range(len(data)):
        img, lab = data[i]['img'][0], data[i]['lab'][3]
        img = cv2.resize(img, (224, 224))
        res['features'].append(img)
        res['labels'].append(lab)
    return res