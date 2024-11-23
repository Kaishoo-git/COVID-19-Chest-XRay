import numpy as np
import random
import cv2
from scipy.ndimage import rotate, shift
from sklearn.model_selection import train_test_split

def generate_new_image(img):

    if random.random() > 0.5:
        img = np.fliplr(img)

    angle = random.uniform(-10, 10)
    img = rotate(img, angle, reshape=False)

    max_shift_x = int(0.1 * img.shape[1])
    max_shift_y = int(0.1 * img.shape[0])
    shift_x = random.randint(-max_shift_x, max_shift_x)
    shift_y = random.randint(-max_shift_y, max_shift_y)
    img = shift(img, shift=(shift_y, shift_x), mode='reflect')
    return img

def get_data(neg_idx, pos_idx, data, random_state, resample = False):
    res, n_pos, n_neg = [], 0, 0
    random.seed(random_state)
    for i in neg_idx:
        img, lab = data[i]['img'][0], data[i]['lab'][3]
        img = cv2.resize(img, (224, 224))
        res.append({'img': img, 'lab': lab})
        n_neg += 1
    for i in pos_idx:
        img, lab = data[i]['img'][0], data[i]['lab'][3]
        img = cv2.resize(img, (224, 224))
        res.append({'img': img, 'lab': lab})
        n_pos += 1
    if resample:
        while n_neg < n_pos:
            i = random.choice(neg_idx)
            img, lab = data[i]['img'][0], data[i]['lab'][3]
            img = cv2.resize(img, (224, 224))
            res.append({'img': generate_new_image(img), 'lab': lab})
            n_neg += 1
    return res

def stratified_split(dataset, resample = False, random_state = 44):
    neg = [i for i, x in enumerate(dataset) if x['lab'][3] == 0]
    pos = [i for i, x in enumerate(dataset) if x['lab'][3] == 1]

    train_neg, temp_neg = train_test_split(neg, train_size = 0.7, random_state = random_state)
    val_neg, test_neg = train_test_split(temp_neg, train_size = 0.5, random_state = random_state)
    train_pos, temp_pos = train_test_split(pos, train_size = 0.7, random_state = random_state)
    val_pos, test_pos = train_test_split(temp_pos, train_size = 0.5, random_state = random_state)

    train = get_data(train_neg, train_pos, dataset, random_state, resample)
    val = get_data(val_neg, val_pos, dataset, random_state)
    test = get_data(test_neg, test_pos, dataset, random_state)

    return train, val, test

