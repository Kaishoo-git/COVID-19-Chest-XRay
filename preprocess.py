import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.ndimage import rotate, shift

def explore_data(data, processed):
    n = len(data)
    p, f = 0, 0
    for i in range(n):
        if processed:
            lab = data[i]['lab']
        else:
            lab = data[i]['lab'][3]

        if lab == 1:
            p += 1
        else:
            f += 1
    print(f"Data Size: {n} | Postives: {p} | Negatives: {f}") 

def find_positive(dataset, skip = 1):
    res = 0
    for i in range(len(dataset)):
        lab = dataset[i]['lab'][3]
        if lab == 1:
            skip -= 1
            if skip == 0:
                return i
    return 'not found'

def find_negative(dataset, skip = 1):
    res = 0
    for i in range(len(dataset)):
        lab = dataset[i]['lab'][3]
        if lab == 0:
            skip -= 1
            if skip == 0:
                return i
    return 'not found'

def show_image(d, idx, processed = False):
    if processed:
        img = d[idx]['img']
    else:
        img = d[idx]['img'][0]
    plt.imshow(img, cmap = 'gray') 
    plt.axis('off')
    plt.show()

def preprocess(vanilla_dataset, resample = False, minority_class = 0):
    n = len(vanilla_dataset)
    resampled_d = []
    for i in range(n):
        img, label = vanilla_dataset[i]['img'][0], vanilla_dataset[i]['lab'][3]
        entry = {
            'img': img,
            'lab': label
        }
        resampled_d.append(entry)

        if label == minority_class and resample:
            new_img = generate_new_image(img)
            new_entry = {
                'img': new_img,
                'lab': label
                }
            resampled_d.append(new_entry)

    return resampled_d

def generate_new_image(img):
    
    # Random Horizontal Flip
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

