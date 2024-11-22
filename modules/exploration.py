import matplotlib.pyplot as plt

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