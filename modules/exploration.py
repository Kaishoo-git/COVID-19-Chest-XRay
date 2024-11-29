import matplotlib.pyplot as plt

def explore_data(data):
    p, f = 0, 0
    for i in range(len(data)):
        lab = data['labels'][i]
        if lab == 1:
            p += 1
        else:
            f += 1
    print(f"Data Size: {len(data)} | Postives: {p} | Negatives: {f}") 

def find_positive(dataset):
    for i in range(len(dataset)):
        lab = dataset['lab'][i]
        if lab == 1:
            return i
    return 'not found'

def find_negative(dataset):
    for i in range(len(dataset)):
        lab = dataset['lab'][i]
        if lab == 0:
            return i
    return 'not found'

def show_image(data, idx):
    img = data['features'][idx]
    plt.imshow(img, cmap = 'gray') 
    plt.axis('off')
    plt.show()