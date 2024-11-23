import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torchvision.transforms as transforms
import pandas as pd
from sklearn.decomposition import PCA
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image

def plot_loss_and_metric(epochs, train_loss, val_loss, train_metric, test_metric, model_name, save_path = None):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6)) 
    
    axes[0].plot([i+1 for i in range(epochs)], train_loss, label='training', linestyle='-', color='blue', marker='o')
    axes[0].plot([i+1 for i in range(epochs)], val_loss, label='validation', linestyle='--', color='red', marker='s')
    axes[0].set_xticks(range(1, epochs+1, 1))
    upper = min(1.1, max(max(train_loss), max(val_loss)))
    spacing = (upper - 0) / 10
    axes[0].set_ylim(0, upper)
    axes[0].set_yticks(np.arange(0, upper + spacing, spacing))
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('BCE Loss')
    axes[0].set_title(f'{model_name} loss')
    axes[0].legend()

    axes[1].plot([i+1 for i in range(epochs)], train_metric, label='training', linestyle='-', color='blue', marker='o')
    axes[1].plot([i+1 for i in range(epochs)], test_metric, label='test', linestyle='--', color='red', marker='s')
    axes[1].set_xticks(range(1, epochs+1, 1))
    axes[1].set_yticks(np.arange(0, 1.1, 0.1))
    axes[1].set_ylim(0, 1.1)
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Metric Value')
    axes[1].set_title(f'{model_name} metric')
    axes[1].legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)  
        print(f"Plot saved to {save_path}")

    plt.show()

def vis_comparison(model, pos_img, neg_img):
    custom_cmap = LinearSegmentedColormap.from_list("black_blue", ["black", "blue"])
    fig, axes = plt.subplots(1, 4, figsize=(20, 6)) 
    pos = 0
    for img in [pos_img, neg_img]:
        for cam in ['gradcam', 'gradcampp']:
            img_rs, img_in = resize(img), tensorize_image(img).unsqueeze(0)
            if cam == 'gradcam':
                heatmap = np.power(cv2.resize(get_gradcam(model, img_in), (img_rs.shape[1], img_rs.shape[0]), interpolation=cv2.INTER_LINEAR), 2)
            else: 
                heatmap = np.power(cv2.resize(get_gradcam_pp(model, img_in), (img_rs.shape[1], img_rs.shape[0]), interpolation=cv2.INTER_LINEAR), 2)
            axes[pos].imshow(img_rs, cmap = 'gray')
            axes[pos].imshow(heatmap, cmap = custom_cmap, alpha = 0.4)
            axes[pos].axis('off')
            pos += 1

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

def create_table(headers, data):
    df = pd.DataFrame(data, columns=headers)
    print(df)

def eigen_smooth(heatmap, n_components=1):
    flat_heatmap = heatmap.reshape(-1, 1)
    
    pca = PCA(n_components=n_components)
    smooth_heatmap = pca.inverse_transform(pca.fit_transform(flat_heatmap))
    
    smooth_heatmap = smooth_heatmap.reshape(heatmap.shape)
    
    return smooth_heatmap
    
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

def resize(img, size = (224, 224)):
    return cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)

def get_gradcam(model, inputs):
    model.eval()
    outputs = model(inputs)
    activations = model.get_activations(inputs)
    outputs.backward(retain_graph=True)
    gradients = model.get_activation_gradients()
    pooled_gradients = torch.mean(gradients, dim = [0, 2, 3])

    for i in range(activations.shape[1]):
        activations[:, i, :, :] *= pooled_gradients[i]

    heatmap = torch.mean(activations, dim = 1).squeeze()
    heatmap = torch.relu(heatmap)    
    heatmap /= torch.max(heatmap + 1e-8)
    heatmap = heatmap.detach()
    heatmap_np = heatmap.numpy()
    smooth_heatmap = eigen_smooth(heatmap_np)
    return smooth_heatmap

def get_gradcam_pp(model, inputs):
    model.eval()
    outputs = model(inputs)
    activations = model.get_activations(inputs)
    outputs.backward()

    first_order_gradients = model.get_activation_gradients()
    gradients_power_2 = first_order_gradients**2
    gradients_power_3 = first_order_gradients**3

    sum_activations = torch.sum(activations, axis=(2, 3))
    alpha = gradients_power_2 / (2 * gradients_power_2 + sum_activations[:, :, None, None] * gradients_power_3 + 1e-8)
    alpha = torch.relu(alpha)

    weights = torch.sum(alpha * torch.relu(first_order_gradients), dim=(2, 3), keepdim=True)
    weighted_activations = activations * weights[:, :, None, None]

    weighted_activations = activations * weights
    heatmap = torch.sum(weighted_activations, dim=1).squeeze()

    heatmap = torch.relu(heatmap)
    heatmap /= torch.max(heatmap + 1e-8)
    heatmap = heatmap.detach()
    heatmap_np = heatmap.numpy()
    smooth_heatmap = eigen_smooth(heatmap_np)
    return smooth_heatmap

def show_heatmap_overlay(img_numpy, heatmap):
    custom_cmap = LinearSegmentedColormap.from_list("black_blue", ["black", "blue"])
    heatmap_resized = cv2.resize(heatmap, (img_numpy.shape[1], img_numpy.shape[0]), interpolation=cv2.INTER_LINEAR)
    heatmap_conc = np.power(heatmap_resized, 2)
    plt.figure(figsize = (img_numpy.shape[1] / 100, img_numpy.shape[0] / 100), dpi = 100)

    plt.imshow(img_numpy, cmap = 'gray')
    plt.imshow(heatmap_conc, cmap= custom_cmap, alpha=0.3)

    plt.axis('off')
    plt.show()

def visualize_heatmaps(model, sample_img, sample_input):
    heatmap_gc = get_gradcam(model, sample_input)
    heatmap_gcpp = get_gradcam_pp(model, sample_input)
    show_heatmap_overlay(sample_img, heatmap_gc)
    show_heatmap_overlay(sample_img, heatmap_gcpp)

