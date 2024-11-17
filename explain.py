import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import time
from sklearn.decomposition import PCA
from matplotlib.colors import LinearSegmentedColormap

custom_cmap = LinearSegmentedColormap.from_list("black_blue", ["black", "blue"])

def eigen_smooth(heatmap, n_components=1):
    flat_heatmap = heatmap.reshape(-1, 1)
    
    pca = PCA(n_components=n_components)
    smooth_heatmap = pca.inverse_transform(pca.fit_transform(flat_heatmap))
    
    smooth_heatmap = smooth_heatmap.reshape(heatmap.shape)
    
    return smooth_heatmap
    
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

def plot_loss(epochs, train_loss, val_loss, title):
    plt.figure(figsize=(8, 6))
    plt.plot([i+1 for i in range(epochs)], train_loss, label='training', linestyle='-', color='blue', marker='o')
    plt.plot([i+1 for i in range(epochs)], val_loss, label='validation', linestyle='--', color='red', marker='s')
    
    plt.xticks(ticks = range(1, epochs+1, 1))
    upper = min(1.1, max( max(train_loss), max(val_loss)))
    spacing = (upper - 0) / 10
    plt.ylim(0, upper)
    plt.yticks(ticks = np.arange(0, upper+spacing, spacing))

    plt.xlabel('Epochs')
    plt.ylabel('BCE loss')
    plt.title(f'Loss (BCE) with each epoch ({title})')
    plt.legend()
    plt.show()

def plot_metric(epochs, train_metric, val_metric, title):
    plt.figure(figsize=(8, 6))
    plt.plot([i+1 for i in range(epochs)], train_metric, label='training', linestyle='-', color='blue', marker='o')
    plt.plot([i+1 for i in range(epochs)], val_metric, label='validation', linestyle='--', color='red', marker='s')
    
    plt.xticks(ticks = range(1, epochs+1, 1))

    plt.yticks(ticks = np.arange(0, 1.1, 0.1))
    plt.ylim(0, 1.1)

    plt.xlabel('Epochs')
    plt.ylabel('Metric value')
    plt.title(title)
    plt.legend()
    plt.show()


def show_heatmap_overlay(img_numpy, heatmap):
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
    
def vis_comparison(model, sample_img, sample_input, title):
    heatmap_gc = get_gradcam(model, sample_input)
    heatmap_gcpp = get_gradcam_pp(model, sample_input)
    heatmap_gc = cv2.resize(heatmap_gc, (sample_img.shape[1], sample_img.shape[0]), interpolation=cv2.INTER_LINEAR)
    heatmap_gcpp = cv2.resize(heatmap_gcpp, (sample_img.shape[1], sample_img.shape[0]), interpolation=cv2.INTER_LINEAR)
    heatmap_gc = np.power(heatmap_gc, 2)
    heatmap_gcpp = np.power(heatmap_gcpp, 2)
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    fig.suptitle(title, fontsize=16, fontweight='bold', ha='center')

    axes[0].imshow(sample_img, cmap='gray')
    axes[0].imshow(heatmap_gc, cmap = custom_cmap, alpha = 0.4)
    axes[0].set_title("Grad-CAM")
    axes[0].axis('off')

    axes[1].imshow(sample_img, cmap='gray')
    axes[1].imshow(heatmap_gcpp, cmap = custom_cmap, alpha = 0.4)
    axes[1].set_title("Grad-CAM++")
    axes[1].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

def create_table(headers, models):
    markdown_table = "| " + " | ".join(headers) + " |\n"
    markdown_table += "| " + " | ".join([":-" for _ in headers]) + " |\n"

    for model_metrics in models:
        markdown_table += "| " + " | ".join(map(lambda x: str(round(x, 4)) if isinstance(x, (int, float)) else str(x), model_metrics)) + " |\n"

    print(markdown_table)

def save_table_as_image(headers, data, filename="table.png"):
    fig, ax = plt.subplots(figsize=(6, len(data) * 0.5))
    ax.axis('tight') 
    ax.axis('off')   
    
    table = plt.table(
        cellText=data,
        colLabels=headers,
        cellLoc='center',
        loc='center'
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.auto_set_column_width(col=list(range(len(headers))))
    
    plt.savefig(filename, bbox_inches='tight')
    plt.close(fig)