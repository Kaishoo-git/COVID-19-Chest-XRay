import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import time
from sklearn.decomposition import PCA
from matplotlib.colors import LinearSegmentedColormap

# Create a custom colormap from white to yellow
black_yellow_cmap = LinearSegmentedColormap.from_list("black_yellow", ["black", "yellow"])

def eigen_smooth(heatmap, n_components=1):
    # Flatten the heatmap for PCA
    flat_heatmap = heatmap.reshape(-1, 1)
    
    # Apply PCA
    pca = PCA(n_components=n_components)
    smooth_heatmap = pca.inverse_transform(pca.fit_transform(flat_heatmap))
    
    # Reshape back to the original heatmap shape
    smooth_heatmap = smooth_heatmap.reshape(heatmap.shape)
    
    return smooth_heatmap

# Note that input_tensor can be a batch of images in a tensor
def infer(model, input_tensor):

    model.eval()
    outputs = model(input_tensor)
    prob = torch.sigmoid(outputs)

    return outputs
    
def get_gradcam(model, outputs, inputs):

    time_s = time.time()
    # print("Generating heatmap")
            
    model.eval()

    # 1.Extract targeted layer
    activations = model.get_activations(inputs)
    # Note that there is no need to create hook as model takes care of that
    # 2. get the gradient of the output with respect to the parameters of the model
    outputs.backward(retain_graph=True)
    
    # 3. pull the gradients out of the model
    gradients = model.get_activation_gradients()

    # 4. pool the gradients across the channels
    pooled_gradients = torch.mean(gradients, dim = [0, 2, 3])

    # 5. weight the channels by corresponding gradients
    for i in range(activations.shape[1]):
        activations[:, i, :, :] *= pooled_gradients[i]
    #   average the channels of the activations
    heatmap = torch.mean(activations, dim = 1).squeeze()

    # 6. relu on top of the heatmap
    #   expression (2) in https://arxiv.org/pdf/1610.02391.pdf
    heatmap = torch.relu(heatmap)    

    # 7. normalize the heatmap
    heatmap /= torch.max(heatmap + 1e-8)

    # draw the heatmap
    heatmap = heatmap.detach()
    heatmap_np = heatmap.numpy()
    smooth_heatmap = eigen_smooth(heatmap_np)
    return smooth_heatmap

def get_gradcam_pp(model, outputs, inputs):
    import time
    time_s = time.time()
    # print("Generating Grad-CAM++ heatmap")
    
    model.eval()

    # 1. Extract the targeted layer's activations
    activations = model.get_activations(inputs)
    
    # 2. Compute the first-order gradients with respect to the output
    outputs.backward()  # This will populate gradients in model.gradients
    first_order_gradients = model.get_activation_gradients()

    # 3. Compute the second-order gradients
    gradients_power_2 = first_order_gradients**2

    # 4. Compute the third-order gradients
    gradients_power_3 = first_order_gradients**3

    # 5. Calculate alpha coefficients using Grad-CAM++ formula
    sum_activations = torch.sum(activations, axis=(2, 3))
    alpha = gradients_power_2 / (2 * gradients_power_2 + sum_activations[:, :, None, None] * gradients_power_3 + 1e-8)
    alpha = torch.relu(alpha)

    # 6. Weight the channels by corresponding alpha * ReLU(gradient)
    weights = torch.sum(alpha * torch.relu(first_order_gradients), dim=(2, 3), keepdim=True)
    weighted_activations = activations * weights[:, :, None, None]

    # Step 7: Combine the weighted activations to get the heatmap
    weighted_activations = activations * weights
    heatmap = torch.sum(weighted_activations, dim=1).squeeze()

    # Step 8: Apply ReLU to keep only positive influence regions
    heatmap = torch.relu(heatmap)

    # Step 9: Normalize the heatmap
    heatmap /= torch.max(heatmap + 1e-8)

    # Draw the heatmap
    heatmap = heatmap.detach()
    heatmap_np = heatmap.numpy()
    smooth_heatmap = eigen_smooth(heatmap_np)
    return smooth_heatmap

def vis_comparison(img_numpy, heatmap_gc, heatmap_gcpp, title):
    # # Display the Grad-CAM and Grad-CAM++ images side by side
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    fig.suptitle(title, fontsize=16, fontweight='bold', ha='center')

    heatmap_gc = cv2.resize(heatmap_gc, (img_numpy.shape[1], img_numpy.shape[0]), interpolation = cv2.INTER_LINEAR)
    axes[0].imshow(img_numpy, cmap='gray')
    axes[0].imshow(heatmap_gc, cmap = black_yellow_cmap, alpha = 0.3)
    axes[0].set_title("Grad-CAM")
    axes[0].axis('off')

    heatmap_gcpp = cv2.resize(heatmap_gcpp, (img_numpy.shape[1], img_numpy.shape[0]), interpolation = cv2.INTER_LINEAR)
    axes[1].imshow(img_numpy, cmap='gray')
    axes[1].imshow(heatmap_gcpp, cmap = black_yellow_cmap, alpha = 0.3)
    axes[1].set_title("Grad-CAM++")
    axes[1].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
