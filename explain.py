import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import time

def get_cam(model, outputs, input):
    time_s = time.time()
    print("Generating heatmap")
    # 1. get the gradient of the output with respect to the parameters of the model
    outputs.backward()

    # 2. get the activations of the last convolutional layer
    activations = model.get_activations(input).detach()

    # 3. pull the gradients out of the model
    gradients = model.get_activations_gradient()

    # 4. pool the gradients across the channels
    pooled_gradients = torch.mean(gradients, dim = [0, 2, 3])


    # 5. weight the channels by corresponding gradients
    for i in range(activations.shape[1]):
        activations[:, i, :, :] *= pooled_gradients[i]
    #   average the channels of the activations
    heatmap = torch.mean(activations, dim = 1).squeeze()

    # 6. relu on top of the heatmap
    #   expression (2) in https://arxiv.org/pdf/1610.02391.pdf
    heatmap = np.maximum(heatmap, 0)

    # 7. normalize the heatmap
    heatmap /= torch.max(heatmap)

    # draw the heatmap
    plt.matshow(heatmap.squeeze())
    run_time = time.time() - time_s
    print(f"Heatmap generated in {run_time:.2f}s")
    return heatmap

def overlay_cam(heatmap, img_numpy):
    # Convert the heatmap to a numpy array and resize to match the input image
    heatmap = heatmap.numpy()
    heatmap = cv2.resize(heatmap, (img_numpy.shape[1], img_numpy.shape[0]))
    
    # Normalize heatmap to be in range [0, 255] and apply colormap
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Superimpose heatmap on the original image
    superimposed_img = heatmap * 0.4 + img_numpy
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)

    # Plot the superimposed image using matplotlib
    plt.imshow(superimposed_img)
    plt.axis('off')
    plt.show()

    return superimposed_img