import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import time

# Note that input_tensor can be a batch of images in a tensor
def infer(model, input_tensor):

    model.eval()
    outputs = model(input_tensor)
    prob = torch.sigmoid(outputs)
    labels = (prob >= 0.5).float()

    for i in range(prob.size(dim=1)):
        print(f"Predicted img {i} {'positive' if labels[i].item() == 1 else 'negative'} with probability {prob[i].item():.4f}")

    return outputs
    
def get_cam(model, outputs, input):

    time_s = time.time()
    print("Generating heatmap")
    
    # Unfreeze gradients for convolution layers
    for param in model.parameters():
        param.requires_grad = True
        
    model.eval()

    # 1.Extract targeted layer
    activations = model.get_activations(input)

    # 2. Set a hook at targeted_layer
    h = activations.register_hook(model.activations_hook)

    # 3. get the gradient of the output with respect to the parameters of the model
    outputs.backward()

    # 4. pull the gradients out of the model
    gradients = model.get_activation_gradients()
    activations = activations.detach()

    # 5. pool the gradients across the channels
    pooled_gradients = torch.mean(gradients, dim = [0, 2, 3])


    # 6. weight the channels by corresponding gradients
    for i in range(activations.shape[1]):
        activations[:, i, :, :] *= pooled_gradients[i]
    #   average the channels of the activations
    heatmap = torch.mean(activations, dim = 1).squeeze()

    # 7. relu on top of the heatmap
    #   expression (2) in https://arxiv.org/pdf/1610.02391.pdf
    heatmap = np.maximum(heatmap, 0)    # Similar to calling relu

    # 8. normalize the heatmap
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
    img_numpy = np.expand_dims(img_numpy, axis = 2)
    # superimposed_img = heatmap * 0.1 + img_numpy
    # superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)

    # Plot the superimposed image using matplotlib
    plt.imshow(img_numpy, cmap = 'gray')
    plt.imshow(heatmap, alpha=0.3)
    plt.axis('off')
    plt.show()

    # return superimposed_img