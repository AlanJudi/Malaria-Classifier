""" Grad-CAM implementation for malaria classifier """

import logging
import cv2
import numpy as np

import torch.nn as nn
import torch

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)


SZ = 128
logger = logging.getLogger(__name__)


def upsample_heatmap(map, img):
    """ Upsample the heatmap to match the size of the input image and apply a color map.
    Args:
        map (np.ndarray): The heatmap to be upsampled.
        img (np.ndarray): The original input image.
    Returns:
        np.ndarray: The upsampled heatmap with color mapping applied."""

    m, M = map.min(), map.max()

    map = 255 * ((map -m) / (M-m))
    map = np.unit8(map)
    map = cv2.resize(map, (SZ, SZ))
    map = cv2.applyColorMap(255-map, cv2.COLORMAP_JET)
    map = np.uint8(map)
    map = np.unit8(map*0.7 + img*0.3)

    return map 


def get_im2fmap(model):
    """Extract the intermediate feature map from the model.
    Args:
        model (nn.Module): The model from which to extract the feature map.
    Returns:
        nn.Sequential: A sequential model containing the layers up to the feature map.
    """
    return nn.Sequential(*(list(model.model[:5].children()) + list(model.model[5][:2].children()))) ## Split the model and expose last convolution before the RelU



def img2grad_cam(x, model):
    """Compute Grad-CAM for a given input image and model."""
    model.eval()
    logits = model(x)
    heatmaps = []
    im2fmap = get_im2fmap(model)
    activations = im2fmap(x)
    logger.info("Activation shape: %s", activations.shape)

    pred = logits.max(-1)[1]
    model.zero_grad()
    logits[0,pred].backward(retain_graph=True)
    pooled_grads= model.model[-6][1].weight.grad.data.mean((1,2,3))

    for i in range(activations.shape[1]):
        activations[:,i,:,:] *= pooled_grads[i]

    heatmap = torch.mean(activations, dim=1)[0].cpu().detach()

    return heatmap, 'Uninfected' if pred.item() else 'Parasitized'

