""" Grad-CAM implementation for malaria classifier """

from pyexpat import model

from xml.parsers.expat import model

import torch.nn as nn




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
    activation = im2fmap(x)
