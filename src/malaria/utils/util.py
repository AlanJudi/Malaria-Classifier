""" This module contains utility functions for the malaria detection project. """


import argparse
import logging
import torch

from pathlib import Path
from malaria.models.Malaria import MalariaClassifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_model(checkpoint_path, device: str) -> MalariaClassifier:
    """Load the best MalariaClassifier checkpoint saved by train.py.
    Args:
        checkpoint_path (str | Path): Path to the checkpoint file (e.g. weights/best.pt)
        device (str): Device to map the weights onto ('cuda' or 'cpu')
    Returns:
        model (MalariaClassifier): Model with trained weights, in eval mode
        epoch (int): Training epoch
        val_loss (float): Validation loss
        val_acc (float): Validation accuracy
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model = MalariaClassifier().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    for val in ["epoch", "val_loss", "val_acc"]:
        if checkpoint.get(val, None) is None:
            raise ValueError(f"{val} not found in checkpoint")

    logger.info(
        "Loaded %s (epoch %s, val_loss %.4f, val_acc %.4f)",
        checkpoint_path,
        checkpoint.get("epoch"),
        checkpoint.get("val_loss"),
        checkpoint.get("val_acc"),
    )


    return model, checkpoint.get("epoch"), checkpoint.get("val_loss"), checkpoint.get("val_acc")

def parse_args():
    """Parse command-line arguments.
    Returns:
        argparse.Namespace: Parsed command-line arguments.  
    """
    parser = argparse.ArgumentParser(description="Malaria Detection Training Script")
    parser.add_argument(
        "--data",
        type=str,
        default="/home/alanj/Projects/classifier/src/config/data.yaml",
        help="Path to the dataset",
    )
    parser.add_argument(
        "--hyp",
        type=str,
        default="/home/alanj/Projects/classifier/src/config/hyp.yaml",
        help="Path to the hyperparameters YAML file   ",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="/home/alanj/Projects/classifier/weights",
        help="Directory where the best model checkpoint (best.pt) is saved",
    )
    return parser.parse_args()