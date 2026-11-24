""" Training script for malaria detection model. """
import glob
import logging
import argparse
import yaml
import torch

import numpy as np
from torchvision import transforms as T
from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split
from malaria.data.datasets import MalariaImages


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

logger = logging.getLogger(__name__)


def get_data(device: str, hyp: dict, data_cfg: dict):
    """" Function to get the training and validation data Loaders
    Args:
        device (str): Device to use for training (e.g., 'cuda' or 'cpu')
        hyp (dict): Hyperparameters loaded from YAML file
        data_cfg (dict): Data configuration loaded from YAML file
    Returns:
        trn_dl (DataLoader): Training DataLoader
        val_dl (DataLoader): Validation DataLoader
    """

    trn_tfms = T.Compose([
        T.ToPILImage(),
        T.Resize(128),
        T.CenterCrop(128),
        T.ColorJitter(brightness=(0.95, 1.05), contrast=(0.95,1.05), saturation=(0.95, 1.05), hue=(0.05)),
        T.RandomAffine(degrees=5, translate=(0.01,0.1)),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    val_tfms = T.Compose([
        T.ToPILImage(),
        T.Resize(128),
        T.CenterCrop(128),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    all_files = glob.glob(data_cfg["images"])
    np.random.seed(10)
    np.random.shuffle(all_files)

    trn_files, val_files = train_test_split(all_files, test_size=0.2, random_state=42)

    trn_ds = MalariaImages(trn_files, transform=trn_tfms)
    val_ds = MalariaImages(val_files, transform=val_tfms)
    trn_dl = DataLoader(trn_ds, batch_size=hyp["batch_size"], shuffle=True, num_workers=4, collate_fn=trn_ds.collate_fn)
    val_dl = DataLoader(val_ds, batch_size=hyp["batch_size"], shuffle=False, num_workers=4, collate_fn=val_ds.collate_fn)

    return trn_dl, val_dl
    

def train(args: argparse.Namespace):
    """ Training Function for malaria detection model. """

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Starting training process using device: %s", device)

    with open(args.hyp, "r", encoding="utf-8") as f:
        hyp = yaml.safe_load(f)

    with open(args.data, "r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)


    train_dl, val_dl = get_data(device=device, hyp=hyp, data_cfg=data_cfg)

    




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Malaria Detection Training Script")
    parser.add_argument("--data", type=str, default="/home/alanj/Projects/classifier/cell_images/*/*.png", help="Path to the dataset")
    parser.add_argument("--hyp", type=str, default="/home/alanj/Projects/classifier/src/config/hyp.yaml", help="Path to the hyperparameters YAML file   ")
    opt = parser.parse_args()

    train(args=opt)