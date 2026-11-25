"""Training script for malaria detection model."""

import glob
import logging
from pathlib import Path
import argparse
import yaml
import torch

import numpy as np
from torchvision import transforms as T
from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split
from malaria.data.datasets import MalariaImages
from malaria.models.Malaria import MalariaClassifier
from malaria.utils.util import parse_args
from torch_snippets import Report

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)

logger = logging.getLogger(__name__)


def get_data(device: str, hyp: dict, data_cfg: dict):
    """ " Function to get the training and validation data Loaders
    Args:
        device (str): Device to use for training (e.g., 'cuda' or 'cpu')
        hyp (dict): Hyperparameters loaded from YAML file
        data_cfg (dict): Data configuration loaded from YAML file
    Returns:
        trn_dl (DataLoader): Training DataLoader
        val_dl (DataLoader): Validation DataLoader
    """

    trn_tfms = T.Compose(
        [
            T.ToPILImage(),
            T.Resize(128),
            T.CenterCrop(128),
            T.ColorJitter(
                brightness=(0.95, 1.05),
                contrast=(0.95, 1.05),
                saturation=(0.95, 1.05),
                hue=(0.05),
            ),
            T.RandomAffine(degrees=5, translate=(0.01, 0.1)),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    val_tfms = T.Compose(
        [
            T.ToPILImage(),
            T.Resize(128),
            T.CenterCrop(128),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    all_files = glob.glob(data_cfg["images"])
    np.random.seed(10)
    np.random.shuffle(all_files)

    trn_files, val_files = train_test_split(all_files, test_size=0.2, random_state=42)

    trn_ds = MalariaImages(trn_files, transform=trn_tfms)
    val_ds = MalariaImages(val_files, transform=val_tfms)
    trn_dl = DataLoader(
        trn_ds,
        batch_size=hyp["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=trn_ds.collate_fn,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=hyp["batch_size"],
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=val_ds.collate_fn,
    )

    return trn_dl, val_dl


def train_batch(model, data, optimizer, criterion):
    """Train the model on a single batch of data.
    Args:
        model (nn.Module): The model to train
        data (tuple): A batch of data containing images and labels
        optimizer (torch.optim.Optimizer): The optimizer for updating model parameters
        criterion (callable): The loss function to compute the loss
    Returns:
        loss (float): The computed loss for the batch
        acc (float): The computed accuracy for the batch
    """
    model.train()
    device = next(model.parameters()).device
    ims, labels, _ = data
    ims, labels = ims.to(device, non_blocking=True), labels.to(device, non_blocking=True)
    _preds = model(ims)
    optimizer.zero_grad()
    loss, acc = criterion(_preds, labels)
    loss.backward()
    optimizer.step()
    return loss.item(), acc.item()

def validate_batch(model, data, criterion):
    """Validate the model on a single batch of data.
    Args:
        model (nn.Module): The model to validate
        data (tuple): A batch of data containing images and labels
        criterion (callable): The loss function to compute the loss
    Returns:
        loss (float): The computed loss for the batch
        acc (float): The computed accuracy for the batch
    """
    model.eval()
    device = next(model.parameters()).device
    ims, labels, _ = data
    ims, labels = ims.to(device, non_blocking=True), labels.to(device, non_blocking=True)
    with torch.no_grad():
        _preds = model(ims)
        loss, acc = criterion(_preds, labels)
    return loss.item(), acc.item()



def train(args: argparse.Namespace):
    """Training Function for malaria detection model."""

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Starting training process using device: %s", device)

    with open(args.hyp, "r", encoding="utf-8") as f:
        hyp = yaml.safe_load(f)

    with open(args.data, "r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)

    train_dl, val_dl = get_data(device=device, hyp=hyp, data_cfg=data_cfg)

    model = MalariaClassifier().to(device)
    criterion = model.compute_metrics
    optimizer = torch.optim.Adam(model.parameters(), lr=hyp["lr"])
    n_epochs = hyp["epochs"]


    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    best_path = save_dir / "best.pt"
    best_val_loss = float("inf")

    log = Report(n_epochs)
    for epoch in range(n_epochs):
        N = len(train_dl)
        for ix, batch in enumerate(train_dl):
            loss, acc = train_batch(model, batch, optimizer, criterion)
            log.record(epoch+(ix+1)/N, trn_loss=loss, trn_acc=acc, end="\r")

        N = len(val_dl)
        val_losses, val_accs = [], []
        for ix, batch in enumerate(val_dl):
            loss, acc = validate_batch(model, batch, criterion)
            val_losses.append(loss)
            val_accs.append(acc)
            log.record(epoch+(ix+1)/N, val_loss=loss, val_acc=acc, end="\r")

        log.report_avgs(epoch+1)

        epoch_val_loss = float(np.mean(val_losses))
        epoch_val_acc = float(np.mean(val_accs))
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": epoch_val_loss,
                    "val_acc": epoch_val_acc,
                    "hyp": hyp,
                },
                best_path,
            )
            logger.info(
                "Epoch %d: val_loss improved to %.4f (val_acc %.4f), saved to %s",
                epoch + 1, epoch_val_loss, epoch_val_acc, best_path,
            )

    logger.info("Training complete. Best val_loss: %.4f (%s)", best_val_loss, best_path)



def main():
    """Main function to run the training process."""
    args = parse_args()
    train(args)

if __name__ == "__main__":
    main()