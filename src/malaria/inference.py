""" Grad-CAM implementation for malaria classifier """

import logging
from pathlib import Path

import torch
import yaml

from malaria.train import get_data
from torch_snippets import resize, subplots
from malaria.cam.grad_cam import img2grad_cam, upsample_heatmap
from malaria.utils.util import parse_args
from malaria.utils.util import load_model

logger = logging.getLogger(__name__)






def infer():
    """Inference function for generating Grad-CAM heatmaps for malaria detection."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args = parse_args()

    model = load_model(Path(args.save_dir) / "best.pt", device)

    with open(args.hyp, "r", encoding="utf-8") as f:
        hyp = yaml.safe_load(f)

    with open(args.data, "r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)

    val_dl, _train_dl = get_data(device=device, hyp=hyp, data_cfg=data_cfg)

    model, _epoch, _val_loss, _val_acc = load_model(Path(args.save_dir) / "best.pt", device=device)

    x, y, z = next(iter(val_dl))

    for i in range(20):
        image = resize(z[i], 128)
        heatmap, pred = img2grad_cam(x[i:i+1], model)

        if pred == 'Uninfected':
            continue
        heatmap = upsample_heatmap(heatmap, image, 128)
        subplots([image, heatmap], nc=2, figsize=(5,3), suptitle=pred)

if __name__ == "__main__":
    infer()