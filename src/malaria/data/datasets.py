""" Dataset Classes for Malaria Detection """

import logging
import random
import torch

from torch.utils.data import Dataset
from torch_snippets import read, fname, parent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

logger = logging.getLogger(__name__)
id2int = {"Parasitized": 1, "Uninfected": 0}


class MalariaImages(Dataset):
    """ Dataset class for Malaria Images """
    def __init__(self, files, transform=None):
        self.files = files
        self.transform = transform
        logger.info("MalariaImages Dataset initialized with %d files.", len(self.files))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fpath = self.files[idx]
        class_label = fname(parent(fpath))
        img = read(fpath)

        return img, class_label

    def choose(self):
        """ Randomly choose an image from the dataset """
        return self[random.randint(0, len(self) - 1)]

    def collate_fn(self, batch):
        """ Custom collate function for DataLoader """
        _imgs, classes = list(zip(*batch))
        if self.transform:
            imgs = [self.transform(img) for img in _imgs]
        else:
            imgs = [torch.tensor(img).permute(2,0,1).float() / 255 for img in _imgs]

        classes = torch.tensor([id2int[c] for c in classes])
        imgs = torch.stack(imgs)

        return imgs, classes, _imgs