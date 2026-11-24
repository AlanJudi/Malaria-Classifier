"""Malaria detection model."""



import torch.nn as nn



def conv_block(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    """Convolutional block with Conv2d, BatchNorm2d, and ReLU."""
    return nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(out_channels),
        nn.MaxPool2d(kernel_size=2)
    )



class MalariaClassifier(nn.Module):
    """ CNN Model for Malaria Detection"""

    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            conv_block(3,64),
            conv_block(64,64),
            conv_block(64,128),
            conv_block(128,256),
            conv_block(256, 512),
            conv_block(512, 64),
            nn.Flatten(),
            nn.Linear(256, 256),
            nn.Dropout(p=0.2),
            nn.ReLU(inplace=True),
            nn.Linear(256, 2)
        )

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        """Forward pass of the model."""
        return self.model(x)

    def compute_metrics(self, preds, targets):
        """Compute loss and accuracy metrics."""
        loss = self.loss_fn(preds, targets)
        acc = (preds.argmax(dim=1) == targets).float().mean()
        return loss, acc