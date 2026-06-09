"""
trainprobe + PyTorch Lightning — image classification

Trains a ResNet-18 on CIFAR-10 using a LightningModule.
LightningTrainProbeCallback captures the probe batch automatically from the
first training batch, so no probe_batch= argument is required.

The on_phase_transition callback is demonstrated by logging to the Lightning
logger when CUSUM detects a structural shift in effective rank.

Install:
    pip install trainprobe pytorch-lightning torchvision

Run:
    python examples/lightning_image_classifier.py
"""
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

import trainprobe as tp
from trainprobe.integrations import LightningTrainProbeCallback


# ── Config ────────────────────────────────────────────────────────────────────

BATCH_SIZE = 128
EPOCHS = 5
LR = 1e-3
DATA_DIR = "./data"


# ── Data ──────────────────────────────────────────────────────────────────────

_MEAN = (0.4914, 0.4822, 0.4465)
_STD  = (0.2023, 0.1994, 0.2010)

train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(_MEAN, _STD),
])
val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(_MEAN, _STD),
])

train_set = datasets.CIFAR10(DATA_DIR, train=True,  download=True, transform=train_transform)
val_set   = datasets.CIFAR10(DATA_DIR, train=False, download=True, transform=val_transform)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
val_loader   = DataLoader(val_set,   batch_size=256,        shuffle=False, num_workers=2)


# ── LightningModule ───────────────────────────────────────────────────────────

class CIFAR10Classifier(pl.LightningModule):
    def __init__(self, lr: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.backbone = models.resnet18(weights=None)
        self.backbone.fc = nn.Linear(512, 10)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.backbone(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)
        loss = self.criterion(logits, labels)
        self.log("train/loss", loss, on_step=True, on_epoch=False, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)
        loss = self.criterion(logits, labels)
        acc = (logits.argmax(1) == labels).float().mean()
        self.log("val/loss", loss, prog_bar=True)
        self.log("val/acc",  acc,  prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
        return [optimizer], [scheduler]


# ── trainprobe callback ───────────────────────────────────────────────────────
#
# LightningTrainProbeCallback wires:
#   on_train_start      → tp.attach(pl_module, ...)
#   on_train_batch_start → scope.capture_batch(batch)   [no-op after first call]
#   on_train_batch_end  → scope.step(trainer.global_step, loss=...)
#
# Because capture_batch is called every batch, the probe batch is always the
# first batch of training — no probe_batch= argument needed.

trainprobe_callback = LightningTrainProbeCallback(
    schedule=tp.composite([
        tp.every(200),
        tp.on_loss_spike(threshold=2.0, cooldown=100),
    ]),
    logger="wandb",   # ← change to "jsonl" if WandB is not configured
)


# ── Trainer ───────────────────────────────────────────────────────────────────

model = CIFAR10Classifier(lr=LR)

trainer = pl.Trainer(
    max_epochs=EPOCHS,
    accelerator="auto",
    callbacks=[trainprobe_callback],
    log_every_n_steps=10,
)

trainer.fit(model, train_loader, val_loader)
