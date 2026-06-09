"""
trainprobe quickstart — raw PyTorch

Trains ResNet-18 on CIFAR-10 and streams representation metrics to JSONL.
Switch logger="wandb" to send to your WandB run instead.

Install:
    pip install trainprobe torchvision

Run:
    python examples/pytorch_quickstart.py
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

import trainprobe as tp
from trainprobe.probes.task import LinearProbeProbe

# ── Config ────────────────────────────────────────────────────────────────────

BATCH_SIZE = 128
PROBE_BATCH_SIZE = 256
EPOCHS = 5
LR = 1e-3
DATA_DIR = "./data"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True)

# Probe batch: clean (no augmentation) validation images.
# Stored on CPU by trainprobe; moved to the model's device at probe time.
probe_loader  = DataLoader(val_set, batch_size=PROBE_BATCH_SIZE, shuffle=False)
probe_images, probe_labels = next(iter(probe_loader))


# ── Model ─────────────────────────────────────────────────────────────────────

model = models.resnet18(weights=None)
model.fc = nn.Linear(512, 10)
model = model.to(DEVICE)


# ── Attach trainprobe ─────────────────────────────────────────────────────────
#
# default_suite() runs five probes:
#   EffectiveRankProbe  — intrinsic dimensionality of each layer's representations
#   CollapseProbe       — norm_cv + sv_ratio (dimensional & hyperspherical collapse)
#   DeadNeuronProbe     — fraction of neurons with zero max-activation
#   GradNormProbe       — per-parameter L2 gradient norm  (runs every step)
#   UpdateRatioProbe    — ||Δw|| / ||w|| per parameter     (runs every step)
#
# LinearProbeProbe (v0.2) is added explicitly — it needs labels= passed to step().

scope = tp.attach(
    model,
    probe_batch=probe_images,
    probes=tp.default_suite() + [LinearProbeProbe()],
    schedule=tp.composite([
        tp.every(200),
        tp.on_loss_spike(threshold=2.0, cooldown=100),
    ]),
    logger="jsonl",   # ← change to "wandb" to log to your active WandB run
)

# Fired whenever CUSUM detects a structural shift in effective rank.
scope.on_phase_transition = lambda step, _metrics: print(
    f"  [trainprobe] Phase transition at step {step}"
)


# ── Training loop ─────────────────────────────────────────────────────────────

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
criterion = nn.CrossEntropyLoss()

global_step = 0
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()

        # ← The only trainprobe line in the loop.
        # Pass labels= so LinearProbeProbe can measure representation separability.
        scope.step(global_step, loss=loss.item(), labels=labels.cpu())

        running_loss += loss.item()
        global_step += 1

    avg_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch + 1}/{EPOCHS}  loss={avg_loss:.4f}")
    scheduler.step()


print(f"\nDone. Metrics written to trainprobe_log.jsonl ({global_step} steps).")
print("Each line is a JSON object: { step, trainprobe/effective_rank/*/rank, ... }")
