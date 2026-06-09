"""
trainprobe + stable-pretraining — SSL encoder observability

Attaches trainprobe to a DINO or SimCLR training run via StablePretrainingPlugin.
This is the highest-signal use case for trainprobe: SSL training is where
effective rank collapse is most consequential and hardest to detect from loss alone.

The plugin hooks into the SSLTrainer lifecycle:
    on_train_start(trainer)      → tp.attach(trainer.model, ...)
    on_batch_start(trainer, batch) → scope.capture_batch(batch)  [no-op after first]
    on_step_end(trainer, step, loss) → scope.step(step, loss=loss)

What to watch in WandB:
  trainprobe/effective_rank/*/rank
    Should rise from ~2-5 (random init) to ~15-40 by step 10k, then plateau.
    If it stays flat near 1-3, the encoder is collapsing.

  trainprobe/collapse/*/sv_ratio
    Should be near 0 at init, rise as features spread, and remain stable.
    A drop after a plateau is the onset of dimensional collapse.

  trainprobe/collapse/*/norm_cv
    Expected near 0 for DINO (L2-normalised representations) — this is correct,
    not a sign of collapse. Use sv_ratio and effective_rank as the primary signals.

  trainprobe/events/phase_transition
    CUSUM marker. Expect 1-3 events around steps 5k-20k as the encoder
    transitions from random to semantic representations.

Install:
    pip install trainprobe stable-pretraining

Run:
    python examples/stable_pretraining_ssl.py
"""
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import trainprobe as tp
from trainprobe.integrations import StablePretrainingPlugin

# stable-pretraining imports — adjust to match your installed version
from stable_pretraining import SSLTrainer
from stable_pretraining.models import DINO            # or SimCLR, VICReg, BarlowTwins
from stable_pretraining.augmentations import DINOAugmentation


# ── Config ────────────────────────────────────────────────────────────────────

BATCH_SIZE = 256
EPOCHS = 100
LR = 1e-3
DATA_DIR = "./data"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Data CIFAR-100 -------------------------------------------------------------

augmentation = DINOAugmentation(image_size=32, global_scale=(0.25, 1.0))

train_set = datasets.CIFAR100(DATA_DIR, train=True, download=True, transform=augmentation)
train_loader = DataLoader(
    train_set,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    drop_last=True,
)


# ── Model ─────────────────────────────────────────────────────────────────────

model = DINO(
    backbone="vit_small",
    patch_size=4,          # smaller patch for 32×32 images
    projection_dim=256,
    num_prototypes=4096,
)


# ── StablePretrainingPlugin ───────────────────────────────────────────────────
#
# StablePretrainingPlugin wraps tp.attach() and wires all three lifecycle hooks.
# The probe batch is captured from the first training batch automatically.
#
# For SSL the probe batch contains augmented view pairs — the hook manager runs
# a forward pass on the student encoder only (not the teacher) to collect
# per-layer activations.  This is a clean representation before any projection head.

trainprobe_plugin = StablePretrainingPlugin(
    schedule=tp.composite([
        tp.every(500),
        tp.on_loss_spike(threshold=1.8, cooldown=200),
    ]),
    logger="wandb",   # ← change to "jsonl" for offline runs on Oscar
    layers=tp.layers.by_name(
        # Probe the output of each transformer block's attention projection.
        # Adjust layer names to match your backbone's named_modules().
        "backbone.blocks.2.attn.proj",
        "backbone.blocks.5.attn.proj",
        "backbone.blocks.8.attn.proj",
        "backbone.blocks.11.attn.proj",
    ),
)

# React to detected phase transitions — useful for checkpointing or logging.
def on_transition(step, metrics):
    print(f"[trainprobe] Phase transition detected at step {step} — "
          f"consider saving a checkpoint.")

# The plugin exposes the scope after on_train_start fires.
# Attach the callback once the trainer has called on_train_start:
#   trainprobe_plugin._scope.on_phase_transition = on_transition
# Or set it on the plugin directly:
trainprobe_plugin._on_phase_transition_fn = on_transition


# ── SSLTrainer ────────────────────────────────────────────────────────────────

trainer = SSLTrainer(
    model=model,
    train_loader=train_loader,
    epochs=EPOCHS,
    lr=LR,
    device=DEVICE,
    plugins=[trainprobe_plugin],
    wandb_project="trainprobe-dino-cifar100",
)

trainer.fit()
