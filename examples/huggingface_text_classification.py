"""
trainprobe + HuggingFace Trainer — text classification

Fine-tunes DistilBERT on SST-2 (binary sentiment) with representation metrics
streamed via TrainProbeCallback.

Key HF integration notes:
  - probe_batch must be passed explicitly because the Trainer callback API
    does not expose the training batch to callbacks.
  - The probe_batch should be tokenized inputs WITHOUT labels so the forward
    pass only computes activations (no loss overhead).
  - Loss is captured via on_log (fires at logging_steps), not on_step_end.
    OnLossSpike scheduling therefore has coarser resolution than in raw PyTorch.

Install:
    pip install trainprobe transformers datasets accelerate

Run:
    python examples/huggingface_text_classification.py
"""
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from trainprobe.integrations import TrainProbeCallback


MODEL_NAME = "distilbert-base-uncased"
PROBE_BATCH_SIZE = 128
MAX_LENGTH = 128
OUTPUT_DIR = "./hf_output"


# ── Data ──────────────────────────────────────────────────────────────────────

raw = load_dataset("glue", "sst2")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def tokenize(batch):
    return tokenizer(batch["sentence"], truncation=True, max_length=MAX_LENGTH)


encoded = raw.map(tokenize, batched=True)
encoded = encoded.rename_column("label", "labels")
encoded.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

train_dataset = encoded["train"]
eval_dataset  = encoded["validation"]


# ── Probe batch ───────────────────────────────────────────────────────────────
#
# Take the first PROBE_BATCH_SIZE samples from the eval set and strip labels.
# trainprobe runs a forward pass on this fixed batch to collect activations;
# including labels would cause the model to also compute the loss (wasted work).

probe_samples = [eval_dataset[i] for i in range(PROBE_BATCH_SIZE)]
probe_batch = tokenizer.pad(
    [{"input_ids": s["input_ids"], "attention_mask": s["attention_mask"]}
     for s in probe_samples],
    return_tensors="pt",
)
# probe_labels mirrors the probe_batch order, passed to scope.step() if you
# add LinearProbeProbe to the suite.
probe_labels = [s["labels"] for s in probe_samples]


# ── Model ─────────────────────────────────────────────────────────────────────

model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)


# ── TrainProbeCallback ────────────────────────────────────────────────────────
#
# probe_batch is required for activation probes (EffectiveRank, Collapse, etc.).
# Without it, only free probes (GradNorm, UpdateRatio) run.
#
# The callback attaches trainprobe in on_train_begin (model is available then),
# captures loss in on_log, and calls scope.step() in on_step_end.

trainprobe_callback = TrainProbeCallback(
    probe_batch=probe_batch,
    logger="wandb",   # ← assumes a WandB run is active; change to "jsonl" otherwise
)


# ── Trainer ───────────────────────────────────────────────────────────────────

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    learning_rate=2e-5,
    weight_decay=0.01,
    logging_steps=50,          # how often HF flushes loss to callbacks
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    report_to="wandb",         # ← set to "none" if not using WandB
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    callbacks=[trainprobe_callback],
)

trainer.train()
