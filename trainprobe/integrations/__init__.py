"""
Framework integrations — zero-touch callbacks for HuggingFace Trainer,
PyTorch Lightning, and stable-pretraining.

HuggingFace notes
-----------------
The Trainer callback API does not expose the training batch to callbacks, so
activation probes require an explicit probe_batch= argument.  For unlabelled
SSL training this is just a batch of images; for supervised training it should
include labels.  Without probe_batch, only free probes (GradNorm, UpdateRatio)
will run until you call scope.capture_batch(batch) externally.

Lightning notes
---------------
on_train_batch_start receives the batch directly, so capture_batch is wired
automatically — no explicit probe_batch required.
"""
import warnings
from typing import List, Optional, Union

import torch.nn as nn


class TrainProbeCallback:
    """HuggingFace Trainer callback.

    Usage::

        from trainprobe.integrations import TrainProbeCallback
        from transformers import Trainer

        trainer = Trainer(
            model=model,
            callbacks=[TrainProbeCallback(probe_batch=val_batch)],
        )
    """

    def __init__(
        self,
        probe_batch=None,
        probes=None,
        schedule=None,
        logger: str = "wandb",
        layers=None,
        **kwargs,
    ):
        self._init_kwargs = dict(
            probe_batch=probe_batch,
            probes=probes,
            schedule=schedule,
            logger=logger,
            layers=layers,
        )
        self._scope = None
        self._last_loss = None   # HF only flushes loss in on_log, not on_step_end

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        import trainprobe as tp

        self._scope = tp.attach(model, **self._init_kwargs)

    def on_log(self, args, state, control, logs=None, **kwargs):
        # HF Trainer passes the loss dict here, not in on_step_end.
        if logs:
            loss = logs.get("loss")
            if loss is not None:
                self._last_loss = float(loss)

    def on_step_end(self, args, state, control, **kwargs):
        if self._scope is None:
            return
        # _last_loss may be None for many steps between HF log flushes —
        # the scope handles this gracefully (OnLossSpike falls back to every-N).
        self._scope.step(state.global_step, loss=self._last_loss)

    def on_train_end(self, args, state, control, **kwargs):
        self._scope = None
        self._last_loss = None


class LightningTrainProbeCallback:
    """PyTorch Lightning callback.

    Usage::

        from trainprobe.integrations import LightningTrainProbeCallback
        import pytorch_lightning as pl

        trainer = pl.Trainer(
            callbacks=[LightningTrainProbeCallback()],
        )

    The first training batch is automatically captured as the probe batch.
    """

    def __init__(
        self,
        probe_batch=None,
        probes=None,
        schedule=None,
        logger: str = "wandb",
        layers=None,
        **kwargs,
    ):
        self._init_kwargs = dict(
            probe_batch=probe_batch,
            probes=probes,
            schedule=schedule,
            logger=logger,
            layers=layers,
        )
        self._scope = None

    def on_train_start(self, trainer, pl_module):
        import trainprobe as tp

        self._scope = tp.attach(pl_module, **self._init_kwargs)

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx, **kwargs):
        if self._scope is not None:
            self._scope.capture_batch(batch)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, **kwargs):
        if self._scope is None:
            return
        step = trainer.global_step
        loss = None
        if isinstance(outputs, dict):
            loss_tensor = outputs.get("loss")
            loss = loss_tensor.item() if loss_tensor is not None else None
        elif hasattr(outputs, "item"):
            loss = outputs.item()
        self._scope.step(step, loss=loss)

    def on_train_end(self, trainer, pl_module):
        self._scope = None


class StablePretrainingPlugin:
    """Plugin for stable-pretraining SSLTrainer.

    Usage::

        from trainprobe.integrations import StablePretrainingPlugin
        trainer = SSLTrainer(model=model, plugins=[StablePretrainingPlugin()])

    Hooks into on_step_end and on_batch_start of the SSLTrainer plugin protocol.
    """

    def __init__(
        self,
        probe_batch=None,
        probes=None,
        schedule=None,
        logger: str = "wandb",
        layers=None,
        **kwargs,
    ):
        self._init_kwargs = dict(
            probe_batch=probe_batch,
            probes=probes,
            schedule=schedule,
            logger=logger,
            layers=layers,
        )
        self._scope = None

    def on_train_start(self, trainer):
        import trainprobe as tp

        self._scope = tp.attach(trainer.model, **self._init_kwargs)

    def on_batch_start(self, trainer, batch):
        if self._scope is not None:
            self._scope.capture_batch(batch)

    def on_step_end(self, trainer, step: int, loss: float):
        if self._scope is not None:
            self._scope.step(step, loss=loss)
