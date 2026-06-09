"""
v0.2 task-level probes.

LinearProbeProbe: closed-form least-squares linear probe (no sklearn).
SpuriousFeatureProbe: genuine-vs-spurious accuracy gap for shortcut detection.

Both probes are inert (return {}) when the required label tensors are absent
from ctx, so they can be registered in the default suite without breaking
label-free training loops.
"""
from typing import Optional

import torch
import torch.nn as nn

from . import Probe


def _lstsq_accuracy(X: torch.Tensor, labels: torch.Tensor) -> float:
    """Closed-form linear probe accuracy via least-squares.

    X     : (N, D) float, mean-centered by caller.
    labels: (N,)   integer class labels, 0-indexed.

    For binary classification a single output is used with a 0.5 threshold.
    For K>2 classes, K one-vs-rest outputs are solved simultaneously.
    This is an in-sample fit — a relative-change signal, not a generalization
    estimate.  Accuracy will be inflated vs. a held-out set.
    """
    if X.shape[0] < 2:
        return 0.0

    n_classes = int(labels.max().item()) + 1
    X_f = X.float()

    if n_classes <= 2:
        y = labels.float().unsqueeze(-1)
        result = torch.linalg.lstsq(X_f, y, driver="gelsd")
        preds = (X_f @ result.solution).squeeze(-1)
        # X_f is mean-centered, so mean(preds) = 0 — threshold at 0, not 0.5.
        predicted = (preds > 0.0).long()
    else:
        y_onehot = torch.zeros(X_f.shape[0], n_classes)
        y_onehot.scatter_(1, labels.long().unsqueeze(1), 1.0)
        result = torch.linalg.lstsq(X_f, y_onehot, driver="gelsd")
        scores = X_f @ result.solution
        predicted = scores.argmax(dim=1)

    return (predicted == labels.long()).float().mean().item()


class LinearProbeProbe(Probe):
    """Per-layer linear probe accuracy via closed-form least-squares solve.

    Pass ``labels`` (integer tensor, shape (N,)) via scope.step():

        scope.step(step, loss=loss.item(), labels=y)

    Accuracy is computed in-sample on the probe batch — use it as a
    relative signal (rising = representations becoming more linearly
    separable) rather than an absolute generalization estimate.

    Returns nothing and warns once when labels are absent.
    """

    needs_activations = True
    name = "linear_probe"

    def __init__(self):
        self._warned = False

    def run(self, activations, model: Optional[nn.Module], step: int, **ctx) -> dict:
        labels = ctx.get("labels")
        if labels is None:
            if not self._warned:
                import warnings
                warnings.warn(
                    "trainprobe: LinearProbeProbe requires labels= in scope.step(); "
                    "returning no metrics. (This warning fires once.)",
                    stacklevel=4,
                )
                self._warned = True
            return {}

        metrics = {}
        for layer_name, acts in activations.items():
            X = acts.float()
            X = X - X.mean(0, keepdim=True)
            acc = _lstsq_accuracy(X, labels)
            metrics[f"{layer_name}/accuracy"] = acc
        return metrics


class SpuriousFeatureProbe(Probe):
    """Genuine-vs-spurious linear probe accuracy gap.

    Fits separate linear probes for genuine task labels and spurious (shortcut)
    labels, then reports each accuracy and their gap:

        gap = genuine_accuracy - spurious_accuracy

    A negative gap (spurious_accuracy > genuine_accuracy) indicates the model's
    representations better encode the shortcut feature than the intended one —
    the classical shortcut learning signature from Geirhos et al. (2020).

    Usage::

        scope.step(step, loss=loss.item(),
                   genuine_labels=y_true, spurious_labels=y_spurious)

    ``genuine_labels`` and ``spurious_labels`` must both be provided; the probe
    returns nothing if either is absent.
    """

    needs_activations = True
    name = "spurious"

    def run(self, activations, model: Optional[nn.Module], step: int, **ctx) -> dict:
        genuine = ctx.get("genuine_labels")
        spurious = ctx.get("spurious_labels")
        if genuine is None or spurious is None:
            return {}

        metrics = {}
        for layer_name, acts in activations.items():
            X = acts.float()
            X = X - X.mean(0, keepdim=True)
            g_acc = _lstsq_accuracy(X, genuine)
            s_acc = _lstsq_accuracy(X, spurious)
            metrics[f"{layer_name}/genuine_accuracy"] = g_acc
            metrics[f"{layer_name}/spurious_accuracy"] = s_acc
            metrics[f"{layer_name}/gap"] = g_acc - s_acc
        return metrics
