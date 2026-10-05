"""Classification losses and batch-level Mixup/CutMix."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class LabelSmoothingCE(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        if not 0 <= smoothing < 1:
            raise ValueError("smoothing must be in [0, 1)")
        self.smoothing = float(smoothing)

    def forward(self, logits, target):
        return F.cross_entropy(logits, target, label_smoothing=self.smoothing)


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None):
        super().__init__()
        if gamma < 0:
            raise ValueError("gamma must be non-negative")
        self.gamma = float(gamma)
        if alpha is None:
            self.register_buffer("alpha", None)
        else:
            self.register_buffer("alpha", torch.as_tensor(alpha, dtype=torch.float32))

    def forward(self, logits, target):
        log_probs = F.log_softmax(logits, dim=1)
        log_pt = log_probs.gather(1, target.long().view(-1, 1)).squeeze(1)
        pt = log_pt.exp()
        loss = -((1.0 - pt).clamp_min(0).pow(self.gamma)) * log_pt
        if self.alpha is not None:
            loss = loss * self.alpha.to(logits.device)[target.long()]
        return loss.mean()


def build_criterion(kind="ce", **kw):
    if kind == "ce":
        return nn.CrossEntropyLoss(
            weight=kw.get("weight"), label_smoothing=kw.get("smoothing", 0.0)
        )
    if kind == "ls":
        return LabelSmoothingCE(kw.get("smoothing", 0.1))
    if kind == "focal":
        return FocalLoss(kw.get("gamma", 2.0), kw.get("alpha"))
    if kind == "ce_weighted":
        weight = kw.get("weight")
        if weight is None:
            raise ValueError("ce_weighted requires class weights")
        return nn.CrossEntropyLoss(
            weight=weight, label_smoothing=kw.get("smoothing", 0.0)
        )
    raise ValueError(f"Unknown criterion: {kind}")


def class_weights(counts, beta=0.0):
    counts = torch.as_tensor(counts, dtype=torch.float32)
    if counts.ndim != 1 or (counts <= 0).any():
        raise ValueError("counts must be a positive one-dimensional vector")
    if beta == 0:
        weights = counts.reciprocal()
    elif 0 < beta < 1:
        weights = (1 - beta) / (-torch.expm1(counts * np.log(beta)))
    else:
        raise ValueError("beta must be 0 or lie strictly between 0 and 1")
    return weights / weights.mean()


def mix_batch(x, y, alpha=1.0, mode="cutmix"):
    if alpha <= 0 or mode not in {"mixup", "cutmix"}:
        raise ValueError("alpha must be positive and mode must be mixup or cutmix")
    lam = float(np.random.beta(alpha, alpha))
    permutation = torch.randperm(x.size(0), device=x.device)
    y_a, y_b = y, y[permutation]
    if mode == "mixup":
        return lam * x + (1 - lam) * x[permutation], (y_a, y_b, lam)
    height, width = x.shape[-2:]
    cut_ratio = np.sqrt(1 - lam)
    cut_w, cut_h = int(width * cut_ratio), int(height * cut_ratio)
    cx = np.random.randint(width)
    cy = np.random.randint(height)
    x1, x2 = max(cx - cut_w // 2, 0), min(cx + cut_w // 2, width)
    y1, y2 = max(cy - cut_h // 2, 0), min(cy + cut_h // 2, height)
    mixed = x.clone()
    mixed[:, :, y1:y2, x1:x2] = x[permutation, :, y1:y2, x1:x2]
    lam = 1.0 - ((x2 - x1) * (y2 - y1) / float(width * height))
    return mixed, (y_a, y_b, lam)


def mixed_loss(criterion, logits, targets):
    if isinstance(targets, tuple) and len(targets) == 3:
        y_a, y_b, lam = targets
        return lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)
    return criterion(logits, targets)
