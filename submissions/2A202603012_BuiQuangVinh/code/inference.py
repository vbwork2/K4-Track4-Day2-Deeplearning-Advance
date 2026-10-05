"""Validation-selected inference transforms, calibration, and Conv-BN fusion."""

from __future__ import annotations

import copy
import numpy as np
import torch
from torch.nn import functional as F


@torch.inference_mode()
def predict_logits(model, loader, device, view=None):
    model.eval()
    names, truths, outputs = [], [], []
    device = torch.device(device)
    for images, labels, filenames in loader:
        images = images.to(device, non_blocking=True)
        if view is not None:
            images = view(images)
        logits = model(images)
        names.extend(filenames)
        truths.append(labels.cpu().numpy())
        outputs.append(logits.float().cpu().numpy())
    if not outputs:
        raise ValueError("Inference loader is empty")
    return names, np.concatenate(truths), np.concatenate(outputs)


def view_identity(x):
    return x


def view_hflip(x):
    return torch.flip(x, dims=(-1,))


def views_multicrop(x, crop: int):
    height, width = x.shape[-2:]
    if crop > height or crop > width:
        raise ValueError("crop cannot exceed input dimensions")
    positions = [
        (0, 0),
        (0, width - crop),
        (height - crop, 0),
        (height - crop, width - crop),
        ((height - crop) // 2, (width - crop) // 2),
    ]
    return [x[..., top : top + crop, left : left + crop] for top, left in positions]


def views_multiscale(x, sizes):
    return [
        F.interpolate(
            x,
            size=(int(size), int(size)),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        for size in sizes
    ]


def method_views(images, method):
    """Use identical view construction for prediction and latency measurement."""
    size = int(images.shape[-1])
    if method in {"single", "fp32"}:
        return [images]
    if method == "hflip":
        return [images, view_hflip(images)]
    if method == "multicrop":
        crops = views_multicrop(images, max(32, int(size * 0.875)))
        return [
            F.interpolate(
                crop,
                size=images.shape[-2:],
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
            for crop in crops
        ]
    if method == "multiscale":
        larger = ((round(size * 1.14) + 31) // 32) * 32
        return views_multiscale(images, (size, larger))
    raise ValueError(f"Unknown inference method: {method}")


@torch.inference_mode()
def forward_method(model, images, method="single", aggregation="prob", amp=False):
    if aggregation not in {"prob", "logit"}:
        raise ValueError("aggregation must be prob or logit")
    use_amp = bool(amp and method != "fp32" and images.device.type == "cuda")
    outputs = []
    for batch in method_views(images, method):
        with torch.autocast(images.device.type, dtype=torch.float16, enabled=use_amp):
            outputs.append(model(batch).float())
    if len(outputs) == 1:
        return outputs[0]
    values = torch.stack(outputs)
    if aggregation == "logit":
        return values.mean(dim=0)
    # log(mean(softmax(logits))) avoids probability underflow.
    return torch.logsumexp(F.log_softmax(values, dim=-1), dim=0) - np.log(len(outputs))


@torch.inference_mode()
def predict_method_logits(
    model, loader, device, method="single", aggregation="prob", amp=False
):
    model.eval()
    device = torch.device(device)
    names, truths, outputs = [], [], []
    for images, labels, filenames in loader:
        images = images.to(device, non_blocking=True)
        logits = forward_method(model, images, method, aggregation, amp)
        names.extend(filenames)
        truths.append(labels.cpu().numpy())
        outputs.append(logits.cpu().numpy())
    if not outputs:
        raise ValueError("Inference loader is empty")
    return names, np.concatenate(truths), np.concatenate(outputs)


def aggregate_views(logits_per_view, space="prob"):
    if not logits_per_view:
        raise ValueError("At least one view is required")
    values = torch.stack([torch.as_tensor(value).float() for value in logits_per_view])
    if space == "prob":
        probs = torch.softmax(values, dim=-1).mean(dim=0)
    elif space == "logit":
        probs = torch.softmax(values.mean(dim=0), dim=-1)
    else:
        raise ValueError("space must be 'prob' or 'logit'")
    return probs / probs.sum(dim=-1, keepdim=True)


def ensemble_probs(list_of_probs):
    if not list_of_probs:
        raise ValueError("At least one probability array is required")
    arrays = [np.asarray(value, dtype=np.float64) for value in list_of_probs]
    if any(value.shape != arrays[0].shape for value in arrays):
        raise ValueError("Ensemble inputs must have identical shapes and row order")
    result = np.mean(arrays, axis=0)
    return result / result.sum(axis=1, keepdims=True)


def fit_temperature(val_logits, val_labels) -> float:
    logits = torch.as_tensor(val_logits, dtype=torch.float64)
    labels = torch.as_tensor(val_labels, dtype=torch.long)
    if logits.ndim != 2 or logits.shape[0] != labels.numel():
        raise ValueError("Validation logits and labels have incompatible shapes")
    if not labels.numel() or not torch.isfinite(logits).all():
        raise ValueError("Validation logits must be non-empty and finite")
    log_t = torch.zeros((), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS(
        [log_t], lr=0.1, max_iter=100, line_search_fn="strong_wolfe"
    )

    def closure():
        optimizer.zero_grad()
        loss = F.cross_entropy(logits / log_t.exp().clamp(0.05, 20), labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_t.detach().exp().clamp(0.05, 20))


def apply_temperature(logits, T):
    if T <= 0:
        raise ValueError("Temperature must be positive")
    return (
        torch.softmax(torch.as_tensor(logits, dtype=torch.float32) / float(T), dim=-1)
        .cpu()
        .numpy()
    )


def expected_calibration_error(probabilities, labels, bins=15):
    probabilities = np.asarray(probabilities)
    labels = np.asarray(labels)
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == labels
    bin_ids = np.clip(np.ceil(confidence * bins).astype(int) - 1, 0, bins - 1)
    return float(
        sum(
            np.mean(bin_ids == i)
            * abs(correct[bin_ids == i].mean() - confidence[bin_ids == i].mean())
            for i in range(bins)
            if np.any(bin_ids == i)
        )
    )


def fuse_conv_bn(model):
    fused = copy.deepcopy(model).eval()

    def recurse(parent):
        children = list(parent.named_children())
        for name, child in children:
            recurse(child)
        # Registration order only guarantees execution order in Sequential modules.
        if not isinstance(parent, torch.nn.Sequential):
            return
        for (first_name, first), (second_name, second) in zip(children, children[1:]):
            if isinstance(first, torch.nn.Conv2d) and isinstance(
                second, torch.nn.BatchNorm2d
            ):
                if (
                    first.out_channels != second.num_features
                    or not second.track_running_stats
                ):
                    continue
                setattr(
                    parent,
                    first_name,
                    torch.nn.utils.fusion.fuse_conv_bn_eval(first, second),
                )
                setattr(parent, second_name, torch.nn.Identity())

    recurse(fused)
    return fused
