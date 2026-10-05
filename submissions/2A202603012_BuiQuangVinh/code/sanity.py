"""Small runtime checks used by the notebook before full experiments."""

from __future__ import annotations

import gc
import math

import numpy as np
import torch
from torch.nn import functional as F

import inference
import losses
import model
import train


def check_mixed_targets(device="cpu"):
    x = torch.arange(4 * 3 * 16 * 16, dtype=torch.float32, device=device).reshape(
        4, 3, 16, 16
    )
    y = torch.arange(4, device=device)
    for seed in range(10):
        train.set_seed(seed)
        for mode in ("mixup", "cutmix"):
            mixed, (first, second, fraction) = losses.mix_batch(x, y, mode=mode)
            assert mixed.shape == x.shape
            assert torch.equal(first, y)
            assert 0 <= fraction <= 1
            if mode == "mixup":
                expected = fraction * x + (1 - fraction) * x[second]
                assert torch.allclose(mixed, expected)
            else:
                # Self-pairs retain every pixel, so check area only on non-self pairs.
                non_self = second != first
                if non_self.any():
                    unchanged = mixed[non_self].eq(x[non_self]).float().mean().item()
                    assert abs(unchanged - fraction) < 1e-6
    return True


def run(images, labels, device="cuda", overfit_steps=40):
    device = torch.device(device)
    train.set_seed(0)
    images = images[:4].to(device)
    labels = labels[:4].to(device)
    assert len(labels) >= 2
    uniform_logits = torch.zeros(len(labels), 9, device=device)
    initial_ce = F.cross_entropy(uniform_logits, labels)
    assert abs(initial_ce.item() - math.log(9)) < 1e-6

    net = model.build_model("resnet18", pretrained=False, init="scratch").to(device)
    net.eval()
    with torch.no_grad():
        logits = net(images)
    assert logits.shape == (len(labels), 9)
    assert torch.allclose(
        F.cross_entropy(logits, labels),
        losses.FocalLoss(0)(logits, labels),
        atol=1e-5,
        rtol=1e-5,
    )
    assert torch.allclose(
        logits.softmax(1).sum(1), torch.ones(len(labels), device=device), atol=1e-6
    )
    groups = model.param_groups(net, 1e-4, 1e-3, 0.05)
    ids = [id(parameter) for group in groups for parameter in group["params"]]
    assert len(ids) == len(set(ids))
    assert set(ids) == {
        id(parameter) for parameter in net.parameters() if parameter.requires_grad
    }
    check_mixed_targets(device)

    frozen = model.build_model("resnet18", pretrained=False, init="frozen").to(device)
    frozen.train()
    model.keep_frozen_batchnorm_eval(frozen)
    head = model._head_parameter_ids(frozen)
    assert all(
        parameter.requires_grad == (id(parameter) in head)
        for parameter in frozen.parameters()
    )
    assert all(
        not module.training
        for module in frozen.modules()
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
    )

    source = (
        torch.nn.Sequential(
            torch.nn.Conv2d(3, 4, 3, padding=1, bias=False), torch.nn.BatchNorm2d(4)
        )
        .to(device)
        .eval()
    )
    fused = inference.fuse_conv_bn(source)
    with torch.no_grad():
        assert torch.allclose(source(images), fused(images), atol=1e-5, rtol=1e-5)
    del frozen, source, fused

    # Keep one fixed small batch; augmentation must not change it between steps.
    small = F.interpolate(images, size=(64, 64), mode="bilinear", align_corners=False)
    optimizer = torch.optim.AdamW(net.parameters(), lr=0.002)
    net.train()
    first_loss = None
    last_loss = None
    for _ in range(overfit_steps):
        optimizer.zero_grad(set_to_none=True)
        value = F.cross_entropy(net(small), labels)
        if first_loss is None:
            first_loss = value.item()
        value.backward()
        optimizer.step()
        last_loss = value.item()
    if last_loss is None or not np.isfinite(last_loss) or last_loss >= first_loss:
        raise AssertionError(
            f"Tiny-batch loss did not decrease: {first_loss} -> {last_loss}"
        )
    result = {
        "status": "PASS",
        "uniform_ce": initial_ce.item(),
        "expected_ce": math.log(9),
        "tiny_batch_first_loss": first_loss,
        "tiny_batch_last_loss": last_loss,
        "checks": [
            "output_shape",
            "focal_gamma_zero",
            "probabilities",
            "optimizer_groups",
            "frozen_batchnorm",
            "mixup",
            "cutmix_non_self_area",
            "bn_fusion",
            "tiny_batch_fit",
        ],
    }
    del net, optimizer
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result
