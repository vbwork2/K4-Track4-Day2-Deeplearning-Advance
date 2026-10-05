"""timm model construction and parameter accounting."""

from __future__ import annotations

import torch
from torch import nn
import timm

SUGGESTED_BACKBONES = {
    "resnet50": "resnet50",
    "resnext50": "resnext50_32x4d",
    "convnext_tiny": "convnext_tiny",
    "deit_small": "deit_small_patch16_224",
    "swin_tiny": "swin_tiny_patch4_window7_224",
    "efficientnet_b0": "efficientnet_b0",
    "mobilenetv3": "mobilenetv3_large_100",
}


def build_model(name, pretrained=True, num_classes=9, drop_rate=0.0, init="finetune"):
    if init not in {"scratch", "frozen", "finetune"}:
        raise ValueError("init must be scratch, frozen, or finetune")
    if name in SUGGESTED_BACKBONES:
        name = SUGGESTED_BACKBONES[name]
    use_pretrained = bool(pretrained and init != "scratch")
    extra = {}
    if name.startswith(("deit_", "vit_")):
        extra = {"dynamic_img_size": True, "dynamic_img_pad": True}
    elif name.startswith("swin_"):
        extra = {"strict_img_size": False}
    model = timm.create_model(
        name,
        pretrained=use_pretrained,
        num_classes=num_classes,
        drop_rate=drop_rate,
        **extra,
    )
    model.pretrained_tag = (
        (
            model.pretrained_cfg.get("tag")
            or model.pretrained_cfg.get("hf_hub_id")
            or model.pretrained_cfg.get("url")
            or "timm-default"
        )
        if use_pretrained
        else "random-initialization"
    )
    model.init_mode = init
    if init == "frozen":
        freeze_backbone(model)
    return model


def _head_parameter_ids(model):
    classifier = model.get_classifier()
    modules = list(classifier.modules()) if isinstance(classifier, nn.Module) else []
    return {id(parameter) for module in modules for parameter in module.parameters()}


def freeze_backbone(model):
    head_ids = _head_parameter_ids(model)
    if not head_ids:
        raise ValueError("Model classifier has no parameters")
    for parameter in model.parameters():
        parameter.requires_grad_(id(parameter) in head_ids)
    model._frozen_backbone = True
    model.eval()


def keep_frozen_batchnorm_eval(model):
    if getattr(model, "_frozen_backbone", False):
        head_ids = _head_parameter_ids(model)
        for module in model.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                if not any(id(p) in head_ids for p in module.parameters()):
                    module.eval()


def param_groups(model, lr_backbone, lr_head, weight_decay):
    head_ids = _head_parameter_ids(model)
    buckets = {
        (False, False): [],
        (False, True): [],
        (True, False): [],
        (True, True): [],
    }
    excluded_names = (
        set(model.no_weight_decay()) if hasattr(model, "no_weight_decay") else set()
    )
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    seen = set()
    for module in model.modules():
        for name, parameter in module.named_parameters(recurse=False):
            if not parameter.requires_grad:
                continue
            if id(parameter) in seen:
                raise ValueError("Duplicate trainable parameter encountered")
            seen.add(id(parameter))
            is_head = id(parameter) in head_ids
            no_decay = (
                parameter.ndim <= 1
                or name.endswith("bias")
                or names[id(parameter)] in excluded_names
            )
            key = (is_head, no_decay)
            buckets[key].append(parameter)
    groups = []
    for key, params in buckets.items():
        if params:
            is_head, no_decay = key
            groups.append(
                {
                    "params": params,
                    "lr": lr_head if is_head else lr_backbone,
                    "weight_decay": 0.0 if no_decay else weight_decay,
                }
            )
    if len(seen) != sum(1 for p in model.parameters() if p.requires_grad):
        raise ValueError(
            "Some trainable parameters were not assigned to an optimizer group"
        )
    return groups


def count_params(model):
    return sum(parameter.numel() for parameter in model.parameters()) / 1e6


def count_gmacs(model, img_size=224):
    """Count Conv/Linear MACs and include attention matrix products for timm ViTs."""
    macs = [0]
    hooks = []

    def conv_hook(module, inputs, output):
        out = output
        per_position = (
            (module.in_channels // module.groups)
            * module.kernel_size[0]
            * module.kernel_size[1]
        )
        macs[0] += out.numel() * per_position

    def linear_hook(module, inputs, output):
        macs[0] += output.numel() * module.in_features

    def attention_hook(module, inputs, output):
        batch, tokens, channels = inputs[0].shape
        macs[0] += 2 * batch * tokens * tokens * channels

    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            hooks.append(module.register_forward_hook(conv_hook))
        elif isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(linear_hook))
        if type(module).__name__ == "Attention" and hasattr(module, "qkv"):
            hooks.append(module.register_forward_hook(attention_hook))
    training_states = {module: module.training for module in model.modules()}
    try:
        model.eval()
        parameter = next(model.parameters())
        with torch.inference_mode():
            model(
                torch.zeros(
                    1,
                    3,
                    img_size,
                    img_size,
                    device=parameter.device,
                    dtype=parameter.dtype,
                )
            )
    finally:
        for hook in hooks:
            hook.remove()
        for module, training in training_states.items():
            module.training = training
    model.gmac_method = (
        "Conv2d/Linear and ViT attention MACs; excludes elementwise operations"
    )
    return macs[0] / 1e9
