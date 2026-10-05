"""Synchronized inference latency measurements with warmup and percentiles."""

from __future__ import annotations

import time
import copy
import numpy as np
import torch


def bench(fn, warmup=10, iters=100, sync=None):
    if warmup < 10 or iters < 50:
        raise ValueError("Use at least 10 warmup and 50 measured iterations")
    synchronize = sync or (lambda: None)
    for _ in range(warmup):
        fn()
    synchronize()
    samples = []
    for _ in range(iters):
        synchronize()
        started = time.perf_counter()
        fn()
        synchronize()
        samples.append((time.perf_counter() - started) * 1000)
    return {
        "p50": float(np.percentile(samples, 50)),
        "p95": float(np.percentile(samples, 95)),
        "p99": float(np.percentile(samples, 99)),
        "mean": float(np.mean(samples)),
        "n": int(iters),
        "warmup": int(warmup),
    }


def latency_report(
    model, batch_size, img_size, dtype="fp32", device="cuda", warmup=10, iters=100
):
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    model = model.to(device).eval()
    x = torch.randn(batch_size, 3, img_size, img_size, device=device)
    if dtype not in {"fp32", "amp", "fp16"}:
        raise ValueError("dtype must be fp32, amp, or fp16")
    if dtype == "fp16":
        if device.type != "cuda":
            raise ValueError("FP16 benchmark requires CUDA")
        model = copy.deepcopy(model).half()
        x = x.half()

    def forward():
        with torch.inference_mode():
            with torch.autocast(
                device.type,
                dtype=torch.float16,
                enabled=dtype == "amp" and device.type == "cuda",
            ):
                return model(x)

    sync = torch.cuda.synchronize if device.type == "cuda" else None
    result = bench(forward, warmup, iters, sync)
    result.update(
        {
            "gpu": torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else "CPU",
            "dtype": dtype,
            "batch": batch_size,
            "img_size": img_size,
            "images_per_s": batch_size / (result["p50"] / 1000),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "preprocessing_included": False,
        }
    )
    return result


def tta_latency(model, k_views, **kw):
    if k_views < 1:
        raise ValueError("k_views must be positive")
    batch = int(kw.get("batch_size", 1))
    size = int(kw.get("img_size", 224))
    device = torch.device(kw.get("device", "cuda"))
    dtype = kw.get("dtype", "fp32")
    warmup = int(kw.get("warmup", 10))
    iters = int(kw.get("iters", 100))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    model = model.to(device).eval()
    x = torch.randn(batch, 3, size, size, device=device)
    if dtype == "fp16":
        model, x = model.half(), x.half()
    if dtype not in {"fp32", "amp", "fp16"}:
        raise ValueError("dtype must be fp32, amp, or fp16")

    def forward():
        with torch.inference_mode():
            with torch.autocast(
                device.type,
                dtype=torch.float16,
                enabled=dtype == "amp" and device.type == "cuda",
            ):
                for _ in range(k_views):
                    model(x)

    result = bench(
        forward,
        warmup,
        iters,
        torch.cuda.synchronize if device.type == "cuda" else None,
    )
    result.update(
        {
            "gpu": torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else "CPU",
            "dtype": dtype,
            "batch": batch,
            "img_size": size,
            "views": k_views,
            "images_per_s": batch / (result["p50"] / 1000),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "preprocessing_included": False,
        }
    )
    return result


def method_latency_report(
    model,
    method="single",
    aggregation="prob",
    batch_size=1,
    img_size=224,
    amp=True,
    temperature=1.0,
    device="cuda",
    warmup=10,
    iters=50,
):
    from inference import forward_method, method_views

    device = torch.device(device)
    model = model.to(device).eval()
    images = torch.randn(batch_size, 3, img_size, img_size, device=device)
    use_amp = bool(amp and method != "fp32" and device.type == "cuda")

    def forward():
        logits = forward_method(model, images, method, aggregation, use_amp)
        return torch.softmax(logits / temperature, dim=-1)

    sync = torch.cuda.synchronize if device.type == "cuda" else None
    result = bench(forward, warmup, iters, sync)
    result.update(
        {
            "method": method,
            "aggregation": aggregation,
            "gpu": torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else "CPU",
            "dtype": "AMP" if use_amp else "FP32",
            "batch": batch_size,
            "img_size": img_size,
            "views": len(method_views(images, method)),
            "images_per_s": batch_size * 1000 / result["p50"],
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "preprocessing_included": False,
            "batchnorm_fused": False,
        }
    )
    return result
