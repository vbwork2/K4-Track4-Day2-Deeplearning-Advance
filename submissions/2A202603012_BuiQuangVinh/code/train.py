"""Single reproducible training pipeline for screening, ablations, and final runs."""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import random
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

import dataset, inference, losses, model as model_lib
from eval import compute_metrics, save_predictions


@dataclass
class Config:
    exp_id: str = "T00"
    seed: int = 0
    fold: int = 0
    backbone: str = "resnet50"
    init: str = "finetune"
    drop_rate: float = 0.0
    img_size: int = 224
    aug: str = "basic"
    sampler: str | None = None
    mix: str | None = None
    mix_alpha: float = 1.0
    loss: str = "ce"
    label_smoothing: float = 0.0
    focal_gamma: float = 2.0
    class_weight_beta: float | None = None
    epochs: int = 12
    batch_size: int = 64
    lr_backbone: float = 1e-4
    lr_head: float = 1e-3
    weight_decay: float = 0.05
    warmup_epochs: float = 1.0
    ema_decay: float | None = None
    amp: bool = True
    num_workers: int = 2
    images_dir: str = "data/images"
    labels_dir: str = "data/labels"
    out_dir: str = "runs"
    pred_dir: str = "predictions"
    curve_dir: str = "curves"
    sync_dir: str | None = None
    inference_method: str = "single"
    inference_aggregation: str = "prob"
    inference_img_size: int | None = None
    inference_amp: bool | None = None
    temperature_scaling: bool = False
    save_test_predictions: bool = False
    resume: bool = True


def run_dir(cfg):
    return Path(cfg.out_dir) / cfg.exp_id / f"seed{cfg.seed}"


def pred_path(cfg, split):
    return Path(cfg.pred_dir) / f"{cfg.exp_id}_seed{cfg.seed}_{split}.csv"


def set_seed(seed):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        torch.use_deterministic_algorithms(True)


def build_optimizer(net, cfg):
    groups = model_lib.param_groups(net, cfg.lr_backbone, cfg.lr_head, cfg.weight_decay)
    return torch.optim.AdamW(groups)


def build_scheduler(optimizer, cfg, steps_per_epoch):
    total = max(1, cfg.epochs * steps_per_epoch)
    warmup = min(total - 1, max(0, int(cfg.warmup_epochs * steps_per_epoch)))

    def scale(step):
        if warmup and step < warmup:
            return max(1e-8, (step + 1) / warmup)
        progress = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1 + np.cos(np.pi * min(1.0, max(0.0, progress))))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, scale)


class EMA:
    def __init__(self, net, decay):
        import copy

        self.model = copy.deepcopy(net).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.decay = float(decay)

    @torch.no_grad()
    def update(self, net):
        source = net.state_dict()
        for key, value in self.model.state_dict().items():
            current = source[key].detach()
            if value.is_floating_point():
                value.mul_(self.decay).add_(current, alpha=1 - self.decay)
            else:
                value.copy_(current)


def _autocast(device, enabled):
    return torch.autocast(
        device_type=device.type,
        dtype=torch.float16,
        enabled=bool(enabled and device.type == "cuda"),
    )


def train_one_epoch(
    net, loader, criterion, optimizer, scheduler, scaler, cfg, device, ema=None
):
    net.train()
    model_lib.keep_frozen_batchnorm_eval(net)
    losses_sum, seen = 0.0, 0
    for images, labels, _ in loader:
        images, labels = (
            images.to(device, non_blocking=True),
            labels.to(device, non_blocking=True),
        )
        optimizer.zero_grad(set_to_none=True)
        targets = labels
        if cfg.mix:
            images, targets = losses.mix_batch(images, labels, cfg.mix_alpha, cfg.mix)
        with _autocast(device, cfg.amp):
            logits = net(images)
            loss = losses.mixed_loss(criterion, logits, targets)
        if not torch.isfinite(loss):
            raise FloatingPointError(
                "Training loss is not finite; inspect the input and configuration"
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            [p for p in net.parameters() if p.requires_grad], 5.0
        )
        old_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if scaler.get_scale() >= old_scale:
            scheduler.step()
            if ema is not None:
                ema.update(net)
        n = labels.shape[0]
        losses_sum += float(loss.detach()) * n
        seen += n
    return {
        "train_loss": losses_sum / max(1, seen),
        "lr": optimizer.param_groups[0]["lr"],
    }


@torch.inference_mode()
def evaluate(net, loader, criterion, device, amp=False):
    net.eval()
    filenames, truths, logits_all = [], [], []
    loss_sum, count = 0.0, 0
    for images, labels, names in loader:
        images, labels = (
            images.to(device, non_blocking=True),
            labels.to(device, non_blocking=True),
        )
        with _autocast(device, amp):
            logits = net(images)
            loss = criterion(logits, labels)
        loss_sum += float(loss) * labels.size(0)
        count += labels.size(0)
        filenames.extend(names)
        truths.append(labels.cpu().numpy())
        logits_all.append(logits.float().cpu().numpy())
    return (
        filenames,
        np.concatenate(truths),
        np.concatenate(logits_all),
        loss_sum / max(count, 1),
    )


def plot_curves(history, path, title):
    from matplotlib.figure import Figure

    frame = pd.DataFrame(history)
    fig = Figure(figsize=(11, 4))
    ax = fig.subplots(1, 2)
    ax[0].plot(frame.epoch, frame.train_loss, label="train")
    ax[0].plot(frame.epoch, frame.val_loss, label="val")
    ax[0].set(xlabel="Epoch", ylabel="Cross-entropy", title="Loss")
    ax[0].legend()
    ax[1].plot(frame.epoch, frame.val_macro_f1, label="val macro-F1")
    ax[1].plot(frame.epoch, frame.val_top1, label="val top-1")
    ax[1].set(xlabel="Epoch", ylabel="Score", title="Validation metrics")
    ax[1].legend()
    fig.suptitle(title)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    fig.clear()


def _save_checkpoint(
    path,
    net,
    optimizer,
    scheduler,
    scaler,
    ema,
    epoch,
    best_score,
    history,
    loader_generator=None,
):
    temporary = Path(path).with_suffix(".pt.tmp")
    torch.save(
        {
            "model": net.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "ema": ema.model.state_dict() if ema else None,
            "epoch": epoch,
            "best_score": best_score,
            "history": history,
            "rng": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all()
                if torch.cuda.is_available()
                else None,
                "loader": loader_generator.get_state()
                if loader_generator is not None
                else None,
            },
        },
        temporary,
    )
    temporary.replace(path)


def _load_checkpoint(
    path, net, optimizer, scheduler, scaler, ema, device, loader_generator=None
):
    state = torch.load(path, map_location=device, weights_only=False)
    net.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    scaler.load_state_dict(state["scaler"])
    if ema is not None and state.get("ema") is not None:
        ema.model.load_state_dict(state["ema"])
    rng = state.get("rng")
    if rng:
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        torch.set_rng_state(rng["torch"].cpu())
        if torch.cuda.is_available() and rng.get("cuda") is not None:
            torch.cuda.set_rng_state_all([value.cpu() for value in rng["cuda"]])
        if loader_generator is not None and rng.get("loader") is not None:
            loader_generator.set_state(rng["loader"].cpu())
    return int(state["epoch"]) + 1, float(state["best_score"]), state["history"]


def _sync_artifacts(source: Path, destination: str | None) -> None:
    if destination is None:
        return
    target = Path(destination) / source.resolve().relative_to(Path.cwd().resolve())
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    shutil.copy2(source, temporary)
    temporary.replace(target)


def training_signature(cfg):
    values = dataclasses.asdict(cfg) if isinstance(cfg, Config) else dict(cfg)
    ignored = {
        "save_test_predictions",
        "resume",
        "sync_dir",
        "out_dir",
        "pred_dir",
        "curve_dir",
        "images_dir",
        "labels_dir",
        "inference_method",
        "inference_aggregation",
        "inference_img_size",
        "temperature_scaling",
        "num_workers",
    }
    ignored.add("inference_amp")
    return {
        field.name: values.get(field.name, field.default)
        for field in dataclasses.fields(Config)
        if field.name not in ignored
    }


def run(cfg: Config) -> dict:
    if cfg.fold != 0:
        raise ValueError("This lab's core workflow is fixed to official fold 0")
    if cfg.epochs < 1 or cfg.batch_size < 1:
        raise ValueError("epochs and batch_size must be positive")
    set_seed(cfg.seed)
    output = run_dir(cfg)
    output.mkdir(parents=True, exist_ok=True)
    config_path = output / "config.json"
    if config_path.exists():
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        if training_signature(previous) != training_signature(cfg):
            raise ValueError(
                f"Configuration changed for {output}; use a new exp_id to preserve existing runs"
            )
    existing_test = pred_path(cfg, "test")
    if cfg.save_test_predictions and existing_test.exists():
        summary_path = output / "summary.json"
        if summary_path.exists():
            return json.loads(summary_path.read_text(encoding="utf-8"))
        raise RuntimeError(
            f"Test output already exists: {existing_test}; refusing a second test pass"
        )
    config_payload = dataclasses.asdict(cfg)
    config_payload["python"] = sys.version
    config_payload["torch"] = torch.__version__
    config_payload["cuda"] = torch.version.cuda
    config_path.write_text(json.dumps(config_payload, indent=2), encoding="utf-8")
    train_df, val_df, test_df = dataset.load_split(cfg.labels_dir, cfg.fold)
    split_report = dataset.check_split(train_df, val_df, test_df, cfg.images_dir)
    (output / "split_check.json").write_text(
        json.dumps(split_report, indent=2), encoding="utf-8"
    )
    train_loader = dataset.make_loader(
        train_df,
        cfg.images_dir,
        dataset.build_transforms(True, cfg.img_size, cfg.aug),
        cfg.batch_size,
        True,
        cfg.sampler,
        cfg.num_workers,
        cfg.seed,
    )
    inference_size = cfg.inference_img_size or cfg.img_size
    inference_amp = cfg.amp if cfg.inference_amp is None else cfg.inference_amp
    val_loader = dataset.make_loader(
        val_df,
        cfg.images_dir,
        dataset.build_transforms(False, cfg.img_size),
        cfg.batch_size,
        False,
        num_workers=cfg.num_workers,
        seed=cfg.seed,
    )
    test_loader = None
    if cfg.save_test_predictions:
        test_loader = dataset.make_loader(
            test_df,
            cfg.images_dir,
            dataset.build_transforms(False, inference_size),
            cfg.batch_size,
            False,
            num_workers=cfg.num_workers,
            seed=cfg.seed,
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_exists = cfg.resume and (output / "last.pt").exists()
    if cfg.sync_dir and cfg.resume:
        checkpoint_exists = (
            checkpoint_exists or (Path(cfg.sync_dir) / output / "last.pt").exists()
        )
    net = model_lib.build_model(
        cfg.backbone,
        pretrained=cfg.init != "scratch" and not checkpoint_exists,
        drop_rate=cfg.drop_rate,
        init=cfg.init,
    ).to(device)
    if checkpoint_exists and config_path.exists():
        net.pretrained_tag = (
            previous.get("pretrained_tag", "restored-checkpoint")
            if "previous" in locals()
            else "restored-checkpoint"
        )
    config_payload["pretrained_tag"] = net.pretrained_tag
    config_path.write_text(json.dumps(config_payload, indent=2), encoding="utf-8")
    counts = (
        train_df.Label.value_counts()
        .reindex(range(dataset.NUM_CLASSES), fill_value=0)
        .values
    )
    class_weight = None
    if cfg.loss == "ce_weighted" or cfg.class_weight_beta is not None:
        class_weight = losses.class_weights(counts, cfg.class_weight_beta or 0).to(
            device
        )
    criterion = losses.build_criterion(
        cfg.loss,
        smoothing=cfg.label_smoothing,
        gamma=cfg.focal_gamma,
        weight=class_weight,
    ).to(device)
    optimizer = build_optimizer(net, cfg)
    scheduler = build_scheduler(optimizer, cfg, len(train_loader))
    scaler = torch.amp.GradScaler(
        "cuda", enabled=bool(cfg.amp and device.type == "cuda")
    )
    ema = EMA(net, cfg.ema_decay) if cfg.ema_decay is not None else None
    start_epoch, best_score, history = 0, -1.0, []
    last_path, best_path = output / "last.pt", output / "best.pt"
    if cfg.sync_dir and cfg.resume:
        drive_run = Path(cfg.sync_dir) / output
        drive_run.mkdir(parents=True, exist_ok=True)
        if not last_path.exists() and (drive_run / "last.pt").exists():
            shutil.copy2(drive_run / "last.pt", last_path)
        if not best_path.exists() and (drive_run / "best.pt").exists():
            shutil.copy2(drive_run / "best.pt", best_path)
    if cfg.resume and last_path.exists():
        start_epoch, best_score, history = _load_checkpoint(
            last_path,
            net,
            optimizer,
            scheduler,
            scaler,
            ema,
            device,
            train_loader.generator,
        )
    started = time.perf_counter()
    durations = []
    for epoch in range(start_epoch, cfg.epochs):
        tick = time.perf_counter()
        train_values = train_one_epoch(
            net, train_loader, criterion, optimizer, scheduler, scaler, cfg, device, ema
        )
        eval_net = ema.model if ema is not None else net
        _, y_val, val_logits, val_loss = evaluate(
            eval_net, val_loader, criterion, device, cfg.amp
        )
        probs = torch.softmax(torch.from_numpy(val_logits), dim=1).numpy()
        metrics = compute_metrics(y_val, probs.argmax(axis=1), probs)
        duration = time.perf_counter() - tick
        durations.append(duration)
        record = {
            "epoch": epoch + 1,
            **train_values,
            "val_loss": val_loss,
            "val_macro_f1": float(metrics["macro_f1"]),
            "val_top1": float(metrics["top1"]),
            "val_balanced_acc": float(metrics["balanced_acc"]),
            "epoch_seconds": duration,
        }
        history.append(record)
        score = record["val_macro_f1"]
        if score > best_score:
            best_score = score
            _save_checkpoint(
                best_path,
                net,
                optimizer,
                scheduler,
                scaler,
                ema,
                epoch,
                best_score,
                history,
                train_loader.generator,
            )
        _save_checkpoint(
            last_path,
            net,
            optimizer,
            scheduler,
            scaler,
            ema,
            epoch,
            best_score,
            history,
            train_loader.generator,
        )
        pd.DataFrame(history).to_csv(output / "history.csv", index=False)
        for artifact in (
            last_path,
            best_path,
            output / "history.csv",
            config_path,
            output / "split_check.json",
        ):
            if artifact.exists():
                _sync_artifacts(artifact, cfg.sync_dir)
        print(
            f"{cfg.exp_id} seed={cfg.seed} epoch={epoch + 1}/{cfg.epochs} val_macro_f1={score:.5f} "
            f"epoch_s={duration:.1f}; estimated_remaining_min={(cfg.epochs - epoch - 1) * duration / 60:.1f}"
        )
    if not best_path.exists():
        raise RuntimeError(
            "Best validation checkpoint is missing; restore best.pt before inference"
        )
    best = torch.load(best_path, map_location=device, weights_only=False)
    net.load_state_dict(best["model"])
    final_net = net
    if ema is not None and best.get("ema") is not None:
        ema.model.load_state_dict(best["ema"])
        final_net = ema.model
    del train_loader
    val_loader = dataset.make_loader(
        val_df,
        cfg.images_dir,
        dataset.build_transforms(False, inference_size),
        cfg.batch_size,
        False,
        num_workers=cfg.num_workers,
        seed=cfg.seed,
    )
    names, y_val, val_logits = inference.predict_method_logits(
        final_net,
        val_loader,
        device,
        cfg.inference_method,
        cfg.inference_aggregation,
        inference_amp,
    )
    val_loss = float(
        F.cross_entropy(torch.from_numpy(val_logits), torch.from_numpy(y_val))
    )
    temperature = (
        inference.fit_temperature(val_logits, y_val) if cfg.temperature_scaling else 1.0
    )
    np.savez_compressed(
        output / "val_logits.npz",
        filenames=np.asarray(names),
        y_true=y_val,
        logits=val_logits,
    )
    probs = inference.apply_temperature(val_logits, temperature)
    save_predictions(pred_path(cfg, "val"), names, y_val, probs)
    # Persist validation and curve evidence before the final test pass starts.
    plot_curves(
        history,
        Path(cfg.curve_dir) / f"{cfg.exp_id}_seed{cfg.seed}.png",
        f"{cfg.exp_id} seed {cfg.seed}",
    )
    for artifact in (
        output / "val_logits.npz",
        pred_path(cfg, "val"),
        Path(cfg.curve_dir) / f"{cfg.exp_id}_seed{cfg.seed}.png",
    ):
        _sync_artifacts(artifact, cfg.sync_dir)
    final_metrics = compute_metrics(y_val, probs.argmax(1), probs)
    summary = {
        "status": "TRAINED_PENDING_TEST" if cfg.save_test_predictions else "COMPLETED",
        "exp_id": cfg.exp_id,
        "seed": cfg.seed,
        "best_epoch": int(best["epoch"]) + 1,
        "val_macro_f1": float(final_metrics["macro_f1"]),
        "val_top1": float(final_metrics["top1"]),
        "val_balanced_acc": float(final_metrics["balanced_acc"]),
        "val_ece": float(final_metrics["ece"]),
        "val_loss": float(val_loss),
        "train_seconds": sum(row["epoch_seconds"] for row in history),
        "mean_epoch_seconds": float(np.mean([row["epoch_seconds"] for row in history])),
        "session_seconds": time.perf_counter() - started,
        "temperature": temperature,
        "inference_method": cfg.inference_method,
        "params_m": model_lib.count_params(net),
        "gmac": model_lib.count_gmacs(net, cfg.img_size),
        "pretrained_tag": getattr(net, "pretrained_tag", "unknown"),
        "split": "official_fold_0",
        "gmac_method": getattr(net, "gmac_method", "unknown"),
        "inference_img_size": inference_size,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    _sync_artifacts(output / "summary.json", cfg.sync_dir)
    if test_loader is not None:
        cached_logits = output / "test_logits.npz"
        if cached_logits.exists():
            with np.load(cached_logits, allow_pickle=False) as saved_test:
                names_test = saved_test["filenames"].tolist()
                y_test = saved_test["y_true"]
                test_logits = saved_test["logits"]
            reference = test_df.set_index("Filename").Label.reindex(names_test)
            if (
                len(names_test) != len(test_df)
                or len(set(names_test)) != len(test_df)
                or not np.array_equal(reference.to_numpy(), y_test)
            ):
                raise ValueError(
                    "Saved test logits do not match the official test split"
                )
        else:
            names_test, y_test, test_logits = inference.predict_method_logits(
                final_net,
                test_loader,
                device,
                cfg.inference_method,
                cfg.inference_aggregation,
                inference_amp,
            )
        if cfg.temperature_scaling:
            save_predictions(
                Path(cfg.pred_dir) / f"{cfg.exp_id}uncal_seed{cfg.seed}_test.csv",
                names_test,
                y_test,
                inference.apply_temperature(test_logits, 1.0),
            )
        test_probs = inference.apply_temperature(test_logits, temperature)
        save_predictions(pred_path(cfg, "test"), names_test, y_test, test_probs)
        np.savez_compressed(
            output / "test_logits.npz",
            filenames=np.asarray(names_test),
            y_true=y_test,
            logits=test_logits,
        )
        # Sync test evidence immediately so a disconnected session never loses the one-time output.
        for artifact in (
            Path(cfg.pred_dir) / f"{cfg.exp_id}uncal_seed{cfg.seed}_test.csv",
            output / "test_logits.npz",
            pred_path(cfg, "test"),
        ):
            if artifact.exists():
                _sync_artifacts(artifact, cfg.sync_dir)
    summary["status"] = "COMPLETED"
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    pd.DataFrame(history).to_csv(output / "history.csv", index=False)
    plot_curves(
        history,
        Path(cfg.curve_dir) / f"{cfg.exp_id}_seed{cfg.seed}.png",
        f"{cfg.exp_id} seed {cfg.seed}",
    )
    for artifact in (
        output / "val_logits.npz",
        output / "summary.json",
        output / "history.csv",
        output / "best.pt",
        output / "last.pt",
        Path(cfg.curve_dir) / f"{cfg.exp_id}_seed{cfg.seed}.png",
        pred_path(cfg, "val"),
    ):
        if artifact.exists():
            _sync_artifacts(artifact, cfg.sync_dir)
    if cfg.save_test_predictions:
        for artifact in (
            output / "test_logits.npz",
            pred_path(cfg, "test"),
            Path(cfg.pred_dir) / f"{cfg.exp_id}uncal_seed{cfg.seed}_test.csv",
        ):
            if artifact.exists():
                _sync_artifacts(artifact, cfg.sync_dir)
    return summary


def parse_overrides(pairs):
    defaults = Config()
    values = {}
    known = {
        field.name: getattr(defaults, field.name)
        for field in dataclasses.fields(Config)
    }
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"Expected KEY=VALUE, got {pair!r}")
        key, raw = pair.split("=", 1)
        if key not in known:
            raise ValueError(f"Unknown Config field: {key}")
        default = known[key]
        if raw.lower() in {"none", "null"}:
            value = None
        elif isinstance(default, bool):
            if raw.lower() not in {"true", "false", "1", "0"}:
                raise ValueError(f"Expected boolean for {key}")
            value = raw.lower() in {"true", "1"}
        elif isinstance(default, int):
            value = int(raw)
        elif isinstance(default, float):
            value = float(raw)
        elif key in {"mix_alpha", "focal_gamma", "ema_decay", "class_weight_beta"}:
            value = float(raw)
        elif key == "inference_amp":
            if raw.lower() not in {"true", "false", "1", "0"}:
                raise ValueError("Expected boolean for inference_amp")
            value = raw.lower() in {"true", "1"}
        elif key == "inference_img_size":
            value = int(raw)
        else:
            value = raw
        values[key] = value
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--set", nargs="*", default=[])
    args = parser.parse_args()
    print(json.dumps(run(Config(**parse_overrides(args.set))), indent=2))


if __name__ == "__main__":
    main()
