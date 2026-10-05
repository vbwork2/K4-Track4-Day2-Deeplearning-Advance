"""Colab setup, durable experiments, validation selection, and final locking."""

from __future__ import annotations

import dataclasses
import gc
import hashlib
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import torch

import benchmark
import dataset
import inference
import model
import train
from eval import check_against_csv, compute_metrics, read_pred

LABEL_HASHES = {
    **dataset.OFFICIAL_CONFLICT_HASHES,
    "val_subset0.csv": "1fa70a03a68def06c921a1dd7f0b53e96490f6dfad51d7d423bb145161ebc478",
    "test_subset0.csv": "fb212793b988bc0455aae2e48cb9eb3168014c2f47fbdb93f5f1e284e0974f75",
}


def file_digest(path, algorithm="sha256"):
    digest = hashlib.new(algorithm)
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def prepare_labels(labels_dir):
    labels_dir = Path(labels_dir)
    labels_dir.mkdir(parents=True, exist_ok=True)
    for filename, digest in LABEL_HASHES.items():
        target = labels_dir / filename
        if target.exists() and file_digest(target) == digest:
            continue
        temporary = target.with_suffix(".csv.download")
        url = (
            "https://raw.githubusercontent.com/AlexOlsen/DeepWeeds/master/labels/"
            + filename
        )
        urllib.request.urlretrieve(url, temporary)
        if file_digest(temporary) != digest:
            raise RuntimeError(
                f"Checksum CSV không khớp: {filename}. Kiểm tra phiên bản nguồn DeepWeeds."
            )
        temporary.replace(target)
    return dataset.load_split(labels_dir)


def prepare_images(zip_path, image_dir, filenames):
    zip_path, image_dir = Path(zip_path), Path(image_dir)
    image_dir.mkdir(parents=True, exist_ok=True)
    names = set(map(str, filenames))
    if len(names) != dataset.EXPECTED_TOTAL:
        raise ValueError("Danh sách ảnh không đủ 17.509 tên duy nhất.")
    missing = {
        name
        for name in names
        if not (image_dir / name).is_file() or (image_dir / name).stat().st_size == 0
    }
    if not missing:
        print("Đã có đủ ảnh local; bỏ qua giải nén.")
        return
    if not zip_path.is_file():
        raise FileNotFoundError(
            "Upload images.zip vào MyDrive/K4_Track4_Day2/data/images.zip rồi chạy lại cell này."
        )
    local_zip = image_dir.parent / "images.zip"
    marker = image_dir.parent / "images_zip_verified.json"
    signature = {
        "size": zip_path.stat().st_size,
        "mtime_ns": zip_path.stat().st_mtime_ns,
    }
    cached = json.loads(marker.read_text(encoding="utf-8")) if marker.exists() else None
    if not local_zip.exists() or cached != signature:
        temporary = local_zip.with_suffix(".zip.download")
        shutil.copy2(zip_path, temporary)
        if file_digest(temporary, "md5") != "b7b30f96d466fba86016aa5a26606e0f":
            raise ValueError("MD5 images.zip không khớp bản DeepWeeds chính thức.")
        temporary.replace(local_zip)
        save_json(marker, signature)
    with zipfile.ZipFile(local_zip) as archive:
        entries = {}
        for entry in archive.infolist():
            name = Path(entry.filename).name
            if name not in names or entry.is_dir():
                continue
            if name in entries:
                raise ValueError(f"ZIP chứa tên ảnh trùng: {name}")
            entries[name] = entry
        if set(entries) != names:
            raise ValueError("Tên ảnh trong ZIP không khớp các CSV fold 0.")
        for name in sorted(missing):
            target = image_dir / name
            temporary = target.with_suffix(".jpg.download")
            with archive.open(entries[name]) as source, temporary.open("wb") as output:
                shutil.copyfileobj(source, output)
            temporary.replace(target)
    print(f"Đã khôi phục {len(missing):,} ảnh; thư mục local: {image_dir}")


def restore_outputs(drive_root):
    drive_root = Path(drive_root)
    for folder in ("runs", "predictions", "curves"):
        Path(folder).mkdir(parents=True, exist_ok=True)
        remote = drive_root / folder
        for source in remote.rglob("*"):
            if not source.is_file() or source.suffix in {
                ".pt",
                ".pth",
                ".ckpt",
                ".tmp",
            }:
                continue
            destination = Path(folder) / source.relative_to(remote)
            destination.parent.mkdir(parents=True, exist_ok=True)
            if (
                not destination.exists()
                or source.stat().st_mtime_ns > destination.stat().st_mtime_ns
            ):
                shutil.copy2(source, destination)
    for filename in (
        "inference_results.json",
        "inference_cache.json",
        "latency.json",
        "latency_cache.json",
        "FINAL_CONFIG_LOCKED.json",
        "per_class.json",
        "sanity_checks.json",
    ):
        source = drive_root / "results" / filename
        if source.exists():
            shutil.copy2(source, Path(filename))


def restore_checkpoint(cfg, name="best.pt"):
    destination = train.run_dir(cfg) / name
    if not destination.exists() and cfg.sync_dir:
        source = Path(cfg.sync_dir) / destination
        if source.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
    if not destination.exists():
        raise FileNotFoundError(f"Thiếu checkpoint: {destination}")
    return destination


def read_run_config(summary):
    path = Path("runs") / summary["exp_id"] / f"seed{summary['seed']}" / "config.json"
    saved = json.loads(path.read_text(encoding="utf-8"))
    fields = {field.name for field in dataclasses.fields(train.Config)}
    return train.Config(**{key: value for key, value in saved.items() if key in fields})


def run_or_resume(cfg, force=False):
    if force:
        if list(Path(cfg.pred_dir).glob("*_test.csv")):
            raise RuntimeError("Không FORCE_RERUN sau khi đã có test output.")
        cfg = dataclasses.replace(cfg, resume=False)
    path = train.run_dir(cfg)
    summary_path = path / "summary.json"
    config_path = path / "config.json"
    if config_path.exists():
        saved = json.loads(config_path.read_text(encoding="utf-8"))
        if train.training_signature(saved) != train.training_signature(cfg):
            raise ValueError(
                f"Cấu hình {cfg.exp_id} đã đổi. Dùng exp_id mới để giữ kết quả cũ."
            )
    if cfg.save_test_predictions and train.pred_path(cfg, "test").exists():
        check_against_csv(
            read_pred(str(train.pred_path(cfg, "test"))),
            str(Path(cfg.labels_dir) / "test_subset0.csv"),
        )
        if not summary_path.exists():
            raise RuntimeError(
                "Đã có dự đoán test nhưng thiếu summary; cần phục hồi, không chạy lại test."
            )
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("status") == "TRAINED_PENDING_TEST":
            summary["status"] = "COMPLETED"
            save_json(summary_path, summary)
            if cfg.sync_dir:
                save_json(Path(cfg.sync_dir) / summary_path, summary)
        return summary
    required = [
        path / "history.csv",
        path / "val_logits.npz",
        train.pred_path(cfg, "val"),
        Path(cfg.curve_dir) / f"{cfg.exp_id}_seed{cfg.seed}.png",
    ]
    if (
        not force
        and summary_path.exists()
        and config_path.exists()
        and all(item.exists() for item in required)
    ):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("status") == "COMPLETED" and not cfg.save_test_predictions:
            check_against_csv(
                read_pred(str(train.pred_path(cfg, "val"))),
                str(Path(cfg.labels_dir) / "val_subset0.csv"),
                "val",
            )
            restore_checkpoint(cfg)
            print(f"Bỏ qua {cfg.exp_id}, seed {cfg.seed}: đủ artifact.")
            return summary
    if not cfg.save_test_predictions and list(Path(cfg.pred_dir).glob("*_test.csv")):
        raise RuntimeError("Đã có test output. Không được chạy lại bước chọn cấu hình.")
    try:
        return train.run(cfg)
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def choose_training_winner(rows):
    candidates = [
        row
        for row in rows
        if row.get("status") == "COMPLETED"
        and np.isfinite(row.get("val_macro_f1", np.nan))
    ]
    if not candidates:
        raise ValueError("Chưa có experiment validation hoàn thành.")
    return max(
        candidates,
        key=lambda row: (row["val_macro_f1"], -row.get("params_m", float("inf"))),
    )


def load_model_for_run(cfg, device="cuda", use_ema=True):
    checkpoint = torch.load(
        restore_checkpoint(cfg), map_location=device, weights_only=False
    )
    net = model.build_model(
        cfg.backbone, pretrained=False, init=cfg.init, drop_rate=cfg.drop_rate
    )
    weights = (
        checkpoint.get("ema")
        if use_ema and checkpoint.get("ema") is not None
        else checkpoint["model"]
    )
    net.load_state_dict(weights)
    return net.to(device).eval()


def inference_experiments(cfg, val_df, device="cuda"):
    specs = [
        ("I00", "fp32", "prob", 224, False),
        ("I01", "hflip", "prob", 224, False),
        ("I02", "multicrop", "prob", 224, False),
        ("I03", "hflip", "logit", 224, False),
        ("I04", "multiscale", "prob", 224, False),
        ("I05", "single", "prob", 224, True),
        ("I06", "fp32", "prob", 256, False),
        ("I07", "fp32", "prob", 288, False),
        ("I08", "temperature_scaling", "prob", 224, False),
    ]
    checkpoint = restore_checkpoint(cfg)
    identity = {
        "checkpoint_sha256": file_digest(checkpoint),
        "specs": specs,
        "inference_source": file_digest(Path(inference.__file__)),
        "benchmark_source": file_digest(Path(benchmark.__file__)),
        "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else "CPU",
    }
    identity_key = hashlib.sha256(
        json.dumps(identity, sort_keys=True).encode()
    ).hexdigest()
    cache_path = Path("inference_cache.json")
    cache = (
        json.loads(cache_path.read_text(encoding="utf-8"))
        if cache_path.exists()
        else {}
    )
    rows = cache.get("rows", []) if cache.get("key") == identity_key else []
    if rows and len(rows) == len(specs):
        return rows
    if list(Path(cfg.pred_dir).glob("*_test.csv")):
        raise RuntimeError(
            "Không chạy thêm inference ablation sau khi đã có test output."
        )
    net = load_model_for_run(cfg, device)
    for tag, method, aggregation, size, amp in specs:
        if any(row["exp_id"] == tag for row in rows):
            continue
        loader = dataset.make_loader(
            val_df,
            cfg.images_dir,
            dataset.build_transforms(False, size),
            min(cfg.batch_size, 32),
            False,
            num_workers=cfg.num_workers,
        )
        actual_method = "fp32" if method == "temperature_scaling" else method
        row = {
            "exp_id": tag,
            "method": method,
            "inference_method": actual_method,
            "aggregation": aggregation,
            "img_size": size,
            "amp": amp,
            "checkpoint": str(checkpoint),
            "temperature_scaling": method == "temperature_scaling",
        }
        try:
            _, labels, logits = inference.predict_method_logits(
                net, loader, device, actual_method, aggregation, amp
            )
            temperature = (
                inference.fit_temperature(logits, labels)
                if row["temperature_scaling"]
                else 1.0
            )
            probs = inference.apply_temperature(logits, temperature)
            metrics = compute_metrics(labels, probs.argmax(1), probs)
            timing = benchmark.method_latency_report(
                net, actual_method, aggregation, 1, size, amp, temperature, device
            )
            row.update(
                {
                    "macro_f1_val": metrics["macro_f1"],
                    "top1_val": metrics["top1"],
                    "balanced_acc_val": metrics["balanced_acc"],
                    "ece_val": metrics["ece"],
                    "ece_uncal_val": compute_metrics(
                        labels,
                        logits.argmax(1),
                        inference.apply_temperature(logits, 1.0),
                    )["ece"],
                    "temperature": temperature,
                    "p50_ms": timing["p50"],
                    "p95_ms": timing["p95"],
                    "p99_ms": timing["p99"],
                    "dtype": timing["dtype"],
                    "views": timing["views"],
                    "images_per_s": timing["images_per_s"],
                    "status": "RECORDED",
                }
            )
        except (RuntimeError, ValueError) as error:
            if tag == "I00":
                raise
            row.update({"status": "FAILED", "reason": str(error)})
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        rows.append(row)
        print(tag, method, row["status"])
        save_json(cache_path, {"key": identity_key, "rows": rows})
        save_json("inference_results.json", rows)
        if cfg.sync_dir:
            save_json(
                Path(cfg.sync_dir) / "results" / cache_path.name,
                {"key": identity_key, "rows": rows},
            )
            save_json(Path(cfg.sync_dir) / "results" / "inference_results.json", rows)
    del net
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows


def lock_final(selected_cfg, baseline_cfg, inference_rows, drive_root):
    path = Path("FINAL_CONFIG_LOCKED.json")
    existing = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    if existing and existing.get("schema_version") == 2:
        if existing.get("status") != "LOCKED_FROM_VALIDATION":
            raise ValueError("File khóa final không hợp lệ.")
        return existing
    if list(Path(selected_cfg.pred_dir).glob("*_test.csv")):
        raise RuntimeError("Đã có test output. Không tạo hoặc nâng cấp khóa final.")
    valid = [row for row in inference_rows if row.get("status") == "RECORDED"]
    if len({row["method"] for row in valid} - {"fp32", "single"}) < 4:
        raise RuntimeError("Cần đủ ít nhất 4 phương pháp inference ngoài baseline.")
    choice = max(
        valid, key=lambda row: (row["macro_f1_val"], -row["ece_val"], -row["p95_ms"])
    )
    lock = {
        "schema_version": 2,
        "status": "LOCKED_FROM_VALIDATION",
        "created_before_test": True,
        "seeds": [0, 1, 2],
        "source_exp_id": selected_cfg.exp_id,
        "training_config": dataclasses.asdict(selected_cfg),
        "baseline_config": dataclasses.asdict(baseline_cfg),
        "selected_inference_row": choice,
        "inference_method": choice["inference_method"],
        "validation_macro_f1": choice["macro_f1_val"],
    }
    save_json(Path(drive_root) / "results" / path.name, lock)
    save_json(path, lock)
    return lock


def final_config(lock, seed, is_final, drive_root, images_dir, labels_dir):
    if (
        lock.get("schema_version") != 2
        or lock.get("status") != "LOCKED_FROM_VALIDATION"
    ):
        raise ValueError("Chạy FINAL CONFIG LOCK trước phase cuối.")
    if seed not in lock["seeds"]:
        raise ValueError("Seed không thuộc cấu hình đã khóa.")
    values = dict(lock["training_config"] if is_final else lock["baseline_config"])
    values.update(
        {
            "exp_id": "F01" if is_final else "T00",
            "seed": seed,
            "images_dir": str(images_dir),
            "labels_dir": str(labels_dir),
            "sync_dir": str(drive_root),
            "save_test_predictions": True,
        }
    )
    if is_final:
        choice = lock["selected_inference_row"]
        values.update(
            {
                "inference_method": choice["inference_method"],
                "inference_aggregation": choice["aggregation"],
                "inference_img_size": choice["img_size"],
                "inference_amp": choice["amp"],
                "temperature_scaling": choice["temperature_scaling"],
            }
        )
    else:
        values.update(
            {
                "inference_method": "single",
                "inference_aggregation": "prob",
                "inference_img_size": values["img_size"],
                "temperature_scaling": False,
            }
        )
    return train.Config(**values)


def final_latency(cfg, device="cuda"):
    checkpoint = restore_checkpoint(cfg)
    summary = json.loads(
        (train.run_dir(cfg) / "summary.json").read_text(encoding="utf-8")
    )
    size = cfg.inference_img_size or cfg.img_size
    amp = cfg.amp if cfg.inference_amp is None else cfg.inference_amp
    identity = {
        "checkpoint": file_digest(checkpoint),
        "method": cfg.inference_method,
        "aggregation": cfg.inference_aggregation,
        "size": size,
        "amp": amp,
        "temperature": summary["temperature"],
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else "CPU",
        "source": file_digest(Path(benchmark.__file__)),
    }
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    cache_path = Path("latency_cache.json")
    cached = (
        json.loads(cache_path.read_text(encoding="utf-8"))
        if cache_path.exists()
        else {}
    )
    if cached.get("key") == key:
        return cached["rows"]
    net = load_model_for_run(cfg, device)
    rows = []
    for batch in (1, 32):
        timing = benchmark.method_latency_report(
            net,
            cfg.inference_method,
            cfg.inference_aggregation,
            batch,
            size,
            amp,
            summary["temperature"],
            device,
        )
        timing["configuration"] = "F01 locked inference: " + cfg.inference_method
        rows.append(timing)
    save_json(cache_path, {"key": key, "rows": rows})
    save_json("latency.json", rows)
    if cfg.sync_dir:
        save_json(
            Path(cfg.sync_dir) / "results" / cache_path.name, {"key": key, "rows": rows}
        )
        save_json(Path(cfg.sync_dir) / "results" / "latency.json", rows)
    del net
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows
