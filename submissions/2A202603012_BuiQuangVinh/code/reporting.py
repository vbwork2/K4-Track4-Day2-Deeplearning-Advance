"""Vietnamese reports and per-seed error analysis from saved predictions."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from PIL import Image

import dataset
from eval import compute_metrics, load_group


def markdown_table(rows, columns):
    if not rows:
        return "PENDING: chưa có kết quả."
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        values = []
        for column in columns:
            value = row.get(column)
            if value is None or (isinstance(value, float) and not np.isfinite(value)):
                text = "PENDING"
            elif isinstance(value, float):
                text = f"{value:.4f}"
            else:
                text = str(value)
            values.append(text.replace("|", "/").replace("\n", " "))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def analyze_tests(
    pred_dir="predictions",
    labels_dir="data/labels",
    out_dir="eval_out",
    images_dir=None,
):
    from workflow import save_json

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    records, errors = [], []
    for tag in ("T00", "F01"):
        group = load_group(
            str(Path(pred_dir) / f"{tag}_seed*_test.csv"),
            str(Path(labels_dir) / "test_subset0.csv"),
        )
        class_values = []
        for prediction in group.preds:
            metrics = compute_metrics(
                prediction.y_true, prediction.y_pred, prediction.probs
            )
            class_values.append(metrics)
            for index, name in enumerate(dataset.CLASS_NAMES):
                records.append(
                    {
                        "exp_id": tag,
                        "seed": prediction.seed,
                        "class": name,
                        "support": int(metrics["support"][index]),
                        "precision": float(metrics["precision"][index]),
                        "recall": float(metrics["recall"][index]),
                        "f1": float(metrics["f1"][index]),
                    }
                )
            if prediction.seed == 0:
                matrix = metrics["confusion"]
                pd.DataFrame(
                    matrix, index=dataset.CLASS_NAMES, columns=dataset.CLASS_NAMES
                ).to_csv(out_dir / f"{tag}_seed0_confusion.csv")
                fig = Figure(figsize=(9, 8))
                axis = fig.subplots()
                image = axis.imshow(matrix, cmap="Blues")
                fig.colorbar(image, ax=axis)
                axis.set_xticks(range(9), dataset.CLASS_NAMES, rotation=70, ha="right")
                axis.set_yticks(range(9), dataset.CLASS_NAMES)
                axis.set(
                    xlabel="Predicted",
                    ylabel="True",
                    title=f"{tag} seed 0: test confusion",
                )
                fig.tight_layout()
                fig.savefig(out_dir / f"{tag}_seed0_confusion.png", dpi=150)
            if tag == "F01":
                for actual, predicted in ((0, 7), (7, 0)):
                    mask = (prediction.y_true == actual) & (
                        prediction.y_pred == predicted
                    )
                    for index in np.flatnonzero(mask)[:10]:
                        errors.append(
                            {
                                "seed": prediction.seed,
                                "Filename": str(prediction.filenames[index]),
                                "direction": f"{dataset.CLASS_NAMES[actual]} -> {dataset.CLASS_NAMES[predicted]}",
                                "confidence": float(prediction.probs[index].max()),
                            }
                        )
        for index, name in enumerate(dataset.CLASS_NAMES):
            values = np.array([item["f1"][index] for item in class_values])
            records.append(
                {
                    "exp_id": tag,
                    "seed": "mean",
                    "class": name,
                    "support": int(class_values[0]["support"][index]),
                    "precision": float(
                        np.mean([item["precision"][index] for item in class_values])
                    ),
                    "recall": float(
                        np.mean([item["recall"][index] for item in class_values])
                    ),
                    "f1": float(values.mean()),
                    "f1_std": float(values.std(ddof=1)),
                }
            )
    save_json("per_class.json", records)
    pd.DataFrame(
        errors, columns=["seed", "Filename", "direction", "confidence"]
    ).to_csv(out_dir / "cross_class_errors.csv", index=False)
    if images_dir and errors:
        examples = errors[:8]
        fig = Figure(figsize=(10, 3 * len(examples)))
        axes = np.atleast_1d(fig.subplots(len(examples), 1))
        for axis, row in zip(axes, examples):
            with Image.open(Path(images_dir) / row["Filename"]) as image:
                axis.imshow(image.convert("RGB"))
            axis.set_title(
                f"Seed {row['seed']}: {row['direction']} | {row['Filename']}"
            )
            axis.axis("off")
        fig.tight_layout()
        fig.savefig(out_dir / "cross_class_error_examples.png", dpi=140)
    return records


def generate_report(path="report.md", results_path="results.xlsx"):
    from artifacts import collect_runs

    runs = collect_runs()
    inference_path = Path("inference_results.json")
    inference_rows = (
        json.loads(inference_path.read_text(encoding="utf-8"))
        if inference_path.exists()
        else []
    )
    latency_path = Path("latency.json")
    latency = (
        json.loads(latency_path.read_text(encoding="utf-8"))
        if latency_path.exists()
        else []
    )
    sanity_path = Path("sanity_checks.json")
    sanity = (
        json.loads(sanity_path.read_text(encoding="utf-8"))
        if sanity_path.exists()
        else {"status": "PENDING"}
    )
    lock_path = Path("FINAL_CONFIG_LOCKED.json")
    lock = (
        json.loads(lock_path.read_text(encoding="utf-8"))
        if lock_path.exists()
        else None
    )
    split_path = next(
        (
            Path(row["run_dir"]) / "split_check.json"
            for row in runs
            if (Path(row["run_dir"]) / "split_check.json").exists()
        ),
        None,
    )
    split = json.loads(split_path.read_text(encoding="utf-8")) if split_path else {}
    test_rows = []
    final_complete = all(
        (Path("predictions") / f"{tag}_seed{seed}_test.csv").exists()
        for tag in ("T00", "F01")
        for seed in (0, 1, 2)
    )
    if final_complete:
        for tag in ("T00", "F01"):
            group = load_group(str(Path("predictions") / f"{tag}_seed*_test.csv"), None)
            for key in ("macro_f1", "top1", "ece"):
                values = [
                    compute_metrics(item.y_true, item.y_pred, item.probs)[key]
                    for item in group.preds
                ]
                test_rows.append(
                    {
                        "cấu hình": tag,
                        "chỉ số": key,
                        "mean": float(np.mean(values)),
                        "std": float(np.std(values, ddof=1)),
                        "seeds": len(values),
                    }
                )
    phase_complete = final_complete and sanity.get("status") == "PASS" and bool(lock)
    status = (
        "EXPERIMENTS COMPLETED"
        if phase_complete
        else "EXPERIMENTS PENDING / IN PROGRESS"
    )
    backbone_rows = [row for row in runs if str(row.get("exp_id", "")).startswith("B")]
    training_rows = [row for row in runs if str(row.get("exp_id", "")).startswith("T")]
    per_class_path = Path("per_class.json")
    per_class = (
        json.loads(per_class_path.read_text(encoding="utf-8"))
        if per_class_path.exists()
        else []
    )
    class_means = [row for row in per_class if row.get("seed") == "mean"]
    selected_latency = next(
        (
            row
            for row in latency
            if str(row.get("configuration", "")).startswith("F01 locked")
        ),
        None,
    )
    delta_text = "PENDING: cần đủ 3 seed cho baseline và final."
    scores = {
        row["cấu hình"]: row["mean"] for row in test_rows if row["chỉ số"] == "macro_f1"
    }
    if len(scores) == 2:
        delta_text = f"Chênh macro-F1 test F01 - T00 = {scores['F01'] - scores['T00']:+.4f}. Đối chiếu với std giữa các seed."
    deployment = "PENDING: cần latency và kết quả cuối."
    if selected_latency:
        p95 = selected_latency["p95"]
        deployment = f"p95 batch 1 = {p95:.3f} ms trên {selected_latency['gpu']}; "
        deployment += (
            "đạt mục tiêu 100 ms trong điều kiện đo."
            if p95 <= 100
            else "vượt mục tiêu 100 ms."
        )
    report = f"""# DeepWeeds Lab Day 2 - 2A202603012 Bui Quang Vinh

Trạng thái: **{status}**. Các số dưới đây lấy từ artifact thật; phần thiếu giữ PENDING.

## Tóm tắt

So sánh backbone, công thức huấn luyện và inference trên fold 0. Chọn cấu hình bằng validation, khóa cấu hình rồi mới chạy test với seed 0, 1, 2.

{markdown_table(test_rows, ["cấu hình", "chỉ số", "mean", "std", "seeds"])}

{delta_text}

## Dữ liệu và thiết lập

- DeepWeeds: 9 lớp, 17.509 ảnh. Train/val/test dùng CSV fold 0 chính thức.
- Số ảnh từng tập: `{split.get("n", "PENDING")}`. Giao các tập: `{split.get("overlap", "PENDING")}`.
- Sai lệch nhãn đã xác minh từ nguồn: `{split.get("official_label_discrepancies", "PENDING")}`. Giữ nguyên nhãn split và kiểm tra checksum.
- Ảnh đọc từ local Colab; checkpoint, log và output lưu trên Drive.
- Phiên bản thư viện, pretrained tag, epoch và batch được ghi trong `config.json` từng run.
- Biểu đồ EDA: `curves/fold0_class_counts.png`, `curves/fold0_examples.png`.

## Kiểm tra pipeline

Trạng thái: **{sanity.get("status", "PENDING")}**. Kiểm tra output 9 lớp, focal gamma 0, Mixup/CutMix, optimizer groups, frozen BatchNorm, fusion và overfit batch nhỏ.

Loss batch nhỏ: `{sanity.get("tiny_batch_first_loss", "PENDING")}` -> `{sanity.get("tiny_batch_last_loss", "PENDING")}`.

## So sánh backbone

{markdown_table(backbone_rows, ["exp_id", "backbone", "pretrained_tag", "params_m", "gmac", "val_macro_f1", "best_val_top1", "mean_epoch_seconds", "latency_p95_ms"])}

GMAC dùng Conv/Linear và tích ma trận attention của ViT; không tính các phép toán elementwise. Chọn winner bằng macro-F1 validation.

## Ablation huấn luyện

{markdown_table(training_rows, ["exp_id", "backbone", "init", "aug", "loss", "mix", "ema_decay", "val_macro_f1", "best_val_top1"])}

Các lần chạy đơn yếu tố so với T00 dùng cùng backbone winner. T_COMBO kết hợp augmentation, loss và EMA được chọn từ validation. Screening một seed chỉ mang tính thăm dò.

## Inference và hiệu chuẩn

{markdown_table(inference_rows, ["exp_id", "method", "aggregation", "img_size", "macro_f1_val", "ece_val", "p95_ms", "status"])}

Temperature chỉ fit trên validation. TTA và latency dùng cùng hàm tạo view. Các phương pháp không chạy được được ghi FAILED cùng nguyên nhân.

## Độ trễ

{markdown_table(latency, ["configuration", "gpu", "dtype", "batch", "img_size", "p50", "p95", "p99", "images_per_s"])}

Đo ít nhất 10 warmup và 50 lượt, đồng bộ GPU. Không tính đọc ảnh và resize ảnh từ đĩa; tính tạo view và forward trên tensor đã chuẩn bị.

## Kết quả cuối và phân tích lỗi

{markdown_table(class_means, ["exp_id", "class", "support", "precision", "recall", "f1", "f1_std"])}

F1 từng lớp là mean giữa các seed, không gộp thành một ensemble khác cấu hình đã khóa. Ma trận nhầm lẫn minh họa dùng seed 0. Ảnh nhầm Chinee Apple/Snake Weed nằm trong `curves/cross_class_error_examples.png` nếu có.

## Kết luận

{delta_text}

{deployment}

## Hạn chế

- Chỉ dùng fold 0; split không tách theo địa điểm, nên kết quả có thể lạc quan khi triển khai nơi khác.
- Screening một seed; final ba seed. Chênh lệch nhỏ cần được đối chiếu với nhiễu giữa các seed.
- Ngân sách 10 epoch ngắn hơn nghiên cứu gốc; không suy ra rằng backbone hoặc recipe nào luôn tốt nhất.
- Latency phụ thuộc GPU và batch; cần đo lại trên phần cứng triển khai.

## Tái lập

Mở `code/lab_day2_colab.ipynb`, bật GPU và chạy từ SECTION 00. Upload `images.zip` vào `MyDrive/K4_Track4_Day2/data/images.zip`. Chi tiết cấu hình và khóa final nằm trong artifact trên Drive. Bảng đầy đủ: `{Path(results_path).name}`.
"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(report, encoding="utf-8")
    return str(target)
