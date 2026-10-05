"""Build an auditable workbook and report from recorded experiment artifacts."""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
from eval import compute_metrics, ece_score, read_pred

SHEETS = [
    "Backbones",
    "Training",
    "Inference",
    "Final",
    "PerClass",
    "Latency",
    "Summary",
]


def collect_runs(root="runs"):
    rows = []
    for config_path in sorted(Path(root).glob("*/*/config.json")):
        run_path = config_path.parent
        summary_path = run_path / "summary.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        summary = (
            json.loads(summary_path.read_text(encoding="utf-8"))
            if summary_path.exists()
            else {}
        )
        history_path = run_path / "history.csv"
        history = pd.read_csv(history_path) if history_path.exists() else pd.DataFrame()
        row = {
            **config,
            **summary,
            "run_dir": str(run_path),
            "best_val_macro_f1": summary.get("val_macro_f1", np.nan),
            "best_val_top1": summary.get(
                "val_top1",
                float(
                    history.loc[
                        history.epoch == summary.get("best_epoch"), "val_top1"
                    ].iloc[0]
                )
                if not history.empty
                and (history.epoch == summary.get("best_epoch")).any()
                else np.nan,
            ),
            "epoch_seconds": summary.get("mean_epoch_seconds", np.nan),
        }
        rows.append(row)
    return rows


def _ablation_description(row, baseline=None):
    default = {
        "init": "finetune",
        "aug": "basic",
        "loss": "ce",
        "sampler": None,
        "mix": None,
        "ema_decay": None,
        "lr_backbone": 1e-4,
        "lr_head": 1e-3,
        "weight_decay": 0.05,
        "img_size": 224,
        "epochs": 10,
    }
    if baseline:
        default.update(
            {key: baseline.get(key, value) for key, value in default.items()}
        )
    changes = [
        f"{key}: {default[key]} → {row.get(key)}"
        for key in default
        if row.get(key, default[key]) != default[key]
    ]
    fields = [part.split(":", 1)[0] for part in changes]
    axis_by_field = {
        "init": "A",
        "aug": "B",
        "loss": "C",
        "sampler": "D",
        "lr_backbone": "E",
        "lr_head": "E",
        "weight_decay": "F",
        "ema_decay": "F",
        "img_size": "G",
        "epochs": "G",
        "mix": "B",
    }
    axes = sorted({axis_by_field[field] for field in fields})
    return ",".join(axes) if axes else "baseline", "; ".join(
        changes
    ) if changes else "baseline recipe"


def generate_results(
    output="results.xlsx",
    runs_dir="runs",
    pred_dir="predictions",
    latency_file="latency.json",
):
    runs = collect_runs(runs_dir)
    run_frame = pd.DataFrame(runs)
    baseline = next(
        (row for row in runs if row.get("exp_id") == "T00" and row.get("seed") == 0),
        None,
    )
    backbones, training, final = [], [], []
    for row in runs:
        if row.get("exp_id", "").startswith("B"):
            backbones.append(
                {
                    "exp_id": row["exp_id"],
                    "backbone": row.get("backbone"),
                    "pretrained tag": row.get("pretrained_tag", "PENDING"),
                    "params (M)": row.get("params_m"),
                    "GMAC": row.get("gmac"),
                    "resolution": row.get("img_size"),
                    "epochs": row.get("epochs"),
                    "seed": row.get("seed"),
                    "macro-F1 val": row.get("val_macro_f1"),
                    "top-1 val": row.get("best_val_top1"),
                    "train seconds/epoch": row.get("epoch_seconds"),
                    "latency batch-1 (ms)": row.get("latency_p95_ms", np.nan),
                    "notes": row.get("status", "RECORDED"),
                }
            )
        elif row.get("exp_id", "").startswith("T"):
            axis, change = _ablation_description(row, baseline)
            training.append(
                {
                    "exp_id": row["exp_id"],
                    "backbone": row.get("backbone"),
                    "axis": axis,
                    "change vs T00": change,
                    "seed": row.get("seed"),
                    "macro-F1 val": row.get("val_macro_f1"),
                    "top-1 val": row.get("best_val_top1"),
                    "delta vs T00": np.nan,
                    "notes": row.get("status", "RECORDED"),
                }
            )
    baseline_val = {
        row.get("seed"): row.get("val_macro_f1")
        for row in runs
        if row.get("exp_id") == "T00"
    }
    for row in training:
        reference = baseline_val.get(row.get("seed"))
        if reference is not None and pd.notna(row.get("macro-F1 val")):
            row["delta vs T00"] = row["macro-F1 val"] - reference
    for exp in ("T00", "F01", "F01uncal"):
        group_rows = []
        for seed in (0, 1, 2):
            path = Path(pred_dir) / f"{exp}_seed{seed}_test.csv"
            if path.exists():
                prediction = read_pred(str(path))
                probs = prediction.probs
                metrics = compute_metrics(prediction.y_true, prediction.y_pred, probs)
                run_match = next(
                    (
                        item
                        for item in runs
                        if item.get("exp_id") == exp
                        and int(item.get("seed", -1)) == seed
                    ),
                    {},
                )
                row = {
                    "exp_id": exp,
                    "seed": seed,
                    "macro-F1 val": run_match.get("val_macro_f1"),
                    "macro-F1 test": metrics["macro_f1"],
                    "top-1 test": metrics["top1"],
                    "ECE test": ece_score(probs, prediction.y_true),
                    "configuration": json.dumps(
                        {
                            key: run_match.get(key)
                            for key in (
                                "backbone",
                                "aug",
                                "loss",
                                "ema_decay",
                                "inference_method",
                                "inference_img_size",
                            )
                        }
                    ),
                    "status": "RECORDED",
                }
                final.append(row)
                group_rows.append(row)
        if len(group_rows) >= 3:
            final.append(
                {
                    "exp_id": exp,
                    "seed": "mean ± sample std",
                    "macro-F1 test": f"{np.mean([r['macro-F1 test'] for r in group_rows]):.6f} ± {np.std([r['macro-F1 test'] for r in group_rows], ddof=1):.6f}",
                    "top-1 test": f"{np.mean([r['top-1 test'] for r in group_rows]):.6f} ± {np.std([r['top-1 test'] for r in group_rows], ddof=1):.6f}",
                    "ECE test": f"{np.mean([r['ECE test'] for r in group_rows]):.6f} ± {np.std([r['ECE test'] for r in group_rows], ddof=1):.6f}",
                    "status": "AGGREGATE",
                }
            )
    latency_path = Path(latency_file)
    latency = (
        json.loads(latency_path.read_text(encoding="utf-8"))
        if latency_path.exists()
        else []
    )
    if isinstance(latency, dict):
        latency = latency.get("measurements", [latency])
    inference_rows = (
        json.loads(Path("inference_results.json").read_text(encoding="utf-8"))
        if Path("inference_results.json").exists()
        else []
    )
    single_view_p50 = next(
        (
            row.get("p50_ms")
            for row in inference_rows
            if row.get("method") == "single"
            and row.get("aggregation", "prob") == "prob"
            and row.get("status") == "RECORDED"
        ),
        None,
    )
    for row in inference_rows:
        row["relative_to_I00"] = (
            row.get("p50_ms") / single_view_p50
            if row.get("p50_ms") and single_view_p50
            else np.nan
        )
    per_class_rows = (
        json.loads(Path("per_class.json").read_text(encoding="utf-8"))
        if Path("per_class.json").exists()
        else []
    )
    columns = {
        "Backbones": [
            "exp_id",
            "backbone",
            "pretrained tag",
            "params (M)",
            "GMAC",
            "resolution",
            "epochs",
            "seed",
            "macro-F1 val",
            "top-1 val",
            "train seconds/epoch",
            "latency batch-1 (ms)",
            "notes",
        ],
        "Training": [
            "exp_id",
            "backbone",
            "axis",
            "change vs T00",
            "seed",
            "macro-F1 val",
            "top-1 val",
            "delta vs T00",
            "notes",
        ],
        "Inference": [
            "exp_id",
            "method",
            "aggregation",
            "checkpoint",
            "views",
            "macro_f1_val",
            "top1_val",
            "ece_val",
            "ece_uncal_val",
            "p50_ms",
            "p95_ms",
            "p99_ms",
            "images_per_s",
            "relative_to_I00",
            "status",
        ],
        "Final": [
            "exp_id",
            "seed",
            "configuration",
            "macro-F1 val",
            "macro-F1 test",
            "top-1 test",
            "ECE test",
            "status",
        ],
        "PerClass": [
            "exp_id",
            "seed",
            "class",
            "support",
            "precision",
            "recall",
            "f1",
            "f1_std",
        ],
        "Latency": [
            "configuration",
            "gpu",
            "dtype",
            "batch",
            "img_size",
            "batchnorm_fused",
            "p50",
            "p95",
            "p99",
            "images_per_s",
        ],
        "Summary": ["item", "value", "status"],
    }
    summary_rows = []
    for exp in ("T00", "F01"):
        records = [
            row for row in final if row["exp_id"] == exp and row["status"] == "RECORDED"
        ]
        if records:
            macro = np.asarray([row["macro-F1 test"] for row in records], dtype=float)
            top1 = np.asarray([row["top-1 test"] for row in records], dtype=float)
            ece = np.asarray([row["ECE test"] for row in records], dtype=float)
            summary_rows.extend(
                [
                    {
                        "item": f"{exp} test macro-F1 mean ± sample std",
                        "value": f"{macro.mean():.6f} ± {macro.std(ddof=1):.6f}"
                        if len(macro) >= 3
                        else "PENDING (need 3 seeds)",
                        "status": f"{len(records)} recorded seeds",
                    },
                    {
                        "item": f"{exp} test accuracy mean ± sample std",
                        "value": f"{top1.mean():.6f} ± {top1.std(ddof=1):.6f}"
                        if len(top1) >= 3
                        else "PENDING (need 3 seeds)",
                        "status": f"{len(records)} recorded seeds",
                    },
                    {
                        "item": f"{exp} test ECE mean ± sample std",
                        "value": f"{ece.mean():.6f} ± {ece.std(ddof=1):.6f}"
                        if len(ece) >= 3
                        else "PENDING (need 3 seeds)",
                        "status": f"{len(records)} recorded seeds",
                    },
                ]
            )
        else:
            summary_rows.append(
                {
                    "item": f"{exp} test result",
                    "value": "PENDING",
                    "status": "No final test artifacts",
                }
            )
    final_scores = [
        float(row["macro-F1 test"])
        for row in final
        if row["exp_id"] == "F01" and row["status"] == "RECORDED"
    ]
    base_scores = [
        float(row["macro-F1 test"])
        for row in final
        if row["exp_id"] == "T00" and row["status"] == "RECORDED"
    ]
    summary_rows.append(
        {
            "item": "F01 minus T00 test macro-F1",
            "value": float(np.mean(final_scores) - np.mean(base_scores))
            if len(final_scores) >= 3 and len(base_scores) >= 3
            else "PENDING",
            "status": "From three final and baseline seeds"
            if len(final_scores) >= 3 and len(base_scores) >= 3
            else "Needs three seeds in both groups",
        }
    )
    if per_class_rows:
        final_class_rows = [
            row
            for row in per_class_rows
            if row.get("exp_id") == "F01" and row.get("seed") == "mean"
        ]
        hardest = sorted(
            final_class_rows or per_class_rows, key=lambda row: row.get("f1", 0)
        )[:2]
        summary_rows.append(
            {
                "item": "Hardest classes by F1",
                "value": ", ".join(row["class"] for row in hardest),
                "status": "From F01 artifacts",
            }
        )
    else:
        summary_rows.append(
            {
                "item": "Hardest classes by F1",
                "value": "PENDING",
                "status": "No per-class test metrics",
            }
        )
    if latency:
        batch_one = next((row for row in latency if row.get("batch") == 1), latency[0])
        summary_rows.append(
            {
                "item": "Selected measured batch-1 p95 latency (ms)",
                "value": batch_one.get("p95", "PENDING"),
                "status": batch_one.get("gpu", "Recorded"),
            }
        )
    else:
        summary_rows.append(
            {
                "item": "Selected measured batch-1 p95 latency (ms)",
                "value": "PENDING",
                "status": "No benchmark artifacts",
            }
        )
    if not run_frame.empty and "best_val_macro_f1" in run_frame.columns:
        for _, row in (
            run_frame.sort_values("best_val_macro_f1", ascending=False)
            .head(10)
            .iterrows()
        ):
            summary_rows.append(
                {
                    "item": f"Validation run {row.get('exp_id')} seed {row.get('seed')}",
                    "value": row.get("best_val_macro_f1", "PENDING"),
                    "status": row.get("status", "RECORDED"),
                }
            )
    frames = {
        "Backbones": pd.DataFrame(backbones, columns=columns["Backbones"])
        if backbones
        else pd.DataFrame([{"notes": "PENDING"}], columns=columns["Backbones"]),
        "Training": pd.DataFrame(training, columns=columns["Training"])
        if training
        else pd.DataFrame([{"notes": "PENDING"}], columns=columns["Training"]),
        "Inference": pd.DataFrame(inference_rows, columns=columns["Inference"])
        if inference_rows
        else pd.DataFrame([{"status": "PENDING"}], columns=columns["Inference"]),
        "Final": pd.DataFrame(final, columns=columns["Final"])
        if final
        else pd.DataFrame([{"status": "PENDING"}], columns=columns["Final"]),
        "PerClass": pd.DataFrame(per_class_rows, columns=columns["PerClass"])
        if per_class_rows
        else pd.DataFrame([{"class": "PENDING"}], columns=columns["PerClass"]),
        "Latency": pd.DataFrame(latency, columns=columns["Latency"])
        if latency
        else pd.DataFrame([{"configuration": "PENDING"}], columns=columns["Latency"]),
        "Summary": pd.DataFrame(summary_rows, columns=columns["Summary"]),
    }
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(target, engine="openpyxl") as writer:
        for sheet in SHEETS:
            frames[sheet].to_excel(writer, sheet_name=sheet, index=False)
            ws = writer.sheets[sheet]
            ws.freeze_panes = "A2"
            ws.auto_filter.ref = ws.dimensions
            for column in ws.columns:
                letter = column[0].column_letter
                width = min(
                    42, max(12, max(len(str(cell.value or "")) for cell in column) + 2)
                )
                ws.column_dimensions[letter].width = width
    return {
        "path": str(target),
        "sheets": SHEETS,
        "recorded_runs": len(runs),
        "recorded_final_test_files": len(final),
    }


def generate_report(path="report.md", results_path="results.xlsx"):
    from reporting import generate_report as build_report

    return build_report(path, results_path)
