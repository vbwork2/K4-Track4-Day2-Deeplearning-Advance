"""Runtime regressions using small synthetic inputs; no lab metrics are produced."""

from __future__ import annotations

import dataclasses
import os
import tempfile
import unittest
import warnings
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from PIL import Image

import artifacts
import reporting
import benchmark
import dataset
import inference
import losses
import model
import sanity
import train
import workflow
from eval import read_pred

torch.set_num_threads(2)


def tiny_model(*args, **kwargs):
    net = torch.nn.Sequential(
        torch.nn.Conv2d(3, 8, 3, padding=1),
        torch.nn.ReLU(),
        torch.nn.AdaptiveAvgPool2d(1),
        torch.nn.Flatten(),
        torch.nn.Linear(8, 9),
    )
    net.get_classifier = lambda: net[-1]
    net.pretrained_tag = "synthetic-test-only"
    return net


class RuntimeTests(unittest.TestCase):
    def test_partial_image_extraction_is_repaired(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            image_dir = root / "data" / "images"
            image_dir.mkdir(parents=True)
            names = ["a.jpg", "b.jpg", "c.jpg"]
            source_zip = root / "source.zip"
            with zipfile.ZipFile(source_zip, "w") as archive:
                for name in names:
                    source = root / name
                    Image.new("RGB", (8, 8)).save(source)
                    archive.write(source, arcname="images/" + name)
            (image_dir / "a.jpg").write_bytes((root / "a.jpg").read_bytes())
            (image_dir / "b.jpg").write_bytes(b"")
            actual_digest = workflow.file_digest

            def fixture_digest(path, algorithm="sha256"):
                if algorithm == "md5":
                    return "b7b30f96d466fba86016aa5a26606e0f"
                return actual_digest(path, algorithm)

            with (
                patch.object(dataset, "EXPECTED_TOTAL", 3),
                patch.object(workflow, "file_digest", side_effect=fixture_digest),
            ):
                workflow.prepare_images(source_zip, image_dir, names)
                times = [(image_dir / name).stat().st_mtime_ns for name in names]
                workflow.prepare_images(source_zip, image_dir, names)
                self.assertEqual(
                    times, [(image_dir / name).stat().st_mtime_ns for name in names]
                )
            self.assertTrue(
                all((image_dir / name).stat().st_size > 0 for name in names)
            )

    def test_focal_zero_and_soft_targets(self):
        torch.manual_seed(0)
        logits = torch.randn(12, 9, requires_grad=True)
        labels = torch.arange(12) % 9
        ce = torch.nn.functional.cross_entropy(logits, labels)
        self.assertTrue(torch.allclose(ce, losses.FocalLoss(0)(logits, labels)))
        value = losses.mixed_loss(
            torch.nn.CrossEntropyLoss(), logits, (labels, labels.flip(0), 0.3)
        )
        value.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertTrue(sanity.check_mixed_targets())

    def test_all_screening_backbones_and_deit_views(self):
        for name in (
            "resnet50",
            "resnext50_32x4d",
            "convnext_tiny",
            "deit_small_patch16_224",
            "efficientnet_b0",
        ):
            with self.subTest(backbone=name):
                net = model.build_model(name, pretrained=False, init="scratch").eval()
                with torch.no_grad():
                    output = net(torch.randn(1, 3, 224, 224))
                self.assertEqual(tuple(output.shape), (1, 9))
                if name.startswith("deit"):
                    for method in ("multicrop", "multiscale"):
                        result = inference.forward_method(
                            net, torch.randn(1, 3, 224, 224), method
                        )
                        self.assertEqual(tuple(result.shape), (1, 9))
                    with torch.no_grad():
                        self.assertEqual(
                            tuple(net(torch.randn(1, 3, 288, 288)).shape), (1, 9)
                        )
                del net

    def test_groups_and_frozen_bn(self):
        net = model.build_model("resnet18", pretrained=False, init="frozen")
        groups = model.param_groups(net, 1e-4, 1e-3, 0.05)
        decay = {
            id(p): group["weight_decay"] for group in groups for p in group["params"]
        }
        self.assertEqual(decay[id(net.fc.bias)], 0)
        self.assertEqual(decay[id(net.fc.weight)], 0.05)
        net.train()
        model.keep_frozen_batchnorm_eval(net)
        before = net.bn1.running_mean.clone()
        with torch.no_grad():
            net(torch.randn(2, 3, 64, 64))
        self.assertTrue(torch.equal(net.bn1.running_mean, before))
        model.count_gmacs(net, 64)
        self.assertFalse(net.bn1.training)

    def test_single_logits_calibration_and_latency(self):
        net = tiny_model().eval()
        inputs = torch.randn(3, 3, 32, 32)
        with torch.no_grad():
            expected = net(inputs)
        actual = inference.forward_method(net, inputs)
        self.assertTrue(torch.equal(expected, actual))
        labels = torch.tensor([0, 1, 2])
        temperature = inference.fit_temperature(actual.numpy(), labels.numpy())
        before = torch.nn.functional.cross_entropy(actual, labels)
        after = torch.nn.functional.cross_entropy(actual / temperature, labels)
        self.assertLessEqual(after.item(), before.item() + 1e-5)
        np.testing.assert_array_equal(
            inference.apply_temperature(actual, temperature).argmax(1), actual.argmax(1)
        )
        timing = benchmark.method_latency_report(
            net, "hflip", "prob", img_size=32, device="cpu"
        )
        self.assertEqual(timing["views"], 2)
        self.assertEqual(timing["n"], 50)
        self.assertGreaterEqual(timing["p99"], timing["p50"])

    def test_bn_fusion_does_not_assume_arbitrary_registration_order(self):
        class Parallel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 3, 1)
                self.bn = torch.nn.BatchNorm2d(3)

            def forward(self, x):
                return self.conv(x) + self.bn(x)

        net = Parallel().eval()
        x = torch.randn(2, 3, 16, 16)
        with torch.no_grad():
            self.assertTrue(torch.equal(net(x), inference.fuse_conv_bn(net)(x)))

    def test_lock_has_timing_and_inference_does_not_change_training_amp(self):
        with tempfile.TemporaryDirectory() as folder:
            previous = Path.cwd()
            try:
                os.chdir(folder)
                base = train.Config(batch_size=32, inference_amp=False)
                methods = (
                    "fp32",
                    "hflip",
                    "multicrop",
                    "multiscale",
                    "temperature_scaling",
                )
                rows = [
                    {
                        "exp_id": f"I{i:02d}",
                        "method": method,
                        "inference_method": "fp32",
                        "aggregation": "prob",
                        "img_size": 256,
                        "amp": False,
                        "temperature_scaling": method == "temperature_scaling",
                        "status": "RECORDED",
                        "macro_f1_val": 0.5,
                        "ece_val": 0.1,
                        "p50_ms": 1.0,
                        "p95_ms": 2.0,
                        "p99_ms": 3.0,
                    }
                    for i, method in enumerate(methods)
                ]
                lock = workflow.lock_final(base, base, rows, Path(folder) / "drive")
                cfg = workflow.final_config(lock, 0, True, "drive", "images", "labels")
                self.assertTrue(cfg.amp)
                self.assertFalse(cfg.inference_amp)
                self.assertEqual(cfg.inference_img_size, 256)
                self.assertIn("selected_inference_row", lock)
                self.assertEqual(workflow.lock_final(base, base, [], "drive"), lock)
            finally:
                os.chdir(previous)

    def test_train_resume_test_once_and_artifact_export(self):
        with tempfile.TemporaryDirectory() as folder:
            previous = Path.cwd()
            try:
                os.chdir(folder)
                images = Path("data/images")
                labels = Path("data/labels")
                images.mkdir(parents=True)
                labels.mkdir(parents=True)
                frame = pd.DataFrame(
                    {
                        "Filename": [f"{i}.jpg" for i in range(45)],
                        "Label": [i % 9 for i in range(45)],
                        "Species": [dataset.CLASS_NAMES[i % 9] for i in range(45)],
                    }
                )
                frame.to_csv(labels / "labels.csv", index=False)
                for name, subset in zip(
                    ("train", "val", "test"),
                    (frame.iloc[:27], frame.iloc[27:36], frame.iloc[36:]),
                ):
                    subset[["Filename", "Label"]].to_csv(
                        labels / f"{name}_subset0.csv", index=False
                    )
                for i, filename in enumerate(frame.Filename):
                    Image.new("RGB", (32, 32), color=(i * 5, i * 3, i * 2)).save(
                        images / filename
                    )
                cfg = train.Config(
                    backbone="resnet18",
                    init="scratch",
                    epochs=2,
                    batch_size=9,
                    img_size=32,
                    num_workers=0,
                    images_dir=str(images),
                    labels_dir=str(labels),
                    sync_dir="drive",
                    amp=False,
                )
                with (
                    patch.object(dataset, "EXPECTED_TOTAL", 45),
                    patch.object(model, "build_model", side_effect=tiny_model),
                ):
                    summary = workflow.run_or_resume(cfg)
                    self.assertEqual(summary["status"], "COMPLETED")
                    inference_rows = workflow.inference_experiments(
                        cfg, frame.iloc[27:36], "cpu"
                    )
                    self.assertEqual(len(inference_rows), 9)
                    self.assertTrue(
                        all(row["status"] == "RECORDED" for row in inference_rows)
                    )
                    self.assertEqual(
                        workflow.inference_experiments(cfg, frame.iloc[27:36], "cpu"),
                        inference_rows,
                    )
                    lock = workflow.lock_final(cfg, cfg, inference_rows, "drive")
                    self.assertEqual(lock["schema_version"], 2)
                    history_before = (train.run_dir(cfg) / "history.csv").read_bytes()
                    final_cfg = dataclasses.replace(cfg, save_test_predictions=True)
                    workflow.run_or_resume(final_cfg)
                    test_path = train.pred_path(final_cfg, "test")
                    self.assertEqual(len(read_pred(str(test_path)).y_true), 9)
                    self.assertEqual(
                        (train.run_dir(cfg) / "history.csv").read_bytes(),
                        history_before,
                    )
                    timestamp = test_path.stat().st_mtime_ns
                    workflow.run_or_resume(final_cfg)
                    self.assertEqual(test_path.stat().st_mtime_ns, timestamp)
                    test_path.unlink()
                    (Path("drive") / test_path).unlink()
                    with patch.object(
                        inference,
                        "predict_method_logits",
                        wraps=inference.predict_method_logits,
                    ) as predict:
                        workflow.run_or_resume(final_cfg)
                    self.assertEqual(predict.call_count, 1)
                    latency = workflow.final_latency(cfg, "cpu")
                    self.assertEqual([row["batch"] for row in latency], [1, 32])
                    self.assertEqual(workflow.final_latency(cfg, "cpu"), latency)
                    changed = dataclasses.replace(cfg, lr_head=0.002)
                    with self.assertRaises(ValueError):
                        workflow.run_or_resume(changed)
                    (train.run_dir(cfg) / "best.pt").unlink()
                    self.assertTrue(workflow.restore_checkpoint(cfg).exists())
                    interrupted = dataclasses.replace(
                        cfg, exp_id="INTERRUPTED", pred_dir="interrupted_predictions"
                    )
                    original_epoch = train.train_one_epoch
                    calls = [0]

                    def stop_after_first(*args, **kwargs):
                        calls[0] += 1
                        if calls[0] == 2:
                            raise RuntimeError("Simulated disconnect")
                        return original_epoch(*args, **kwargs)

                    with patch.object(
                        train, "train_one_epoch", side_effect=stop_after_first
                    ):
                        with self.assertRaisesRegex(
                            RuntimeError, "Simulated disconnect"
                        ):
                            workflow.run_or_resume(interrupted)
                    resumed = workflow.run_or_resume(interrupted)
                    self.assertEqual(resumed["status"], "COMPLETED")
                    self.assertEqual(
                        len(pd.read_csv(train.run_dir(interrupted) / "history.csv")), 2
                    )
                for tag in ("T00", "F01"):
                    for seed in (0, 1, 2):
                        shutil_source = pd.read_csv(test_path)
                        shutil_source.to_csv(
                            Path("predictions") / f"{tag}_seed{seed}_test.csv",
                            index=False,
                        )
                records = reporting.analyze_tests(labels_dir=labels, images_dir=images)
                self.assertEqual(
                    len([row for row in records if row["seed"] == "mean"]), 18
                )
                result = artifacts.generate_results("results.xlsx")
                self.assertEqual(len(result["sheets"]), 7)
                artifacts.generate_report("report.md", "results.xlsx")
                self.assertTrue(Path("report.md").exists())
                self.assertTrue((Path("drive") / test_path).exists())
            finally:
                os.chdir(previous)

    def test_optional_cli_values_are_typed(self):
        parsed = train.parse_overrides(
            ["ema_decay=0.99", "class_weight_beta=0.999", "inference_amp=false"]
        )
        self.assertEqual(
            parsed,
            {"ema_decay": 0.99, "class_weight_beta": 0.999, "inference_amp": False},
        )


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        unittest.main(verbosity=2)
