#!/usr/bin/env python3
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


def load_module(module_name, path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ExamplePipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("PyTorch not installed; skipping example tests.")
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("matplotlib not installed; skipping example tests.")

        cls.root = Path(__file__).resolve().parents[1]
        cls.data_root = cls.root / "data" / "dataset_example"
        if not cls.data_root.exists():
            raise unittest.SkipTest("Example dataset missing; skipping example tests.")

    def test_train_and_test_pipeline(self):
        train_script = self.root / "src" / "03_train_test" / "train_example.py"
        test_script = self.root / "src" / "03_train_test" / "test_example.py"

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            subprocess.run(
                [
                    sys.executable,
                    str(train_script),
                    "--data-root",
                    str(self.data_root),
                    "--output-dir",
                    str(tmp_path),
                    "--epochs",
                    "40",
                    "--batch-size",
                    "2",
                    "--min-ssim",
                    "0.9",
                ],
                check=True,
            )

            weights_path = tmp_path / "model_best.pt"
            self.assertTrue(weights_path.exists(), "Expected model_best.pt to exist.")

            subprocess.run(
                [
                    sys.executable,
                    str(test_script),
                    "--data-root",
                    str(self.data_root),
                    "--weights",
                    str(weights_path),
                    "--batch-size",
                    "2",
                ],
                check=True,
            )

            metrics_path = tmp_path / "test_metrics.json"
            self.assertTrue(metrics_path.exists(), "Expected test_metrics.json to exist.")

            with open(metrics_path, "r", encoding="utf-8") as handle:
                metrics = json.load(handle)

            self.assertIn("test_ssim", metrics)
            self.assertGreaterEqual(metrics["test_ssim"], 0.0)
            self.assertLessEqual(metrics["test_ssim"], 1.001)

            self.assertTrue((tmp_path / "loss_curve.png").exists())
            self.assertTrue((tmp_path / "ssim_curve.png").exists())

            train_metrics_path = tmp_path / "train_metrics.json"
            with open(train_metrics_path, "r", encoding="utf-8") as handle:
                train_metrics = json.load(handle)

            self.assertIn("train_ssim", train_metrics)
            self.assertGreaterEqual(max(train_metrics["train_ssim"]), 0.9)

    def test_ssim_identity(self):
        metrics_path = self.root / "src" / "03_train_test" / "metrics_example.py"
        metrics_module = load_module("metrics_example", metrics_path)

        import torch

        sample = torch.rand(1, 3, 8, 8)
        value = metrics_module.ssim(sample, sample).item()
        self.assertAlmostEqual(value, 1.0, places=4)


if __name__ == "__main__":
    unittest.main()
