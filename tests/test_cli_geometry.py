import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch


class ContractTests(unittest.TestCase):
    def test_explicit_cuda_does_not_fall_back(self):
        from pytorch3d_drr.utilities import determine_device
        with patch("torch.cuda.is_available", return_value=False):
            with self.assertRaises(RuntimeError):
                determine_device("cuda")
            self.assertEqual(str(determine_device("auto")), "cpu")
            self.assertEqual(str(determine_device("cpu")), "cpu")

    def test_missing_dependencies_still_produce_json(self):
        env = os.environ.copy()
        env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
        result = subprocess.run([sys.executable, "-S", "-m", "pytorch3d_drr",
                                 "doctor", "--json"], env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 1)
        report = json.loads(result.stdout)
        self.assertFalse(report["ok"])
        self.assertTrue(report["errors"])

    def test_camera_contract(self):
        from pytorch3d_drr import Camera
        camera = Camera("test", (0, 0, 400), (0, 0, -1), (0, 1, 0),
                        160., 160., 0., 0., 400.)
        torch.testing.assert_close(camera.get_cam_pos(), torch.zeros(1, 3))
        torch.testing.assert_close(camera.get_cam_target(), torch.tensor([[0., 0., 400.]]))
        points = torch.tensor([[[0., 0., 200.], [10., 0., 200.]]])
        ndc = camera.transform_points_ndc(points)
        torch.testing.assert_close(ndc[0, 0, :2], torch.zeros(2), atol=1e-6, rtol=0)
        self.assertAlmostEqual(abs(float(ndc[0, 1, 0])), .25, places=5)

    def test_offline_demo(self):
        import numpy as np
        from pytorch3d_drr.demo import run_demo
        with tempfile.TemporaryDirectory() as folder:
            report = run_demo(folder, "cpu", 16)
            self.assertTrue(report["ok"])
            arrays = {mode: np.load(info["array"]) for mode, info in report["images"].items()}
            for array in arrays.values():
                self.assertEqual(array.shape, (1, 16, 16))
                self.assertTrue(np.isfinite(array).all())
                self.assertGreater(float(np.ptp(array)), 0)
            np.testing.assert_allclose(arrays["exp"], 1 - np.exp(-.0192 * arrays["linear"]), atol=1e-6)
            self.assertGreaterEqual(float(arrays["surface"].min()), 0)
            self.assertLessEqual(float(arrays["surface"].max()), 1)

    def test_cli_invalid_resolution(self):
        from pytorch3d_drr.cli import main
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = main(["demo", "--resolution", "0"])
        self.assertEqual(code, 1)
        self.assertFalse(json.loads(output.getvalue())["ok"])


if __name__ == "__main__":
    unittest.main()
