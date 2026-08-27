import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from scipy.io import savemat

from model_3d import FNO3d, IPHI
from ram_dataset_loader import TIME_LEVELS, load_dataset, load_ood_dataset


def write_taylor_green_data(root):
    problem_dir = Path(root) / "taylor_green"
    problem_dir.mkdir()
    points = np.asarray([
        [0.2, 0.3],
        [0.4, 0.7],
        [0.8, 1.1],
        [1.2, 0.5],
        [1.5, 1.8],
        [2.0, 1.4],
    ])
    viscosity = np.linspace(0.01, 0.06, 8)
    amplitude = np.linspace(1.0, 2.4, 8)
    base = np.stack([
        np.sin(points[:, 0]) * np.cos(points[:, 1]),
        -np.cos(points[:, 0]) * np.sin(points[:, 1]),
    ], axis=-1)
    initial = amplitude[:, None, None] * base[None]
    data = {"points": points, "init_velocity": initial}
    for time in TIME_LEVELS:
        data[f"vel_{round(10 * time)}"] = (
            initial * np.exp(-2 * viscosity * time)[:, None, None]
        )
    savemat(problem_dir / "data_time.mat", data)
    savemat(
        problem_dir / "data_coeffs.mat",
        {"init_coeffs": np.column_stack((viscosity, amplitude))},
    )
    savemat(problem_dir / "data_time_ood.mat", data)


class TaylorGreenSpacetimeSmokeTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        write_taylor_green_data(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def load(self, name):
        return load_dataset(name, 2, 4, self.temp_dir.name, test_count=2)

    def test_loader_shapes_and_time_order(self):
        velocity = self.load("taylor_green_spacetime")
        coefficients = self.load("taylor_green_spacetime_coeffs")
        self.assertEqual(velocity.train_input.shape, (2, 4, 2))
        self.assertEqual(coefficients.train_input.shape, (2, 2))
        self.assertEqual(velocity.train_output.shape, (2, 16, 2))
        self.assertEqual(coefficients.train_output.shape, (2, 16, 2))
        np.testing.assert_allclose(
            velocity.output_points[:4, 2], np.asarray(TIME_LEVELS)
        )

    def test_ood_uses_ood_files(self):
        _, _, inputs, outputs = load_ood_dataset(
            "taylor_green_spacetime", 4, self.temp_dir.name, test_count=2
        )
        self.assertEqual(inputs.shape, (2, 4, 2))
        self.assertEqual(outputs.shape, (2, 16, 2))
        with self.assertRaisesRegex(FileNotFoundError, "data_coeffs_ood.mat"):
            load_ood_dataset(
                "taylor_green_spacetime_coeffs",
                4,
                self.temp_dir.name,
                test_count=2,
            )

    def test_tiny_forward_and_backward(self):
        for name in (
            "taylor_green_spacetime",
            "taylor_green_spacetime_coeffs",
        ):
            dataset = self.load(name)
            inputs = torch.tensor(dataset.train_input, dtype=torch.float32)
            if inputs.ndim == 2:
                inputs = inputs[..., None]
            input_points = torch.tensor(
                dataset.input_points, dtype=torch.float32
            ).repeat(len(inputs), 1, 1)
            output_points = torch.tensor(
                dataset.output_points, dtype=torch.float32
            ).repeat(len(inputs), 1, 1)
            targets = torch.tensor(dataset.train_output, dtype=torch.float32)
            model = FNO3d(
                2,
                4,
                in_channels=inputs.shape[-1] + 3,
                out_channels=targets.shape[-1],
                s1=4,
                s2=4,
                s3=4,
            )
            iphi = IPHI(width=4)
            predictions = model(
                torch.cat((inputs, input_points), dim=-1),
                x_in=input_points,
                x_out=output_points,
                iphi=iphi,
            )
            self.assertEqual(predictions.shape, targets.shape)
            loss = torch.mean((predictions - targets) ** 2)
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(all(
                parameter.grad is None or torch.isfinite(parameter.grad).all()
                for parameter in list(model.parameters()) + list(iphi.parameters())
            ))


if __name__ == "__main__":
    unittest.main()
