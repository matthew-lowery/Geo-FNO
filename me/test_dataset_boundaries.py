import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.io import savemat

from dataset_boundaries import divergence_interior_mask
from divergence_metrics import summarize_divergence
from ram_dataset_loader import _point_filter
import torch


class BoundaryMaskTest(unittest.TestCase):
    def test_fixed_geometry_not_sample_extrema(self):
        points = np.array([[.2, .2], [.8, .8], [0., .5], [1., .5]])
        mask = divergence_interior_mask(points, "lid_cavity_flow", ".")
        np.testing.assert_array_equal(mask, [True, True, False, False])

    def test_taylor_green_keeps_functions_but_masks_all_edges(self):
        points = np.array([[0., .5], [.5, 0.], [2 * np.pi, .5], [.5, 2 * np.pi], [1., 1.]])
        grid, _, _ = _point_filter("taylor_green_exact", points, None, ".")
        np.testing.assert_array_equal(grid, points)
        mask = divergence_interior_mask(grid, "taylor_green", ".")
        np.testing.assert_array_equal(mask, [False, False, False, False, True])

    def test_forced_turbulence_periodic_copy_is_divergence_only(self):
        points = np.array([[0., 0., 0.], [2 * np.pi, 0., 0.], [1., 1., 1.]])
        grid, _, _ = _point_filter("forced_turb", points, 3, ".")
        np.testing.assert_array_equal(grid, points)
        np.testing.assert_array_equal(
            divergence_interior_mask(grid, "forced_turb", "."), [True, False, True],
        )

    def test_species_boundary_list(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "species_transport"
            path.mkdir()
            boundary = np.array([[.1, .2, .0], [.2, .3, .4]])
            savemat(path / "full_domain_boundary_points.mat", {"boundary_points": boundary})
            points = np.vstack((boundary, [[.1, .1, .1]]))
            np.testing.assert_array_equal(
                divergence_interior_mask(points, "species_transport", directory),
                [False, False, True],
            )

    def test_reference_cylinder_and_step_rectangles(self):
        for name, points in [
            ("flow_cylinder_laminar", [[0., 7.], [20., 7.], [10., 7.]]),
            ("flow_cylinder_shedding", [[10., 0.], [10., 14.], [10., 7.]]),
            ("backward_facing_step", [[0., .1], [10., -.5], [10., 0.]]),
        ]:
            with self.subTest(name=name):
                np.testing.assert_array_equal(
                    divergence_interior_mask(points, name, "."), [False, False, True],
                )

    def test_reported_divergence_excludes_boundaries(self):
        predictions = torch.tensor([[[100., 0.], [2., 0.], [100., 0.]]])
        operators = (torch.eye(3).to_sparse(), torch.zeros(3, 3).to_sparse())
        metrics = summarize_divergence(predictions, operators, torch.tensor([False, True, False]))
        self.assertEqual(metrics, {
            "test_div/max_abs_interior": 2.,
            "test_div/median_abs_interior": 2.,
        })


if __name__ == "__main__":
    unittest.main()
