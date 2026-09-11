import unittest

import numpy as np
import torch

from transolver.train import divergence_loss, pad_inputs, union_grid


class TransolverRuntimeTest(unittest.TestCase):
    def test_union_retains_boundary_inputs_and_output_order(self):
        inputs = np.array([[0., 0.], [0., 1.]])
        outputs = np.array([[.5, .5], [0., 0.], [1., 1.]])
        points, input_indices, output_indices = union_grid(inputs, outputs)
        np.testing.assert_array_equal(points[output_indices], outputs)
        values = np.array([[[2.], [3.]]])
        padded = pad_inputs(values, len(points), input_indices)
        np.testing.assert_array_equal(padded[:, input_indices], values)

    def test_boundary_divergence_has_no_training_gradient(self):
        predictions = torch.ones(1, 3, 2, requires_grad=True)
        operators = (torch.eye(3).to_sparse(), torch.zeros(3, 3).to_sparse())
        mask = torch.tensor([False, True, False])
        loss = divergence_loss(predictions, operators, mask, 1)
        loss.backward()
        self.assertEqual(loss.item(), 1.)
        self.assertTrue(torch.equal(predictions.grad[:, [0, 2]], torch.zeros(1, 2, 2)))
        self.assertEqual(predictions.grad[0, 1, 0].item(), 2.)


if __name__ == "__main__":
    unittest.main()
