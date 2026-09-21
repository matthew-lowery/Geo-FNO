import unittest

import numpy as np
import torch

from transolver.train import divergence_loss, evaluate, pad_inputs, union_grid
from transolver.model.Physics_Attention import Physics_Attention_Irregular_Mesh


class TransolverRuntimeTest(unittest.TestCase):
    def test_pooled_ood_loss_handles_zero_target_functions(self):
        class ConstantModel(torch.nn.Module):
            def forward(self, positions, fx):
                return torch.ones(len(fx), positions.shape[1], 2)

        inputs = torch.zeros(2, 2, 1)
        targets = torch.zeros(2, 2, 2)
        targets[1] = 1
        normalizer = type("Identity", (), {"decode": lambda self, x: x})()
        positions = torch.zeros(1, 2, 2)
        _, loss = evaluate(ConstantModel(), positions, inputs, targets,
                           normalizer, torch.arange(2), batch_size=1, pooled=True)
        self.assertAlmostEqual(loss, 1.)

    def test_zero_slice_temperature_has_finite_output_and_gradients(self):
        attention = Physics_Attention_Irregular_Mesh(8, heads=2, dim_head=4, slice_num=4)
        attention.temperature.data.zero_()
        x = torch.randn(2, 16, 8, requires_grad=True)
        y = attention(x)
        y.square().sum().backward()
        self.assertTrue(torch.isfinite(y).all())
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all()
                            for p in attention.parameters()))

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
