from dataclasses import replace

import numpy as np

import ram_dataset_loader as shared


def _coefficient_input(points, values, problem):
    if "spacetime" in problem or "_time" in problem:
        points = points.reshape(-1, 4, 3)[:, 0].copy()
        points[:, 2] = 0
    return points, np.broadcast_to(values[:, None, :], (len(values), len(points), values.shape[-1]))


def load_dataset(problem, *args, **kwargs):
    data = shared.load_dataset(problem, *args, **kwargs)
    if problem.endswith("coeffs"):
        points, train = _coefficient_input(data.output_points, data.train_input, problem)
        _, test = _coefficient_input(data.output_points, data.test_input, problem)
        data = replace(data, input_points=points, train_input=train, test_input=test)
    return data


def load_ood_dataset(problem, *args, **kwargs):
    data = shared.try_load_ood_dataset(problem, *args, **kwargs)
    if data is None:
        return None
    input_points, output_points, inputs, targets = data
    if problem.endswith("coeffs"):
        input_points, inputs = _coefficient_input(output_points, inputs, problem)
    return input_points, output_points, inputs, targets
