"""CPU forward/backward checks using the RAM MATLAB datasets."""

import argparse

import torch

from model import FNO2d
from model_3d import FNO3d, IPHI as IPHI3d
from ram_dataset_loader import DEFAULT_DATA_ROOT, TRAIN_SAMPLE_COUNTS, load_dataset


def smoke_test(name, root, count):
    dataset = load_dataset(name, 2, count, root, test_count=2)
    inputs = dataset.train_input
    if inputs.ndim == 2:
        inputs = inputs[..., None]
    inputs = torch.tensor(inputs, dtype=torch.float32)
    targets = torch.tensor(dataset.train_output, dtype=torch.float32)
    input_points = torch.tensor(dataset.input_points, dtype=torch.float32).repeat(2, 1, 1)
    output_points = torch.tensor(dataset.output_points, dtype=torch.float32).repeat(2, 1, 1)
    dimension = input_points.shape[-1]
    channels = dict(in_channels=inputs.shape[-1] + dimension, out_channels=targets.shape[-1])
    if dimension == 2:
        model = FNO2d(2, 2, 4, is_mesh=False, s1=4, s2=4, **channels)
        iphi = None
    else:
        model = FNO3d(2, 4, is_mesh=False, s1=4, s2=4, s3=4, **channels)
        iphi = IPHI3d(width=4)
    predictions = model(
        torch.cat((inputs, input_points), dim=-1),
        x_in=input_points, x_out=output_points, iphi=iphi,
    )
    if predictions.shape != targets.shape:
        raise AssertionError(f"{name}: {predictions.shape} != {targets.shape}")
    loss = (predictions - targets).square().mean()
    loss.backward()
    if not torch.isfinite(loss) or not all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in list(model.parameters()) + ([] if iphi is None else list(iphi.parameters()))
    ):
        raise AssertionError(f"{name}: nonfinite loss or gradient")
    print(f"{name}: output={tuple(predictions.shape)}, finite loss/gradients")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--npoints", type=int, default=32)
    parser.add_argument("datasets", nargs="*", default=list(TRAIN_SAMPLE_COUNTS))
    args = parser.parse_args()
    for name in args.datasets:
        smoke_test(name, args.data_root, args.npoints)
