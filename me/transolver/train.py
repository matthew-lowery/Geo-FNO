"""Train Transolver on the same MATLAB datasets as the Geo-FNO runners."""

import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch
import wandb
from scipy.io import savemat

from dataset_boundaries import divergence_interior_mask
from divergence_metrics import build_rbf_fd_gradient, summarize_divergence
from ram_dataset_loader import DEFAULT_DATA_ROOT
from run_artifacts import (RunArtifacts, add_runtime_arguments, validate_runtime,
                           require_finite, check_gradients)
from .data import load_dataset, load_ood_dataset
from .model_dict import get_model
from .utils.normalizer import UnitTransformer
from .utils.testloss import TestLoss


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--ntrain", type=int, required=True)
    parser.add_argument("--npoints", type=int, default=7000)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--model", default="Transolver_Irregular_Mesh")
    parser.add_argument("--n-hidden", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=5)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--slice-num", type=int, default=32)
    parser.add_argument("--ref", type=int, default=8)
    parser.add_argument("--mlp_ratio", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--project-name", default="transolver_div_loss")
    parser.add_argument("--div-order", type=int, default=4)
    parser.add_argument("--div-loss-weight", type=float, default=0.)
    parser.add_argument("--div-folder", type=Path, default=Path("results/transolver/divs"))
    parser.add_argument("--model-folder", type=Path, default=Path("results/transolver/models"))
    for flag in ["wandb", "save", "norm-grid", "div-loss", "calc-div", "no-ood"]:
        parser.add_argument(f"--{flag}", action="store_true")
    add_runtime_arguments(parser)
    return parser.parse_args()


def union_grid(input_points, output_points):
    points, inverse = np.unique(
        np.concatenate((input_points, output_points)), axis=0, return_inverse=True,
    )
    return points, inverse[:len(input_points)], inverse[len(input_points):]


def pad_inputs(inputs, count, indices):
    padded = np.zeros((len(inputs), count, inputs.shape[-1]), dtype=np.float32)
    padded[:, indices] = inputs
    return torch.tensor(padded)


def divergence_loss(predictions, operators, mask, steps):
    batch, _, channels = predictions.shape
    values = predictions.reshape(batch, -1, steps, channels)
    values = values.permute(0, 2, 1, 3).reshape(batch * steps, -1, channels)
    divergence = sum(
        torch.sparse.mm(operator, values[..., axis].T).T
        for axis, operator in enumerate(operators)
    )
    return divergence[:, mask].square().mean(dim=1).sum() / steps


def evaluate(model, positions, inputs, targets, normalizer, output_indices, batch_size,
             pooled=False):
    loss_fn = TestLoss(size_average=False)
    predictions, total = [], 0.
    error_sq = target_sq = 0.
    with torch.no_grad():
        for start in range(0, len(inputs), batch_size):
            x = inputs[start:start + batch_size]
            y = targets[start:start + batch_size]
            output = model(positions.expand(len(x), -1, -1), fx=x)
            output = normalizer.decode(output)[:, output_indices]
            require_finite(output, "evaluation prediction")
            output_magnitude = output.double().norm(dim=-1)
            target_magnitude = y.double().norm(dim=-1)
            if pooled:
                error_sq += (output_magnitude - target_magnitude).square().sum().item()
                target_sq += target_magnitude.square().sum().item()
            else:
                total += loss_fn(output_magnitude, target_magnitude).item()
            predictions.append(output.cpu())
    if pooled:
        if target_sq == 0:
            raise ValueError("OOD targets have zero pooled magnitude norm")
        total = (error_sq / target_sq) ** .5
    else:
        total /= len(inputs)
    return torch.cat(predictions), total


def main():
    args = arguments()
    validate_runtime(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    wandb.init(
        project=args.project_name, config=vars(args),
        mode="online" if args.wandb else "disabled",
    )
    name = f"{args.dataset}_{args.seed}_{args.ntrain}_{args.npoints}"
    artifacts = RunArtifacts(args, name)
    count = args.npoints or None
    data = load_dataset(args.dataset, args.ntrain, count, args.data_root)
    grid, input_indices, output_indices = union_grid(data.input_points, data.output_points)
    inputs = pad_inputs(data.train_input, len(grid), input_indices)
    tests = pad_inputs(data.test_input, len(grid), input_indices)
    targets = torch.tensor(data.train_output, dtype=torch.float32)
    test_targets = torch.tensor(data.test_output, dtype=torch.float32, device=device)
    input_normalizer, output_normalizer = UnitTransformer(inputs), UnitTransformer(targets)
    inputs = input_normalizer.encode(inputs)
    tests = input_normalizer.encode(tests).to(device)
    targets = output_normalizer.encode(targets)
    input_normalizer.to(device)
    output_normalizer.to(device)
    minimum = grid.min(axis=0)
    span = np.maximum(grid.max(axis=0) - minimum, 1e-12)
    positions = (grid - minimum) / span if args.norm_grid else grid
    positions = torch.tensor(positions, dtype=torch.float32, device=device)[None]
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(inputs, targets),
        batch_size=args.batch_size, shuffle=True,
    )
    model = get_model(args).Model(
        space_dim=grid.shape[1], n_layers=args.n_layers, n_hidden=args.n_hidden,
        dropout=args.dropout, n_head=args.n_heads, Time_Input=False,
        mlp_ratio=args.mlp_ratio, fun_dim=inputs.shape[-1],
        out_dim=targets.shape[-1], slice_num=args.slice_num, ref=args.ref,
    ).to(device)
    operators, mask, steps = None, None, 1
    if args.div_loss or args.calc_div:
        physical = data.output_points
        if args.dataset.startswith("taylor_green") and physical.shape[1] == 3:
            steps = 4
            physical = physical.reshape(-1, steps, 3)[:, 0, :2]
        mask = divergence_interior_mask(physical, args.dataset, args.data_root).to(device)
        operators = tuple(
            operator.to(device) for operator in build_rbf_fd_gradient(physical, args.div_order)
        )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=len(train_loader),
    )
    loss_fn = TestLoss(size_average=False)
    normalizers = (input_normalizer, output_normalizer)
    geometry = {"input_points": data.input_points, "output_points": data.output_points,
                "grid": grid, "minimum": minimum, "span": span,
                "input_indices": input_indices, "output_indices": output_indices}
    start_epoch = artifacts.restore(model, normalizers, (optimizer,), (scheduler,))
    epoch = start_epoch - 1
    start_time = time.perf_counter()
    for epoch in range(start_epoch, start_epoch if args.eval_only else args.epochs):
        model.train()
        data_total, div_total = 0., 0.
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output = output_normalizer.decode(model(positions.expand(len(x), -1, -1), fx=x))
            output = output[:, output_indices]
            target = output_normalizer.decode(y)
            data_loss = loss_fn(output, target)
            div_loss = divergence_loss(output, operators, mask, steps) if args.div_loss else output.new_zeros(())
            loss = data_loss + args.div_loss_weight * div_loss
            require_finite(loss, f"training loss at epoch {epoch}")
            loss.backward()
            check_gradients(model)
            optimizer.step()
            scheduler.step()
            data_total += data_loss.item()
            div_total += div_loss.item()
        wandb.log({
            "train_data_loss": data_total / args.ntrain,
            "train_div_loss": div_total / args.ntrain,
            "train_loss": (data_total + args.div_loss_weight * div_total) / args.ntrain,
        }, step=epoch)
        print(f"epoch={epoch} data_loss={data_total / args.ntrain:.6g}", flush=True)
        if (epoch + 1) % args.checkpoint_every == 0:
            artifacts.save(epoch, model, normalizers, geometry, (optimizer,), (scheduler,))
    artifacts.save(epoch, model, normalizers, geometry, (optimizer,), (scheduler,))
    artifacts.log({"total_train_time": artifacts.elapsed + time.perf_counter() - start_time})
    model.eval()
    predictions, test_loss = evaluate(
        model, positions, tests, test_targets, output_normalizer, output_indices, args.batch_size,
    )
    artifacts.log({"test_loss": test_loss})
    if args.calc_div:
        artifacts.log(summarize_divergence(predictions.to(device), operators, mask, steps))
    if not args.no_ood:
        ood = load_ood_dataset(args.dataset, count, args.data_root)
        artifacts.log({"ood_available": ood is not None})
        if ood is not None:
            x_points, y_points, x, y = ood
            if not (np.allclose(x_points, data.input_points) and np.allclose(y_points, data.output_points)):
                raise ValueError("OOD grid differs from the training grid")
            x = input_normalizer.encode(pad_inputs(x, len(grid), input_indices).to(device))
            y = torch.tensor(y, dtype=torch.float32, device=device)
            pooled_ood = args.dataset in {"taylor_green", "taylor_green_spacetime"}
            _, ood_loss = evaluate(
                model, positions, x, y, output_normalizer, output_indices, args.batch_size,
                pooled=pooled_ood,
            )
            artifacts.log({"ood_loss": ood_loss,
                           "ood_metric": "pooled_magnitude_relative_l2" if pooled_ood
                           else "mean_magnitude_relative_l2"})
    if args.save and args.calc_div:
        args.div_folder.mkdir(parents=True, exist_ok=True)
        savemat(args.div_folder / f"{name}.mat", {
            "x_grid": data.output_points, "y_preds_test": predictions.numpy(),
        })
    wandb.finish()


if __name__ == "__main__":
    main()
