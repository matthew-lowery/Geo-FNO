import torch.nn.functional as F
import matplotlib.pyplot as plt
from timeit import default_timer
import sys
sys.path.append('..')
from utilities3 import *
from Adam import Adam
import numpy as np
import os, copy
from model_3d import FNO3d, IPHI
import wandb
import time
import scipy
from itertools import product
from scipy.linalg import lstsq
from scipy.spatial import cKDTree
from ram_dataset_loader import load_dataset, load_ood_dataset


def build_rbf_fd_gradient(points, order=5):
    points = np.asarray(points, dtype=np.float64)
    spatial_dim = points.shape[1]
    poly_powers = np.asarray([
        powers for powers in product(range(order + 1), repeat=spatial_dim)
        if sum(powers) <= order
    ])
    poly_count = len(poly_powers)
    stencil_size = 2 * poly_count + 1
    if stencil_size > len(points):
        raise ValueError(f"RBF-FD stencil needs {stencil_size} points, got {len(points)}")

    rbf_power = order if order % 2 else order - 1
    rbf_power = min(max(rbf_power, 5), 11)
    tree = cKDTree(points)
    rows = np.repeat(np.arange(len(points)), stencil_size)
    columns = np.empty_like(rows)
    weights = np.empty((spatial_dim, len(rows)), dtype=np.float64)
    eps = np.finfo(np.float64).eps

    for center_idx, center in enumerate(points):
        distances, stencil = tree.query(center, k=stencil_size)
        stencil_points = points[stencil]
        scale = distances[-1]
        if not np.isfinite(scale) or scale <= eps:
            raise ValueError("RBF-FD stencil contains coincident points")
        local_points = (stencil_points - center) / scale
        pairwise = np.linalg.norm(
            local_points[:, None] - local_points[None, :], axis=-1
        )
        polys = np.prod(
            local_points[:, None, :] ** poly_powers[None, :, :], axis=-1
        )
        system = np.block([
            [pairwise ** rbf_power, polys],
            [polys.T, np.zeros((poly_count, poly_count))],
        ])

        derivative = np.zeros((stencil_size + poly_count, spatial_dim))
        derivative[:stencil_size] = (
            -local_points
            * rbf_power
            * (pairwise[0, :, None] + eps) ** (rbf_power - 2)
            / scale
        )
        for axis in range(spatial_dim):
            first_power = np.zeros(spatial_dim, dtype=int)
            first_power[axis] = 1
            poly_idx = np.flatnonzero(np.all(poly_powers == first_power, axis=1))[0]
            derivative[stencil_size + poly_idx, axis] = 1 / scale

        local_weights = lstsq(
            system, derivative, lapack_driver='gelsy', check_finite=False
        )[0]
        block = slice(center_idx * stencil_size, (center_idx + 1) * stencil_size)
        columns[block] = stencil
        weights[:, block] = local_weights[:stencil_size].T

    indices = torch.tensor(np.stack((rows, columns)), dtype=torch.long)
    return tuple(
        torch.sparse_coo_tensor(
            indices,
            torch.tensor(axis_weights, dtype=torch.float32),
            (len(points), len(points)),
        ).coalesce()
        for axis_weights in weights
    )


def build_interior_mask(points, cylindrical=False):
    points = np.asarray(points)
    if cylindrical:
        radius = np.linalg.norm(points[:, :2], axis=1)
        boundary = (
            (points[:, 2] == points[:, 2].min())
            | (points[:, 2] == points[:, 2].max())
            | np.isclose(radius, radius.max())
        )
    else:
        spans = np.ptp(points, axis=0)
        active_axes = spans > 100 * np.finfo(points.dtype).eps
        active_points = points[:, active_axes]
        boundary = np.any(
            (active_points == active_points.min(axis=0))
            | (active_points == active_points.max(axis=0)),
            axis=1,
        )
    if boundary.all():
        raise ValueError("Divergence loss has no interior points")
    return torch.tensor(~boundary, dtype=torch.bool)


def divergence_loss(vector_field, gradient_operators, interior_mask, time_steps=1):
    batch_size, _, vector_dim = vector_field.shape
    vector_field = vector_field.reshape(
        batch_size, -1, time_steps, vector_dim
    ).permute(0, 2, 1, 3).reshape(batch_size * time_steps, -1, vector_dim)
    divergence = sum(
        torch.sparse.mm(operator, vector_field[..., axis].T).T
        for axis, operator in enumerate(gradient_operators)
    )
    divergence = divergence[:, interior_mask]
    return divergence.square().mean(dim=1).sum() / time_steps

def set_seed(seed):    
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)

torch.backends.cudnn.deterministic = True

################################################################
# configs
################################################################

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--modes', type=int, default=12)
parser.add_argument('--res1d', type=int, default=40)
parser.add_argument('--width', type=int, default=32)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--lr-phi', type=float, default=1e-4)
parser.add_argument('--lr-fno', type=float, default=1e-3)
parser.add_argument('--ntrain', type=int, default=1_000)
parser.add_argument('--npoints', type=str, default='2700')
parser.add_argument('--data-root', type=str, default='/projects/bgcs/mlowery/ram_dataset')
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--norm-grid', action='store_true')
parser.add_argument('--batch-size', type=int, default=20)
parser.add_argument('--wandb', action='store_true')
parser.add_argument('--save', action='store_true')
parser.add_argument('--calc-div', action='store_true')
parser.add_argument('--div-loss', action='store_true')
parser.add_argument('--div-loss-weight', type=float, default=1.0)
parser.add_argument('--project-name', type=str, default='ramansh')
parser.add_argument('--div-folder', type=str, default='/projects/bfel/mlowery/geo-fno_divs')
parser.add_argument('--dir', type=str)
parser.add_argument('--model-folder', type=str, default='/projects/bfel/mlowery/geo-fno_models')
parser.add_argument('--dataset', type=str, default='taylor_green_time', choices=['taylor_green_time', 'taylor_green_spacetime', 'species_transport', 'taylor_green_time_coeffs', 'taylor_green_spacetime_coeffs', 'forced_turb'])
parser.add_argument('--no-ood', dest='eval_ood', action='store_false')
parser.set_defaults(eval_ood=True)
                                                                      
args = parser.parse_args()
print(args)
name = f"{args.dataset}_{args.seed}_{args.ntrain}_{args.npoints}"
if not args.wandb:
    os.environ["WANDB_MODE"] = "disabled"
wandb.login(key='d612cda26a5690e196d092756d668fc2aee8525b')
wandb.init(project=args.project_name, name=f'{name}')
wandb.config.update(args)

set_seed(args.seed)
batch_size = args.batch_size
learning_rate_fno = args.lr_fno
learning_rate_iphi = args.lr_phi

epochs = args.epochs
ntrain,ntest = args.ntrain, 200 ### ntest is always 200 for ram's problems

modes = args.modes
width = args.width

########### load data ########################################################################
point_count = None if args.npoints == 'all' else int(args.npoints)
legacy_filename = {
    'taylor_green_time': 'taylor_green_time',
    'taylor_green_spacetime': 'taylor_green_time',
    'taylor_green_time_coeffs': 'taylor_green_time_coeffs',
    'taylor_green_spacetime_coeffs': 'taylor_green_time_coeffs',
    'species_transport': 'species_transport',
}.get(args.dataset)
if legacy_filename:
    legacy_dir = args.dir or (
        '/projects/bfel/mlowery/geo-fno'
        if args.dataset == 'species_transport'
        else '/projects/bfel/mlowery/geo-fno-new'
    )
    data = np.load(os.path.join(legacy_dir, f'{legacy_filename}.npz'))
    x_grid, y_grid = data['x_grid'], data['y_grid']
    train_x, test_x = data['x_train'], data['x_test']
    train_y, test_y = data['y_train'], data['y_test']
    train_x, train_y = train_x[:ntrain], train_y[:ntrain]
else:
    dataset = load_dataset(args.dataset, ntrain, point_count, args.data_root)
    x_grid = dataset.input_points
    y_grid = dataset.output_points
    train_x, test_x = dataset.train_input, dataset.test_input
    train_y, test_y = dataset.train_output, dataset.test_output
physical_input_grid = x_grid.copy()
physical_output_grid = y_grid.copy()
ntest = len(test_x)

if train_x.ndim == 2: train_x = train_x[..., None]
if test_x.ndim == 2: test_x = test_x[..., None]

### basically norm domain to \in [0,1]^d
is_spacetime = args.dataset in {'taylor_green_time', 'taylor_green_spacetime'}
is_spacetime_coeffs = args.dataset in {'taylor_green_time_coeffs', 'taylor_green_spacetime_coeffs'}
if is_spacetime:
    xs = x_grid_spatial = x_grid[:,:2] ## t is already in [0,1]
    ys = y_grid[:,:2]
    grid_min, grid_max = np.min(xs, axis=0, keepdims=True), np.max(xs, axis=0, keepdims=True)
    xs_norm = (xs - grid_min) / (grid_max - grid_min)
    ys_norm = (ys - grid_min) / (grid_max - grid_min)
    x_grid[:,:2] = xs_norm
    y_grid[:,:2] = ys_norm
elif args.dataset == 'species_transport':
    ## x_grid is a subset of y_grid so norm with y_grid
    grid_min, grid_max = np.min(y_grid, axis=0, keepdims=True), np.max(y_grid, axis=0, keepdims=True)
    x_grid = (x_grid - grid_min) / (grid_max - grid_min)
    y_grid = (y_grid - grid_min) / (grid_max - grid_min)
elif is_spacetime_coeffs:
    grid_min = np.min(y_grid, axis=0, keepdims=True)
    grid_max = np.max(y_grid, axis=0, keepdims=True)
    y_grid = (y_grid - grid_min) / (grid_max - grid_min)
elif args.dataset == 'forced_turb' and args.norm_grid:
    grid_min, grid_max = np.min(y_grid, axis=0, keepdims=True), np.max(y_grid, axis=0, keepdims=True)
    x_grid = (x_grid - grid_min) / (grid_max - grid_min)
    y_grid = (y_grid - grid_min) / (grid_max - grid_min)


### move to torch as the normalizers are written in torch and everything subsequently also
train_x = torch.tensor(train_x, dtype=torch.float32)
test_x =  torch.tensor(test_x, dtype=torch.float32)
train_y = torch.tensor(train_y, dtype=torch.float32)
test_y = torch.tensor(test_y, dtype=torch.float32)
x_grid = torch.tensor(x_grid, dtype=torch.float32)
y_grid = torch.tensor(y_grid, dtype=torch.float32)

x_normalizer = UnitGaussianNormalizer(train_x)
train_x = x_normalizer.encode(train_x) ### normalize x before subsampling
test_x = x_normalizer.encode(test_x)
y_normalizer = UnitGaussianNormalizer(train_y)
y_normalizer.cuda()

train_x_grid = x_grid.unsqueeze(0).repeat(ntrain, *([1] * x_grid.ndim))
train_y_grid = y_grid.unsqueeze(0).repeat(ntrain, *([1] * y_grid.ndim))
test_x_grid = x_grid.unsqueeze(0).repeat(ntest, *([1] * x_grid.ndim))
test_y_grid = y_grid.unsqueeze(0).repeat(ntest, *([1] * y_grid.ndim))

print(f'{train_x.shape=}, {train_x_grid.shape=}, {train_y.shape=}, {train_y_grid.shape=}, {test_x.shape=}, {test_y.shape=}')

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_x, train_x_grid, train_y, train_y_grid), 
                                                                            batch_size=batch_size, shuffle=True) 

test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_x, test_x_grid, test_y, test_y_grid),
                                              batch_size=batch_size, shuffle=False) 

################################################################
# training and evaluation
################################################################
### in dimensions and out dimensions
in_channels = train_x.shape[-1] + x_grid.shape[-1]
out_channels = train_y.shape[-1]
gradient_operators = None
interior_mask = None
div_time_steps = 1
if args.div_loss:
    if is_spacetime or is_spacetime_coeffs:
        div_time_steps = 4
        physical_grid = physical_output_grid.reshape(-1, div_time_steps, 3)[:, 0, :2]
    elif args.dataset == 'forced_turb':
        physical_grid = physical_output_grid
    else:
        physical_grid = physical_output_grid[:, :out_channels]
    gradient_operators = tuple(
        operator.cuda() for operator in build_rbf_fd_gradient(physical_grid)
    )
    interior_mask = build_interior_mask(
        physical_grid, cylindrical=args.dataset == 'species_transport'
    ).cuda()

model = FNO3d(modes, width, in_channels=in_channels, out_channels=out_channels, is_mesh=False, s1=args.res1d, s2=args.res1d, s3=args.res1d).cuda()
model_iphi = IPHI().cuda()
print(count_params(model), count_params(model_iphi))

optimizer_fno = Adam(model.parameters(), lr=learning_rate_fno, weight_decay=1e-4)
scheduler_fno = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_fno, T_max = epochs)
optimizer_iphi = Adam(model_iphi.parameters(), lr=learning_rate_iphi, weight_decay=1e-4)
scheduler_iphi = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_iphi, T_max = epochs)

myloss = LpLoss(size_average=False)

t1 = time.perf_counter()
for ep in range(epochs):
    model.train()
    train_l2 = 0
    train_div = 0
    train_total = 0
    train_t1 = time.perf_counter()
    for x, x_grid, y, y_grid in train_loader:
        x, x_grid, y, y_grid = x.cuda(), x_grid.cuda(), y.cuda(), y_grid.cuda()

        optimizer_fno.zero_grad()
        optimizer_iphi.zero_grad() 
        inp = torch.concat((x, x_grid), axis=-1) ### nbatch, n, 3
        out = model(inp, code=None, x_in=x_grid, x_out=y_grid, iphi=model_iphi)
        out = y_normalizer.decode(out)
        data_loss = myloss(out.view(batch_size, -1), y.view(batch_size, -1))
        div_loss = divergence_loss(
            out, gradient_operators, interior_mask, div_time_steps
        ) if args.div_loss else out.new_zeros(())
        loss = data_loss + args.div_loss_weight * div_loss
        loss.backward()
        optimizer_fno.step()
        optimizer_iphi.step()
        train_l2 += data_loss.item()
        train_div += div_loss.item()
        train_total += loss.item()
        print(loss.item())
    train_t2 = time.perf_counter() 

    scheduler_fno.step()
    scheduler_iphi.step()
     
    train_l2 /= ntrain
    train_div /= ntrain
    train_total /= ntrain
    print(ep, 'train_time:', train_t2-train_t1, f'{train_total=}', f'{train_l2=}', f'{train_div=}')
    wandb.log({"train_loss": train_total, "train_data_loss": train_l2,
               "train_div_loss": train_div, "train_time": train_t2-train_t1}, step=ep)

### eval when training is over
model.eval()

test_l2 = 0.0
eval_t1 = time.perf_counter()
with torch.no_grad():
    for x, x_grid, y, y_grid in test_loader:
        x, x_grid, y, y_grid = x.cuda(), x_grid.cuda(), y.cuda(), y_grid.cuda()
        inp = torch.concat((x, x_grid), axis=-1) ### nbatch, n, 3
        out = model(inp, code=None, x_in=x_grid, x_out=y_grid, iphi=model_iphi) 
        out = y_normalizer.decode(out)
        out = torch.linalg.norm(out, dim=-1) ### (batch, pts, 3) --> (batch, pts)
        y = torch.linalg.norm(y, dim=-1)
        test_l2 += myloss(out.view(batch_size, -1), y.view(batch_size, -1)).item()
eval_t2 = time.perf_counter()
test_l2 /= ntest

print(ep, 'eval_time', eval_t2-eval_t1, f'{test_l2=}')
wandb.log({"test_loss": test_l2, "eval_time":eval_t2-eval_t1,
            }, step=ep)

t2 = time.perf_counter()
wandb.log({"total_train_time": t2-t1}, step=ep)
print('total_train_time', t2-t1)

### collect model output for divergence calculation
if args.calc_div:
    y_preds_test = []
    with torch.no_grad():
        for x, x_grid, y, y_grid in test_loader:
            x, x_grid, y, y_grid = x.cuda(), x_grid.cuda(), y.cuda(), y_grid.cuda()
            inp = torch.concat((x, x_grid), axis=-1) ### nbatch, n, 3
            out = model(inp, code=None, x_in=x_grid, x_out=y_grid, iphi=model_iphi) 
            out = y_normalizer.decode(out)
            y_preds_test.append(out)
    y_preds_test = torch.stack(y_preds_test).reshape(ntest, -1, out.shape[-1])

ood_dataset = None
if args.eval_ood:
    ood_dataset = load_ood_dataset(args.dataset, point_count, args.data_root)

if ood_dataset is not None:
    ood_x_grid, ood_y_grid, ood_x, ood_y = ood_dataset
    ood_x = torch.tensor(ood_x, dtype=torch.float32)
    ood_y = torch.tensor(ood_y, dtype=torch.float32)
    if ood_x.ndim == 2:
        ood_x = ood_x[..., None]
    ood_x = x_normalizer.encode(ood_x)
    if args.norm_grid:
        if is_spacetime:
            ood_x_grid = ood_x_grid.copy()
            ood_y_grid = ood_y_grid.copy()
            ood_x_grid[:, :2] = (ood_x_grid[:, :2] - grid_min[:, :2]) / (grid_max[:, :2] - grid_min[:, :2])
            ood_y_grid[:, :2] = (ood_y_grid[:, :2] - grid_min[:, :2]) / (grid_max[:, :2] - grid_min[:, :2])
        elif is_spacetime_coeffs:
            ood_y_grid = (ood_y_grid - grid_min) / (grid_max - grid_min)
        else:
            ood_x_grid = (ood_x_grid - grid_min) / (grid_max - grid_min)
            ood_y_grid = (ood_y_grid - grid_min) / (grid_max - grid_min)
    ood_x_grid = torch.tensor(ood_x_grid, dtype=torch.float32)
    ood_y_grid = torch.tensor(ood_y_grid, dtype=torch.float32)
    ood_x_grid = ood_x_grid.unsqueeze(0).repeat(len(ood_x), 1, 1)
    ood_y_grid = ood_y_grid.unsqueeze(0).repeat(len(ood_x), 1, 1)
    ood_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(ood_x, ood_x_grid, ood_y, ood_y_grid),
        batch_size=batch_size, shuffle=False
    )
    ood_loss = 0.0
    with torch.no_grad():
        for x, x_grid_batch, y, y_grid_batch in ood_loader:
            x, x_grid_batch = x.cuda(), x_grid_batch.cuda()
            y, y_grid_batch = y.cuda(), y_grid_batch.cuda()
            inp = torch.concat((x, x_grid_batch), axis=-1)
            out = model(inp, code=None, x_in=x_grid_batch, x_out=y_grid_batch, iphi=model_iphi)
            out = y_normalizer.decode(out)
            ood_loss += myloss(out.reshape(len(x), -1), y.reshape(len(x), -1)).item()
    ood_loss /= len(ood_x)
    wandb.log({f'ood/{args.dataset}': ood_loss}, step=ep)

### saving model for later use
if args.save:
    os.makedirs(args.model_folder, exist_ok=True)
    torch.save({
    "model_state_dict": model.state_dict(),
    }, os.path.join(args.model_folder, f'{name}.torch'))

    ### saving test output functions for div calc 
    os.makedirs(args.div_folder, exist_ok=True)
    scipy.io.savemat(os.path.join(args.div_folder, f'{name}.mat'), {'x_grid': physical_output_grid,
                                                           'y_preds_test': y_preds_test.cpu().numpy().astype(np.float64)})
