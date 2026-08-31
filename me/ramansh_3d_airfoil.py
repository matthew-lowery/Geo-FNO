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
from divergence_metrics import summarize_divergence
from itertools import product
from scipy.linalg import lstsq
from scipy.spatial import cKDTree


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
        pairwise = np.linalg.norm(
            stencil_points[:, None] - stencil_points[None, :], axis=-1
        )
        scale = distances[-1]
        local_points = (stencil_points - center) / scale
        polys = np.prod(
            local_points[:, None, :] ** poly_powers[None, :, :], axis=-1
        )
        system = np.block([
            [pairwise ** rbf_power, polys],
            [polys.T, np.zeros((poly_count, poly_count))],
        ])

        derivative = np.zeros((stencil_size + poly_count, spatial_dim))
        derivative[:stencil_size] = (
            (center - stencil_points)
            * rbf_power
            * (pairwise[0, :, None] + eps) ** (rbf_power - 2)
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
parser.add_argument('--npoints', type=str, default='all')
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--norm-grid', action='store_true')
parser.add_argument('--batch-size', type=int, default=20)
parser.add_argument('--project-name', type=str, default='ramansh')
parser.add_argument('--wandb', action='store_true')
parser.add_argument('--save', action='store_true')
parser.add_argument('--calc-div', action='store_true')
parser.add_argument('--div-loss', action='store_true')
parser.add_argument('--div-loss-weight', type=float, default=1.0)
parser.add_argument('--dir', type=str, default='/projects/bfel/mlowery/geo-fno')
parser.add_argument('--div-folder', type=str, default='/projects/bfel/mlowery/geo-fno_divs')
parser.add_argument('--model-folder', type=str, default='/projects/bfel/mlowery/geo-fno_models')
parser.add_argument('--dataset', type=str, default='airfoil')


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

########## load data ########################################################################
#data = np.load(os.path.join(args.dir, f'{args.dataset}.npz'))
#data = np.load(f'../../ram_dataset/geo-fno/{args.dataset}.npz')
data = np.load('data/geo-fno-new/airfoil.npz')
x_grid = data['x_grid']; y = data['y']; y_grid = data['y_grid'] # (10210, 2000, 2) (10210, 1000, 3) (1000, 3)

x_grid_train = x_grid[:ntrain]
x_grid_test = x_grid[-ntest:]
y_train, y_test = y[:ntrain], y[-ntest:]

#### range normalize input coords to [0,1]^2
grid_min, grid_max = np.min(x_grid_train, axis=(0,1), keepdims=True), np.max(x_grid_train, axis=(0,1),keepdims=True)
x_grid_train = (x_grid_train - grid_min) / ((grid_max-grid_min) + 1e-6)
x_grid_test = (x_grid_test - grid_min) / ((grid_max-grid_min) + 1e-6)

### range normalize output coords to [0,1]^3
grid_min, grid_max = np.min(y_grid, axis=(0,1), keepdims=True), np.max(y_grid, axis=(0,1),keepdims=True)
y_grid = (y_grid - grid_min) / ((grid_max-grid_min) + 1e-6)
print(y_grid.shape)
x_grid_train = np.concatenate((x_grid_train, np.zeros((len(x_grid_train),x_grid_train.shape[1], 1))), axis=-1) ## pad to make 3d
x_grid_test = np.concatenate((x_grid_test, np.zeros((len(x_grid_test),x_grid_test.shape[1], 1))), axis=-1) ## pad to make 3d
print(x_grid_train.shape, x_grid_test.shape)

### in dimensions and out dimensions
in_channels = x_grid_train.shape[-1]
out_channels = y_train.shape[-1]
gradient_operators = None
interior_mask = None
if args.div_loss or args.calc_div:
    physical_grid = data['y_grid'][:, :out_channels]
    gradient_operators = tuple(
        operator.cuda()
        for operator in build_rbf_fd_gradient(physical_grid)
    )
    interior_mask = build_interior_mask(physical_grid).cuda()

### move to torch as the normalizers are written in torch and everything subsequently also
x_grid_train = torch.tensor(x_grid_train, dtype=torch.float32)
x_grid_test =  torch.tensor(x_grid_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)
y_grid = torch.tensor(y_grid, dtype=torch.float32)

y_grid_train = y_grid.unsqueeze(0).repeat(ntrain,1,1)
y_grid_test = y_grid.unsqueeze(0).repeat(ntest,1,1)

y_normalizer = UnitGaussianNormalizer(y_train)
y_normalizer.cuda()


train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_grid_train, y_train, y_grid_train), 
                                                                            batch_size=batch_size, shuffle=True) 

test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_grid_test, y_test, y_grid_test), 
                                            batch_size=batch_size, shuffle=False) 


################################################################
# training and evaluation
################################################################

model = FNO3d(modes, width, in_channels=in_channels, out_channels=out_channels, is_mesh=False, s1=args.res1d, s2=args.res1d).cuda()
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
    for x_grid, y, y_grid in train_loader:
        x_grid, y, y_grid = x_grid.cuda(), y.cuda(), y_grid.cuda()

        optimizer_fno.zero_grad()
        optimizer_iphi.zero_grad() 
        out = model(x_grid, code=None, x_in=x_grid, x_out=y_grid, iphi=model_iphi)
        out = y_normalizer.decode(out)
        data_loss = myloss(out.view(batch_size, -1), y.view(batch_size, -1))
        div_loss = divergence_loss(out, gradient_operators, interior_mask) if args.div_loss else out.new_zeros(())
        loss = data_loss + args.div_loss_weight * div_loss
        loss.backward()
        optimizer_fno.step()
        optimizer_iphi.step()
        train_l2 += data_loss.item()
        train_div += div_loss.item()
        train_total += loss.item()
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
    for x_grid, y, y_grid in test_loader:
        x_grid, y, y_grid = x_grid.cuda(), y.cuda(), y_grid.cuda()
        out = model(x_grid, code=None, x_in=x_grid, x_out=y_grid, iphi=model_iphi) 
        out = y_normalizer.decode(out)
        out = torch.linalg.norm(out, dim=-1) ### (batch, pts, 2) --> (batch, pts)
        y = torch.linalg.norm(y, dim=-1)
        test_l2 += myloss(out.view(batch_size, -1), y.view(batch_size, -1)).item()
eval_t2 = time.perf_counter()
test_l2 /= ntest

print(ep, 'eval_time:', eval_t2-eval_t1, f'{test_l2=}')
wandb.log({"test_loss": test_l2, "eval_time": eval_t2 - eval_t1}, step=ep)


t2 = time.perf_counter()
wandb.log({"total_train_time": t2-t1}, step=ep)
print('total_train_time', t2-t1)

### collect model output for divergence calculation
if args.calc_div:
    y_preds_test = []
    
    with torch.no_grad():
        for x_grid, y, y_grid in test_loader:
            x_grid, y, y_grid = x_grid.cuda(), y.cuda(), y_grid.cuda()
            out = model(x_grid, code=None, x_in=x_grid, x_out=y_grid, iphi=model_iphi) 
            out = y_normalizer.decode(out)
            y_preds_test.append(out)
    y_preds_test = torch.stack(y_preds_test).reshape(ntest, -1, 2)
    wandb.log(summarize_divergence(y_preds_test, gradient_operators, interior_mask), commit=True)

### saving model for later use
if args.save:
    os.makedirs(args.model_folder, exist_ok=True)
    torch.save({
    "model_state_dict": model.state_dict(),
    }, os.path.join(args.model_folder, f'{name}.torch'))

    ### saving test output functions for div calc 
    os.makedirs(args.div_folder, exist_ok=True)
    scipy.io.savemat(os.path.join(args.div_folder, f'{name}.mat'), {'x_grid': data['x_grid'],
                                                                    'y_preds_test': y_preds_test.cpu().numpy().astype(np.float64)})
