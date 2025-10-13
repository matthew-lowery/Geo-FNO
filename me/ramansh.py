import torch.nn.functional as F
import matplotlib.pyplot as plt
from timeit import default_timer
import sys
sys.path.append('..')
from utilities3 import *
from Adam import Adam
import numpy as np
import os, copy
from model_cpu import FNO2d, IPHI
import wandb

def set_seed(seed):    
    torch.manual_seed(seed)
    np.random.seed(seed)
    # torch.cuda.manual_seed(seed)

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
parser.add_argument('--epochs', type=int, default=1_000)
parser.add_argument('--norm-grid', action='store_true')
parser.add_argument('--batch-size', type=int, default=20)
parser.add_argument('--wandb', action='store_true')
parser.add_argument('--dataset', type=str, default='backward_facing_step', choices=['backward_facing_step', 
                                                                                    'buoyancy_cavity_flow', 
                                                                                    'flow_cylinder_laminar', 
                                                                                    'flow_cylinder_shedding', 
                                                                                    'lid_cavity_flow', 
                                                                                    'merge_vortices', 
                                                                                    'taylor_green_exact', 
                                                                                    'taylor_green_numerical'])


args = parser.parse_args()
print(args)

if not args.wandb:
    os.environ["WANDB_MODE"] = "disabled"
wandb.login(key='d612cda26a5690e196d092756d668fc2aee8525b')
wandb.init(project=args.dataset)
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
data = np.load(f'./data/{args.dataset}.npz')

x_grid = data['x_grid']
train_x, test_x, train_y, test_y = data['x_train'], data['x_test'], data['y_train'], data['y_test']
train_x, train_y = train_x[:ntrain], train_y[:ntrain]

### norm rect domain to [0,1]^2
if args.norm_grid:
    x_grid_min, x_grid_max = np.min(x_grid, axis=0, keepdims=True), np.max(x_grid, axis=0, keepdims=True)
    x_grid = (x_grid- x_grid_min) / (x_grid_max - x_grid_min)

### in dimensions and out dimensions
in_channels = train_x.shape[-1] + x_grid.shape[-1]
out_channels = train_y.shape[-1]

########################################################################################
### Here we are subsampling points in the training functions, but not in the test functions, which means our pointwise normalizer,
### made from the training functions, and used on both training+test functions, must be amenable to both the subsampled grid and the full grid. 
### So we just make a normalizer for both.

if args.npoints != 'all':
    def subsample_points(arr, d):
        t = arr.shape[0] // d
        kept_indices = np.arange(arr.shape[0])[::t]
        r = arr[kept_indices, :]
        removed_indices = np.linspace(0, kept_indices.shape[0] - 1, kept_indices.shape[0] - d, dtype=np.int32)
        removed_from_kept = kept_indices[removed_indices]
        final_kept_indices = np.setdiff1d(kept_indices, removed_from_kept)
        r = arr[final_kept_indices]
        return r, final_kept_indices
    _, subsample_idx = subsample_points(x_grid, int(args.npoints))
else:
    ### full grid indices
    subsample_idx = np.arange(len(x_grid))

### move to torch as the normalizers are written in torch and everything subsequently also
train_x = torch.tensor(train_x, dtype=torch.float32)
test_x =  torch.tensor(test_x, dtype=torch.float32)
train_y = torch.tensor(train_y, dtype=torch.float32)
test_y = torch.tensor(test_y, dtype=torch.float32)
x_grid = torch.tensor(x_grid, dtype=torch.float32)
subsample_idx = torch.tensor(subsample_idx, dtype=torch.int32)

x_normalizer = UnitGaussianNormalizer(train_x)
train_x = x_normalizer.encode(train_x)
test_x = x_normalizer.encode(test_x)

### each normalizer based on train, but each used on the model's predicted train/test output functions
y_normalizer = UnitGaussianNormalizer(train_y)
y_normalizer_train = copy.deepcopy(y_normalizer)
y_normalizer_train.mean = y_normalizer_train.mean[subsample_idx]
y_normalizer_train.std = y_normalizer_train.std[subsample_idx]

y_normalizer.cuda()
y_normalizer_train.cuda()

### subsampling the training functions now
train_x = train_x[:, subsample_idx]
train_x_grid = x_grid[subsample_idx]
train_x_grid = train_x_grid.unsqueeze(0).repeat(ntrain, *([1] * x_grid.ndim))
train_y =  train_y[:, subsample_idx]

test_x_grid = x_grid.unsqueeze(0).repeat(ntest, *([1] * x_grid.ndim))

print(f'{train_x.shape=}, {train_x_grid.shape=}, {train_y.shape=}, {test_x.shape=}, {test_y.shape=}, {test_x_grid.shape=}')

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_x, train_x_grid, train_y, train_x_grid), 
                                                                            batch_size=batch_size, shuffle=True) 

test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_x, test_x_grid, test_y, test_x_grid), 
                                            batch_size=batch_size, shuffle=False) 

################################################################
# training and evaluation
################################################################

model = FNO2d(modes, modes, width, in_channels=in_channels, out_channels=out_channels, is_mesh=False, s1=args.res1d, s2=args.res1d).cuda()
model_iphi = IPHI().cuda()
print(count_params(model), count_params(model_iphi))

optimizer_fno = Adam(model.parameters(), lr=learning_rate_fno, weight_decay=1e-4)
scheduler_fno = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_fno, T_max = epochs)
optimizer_iphi = Adam(model_iphi.parameters(), lr=learning_rate_iphi, weight_decay=1e-4)
scheduler_iphi = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_iphi, T_max = epochs)

myloss = LpLoss(size_average=False)
for ep in range(epochs):
    model.train()
    t1 = default_timer()
    train_l2 = 0
    train_reg = 0

    for x, x_grid, y, y_grid in train_loader:
        x, x_grid, y, y_grid = x.cuda(), x_grid.cuda(), y.cuda(), y_grid.cuda()

        optimizer_fno.zero_grad()
        optimizer_iphi.zero_grad() 
        inp = torch.concat((x, x_grid), axis=-1) ### nbatch, n, 3
        out = model(inp, code=None, x_in=x_grid, x_out=y_grid, iphi=model_iphi)
        out = y_normalizer_train.decode(out)
        loss = myloss(out.view(batch_size, -1), y.view(batch_size, -1))
        loss.backward()
        # print(loss)
        optimizer_fno.step()
        optimizer_iphi.step()
        train_l2 += loss.item()

    scheduler_fno.step()
    scheduler_iphi.step()

    model.eval()
    test_l2 = 0.0
    with torch.no_grad():
        for x, x_grid, y, y_grid in test_loader:
            x, x_grid, y, y_grid = x, x_grid, y, y_grid
            # print(rr.shape, sigma.shape, mesh.shape) ## 20,42 ; 20, 972, 1 ; 20, 972, 2
            # rr, sigma, mesh = rr, sigma, mesh
            inp = torch.concat((x, x_grid), axis=-1) ### nbatch, n, 3
            out = model(inp, code=None, x_in=x_grid, x_out=y_grid, iphi=model_iphi) ### self, u, code=None, x_in=None, x_out=None, iphi=None
            out = y_normalizer.decode(out)
            test_l2 += myloss(out.view(batch_size, -1), y.view(batch_size, -1)).item()

    train_l2 /= ntrain
    test_l2 /= ntest

    t2 = default_timer()
    print(ep, t2 - t1, train_l2, f'{test_l2=}')
    wandb.log({"test_loss": test_l2}, step=ep)