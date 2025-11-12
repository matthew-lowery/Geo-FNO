import torch.nn.functional as F
import matplotlib.pyplot as plt
from timeit import default_timer
import sys
sys.path.append('..')
from utilities3 import *
from Adam import Adam
import numpy as np
import os, copy
from model import FNO2d, IPHI
import wandb
import time
from calc_div_local_wls_poly import calc_div

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
parser.add_argument('--epochs', type=int, default=1_000)
parser.add_argument('--norm-grid', action='store_true')
parser.add_argument('--batch-size', type=int, default=50)
parser.add_argument('--wandb', action='store_true')
parser.add_argument('--save', action='store_true')
parser.add_argument('--calc-div', action='store_true')
parser.add_argument('--div-folder', type=str, default='/projects/bfel/mlowery/geo-fno_divs')
parser.add_argument('--model-folder', type=str, default='/projects/bfel/mlowery/geo-fno_models')
parser.add_argument('--dataset', type=str, default='backward_facing_step', choices=['backward_facing_step', 
                                                                                    'buoyancy_cavity_flow', 
                                                                                    'flow_cylinder_laminar', 
                                                                                    'flow_cylinder_shedding', 
                                                                                    'lid_cavity_flow', 
                                                                                    'merge_vortices', 
                                                                                    'taylor_green_exact', 
                                                                                    'taylor_green_numerical',
                                                                                    "merge_vortices_easier",
                                                                                    "backward_facing_step_ood"
                                                                                    ])


args = parser.parse_args()
print(args)
name = f"{args.dataset}_{args.seed}_{args.ntrain}_{args.npoints}"
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
train_x = x_normalizer.encode(train_x) ### normalize x before subsampling
test_x = x_normalizer.encode(test_x)

### training functions setup
train_x_sub = train_x[:, subsample_idx]
x_grid_sub = x_grid[subsample_idx]
train_x_grid_sub = x_grid_sub.unsqueeze(0).repeat(ntrain, *([1] * x_grid.ndim))
train_y_sub = train_y[:, subsample_idx]

### testing functions setup
test_x_sub = test_x[:, subsample_idx]
test_x_grid_sub = x_grid_sub.unsqueeze(0).repeat(ntest, *([1] * x_grid.ndim))
test_y_sub =  test_y[:, subsample_idx]
test_x_grid = x_grid.unsqueeze(0).repeat(ntest, *([1] * x_grid.ndim))

print(f'{train_x_sub.shape=}, {train_x_grid_sub.shape=}, {train_y_sub.shape=}, {test_x.shape=}, {test_y.shape=}, {test_x_grid.shape=}')

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_x_sub, train_x_grid_sub, train_y_sub, train_x_grid_sub), 
                                                                            batch_size=batch_size, shuffle=True) 

test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_x, test_x_grid, test_y, test_x_grid), 
                                            batch_size=batch_size, shuffle=False) 

test_loader_sub = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_x_sub, test_x_grid_sub, test_y_sub, test_x_grid_sub),
                                              batch_size=batch_size, shuffle=False) 

### each normalizer based on train, but each used on the model's predicted train/test output functions
y_normalizer = UnitGaussianNormalizer(train_y)
y_normalizer_sub = UnitGaussianNormalizer(train_y_sub)
y_normalizer.cuda(); y_normalizer_sub.cuda()


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

t1 = time.perf_counter()
for ep in range(epochs):
    model.train()
    train_l2 = 0
    train_reg = 0

    for x, x_grid, y, y_grid in train_loader:
        x, x_grid, y, y_grid = x.cuda(), x_grid.cuda(), y.cuda(), y_grid.cuda()

        optimizer_fno.zero_grad()
        optimizer_iphi.zero_grad() 
        inp = torch.concat((x, x_grid), axis=-1) ### nbatch, n, 3
        out = model(inp, code=None, x_in=x_grid, x_out=y_grid, iphi=model_iphi)
        out = y_normalizer_sub.decode(out)
        loss = myloss(out.view(batch_size, -1), y.view(batch_size, -1))
        loss.backward()
        optimizer_fno.step()
        optimizer_iphi.step()
        train_l2 += loss.item()

    scheduler_fno.step()
    scheduler_iphi.step()

    model.eval()
    test_l2 = 0.0
   
    eval_t1 = time.perf_counter()
    with torch.no_grad():
        for x, x_grid, y, y_grid in test_loader:
            x, x_grid, y, y_grid = x.cuda(), x_grid.cuda(), y.cuda(), y_grid.cuda()
            inp = torch.concat((x, x_grid), axis=-1) ### nbatch, n, 3
            out = model(inp, code=None, x_in=x_grid, x_out=y_grid, iphi=model_iphi) 
            out = y_normalizer.decode(out)
            out = torch.linalg.norm(out, dim=-1) ### (batch, pts, 2) --> (batch, pts)
            y = torch.linalg.norm(y, dim=-1)
            test_l2 += myloss(out.view(batch_size, -1), y.view(batch_size, -1)).item()
    eval_t2 = time.perf_counter()

    test_l2_sub = 0.0
    eval_t1_sub = time.perf_counter()
    with torch.no_grad():
        for x, x_grid, y, y_grid in test_loader_sub:
            x, x_grid, y, y_grid = x.cuda(), x_grid.cuda(), y.cuda(), y_grid.cuda()
            inp = torch.concat((x, x_grid), axis=-1) ### nbatch, n, 3
            out = model(inp, code=None, x_in=x_grid, x_out=y_grid, iphi=model_iphi) 
            out = y_normalizer_sub.decode(out)
            out = torch.linalg.norm(out, dim=-1) ### (batch, pts, 2) --> (batch, pts)
            y = torch.linalg.norm(y, dim=-1)
            test_l2_sub += myloss(out.view(batch_size, -1), y.view(batch_size, -1)).item()
    eval_t2_sub = time.perf_counter()

    train_l2 /= ntrain
    test_l2 /= ntest
    test_l2_sub /= ntest

    print(ep, 'eval_time:', eval_t2-eval_t1, f'{train_l2=}', f'{test_l2=}', f'{test_l2_sub=}')
    wandb.log({"train_loss": train_l2, "test_loss": test_l2, "test_loss_sub": test_l2_sub, "eval_time": eval_t2 - eval_t1}, step=ep)


t2 = time.perf_counter()
wandb.log({"total_train_time": t2-t1}, step=ep)
print('total_train_time', t2-t1)

### collect model output for divergence calculation
if args.calc_div:
    y_preds_test = []
    
    with torch.no_grad():
        for x, x_grid, y, y_grid in test_loader_sub:
            x, x_grid, y, y_grid = x.cuda(), x_grid.cuda(), y.cuda(), y_grid.cuda()
            inp = torch.concat((x, x_grid), axis=-1) ### nbatch, n, 3
            out = model(inp, code=None, x_in=x_grid, x_out=y_grid, iphi=model_iphi) 
            out = y_normalizer_sub.decode(out)
            y_preds_test.append(out)
    y_preds_test = torch.stack(y_preds_test).reshape(ntest, -1, 2)
    ### divergence calculation in jax and saving
    import jax; import jax.numpy as jnp
    y_preds_test_jnp = jnp.asarray(y_preds_test, dtype=jnp.float64)
    x_grid_jnp = jnp.asarray(data['x_grid'][subsample_idx], dtype=jnp.float64) ### use the original f64 points, might as well
    torch.cuda.empty_cache()
    divs = jax.vmap(calc_div, in_axes=(0, None))(y_preds_test_jnp, x_grid_jnp)
    print(f'{jnp.max(jnp.abs(divs))=}, {jnp.mean(divs)=}')
    os.makedirs(args.div_folder, exist_ok=True)
    jnp.save(os.path.join(args.div_folder, name), divs)

### saving model for later use
if args.save:
    os.makedirs(args.model_folder, exist_ok=True)
    torch.save({
    "model_state_dict": model.state_dict(),
    }, os.path.join(args.model_folder, f'{name}.torch'))


