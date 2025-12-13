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
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--norm-grid', action='store_true')
parser.add_argument('--batch-size', type=int, default=20)
parser.add_argument('--wandb', action='store_true')
parser.add_argument('--save', action='store_true')
parser.add_argument('--calc-div', action='store_true')
parser.add_argument('--project-name', type=str, default='ramansh')
parser.add_argument('--div-folder', type=str, default='/projects/bfel/mlowery/geo-fno_divs')
parser.add_argument('--dir', type=str, default='/projects/bfel/mlowery/geo-fno')
parser.add_argument('--model-folder', type=str, default='/projects/bfel/mlowery/geo-fno_models')
parser.add_argument('--dataset', type=str, default='taylor_green_time', choices=['taylor_green_time', 'species_transport', 'taylor_green_time_coeffs'])
                                                                      
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
data = np.load(os.path.join(args.dir, f'{args.dataset}.npz'))
# data = np.load(f'../../ram_dataset/geo-fno/{args.dataset}.npz')

x_grid = data['x_grid']; y_grid = data['y_grid']
train_x, test_x, train_y, test_y = data['x_train'], data['x_test'], data['y_train'], data['y_test']

if train_x.ndim == 2: train_x = train_x[...,None]
if test_x.ndim == 2: test_x = test_x[...,None]

train_x, train_y = train_x[:ntrain], train_y[:ntrain]

### basically norm domain to \in [0,1]^d
if args.dataset == 'taylor_green_time':
    xs = x_grid_spatial = x_grid[:,:2] ## t is already in [0,1]
    ys = y_grid[:,:2]
    min, max = np.min(xs, axis=0, keepdims=True), np.max(xs, axis=0, keepdims=True)
    xs_norm = (xs - min) / (max - min)
    ys_norm = (ys - min) / (max - min)
    x_grid[:,:2] = xs_norm
    y_grid[:,:2] = ys_norm
elif args.dataset == 'species_transport':
    ## x_grid is a subset of y_grid so norm with y_grid
    min, max = np.min(y_grid, axis=0, keepdims=True), np.max(y_grid, axis=0, keepdims=True)
    x_grid = (x_grid - min) / (max - min)
    y_grid = (y_grid - min) / (max - min)
elif args.dataset == 'taylor_green_time_coeffs':
    ## x_grid is a subset of y_grid so norm with y_grid
    min, max = np.min(y_grid, axis=0, keepdims=True), np.max(y_grid, axis=0, keepdims=True)
    y_grid = (y_grid - min) / (max - min) ## x grid is already [0,1]^3


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
    train_t1 = time.perf_counter()
    for x, x_grid, y, y_grid in train_loader:
        x, x_grid, y, y_grid = x.cuda(), x_grid.cuda(), y.cuda(), y_grid.cuda()

        optimizer_fno.zero_grad()
        optimizer_iphi.zero_grad() 
        inp = torch.concat((x, x_grid), axis=-1) ### nbatch, n, 3
        out = model(inp, code=None, x_in=x_grid, x_out=y_grid, iphi=model_iphi)
        out = y_normalizer.decode(out)
        loss = myloss(out.view(batch_size, -1), y.view(batch_size, -1))
        loss.backward()
        optimizer_fno.step()
        optimizer_iphi.step()
        train_l2 += loss.item()
        print(loss.item())
    train_t2 = time.perf_counter() 

    scheduler_fno.step()
    scheduler_iphi.step()
     
    train_l2 /= ntrain
    print(ep, 'train_time:', train_t2-train_t1, f'{train_l2=}')
    wandb.log({"train_loss": train_l2, "train_time": train_t2-train_t1}, step=ep)

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

