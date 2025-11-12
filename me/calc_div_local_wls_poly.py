from jax import numpy as jnp, random as jr
import jax 
# import equinox as eqx
import orthojax as ojax
import math
from scipy.spatial import cKDTree
import legendre as lg
from matplotlib import pyplot as plt
import os
jax.config.update("jax_enable_x64", True)

def _knn_indices(X, x0, k):
    dists = jnp.linalg.norm(X - x0[None, :], axis=1)
    return jnp.argsort(dists)[:k].astype(jnp.int32)
     
def _wendland_c2(r):
    t = jnp.clip(1.0 - r, a_min=0.0)
    return (t**4) * (4.0*(1.0 - t) + 1.0)

def calc_div_local_wls(f, X, p=8, neighbor_idx=None, rho=1.0, rcond=None):
    """
    Local weighted LS via lstsq on each stencil; divergence at centers.
    X: (M,d), f: (M,d)
    p: target divergence degree; fit degree L = p+1
    neighbor_idx: optional (M,k) int array of stencil indices; else k-NN with given k
    rho: bandwidth scale; h = rho * max physical distance in stencil
    rcond: passed to jnp.linalg.lstsq (None -> default)
    Returns: div (M,)
    """
    
    M, d = X.shape
    L = p + 1 ### polynomial total degree for 1st order differential operator to achieve convergence rate of p
    I = lg.generate_total_degree_multi_indices(L, d)  # |I| = n
    k = n_stencil = 2 * math.comb(L+d, d) + 1
    X = jnp.asarray(X); f = jnp.asarray(f) 
    def div_at_i(i):
        x0 = X[i, :]
        idx = neighbor_idx[i] if neighbor_idx is not None else _knn_indices(X, x0, k) ### knn indices 
        Xloc = X[idx, :]          # (k,d)
        Uloc = f[idx, :]          # (k,d)

        # Wendland-C2 weights in PHYSICAL coords
        r_phys = jnp.linalg.norm(Xloc - x0[None, :], axis=1)
        h = rho * jnp.maximum(r_phys.max(), 1e-12)
        w = _wendland_c2(r_phys / h)                 # (k,)
        W12 = jnp.sqrt(w)[:, None]                   # (k,1)

        # Affine map stencil -> [-1,1]^d for orthogonal Legendre basis
        Xmin = Xloc.min(axis=0, keepdims=True)
        Xmax = Xloc.max(axis=0, keepdims=True)
        span = jnp.maximum(Xmax - Xmin, 1e-12)
        Xhat  = 2.0*(Xloc - Xmin)/span - 1.0         # (k,d)
        xhat0 = 2.0*(x0 - Xmin[0])/span[0] - 1.0     # (d,)
        dxhat_dx = 2.0/span                           # (1,d) chain factor

        # Design and analytic gradients (w.r.t. xhat)
        PHI = lg.legendre_poly_eval(Xhat, I)                         # (k,n)
        PHI0_hat_grad = lg.legendre_poly_grad_eval(xhat0[None,:], I)[0]  # (n,d)

        # Weighted least squares by lstsq: min || W^{1/2}(PHI c - y) ||_2
        A = W12 * PHI                                                # (k,n)

        # Solve for each component independently
        def dcomp(m):
            y = W12.squeeze() * Uloc[:, m]
            c, *_ = jnp.linalg.lstsq(A, y, rcond=rcond)              # (n,)
            # ∂u_m/∂x_m at center = (dxhat/dx)_m * (∂φ/∂xhat_m · c)
            return dxhat_dx[0, m] * (PHI0_hat_grad[:, m] @ c)

        return jnp.add.reduce(jax.vmap(dcomp)(jnp.arange(d)))
    all_indices = jnp.arange(M).astype(jnp.int32) 
    return jax.vmap(div_at_i)(all_indices)


### streamwise-periodic is 3d hexahedral, taylor_green_temporal is just 4 output functions that 
def plot_div(dataset, divergences, centroids):
    plt.figure(figsize=(6,5))
    plt.scatter(centroids[:,0], centroids[:,1], c=divergences, cmap='viridis', s=25)
    plt.colorbar(label='div')
    plt.title(dataset)
    plt.axis('equal')
    plt.savefig(os.path.join('div_plots', dataset))

# buoyancy_cavity_flow is quad mesh
### ['merge_vortices', 'backward_facing_step', 'buoyancy_cavity_flow'***quadmesh, 'flow_cylinder_laminar', 'flow_cylinder_shedding', 'lid_cavity_flow','taylor_green_exact',]
#
calc_div = lambda f, X: calc_div_local_wls(f, X, p=8, rho=1.0, rcond=None)
calc_div = jax.jit(calc_div)
#dataset = 'merge_vortices'
#data =jnp.load(f'./geo-fno/{dataset}.npz')
#f = data['y_train'][0].astype(jnp.float64); x = data['x_grid'].astype(jnp.float64)
#divergences = calc_div(f,x)
#print(dataset,jnp.mean(divergences),jnp.max(jnp.abs(divergences)))
#plot_div(dataset, divergences, x)
#
#dataset = 'test_case f = [x,-y]'
#x0,x1 = x[:,0], x[:,1]
#f =jnp.stack((x0, -x1), axis=-1)
#divergences = calc_div(f,x)
#print(dataset,jnp.mean(divergences),jnp.max(jnp.abs(divergences)))
#plot_div(dataset, divergences, x)
#
#dataset = 'test_case f = [6x^2y,-6xy^2]'
#f =jnp.stack((6*x1*x0**2, -6*x0*x1**2), axis=-1)
#divergences = calc_div(f,x)
#print(dataset,jnp.mean(divergences),jnp.max(jnp.abs(divergences)))
#plot_div(dataset, divergences, x)
#
#dataset = 'backward_facing_step'
#data =jnp.load(f'./geo-fno/{dataset}.npz')
#f = data['y_train'][0].astype(jnp.float64); x = data['x_grid'].astype(jnp.float64) 
#divergences = calc_div(f,x)
#print(dataset,jnp.mean(divergences),jnp.max(jnp.abs(divergences)))
#plot_div(dataset, divergences, x)
#
#dataset = 'flow_cylinder_laminar'
#data =jnp.load(f'./geo-fno/{dataset}.npz')
#f = data['y_train'][0].astype(jnp.float64); x = data['x_grid'].astype(jnp.float64) 
#divergences = calc_div(f,x)
#print(dataset,jnp.mean(divergences),jnp.max(jnp.abs(divergences)))
#plot_div(dataset, divergences, x)
#
#dataset = 'flow_cylinder_shedding'
#data =jnp.load(f'./geo-fno/{dataset}.npz')
#f = data['y_train'][0].astype(jnp.float64); x = data['x_grid'].astype(jnp.float64) 
#divergences = calc_div(f,x)
#print(dataset,jnp.mean(divergences),jnp.max(jnp.abs(divergences)))
#plot_div(dataset, divergences, x)
#
#dataset = 'taylor_green_exact'
#data =jnp.load(f'./geo-fno/{dataset}.npz')
#f = data['y_train'][0].astype(jnp.float64); x = data['x_grid'].astype(jnp.float64) 
#divergences = calc_div(f,x)
#print(dataset,jnp.mean(divergences),jnp.max(jnp.abs(divergences)))
#plot_div(dataset, divergences, x)
#
#dataset = 'lid_cavity_flow'
#data =jnp.load(f'./geo-fno/{dataset}.npz')
#f = data['y_train'][0].astype(jnp.float64); x = data['x_grid'].astype(jnp.float64) 
#divergences = calc_div(f,x)
#print(dataset,jnp.mean(divergences),jnp.max(jnp.abs(divergences)))
#plot_div(dataset, divergences, x)
