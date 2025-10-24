from jax import numpy as jnp, random as jr
import jax 
import equinox as eqx
import orthojax as ojax
import math
from scipy.spatial import cKDTree

jax.config.update("jax_enable_x64", True)

def div_weights(*, X, p=4):
    d = spatial_dimension = X.shape[-1]
    theta = differential_operator_order = 1
    l = total_polynomial_degree = p + theta - 1 ### follows from p and theta

    ### function that evaluates all basis functions on the coordinates to return (n_x, num_bases)
    poly_basis = ojax.TensorProduct(l, [ojax.make_hermite_polynomial(deg) for deg in (l,) * d])

    def grad_poly_basis(x):
        dfdxi = [jax.jacfwd(b.eval)(xi) for b, xi in zip(poly_basis.bases, x)]
        fs = [b.eval(xi) for b, xi in zip(poly_basis.bases, x)] ### 
        dim_range = range(poly_basis.num_dim)
        grad = jnp.ones((d, poly_basis.num_basis))
        for i in dim_range:
            tmp = jnp.ones((poly_basis.num_basis))
            for j in dim_range:
                if j == i:
                    tmp *= dfdxi[i][poly_basis.terms[:, i]]
                else:
                    tmp *= fs[j][poly_basis.terms[:, j]]
            grad = grad.at[i].set(tmp)
        return grad.T ### n_bases, n_dim
    grad_poly_basis = jax.vmap(grad_poly_basis) #### takes in vector of coordinates X, returns (n_x, n_bases, n_dim,)

    class PolyharmonicSpline(eqx.Module):
        m: int
        d: int
        def __init__(self, *, m, d, **kwargs):
            self.m = m
            self.d = d

        def one(self, x, y,):
            eps = jnp.finfo(x.dtype).eps
            r = jnp.linalg.norm((x-y) + eps)
            return r**self.m 
        
        def __call__(self, X, Y):
            k_xy = jax.vmap(jax.vmap(self.one, (0, None)), (None, 0))(Y,X)
            return k_xy
        
        def dphi_dx_autograd(self, X, Y):
            dphi_dx = jax.grad(lambda x, y: self.one(x, y), argnums=0) ### plug in y, i.e. the centers
            return jax.vmap(jax.vmap(dphi_dx, (0, None)), (None, 0))(Y,X)
        
        def dphi_dx_analytic(self, X, Y):
            eps = jnp.finfo(X.dtype).eps
            dphi_dx = lambda x,y: self.m*(x-y)*(jnp.linalg.norm((x-y))+eps)**(self.m-2)
            return jax.vmap(jax.vmap(dphi_dx, (0, None)), (None, 0))(Y,X)
        
    m=5
    rbf = PolyharmonicSpline(m=m, d=d)

    n = n_stencil = 2 * math.comb(l+d, d) + 1

    tree = cKDTree(X)
    _, indices = tree.query(X, k=n_stencil) 

    @jax.jit
    def div_weights_one_center(center, neighbors):
        A = rbf(neighbors, neighbors) ### n_neighbors, n_neighbors
        P = poly_basis(neighbors) ### n_neighbors, n_bases
        zero_block = jnp.zeros((poly_basis.num_basis, poly_basis.num_basis)) ###
        lhs = jnp.block([[A, P],
                        [P.T, zero_block]]) 
        # lhs += jnp.eye(len(lhs)) * 1e-8

        Lq = grad_poly_basis(center[None]).squeeze() ### 1, n_bases, 2 --> n_bases, 2
        # La = rbf.dphi_dx_autograd(center[None], neighbors).squeeze() ### 1, n_stencil/n_bases, 2 --> n_stencil,2
        La = rbf.dphi_dx_analytic(center[None], neighbors).squeeze() ### 1, n_stencil/n_bases, 2 --> n_stencil,2
        rhs = jnp.concatenate((La, Lq), axis=0) ### n_kernel_bases + n_poly_bases, 2    
        weights = jnp.linalg.solve(lhs, rhs)[:n] 
        return weights ### n_neighbors, 2

    neighbors = X[indices] 
    weights = jax.vmap(div_weights_one_center)(X, neighbors) ### n_x, n_neighbors, 2

    make_D = lambda weights: jnp.zeros((len(X), len(X))).at[
        jnp.repeat(jnp.arange(len(X))[:, None], indices.shape[1], 1), indices 
    ].set(weights)
    print(weights.shape)
    Ds = [make_D(weights[...,i]) for i in range(weights.shape[-1])]
    return Ds


def calc_div(*, f, X, p=4):
    Ds = div_weights(X=X,p=p)
    div = sum([Ds[i] @ f[...,i] for i in range(len(Ds))])
    return div

k = jr.PRNGKey(0)
X = jr.uniform(k, shape=(500,2), minval=-1., maxval=1.)
f = jr.uniform(k, shape=(500,2))
print(calc_div(X=X,f=f).shape)