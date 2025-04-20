import jax
from jax import numpy as jnp
import equinox as eqx
import optax
from jaxtyping import Array, Float, Int, PyTree

import numpy as np
import matplotlib.pyplot as plt

import tqdm

SEED = 5678
key = jax.random.PRNGKey(SEED)
rho = jnp.array(1.0)
nu = jnp.array(0.4)

xrange = [0, 1]
yrange = [0, 1]

def dataset_gen(n_slices=20):
    x = jnp.linspace(xrange[0], xrange[1], n_slices + 1)
    y = jnp.linspace(yrange[0], yrange[1], n_slices + 1)
    x_col, y_col = jnp.meshgrid(x, y, indexing='ij')
    
    x_bc_down = x_col[:, 0]
    y_bc_down = y_col[:, 0]
    u_bc_down = jnp.ones_like(x_bc_down)
    v_bc_down = jnp.zeros_like(x_bc_down)
    
    x_bc_up = x_col[:, -1]
    y_bc_up = y_col[:, -1]
    u_bc_up = jnp.zeros_like(x_bc_down)
    v_bc_up = jnp.zeros_like(x_bc_down)
    
    x_bc_left = x_col[0, :]
    y_bc_left = y_col[0, :]
    u_bc_left = jnp.zeros_like(x_bc_down)
    v_bc_left = jnp.zeros_like(x_bc_down)
    
    x_bc_right = x_col[-1, :]
    y_bc_right = y_col[-1, :]
    u_bc_right = jnp.zeros_like(x_bc_down)
    v_bc_right = jnp.zeros_like(x_bc_down)
    
    x_bc = jnp.concatenate([x_bc_down, x_bc_up, x_bc_left, x_bc_right])
    y_bc = jnp.concatenate([y_bc_down, y_bc_up, y_bc_left, y_bc_right])
    u_bc = jnp.concatenate([u_bc_down, u_bc_up, u_bc_left, u_bc_right])
    v_bc = jnp.concatenate([v_bc_down, v_bc_up, v_bc_left, v_bc_right])
     
    x_col = x_col.flatten()
    y_col = y_col.flatten()
    
    return x_col, y_col, x_bc, y_bc, u_bc, v_bc


class PINN(eqx.Module):
    net: eqx.nn.MLP
    
    def __init__(self, key):
        
        self.net = eqx.nn.MLP(
            in_size=2,
            out_size=2,
            width_size=64,
            depth=5,
            activation=jax.nn.tanh,
            key=key
        )
    
    def __call__(self, x, y):
        xy = jnp.stack((x, y), axis=-1) 
        out = self.net(xy)
        psi = out[..., 0]
        p   = out[..., 1]
        return psi, p

def residual_bc(model, x, y, u_bc, v_bc):
    psi_fn = lambda x, y: model(x, y)[0]
    p_fn   = lambda x, y: model(x, y)[1]
    
    psi_x  = jax.grad(psi_fn, 0)
    psi_y  = jax.grad(psi_fn, 1)
    
    u_pred = psi_y(x, y)
    v_pred = -psi_x(x, y)
    
    res_bc = (u_pred - u_bc) ** 2 + (v_pred - v_bc) ** 2
    return res_bc

def residual_pde(model, x, y):
    # zeroth
    psi_fn = lambda x, y: model(x, y)[0]
    p_fn   = lambda x, y: model(x, y)[1]
    
    # first
    psi_x  = jax.grad(psi_fn, 0)
    psi_y  = jax.grad(psi_fn, 1)
    p_x    = jax.grad(p_fn,   0)
    p_y    = jax.grad(p_fn,   1)
    
    # second
    psi_xx = jax.grad(psi_x, 0)
    psi_xy = jax.grad(psi_x, 1)
    psi_yy = jax.grad(psi_y, 1)
    
    # third
    psi_xxx = jax.grad(psi_xx, 0)
    psi_xxy = jax.grad(psi_xx, 1)
    psi_yyy = jax.grad(psi_yy, 1)
    psi_xyy = jax.grad(psi_xy, 1) 
    
    u = psi_y(x, y)
    v = -psi_x(x, y)
    
    u_x = psi_xy(x, y)
    u_y = psi_yy(x, y)
    v_x = -psi_xx(x, y)
    v_y = -psi_xy(x, y)
    
    u_xx = psi_xxy(x, y)
    u_yy = psi_yyy(x, y)
    v_xx = -psi_xxx(x, y)
    v_yy = -psi_xyy(x, y)
    
    res_pde = (u * u_x + v * u_y + p_x(x, y) / rho - nu * (u_xx + u_yy)) ** 2 \
            + (u * v_x + v * v_y + p_y(x, y) / rho - nu * (v_xx + v_yy)) ** 2
    return res_pde

def loss_fn(model, dataset):
    x_col, y_col, x_bc, y_bc, u_bc, v_bc = dataset
    res_bc  = jax.vmap(residual_bc,  (None, 0, 0, 0, 0), 0)(model, x_bc, y_bc, u_bc, v_bc)
    res_pde = jax.vmap(residual_pde, (None, 0, 0),       0)(model, x_col, y_col)
    
    return jnp.mean(res_bc) + jnp.mean(res_pde)

def train(
    model: PINN,
    dataset: tuple,
    optim: optax.GradientTransformation,
    steps: int,
):
    opt_state = optim.init(eqx.filter(model, eqx.is_array))
    
    @eqx.filter_jit
    def make_step(
        model: PINN,
        opt_state: PyTree,
        dataset: tuple,
    ):
        loss_value, grads = eqx.filter_value_and_grad(loss_fn)(model, dataset)
        updates, opt_state = optim.update(
            grads, opt_state, eqx.filter(model, eqx.is_array)
        )
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss_value
    
    pbar = tqdm.trange(steps, desc="training", unit="step")
    for step in pbar:
        model, opt_state, loss_value = make_step(model, opt_state, dataset)
        pbar.set_postfix(loss=float(loss_value))
            
    return model

key, subkey = jax.random.split(key, 2)
model = PINN(subkey)

dataset = dataset_gen(n_slices=20)

# hyperparameters
n_epochs = 20000
schedule = optax.piecewise_constant_schedule(
    init_value=1e-3,
    boundaries_and_scales={2000: 0.1}
)
optim = optax.adamw(schedule)

model = train(model, dataset, optim, n_epochs)


# visualization

psi_fn = lambda x, y: model(x, y)[0]
p_fn   = lambda x, y: model(x, y)[1]

psi_x  = jax.grad(psi_fn, 0)
psi_y  = jax.grad(psi_fn, 1)

n = 101
x = np.linspace(0, 1, n)
y = np.linspace(0, 1, n)
X, Y = np.meshgrid(x, y, indexing="xy")
Xf = X.ravel()       # (n*n,)`  `
Yf = Y.ravel()

U = jax.vmap(psi_y)(Xf, Yf).reshape(n, n)
V = jax.vmap(psi_x)(Xf, Yf).reshape(n, n)
P = jax.vmap(p_fn)(Xf, Yf).reshape(n, n)

fig, ax = plt.subplots(figsize=(6, 5))

# pressure heat map
im = ax.imshow(
    P,
    origin="lower",
    extent=[0, 1, 0, 1],
    cmap="viridis",
    aspect="equal"
)
cbar = fig.colorbar(im, ax=ax, label="p(x, y)")

# velocity quiver
step = 5
ax.quiver(
    X[::step, ::step],
    Y[::step, ::step],
    U[::step, ::step],
    V[::step, ::step],
    color="white",
    scale_units="xy",
    angles="xy",
    pivot='mid',
    scale=5.0 
)

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Velocity (quiver) + Pressure (heat map)")

plt.tight_layout()
plt.show()
    