import torch
from torch import nn
from torch.optim import AdamW
from torch.func import vmap, jacrev
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange


SEED = 5678
torch.manual_seed(SEED)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"{device=}")

rho = torch.tensor(1.0, device=device)
nu  = torch.tensor(0.4, device=device)

xrange = [0.0, 1.0]
yrange = [0.0, 1.0]

def dataset_gen(n_slices: int = 20):
    x = torch.linspace(*xrange, n_slices + 1, device=device)
    y = torch.linspace(*yrange, n_slices + 1, device=device)
    x_col, y_col = torch.meshgrid(x, y, indexing="ij")

    # --- 경계 (Dirichlet) 조건 ---
    x_bc_down, y_bc_down = x_col[:, 0], y_col[:, 0]
    x_bc_up,   y_bc_up   = x_col[:, -1], y_col[:, -1]
    x_bc_left, y_bc_left = x_col[0, :],  y_col[0, :]
    x_bc_right, y_bc_right = x_col[-1, :], y_col[-1, :]

    u_bc_down = torch.ones_like(x_bc_down)
    v_bc_down = torch.zeros_like(x_bc_down)

    zeros_col = torch.zeros_like(x_bc_down)
    u_bc_up, v_bc_up       = zeros_col, zeros_col
    u_bc_left, v_bc_left   = zeros_col, zeros_col
    u_bc_right, v_bc_right = zeros_col, zeros_col

    x_bc = torch.cat([x_bc_down, x_bc_up, x_bc_left, x_bc_right])
    y_bc = torch.cat([y_bc_down, y_bc_up, y_bc_left, y_bc_right])
    u_bc = torch.cat([u_bc_down, u_bc_up, u_bc_left, u_bc_right])
    v_bc = torch.cat([v_bc_down, v_bc_up, v_bc_left, v_bc_right])

    # 컬로케이션(내부) 점
    x_col = x_col.flatten()
    y_col = y_col.flatten()

    return (x_col, y_col, x_bc, y_bc, u_bc, v_bc)


class PINN(nn.Module):
    def __init__(self, width=64, depth=5):
        super().__init__()
        layers = [nn.Linear(2, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.Tanh()]
        layers += [nn.Linear(width, 2)]
        self.net = nn.Sequential(*layers)

    def forward(self, x, y):
        xy = torch.stack((x, y), dim=-1)
        out = self.net(xy)
        psi, p = out[..., 0], out[..., 1]
        return psi, p

# torch.autograd.grad 호출 시 편의를 위한 helper
def grad(outputs, inputs, retain_graph=True):
    return torch.autograd.grad(
        outputs, inputs,
        grad_outputs=torch.ones_like(outputs),
        create_graph=True,
        retain_graph=retain_graph,
    )[0]
    
    
def residual_bc(model, x, y, u_bc, v_bc):
    # x, y는 스칼라 텐서 (requires_grad=False)
    x_r = x.detach().clone().requires_grad_(True)
    y_r = y.detach().clone().requires_grad_(True)

    psi, _ = model(x_r, y_r)
    psi_x = grad(psi, x_r)
    psi_y = grad(psi, y_r)

    u_pred = psi_y
    v_pred = -psi_x
    return (u_pred - u_bc) ** 2 + (v_pred - v_bc) ** 2

def residual_pde(model, x, y):
    x_r = x.detach().clone().requires_grad_(True)
    y_r = y.detach().clone().requires_grad_(True)

    psi, p = model(x_r, y_r)
    psi_x = grad(psi, x_r)
    psi_y = grad(psi, y_r)
    p_x   = grad(p,   x_r)
    p_y   = grad(p,   y_r)

    # 2차 도함수
    psi_xx = grad(psi_x, x_r)
    psi_xy = grad(psi_x, y_r)
    psi_yy = grad(psi_y, y_r)

    # 3차 도함수
    psi_xxx = grad(psi_xx, x_r)
    psi_xxy = grad(psi_xx, y_r)
    psi_yyy = grad(psi_yy, y_r)
    psi_xyy = grad(psi_xy, y_r)

    u = psi_y
    v = -psi_x

    u_x = psi_xy
    u_y = psi_yy
    v_x = -psi_xx
    v_y = -psi_xy

    u_xx = psi_xxy
    u_yy = psi_yyy
    v_xx = -psi_xxx
    v_yy = -psi_xyy

    res1 = u * u_x + v * u_y + p_x / rho - nu * (u_xx + u_yy)
    res2 = u * v_x + v * v_y + p_y / rho - nu * (v_xx + v_yy)
    return res1 ** 2 + res2 ** 2

def loss_fn(model, dataset):
    x_col, y_col, x_bc, y_bc, u_bc, v_bc = dataset
    # 경계 손실
    bc_loss = torch.stack(
        [residual_bc(model, x, y, u, v)
         for x, y, u, v in zip(x_bc, y_bc, u_bc, v_bc)]
    ).mean()
    # PDE 손실
    pde_loss = torch.stack(
        [residual_pde(model, x, y)
         for x, y in zip(x_col, y_col)]
    ).mean()

    return bc_loss + pde_loss

def train(model, dataset, epochs=5000):
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[1500], gamma=0.1
    )

    pbar = trange(epochs, desc="training", unit="step")
    for step in pbar:
        optimizer.zero_grad()
        loss = loss_fn(model, dataset)
        loss.backward()
        optimizer.step()
        scheduler.step()

        if step % 20 == 0:
            pbar.set_postfix(loss=float(loss.item()))

    return model

dataset = dataset_gen(n_slices=20)
model = PINN().to(device)

model = train(model, dataset, epochs=5000)


# visualization

@torch.no_grad()
def predict_pressure_velocity(model, n=101):
    x = torch.linspace(0, 1, n, device=device)
    y = torch.linspace(0, 1, n, device=device)
    X, Y = torch.meshgrid(x, y, indexing="xy")
    Xf, Yf = X.flatten(), Y.flatten()

    # grad 계산을 위해 requires_grad 설정
    Xf_r = Xf.detach().clone().requires_grad_(True)
    Yf_r = Yf.detach().clone().requires_grad_(True)

    psi, p = model(Xf_r, Yf_r)
    psi_x = grad(psi, Xf_r)
    psi_y = grad(psi, Yf_r)

    U = psi_y.reshape(n, n).cpu().numpy()
    V = psi_x.reshape(n, n).cpu().numpy()
    P = p.reshape(n, n).cpu().numpy()
    X_np, Y_np = X.cpu().numpy(), Y.cpu().numpy()
    return X_np, Y_np, U, V, P

X, Y, U, V, P = predict_pressure_velocity(model, n=101)

fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(
    P,
    origin="lower",
    extent=[0, 1, 0, 1],
    cmap="viridis",
    aspect="equal",
)
fig.colorbar(im, ax=ax, label="p(x, y)")

step = 5
ax.quiver(
    X[::step, ::step],
    Y[::step, ::step],
    U[::step, ::step],
    V[::step, ::step],
    color="white",
    scale_units="xy",
    angles="xy",
    pivot="mid",
    scale=5.0,
)

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Velocity (quiver) + Pressure (heat map)")
plt.tight_layout()
plt.show()