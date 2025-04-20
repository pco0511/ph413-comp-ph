import torch
from torch import nn
from torch.optim import AdamW
from torch.func import vmap, jacrev
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange

# Reproducibility
torch.manual_seed(5678)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Physical parameters
rho = 1.0
nu = 0.4

# Mesh generation

def mesh(n):
    x = torch.linspace(0, 1, n, device=device)
    y = torch.linspace(0, 1, n, device=device)
    X, Y = torch.meshgrid(x, y, indexing="ij")
    pts = torch.stack((X.flatten(), Y.flatten()), dim=-1)
    return pts

# Collocation and boundary points
dof = 21
col_pts = mesh(dof).requires_grad_(True)
mask = (col_pts[:,0] == 0) | (col_pts[:,0] == 1) | (col_pts[:,1] == 0) | (col_pts[:,1] == 1)
bc_pts = col_pts[mask].requires_grad_(True)

# Boundary conditions: u=1 on bottom, u=0 elsewhere
u_bc = torch.zeros(bc_pts.size(0), device=device)
v_bc = torch.zeros_like(u_bc := u_bc if 'u_bc' in globals() else torch.zeros(bc_pts.size(0), device=device))
v_bc = torch.zeros_like(v_bc)
u_bc[bc_pts[:,1] == 0] = 1.0

# PINN model definition
class PINN(nn.Module):
    def __init__(self, width=64, depth=5):
        super().__init__()
        layers = [nn.Linear(2, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.Tanh()]
        layers += [nn.Linear(width, 2)]
        self.net = nn.Sequential(*layers)

    def forward(self, xy):
        return self.net(xy)  # returns [psi, p]

# Differential operators using functorch

def psi_p(model, xy):
    out = model(xy)      # (B,2)
    psi = out[:, 0]
    p = out[:, 1]
    return psi, p

# 1st derivatives
def grads(model, xy):
    def psi_fn(v): return model(v.unsqueeze(0))[0][0]
    def p_fn(v):   return model(v.unsqueeze(0))[0][1]
    grad_psi = vmap(jacrev(psi_fn))(xy)  # (B,2)
    grad_p   = vmap(jacrev(p_fn))(xy)    # (B,2)
    psi_x, psi_y = grad_psi[:,0], grad_psi[:,1]
    p_x, p_y     = grad_p[:,0],   grad_p[:,1]
    return psi_x, psi_y, p_x, p_y

# 2nd derivatives (Hessian of psi)
def hess_psi(model, xy):
    def psi_fn(v): return model(v.unsqueeze(0))[0][0]
    H = vmap(jacrev(jacrev(psi_fn)))(xy)  # (B,2,2)
    psi_xx = H[:,0,0]
    psi_xy = H[:,0,1]
    psi_yy = H[:,1,1]
    return psi_xx, psi_xy, psi_yy

# 3rd derivatives of psi
def third_psi(model, xy):
    def psi_fn(v): return model(v.unsqueeze(0))[0][0]
    J3 = vmap(jacrev(jacrev(jacrev(psi_fn))))(xy)  # (B,2,2,2)
    psi_xxx = J3[:,0,0,0]
    psi_xxy = J3[:,0,0,1]
    psi_xyy = J3[:,0,1,1]
    psi_yyy = J3[:,1,1,1]
    return psi_xxx, psi_xxy, psi_xyy, psi_yyy

# Loss function combining PDE residual and BC residual
def loss_fn(model):
    # PDE residual
    psi_x, psi_y, p_x, p_y = grads(model, col_pts)
    psi_xx, psi_xy, psi_yy = hess_psi(model, col_pts)
    psi_xxx, psi_xxy, psi_xyy, psi_yyy = third_psi(model, col_pts)

    u = psi_y
    v = -psi_x
    u_x, u_y = psi_xy, psi_yy
    v_x, v_y = -psi_xx, -psi_xy
    u_xx = psi_xxy
    u_yy = psi_yyy
    v_xx = -psi_xxx
    v_yy = -psi_xyy

    res1 = u * u_x + v * u_y + p_x / rho - nu * (u_xx + u_yy)
    res2 = u * v_x + v * v_y + p_y / rho - nu * (v_xx + v_yy)
    pde_loss = (res1.pow(2) + res2.pow(2)).mean()

    # BC residual
    psi_x_b, psi_y_b, _, _ = grads(model, bc_pts)
    u_b, v_b = psi_y_b, -psi_x_b
    bc_loss = ((u_b - u_bc).pow(2) + (v_b - v_bc).pow(2)).mean()

    return pde_loss + bc_loss

# Training setup
model = PINN().to(device)
optimizer = AdamW(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2000], gamma=0.1)

# Optional compilation (PyTorch 2.2+). If Triton not available, suppress errors and fall back
import torch._dynamo
torch._dynamo.config.suppress_errors = True
try:
    compiled_loss = torch.compile(loss_fn)
except Exception:
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1")
    compiled_loss = loss_fn

# Training loop
n_epochs = 20000
for epoch in trange(n_epochs, desc="training", unit="step"):
    optimizer.zero_grad()
    loss = compiled_loss(model)
    loss.backward()
    optimizer.step()
    scheduler.step()
    if epoch % 500 == 0:
        print(f"Epoch {epoch}: loss={loss.item():.4e}")

# Prediction & Visualization
@torch.no_grad()
def predict(model, n=101):
    pts = mesh(n)
    pts.requires_grad_(True)
    psi, p = psi_p(model, pts)
    # velocity = (-psi_x, psi_y)
    psi_x = vmap(jacrev(lambda v: model(v.unsqueeze(0))[0][0]))(pts)[:,1]
    psi_y = vmap(jacrev(lambda v: model(v.unsqueeze(0))[0][0]))(pts)[:,0]
    U = psi_y.reshape(n, n).cpu().numpy()
    V = -psi_x.reshape(n, n).cpu().numpy()
    P = p.reshape(n, n).cpu().numpy()
    X, Y = torch.meshgrid(torch.linspace(0,1,n), torch.linspace(0,1,n), indexing='xy')
    return X.cpu().numpy(), Y.cpu().numpy(), U, V, P

X, Y, U, V, P = predict(model)
fig, ax = plt.subplots(figsize=(6,5))
im = ax.imshow(P, origin='lower', extent=[0,1,0,1], cmap='viridis', aspect='equal')
fig.colorbar(im, ax=ax, label='p(x,y)')
step = 5
ax.quiver(X[::step,::step], Y[::step,::step], U[::step,::step], V[::step,::step], color='white', scale_units='xy', angles='xy', pivot='mid', scale=5.0)
ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_title('Velocity + Pressure')
plt.tight_layout(); plt.show()
