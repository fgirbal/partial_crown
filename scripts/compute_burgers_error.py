import argparse

import scipy.io
import numpy as np
import torch

from pinn_verifier.utils import load_compliant_model

parser = argparse.ArgumentParser()
parser.add_argument('--network-filename', required=True, type=str, help='onnx file to load.')
args = parser.parse_args()

model, layers = load_compliant_model(args.network_filename)

data = scipy.io.loadmat('../../PINNs/appendix/Data/burgers_shock.mat')
    
t = data['t'].flatten()
x = data['x'].flatten()
usol = data['usol'].T.flatten()

X, T = np.meshgrid(x, t)
X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))
grid_points = torch.tensor(np.stack([X_star[:, 1], X_star[:, 0]], axis=1), dtype=torch.float32)
# grid_points = torch.tensor(np.stack([X_star[:, 0], X_star[:, 1]], axis=1), dtype=torch.float32)

outputs = model(grid_points)

errors = np.abs(usol - outputs.detach().numpy().flatten())

print("---- solution ----")
print("Max error:", errors.max())
print("Mean error:", errors.mean())
print("l_2 error:", np.linalg.norm(usol - outputs.detach().numpy().flatten(),2)/np.linalg.norm(usol,2))

def model_get_residual(model, X_r):
    t, x = X_r[:, 0:1], X_r[:, 1:2]
    t.requires_grad_()
    x.requires_grad_()

    u = model(torch.hstack([t, x]))

    u_x = torch.autograd.grad(u.sum(), x, create_graph=True, retain_graph=True, allow_unused=True)[0]
    u_t = torch.autograd.grad(u.sum(), t, create_graph=True, retain_graph=True, allow_unused=True)[0]

    u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True, retain_graph=True, allow_unused=True)[0]

    return u_t + u * u_x - .01/np.pi * u_xx

def compute_torch_gradient_dt(model, X_r):
    t, x = X_r[:, 0:1], X_r[:, 1:2]
    t.requires_grad_()
    x.requires_grad_()

    u = model(torch.hstack([t, x]))
    u_t = torch.autograd.grad(u.sum(), t, create_graph=True, retain_graph=True, allow_unused=True)[0]

    return u_t

def compute_torch_gradient_dx(model, X_r):
    t, x = X_r[:, 0:1], X_r[:, 1:2]
    t.requires_grad_()
    x.requires_grad_()

    u = model(torch.hstack([t, x]))
    u_x = torch.autograd.grad(u.sum(), x, create_graph=True, retain_graph=True, allow_unused=True)[0]

    return u_x

def compute_torch_gradient_dxdx(model, X_r):
    t, x = X_r[:, 0:1], X_r[:, 1:2]
    t.requires_grad_()
    x.requires_grad_()

    u = model(torch.hstack([t, x]))
    u_x = torch.autograd.grad(u.sum(), x, create_graph=True, retain_graph=True, allow_unused=True)[0]
    u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True, retain_graph=True, allow_unused=True)[0]

    return u_xx

domain_bounds = torch.tensor([[0, -1], [1, 1]])

ts = torch.linspace(domain_bounds[0, 0], domain_bounds[1, 0], 1000)
xs = torch.linspace(domain_bounds[0, 1], domain_bounds[1, 1], 1000)
grid_ts, grid_xs = torch.meshgrid(ts, xs, indexing='ij')
all_grid_points = torch.dstack([grid_ts, grid_xs]).reshape(-1, 2)

model_pts = model_get_residual(model, all_grid_points)
all_min, all_max = model_pts.min(), model_pts.max()

print("---- residual ----")
print("f min:", all_min)
print("f max:", all_max)

u_thetas = model(all_grid_points)
all_min, all_max = u_thetas.min(), u_thetas.max()

print("---- u_theta ----")
print("u_theta min:", all_min)
print("u_theta max:", all_max)

# u_dt_thetas = compute_torch_gradient_dt(model, all_grid_points)
# all_min, all_max = u_dt_thetas.min(), u_dt_thetas.max()

# print("---- u_dt_theta ----")
# print("u_dt_theta min:", all_min)
# print("u_dt_theta max:", all_max)

# u_dx_thetas = compute_torch_gradient_dx(model, all_grid_points)
# all_min, all_max = u_dx_thetas.min(), u_dx_thetas.max()

# print("---- u_dx_theta ----")
# print("u_dx_theta min:", all_min)
# print("u_dx_theta max:", all_max)

# u_dxdx_thetas = compute_torch_gradient_dxdx(model, all_grid_points)
# all_min, all_max = u_dxdx_thetas.min(), u_dxdx_thetas.max()

# print("---- u_dxdx_theta ----")
# print("u_dxdx_theta min:", all_min)
# print("u_dxdx_theta max:", all_max)

import matplotlib
import matplotlib.colors as colors
import matplotlib.pyplot as plt

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"],
    "font.size": 14
})

fig, ax = plt.subplots(1, 2, sharex=False)
fig.set_figheight(3)
fig.set_figwidth(15)

thetas_plot = ax[0].imshow(u_thetas.detach().numpy().reshape(grid_ts.shape).T, extent=[0, 1, -1, 1], aspect=0.2)
fig.colorbar(thetas_plot, ax=ax[0])
# ax[0].scatter(grid_points[:, 0], grid_points[:, 1], s=1, c="red")
ax[0].set_ylabel(r"$x$")
ax[0].set_xlabel(r"$t$")
ax[0].set_title(r"$u_{\theta}$")

# cmap = matplotlib.cm.get_cmap('seismic')
fs_plot = ax[1].imshow(model_pts.detach().numpy().reshape(grid_ts.shape).T, extent=[0, 1, -1, 1], aspect=0.2, cmap='seismic', norm=colors.CenteredNorm())
fig.colorbar(fs_plot, ax=ax[1])
# ax[1].scatter(grid_points[:, 0], grid_points[:, 1], s=1, c="red")
ax[1].set_ylabel(r"$x$")
ax[1].set_xlabel(r"$t$")
ax[1].set_title(r"$f_{\theta}$")

plt.tight_layout()
plt.show()

import pdb
pdb.set_trace()