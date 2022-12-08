"""
Adapted from the error computations of https://github.com/maziarraissi/PINNs, to take into account that the input is always
(t, x) instead of (x, t). Computes error and plots u_theta and f_theta accross the domain.
"""
import argparse

import scipy.io
import numpy as np
import torch

from pinn_verifier.utils import load_compliant_model

parser = argparse.ArgumentParser()
parser.add_argument('--network-filename', required=True, type=str, help='onnx file to load.')
args = parser.parse_args()

model, layers = load_compliant_model(args.network_filename)

data = scipy.io.loadmat('data/schrodinger.mat')
    
t = data['tt'].flatten()[:,None]
x = data['x'].flatten()[:,None]
Exact = data['uu']
Exact_u = np.real(Exact)
Exact_v = np.imag(Exact)
Exact_h = np.sqrt(Exact_u**2 + Exact_v**2)

X, T = np.meshgrid(x,t)

X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))
u_star = Exact_u.T.flatten()[:,None].flatten()
v_star = Exact_v.T.flatten()[:,None].flatten()
h_star = Exact_h.T.flatten()[:,None].flatten()

# IMPORTANT: input is (t, x) instead of (x, t), so must flip the grid generation
grid_points = torch.tensor(np.stack([X_star[:, 1], X_star[:, 0]], axis=1), dtype=torch.float32)
# grid_points = torch.tensor(np.stack([X_star[:, 0], X_star[:, 1]], axis=1), dtype=torch.float32)

outputs = model(grid_points)
us, vs = outputs[:, 0], outputs[:, 1]
hs = torch.sqrt(us**2 + vs**2)

errors_u = np.abs(u_star - us.detach().numpy())
errors_v = np.abs(v_star - vs.detach().numpy())
errors_h = np.abs(h_star - hs.detach().numpy())

print("---- solution ----")
print("Max error_u:", errors_u.max())
print("Mean error_u:", errors_u.mean())
print("l_2 error_u:", np.linalg.norm(u_star - us.detach().numpy(),2)/np.linalg.norm(u_star,2))

print("Max error_v:", errors_v.max())
print("Mean error_v:", errors_v.mean())
print("l_2 error_v:", np.linalg.norm(v_star - vs.detach().numpy(),2)/np.linalg.norm(v_star,2))

print("Max error_h:", errors_h.max())
print("Mean error_h:", errors_h.mean())
print("l_2 error_h:", np.linalg.norm(h_star - hs.detach().numpy(),2)/np.linalg.norm(h_star,2))

def model_get_residual(model, X_r):
    t, x = X_r[:, 0:1], X_r[:, 1:2]
    t.requires_grad_()
    x.requires_grad_()

    h = model(torch.hstack([t, x]))
    u, v = h[:, 0:1], h[:, 1:2]

    u_x = torch.autograd.grad(u.sum(), x, create_graph=True, retain_graph=True, allow_unused=True)[0]
    v_x = torch.autograd.grad(v.sum(), x, create_graph=True, retain_graph=True, allow_unused=True)[0]

    u_t = torch.autograd.grad(u.sum(), t, create_graph=True, retain_graph=True, allow_unused=True)[0]
    v_t = torch.autograd.grad(v.sum(), t, create_graph=True, retain_graph=True, allow_unused=True)[0]

    u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True, retain_graph=True, allow_unused=True)[0]
    v_xx = torch.autograd.grad(v_x.sum(), x, create_graph=True, retain_graph=True, allow_unused=True)[0]

    f_u = u_t + 0.5*v_xx + (u**2 + v**2)*v
    f_v = v_t - 0.5*u_xx - (u**2 + v**2)*u

    return f_u, f_v

domain_bounds = torch.tensor([[0, -5], [np.pi/2, 5]])

ts = torch.linspace(domain_bounds[0, 0], domain_bounds[1, 0], 100)
xs = torch.linspace(domain_bounds[0, 1], domain_bounds[1, 1], 100)
grid_ts, grid_xs = torch.meshgrid(ts, xs, indexing='ij')
all_grid_points = torch.dstack([grid_ts, grid_xs]).reshape(-1, 2)

f_us, f_vs = model_get_residual(model, all_grid_points)
f_hs = f_us**2 + f_vs**2
f_u_min, f_u_max = f_us.min(), f_us.max()
f_v_min, f_v_max = f_vs.min(), f_vs.max()
f_h_min, f_h_max = f_hs.min(), f_hs.max()

print("---- residual ----")
print("f_u min:", f_u_min)
print("f_u max:", f_u_max)
print("f_v min:", f_v_min)
print("f_v max:", f_v_max)
print("f_h min:", f_h_min)
print("f_h max:", f_h_max)

h_thetas = model(all_grid_points)
u_thetas, v_thetas = h_thetas[:, 0], h_thetas[:, 1]
h_theta_norm = u_thetas**2 + v_thetas**2

all_min_u, all_max_u = u_thetas.min(), u_thetas.max()
all_min_v, all_max_v = v_thetas.min(), v_thetas.max()
all_min_h, all_max_h = h_theta_norm.min(), h_theta_norm.max()

print("---- u_theta/v_theta/|h_theta| ----")
print("u_theta min:", all_min_u)
print("u_theta max:", all_max_u)
print("v_theta min:", all_min_v)
print("v_theta max:", all_max_v)
print("h_theta_norm min:", all_min_h)
print("h_theta_norm max:", all_max_h)

import pdb
pdb.set_trace()

# def compute_torch_gradient_dt(model, X_r):
#     t, x = X_r[:, 0:1], X_r[:, 1:2]
#     t.requires_grad_()
#     x.requires_grad_()

#     h = model(torch.hstack([t, x]))
#     u_t = torch.autograd.grad(h[:, 0].sum(), t, create_graph=True, retain_graph=True, allow_unused=True)[0]
#     v_t = torch.autograd.grad(h[:, 1].sum(), t, create_graph=True, retain_graph=True, allow_unused=True)[0]

#     return u_t, v_t

# u_dt_thetas, v_dt_thetas = compute_torch_gradient_dt(model, all_grid_points)
# u_dt_thetas_min, u_dt_thetas_max = u_dt_thetas.min(), u_dt_thetas.max()
# v_dt_thetas_min, v_dt_thetas_max = v_dt_thetas.min(), v_dt_thetas.max()

# print("---- u_dt_theta/v_dt_theta ----")
# print("u_dt_theta min:", u_dt_thetas_min)
# print("u_dt_theta max:", u_dt_thetas_max)
# print("v_dt_theta min:", v_dt_thetas_min)
# print("v_dt_theta max:", v_dt_thetas_max)

# def compute_torch_gradient_dxdx(model, X_r):
#     t, x = X_r[:, 0:1], X_r[:, 1:2]
#     t.requires_grad_()
#     x.requires_grad_()

#     h = model(torch.hstack([t, x]))
#     u_x = torch.autograd.grad(h[:, 0].sum(), x, create_graph=True, retain_graph=True, allow_unused=True)[0]
#     u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True, retain_graph=True, allow_unused=True)[0]

#     v_x = torch.autograd.grad(h[:, 1].sum(), x, create_graph=True, retain_graph=True, allow_unused=True)[0]
#     v_xx = torch.autograd.grad(v_x.sum(), x, create_graph=True, retain_graph=True, allow_unused=True)[0]

#     return u_xx, v_xx

# u_dxdx_thetas, v_dxdx_thetas = compute_torch_gradient_dxdx(model, all_grid_points)
# u_dxdx_thetas_min, u_dxdx_thetas_max = u_dxdx_thetas.min(), u_dxdx_thetas.max()
# v_dxdx_thetas_min, v_dxdx_thetas_max = v_dxdx_thetas.min(), v_dxdx_thetas.max()

# print("---- u_dxdx_theta/v_dxdx_theta ----")
# print("u_dxdx_theta min:", u_dxdx_thetas_min)
# print("u_dxdx_theta max:", u_dxdx_thetas_max)
# print("v_dxdx_theta min:", v_dxdx_thetas_min)
# print("v_dxdx_theta max:", v_dxdx_thetas_max)

import matplotlib
import matplotlib.colors as colors
import matplotlib.pyplot as plt

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"],
    "font.size": 14
})

fig, ax = plt.subplots(1, 2, sharex=True)
fig.set_figheight(3)
fig.set_figwidth(15)

thetas_plot = ax[0].imshow(h_theta_norm.detach().numpy().reshape(grid_ts.shape).T, extent=[0, np.pi/2, -5, 5], aspect=0.05)
fig.colorbar(thetas_plot, ax=ax[0])
# ax[0].scatter(grid_points[:, 0], grid_points[:, 1], s=1, c="red")
ax[0].set_ylabel(r"$x$")
ax[0].set_xlabel(r"$t$")
ax[0].set_title(r"$|h_{\theta}|$")

# cmap = matplotlib.cm.get_cmap('seismic')
fs_plot = ax[1].imshow(f_hs.detach().numpy().reshape(grid_ts.shape).T, extent=[0, np.pi/2, -5, 5], aspect=0.05, cmap='seismic', norm=colors.CenteredNorm())
fig.colorbar(fs_plot, ax=ax[1])
# ax[1].scatter(grid_points[:, 0], grid_points[:, 1], s=1, c="red")
ax[1].set_ylabel(r"$x$")
ax[1].set_xlabel(r"$t$")
ax[1].set_title(r"$|f_{\theta}|^2$")

plt.tight_layout()
plt.show()

import pdb
pdb.set_trace()