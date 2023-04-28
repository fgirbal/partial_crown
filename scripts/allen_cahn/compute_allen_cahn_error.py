"""
Computes error and plots u_theta and f_theta accross the domain.
"""
import argparse

from scipy.integrate import odeint
import numpy as np
import torch

from pinn_verifier.utils import load_compliant_model

def function(u0: str):
    """Initial condition, string --> function."""

    funs = {
        'sin(x)': lambda x: np.sin(x),
        '-sin(x)': lambda x: -np.sin(x),
        '-sin(pix)': lambda x: -np.sin(np.pi*x),
        '-sin(2pix)': lambda x: -np.sin(2*np.pi*x),
        'sin(pix)': lambda x: np.sin(np.pi*x),
        'sin^2(x)': lambda x: np.sin(x)**2,
        'sin(x)cos(x)': lambda x: np.sin(x)*np.cos(x),
        'x^2*cos(pix)': lambda x: x**2 * np.cos(np.pi*x),
        '0.1sin(x)': lambda x: 0.1*np.sin(x),
        '0.5sin(x)': lambda x: 0.5*np.sin(x),
        '10sin(x)': lambda x: 10*np.sin(x),
        '50sin(x)': lambda x: 50*np.sin(x),
        '1+sin(x)': lambda x: 1 + np.sin(x),
        '2+sin(x)': lambda x: 2 + np.sin(x),
        '6+sin(x)': lambda x: 6 + np.sin(x),
        '10+sin(x)': lambda x: 10 + np.sin(x),
        'sin(2x)': lambda x: np.sin(2*x),
        'tanh(x)': lambda x: np.tanh(x),
        '2x': lambda x: 2*x,
        'x^2': lambda x: x**2,
        'gauss':  lambda x: np.exp(-np.power(x - np.pi, 2.) / (2 * np.power(np.pi/4, 2.)))
    }
    return funs[u0]


def ac_system(u, t, k, nu, rho):
    #Spatial derivative in the Fourier domain
    u_hat = np.fft.fft(u)
    u_hat_xx = -k**2*u_hat
    
    #Switching in the spatial domain
    u_xx = np.fft.ifft(u_hat_xx)
    
    #ODE resolution
    u_t = -rho * (u**3 - u) + nu*u_xx
    return u_t.real


def fft2_discrete_sol(x, t, u0 : str, system, **sys_args):
    #Wave number discretization
    N_x = len(x)
    dx = (x[-1] - x[0])/N_x

    k = 2*np.pi*np.fft.fftfreq(N_x, d=dx)

    #Def of the initial condition
    u0 = function(u0)(x)
    
    if system == 'allencahn':
        Ft = ac_system
    else:
        raise NotImplementedError("System not valid.")

    return odeint(Ft, u0, t, args=(k, *sys_args.values()),
                  mxstep=5000)  


parser = argparse.ArgumentParser()
parser.add_argument('--network-filename', required=True, type=str, help='onnx file to load.')
args = parser.parse_args()

model, layers = load_compliant_model(args.network_filename)

t = np.linspace(0, 1, 1000).reshape(-1, 1)
x = np.linspace(-1, 1, 1000, endpoint=False).reshape(-1, 1)
U_field = fft2_discrete_sol(
    x.ravel(), t.ravel(), "x^2*cos(pix)", system="allencahn",
    nu=0.0001, rho=5.0
).flatten()

X, T = np.meshgrid(x,t)
X_star = torch.tensor(np.hstack((X.flatten()[:,None], T.flatten()[:,None])), dtype=torch.float32)
u_pred = model(X_star).flatten()

errors_u = np.abs(u_pred.detach().numpy() - U_field)

print("---- solution ----")
print("Max error_u:", errors_u.max())
print("Mean error_u:", errors_u.mean())
print("l_2 error_u:", np.power(u_pred.detach().numpy() - U_field, 2).mean())

def model_get_residual(model, X_r):
    x, t = X_r[:, 0:1], X_r[:, 1:2]
    x.requires_grad_()
    t.requires_grad_()

    u = model(torch.hstack([x, t]))
    u_t = torch.autograd.grad(
        u, t,
        grad_outputs=torch.ones_like(u),
        retain_graph=True,
        create_graph=True
    )[0]
    u_x = torch.autograd.grad(
        u, x,
        grad_outputs=torch.ones_like(u),
        retain_graph=True,
        create_graph=True
    )[0]

    u_xx = torch.autograd.grad(
        u_x, x,
        grad_outputs=torch.ones_like(u_x),
        retain_graph=True,
        create_graph=True
    )[0]

    return u_t + 5 * u**3 - 5 * u - 0.0001 * u_xx

domain_bounds = torch.tensor([[-1, 0], [1, 1]])

xs = torch.linspace(domain_bounds[0, 0], domain_bounds[1, 0], 1000)
ts = torch.linspace(domain_bounds[0, 1], domain_bounds[1, 1], 1000)
grid_xs, grid_ts = torch.meshgrid(xs, ts, indexing='ij')
all_grid_points = torch.dstack([grid_xs, grid_ts]).reshape(-1, 2)

f_us = model_get_residual(model, all_grid_points)
f_u_min, f_u_max = f_us.min(), f_us.max()

print("---- residual ----")
print("f_u min:", f_u_min)
print("f_u max:", f_u_max)

f_us_squared = (f_us)**2

u_thetas = model(all_grid_points)

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

thetas_plot = ax[0].imshow(u_thetas.detach().numpy().reshape(grid_ts.shape), extent=[0, 1, -1, 1], aspect=0.2)
fig.colorbar(thetas_plot, ax=ax[0])
# ax[0].scatter(grid_points[:, 0], grid_points[:, 1], s=1, c="red")
ax[0].set_ylabel(r"$x$")
ax[0].set_xlabel(r"$t$")
ax[0].set_title(r"$|u_{\theta}|$")

# cmap = matplotlib.cm.get_cmap('seismic')
fs_plot = ax[1].imshow(f_us_squared.detach().numpy().reshape(grid_ts.shape), extent=[0, 1, -1, 1], aspect=0.2, cmap='hot')
fig.colorbar(fs_plot, ax=ax[1])
# ax[1].scatter(grid_points[:, 0], grid_points[:, 1], s=1, c="red")
ax[1].set_ylabel(r"$x$")
ax[1].set_xlabel(r"$t$")
ax[1].set_title(r"$|f_{\theta}|^2$")

plt.tight_layout()
plt.show()

import pdb
pdb.set_trace()