import copy
import argparse
import time
import json

from pyparsing import empty

from scipy import optimize
from tqdm import tqdm
import numpy as np
import torch
import gurobipy as grb

import tools.bab_tools.vnnlib_utils as vnnlib_utils
from activation_relaxations import ActivationRelaxationType, SoftplusRelaxation
from pinn_verifier import BurgersVerifier

debug = False
torch.manual_seed(43)

parser = argparse.ArgumentParser()
parser.add_argument('--network-filename', required=True, type=str, help='onnx file to load.')
parser.add_argument('--greedy-input-pieces', type=str, help='if passed; load the pieces from this file and continue from here')
args = parser.parse_args()

assert args.network_filename.endswith(".onnx")
model, in_shape, out_shape, dtype, model_correctness = vnnlib_utils.onnx_to_pytorch(args.network_filename)
model.eval()

if not model_correctness:
    raise ValueError(
        "model has not been loaded successfully; some operations are not compatible, please check manually"
    )

# should fail if some maxpools are actually in the network
layers = vnnlib_utils.remove_maxpools(copy.deepcopy(list(model.children())), [], dtype=dtype)

# Skip all gradient computation for the weights of the Net
for param in model.parameters():
    param.requires_grad = False

def model_get_residual(model, X_r):
    t, x = X_r[:, 0:1], X_r[:, 1:2]
    t.requires_grad_()
    x.requires_grad_()

    u = model(torch.hstack([t, x]))

    u_x = torch.autograd.grad(u.sum(), x, create_graph=True, retain_graph=True, allow_unused=True)[0]
    u_t = torch.autograd.grad(u.sum(), t, create_graph=True, retain_graph=True, allow_unused=True)[0]

    u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True, retain_graph=True, allow_unused=True)[0]

    return u_t + u * u_x - .01/np.pi * u_xx

# ts = torch.linspace(0, 1, 1000)
# xs = torch.linspace(-1, 1, 1000)
# grid_ts, grid_xs = torch.meshgrid(ts, xs, indexing='ij')
# grid_points = torch.dstack([grid_ts, grid_xs]).reshape(-1, 2)

# grid_residuals = model_get_residual(model, grid_points)
# print(torch.abs(grid_residuals).max())

# format should be [[lb_1, ..., lb_n], [ub_1, ..., ub_n]]
domain_bounds = torch.tensor([[0, -1], [1, 1]])

softplus = torch.nn.Softplus()
sigmoid = torch.nn.Sigmoid()

ts = torch.linspace(0, 1, 1000)
xs = torch.linspace(-1, 1, 1000)
grid_ts, grid_xs = torch.meshgrid(ts, xs, indexing='ij')
grid_points = torch.dstack([grid_ts, grid_xs]).reshape(-1, 2)

def compute_gradient_dt(x, layers):
    grad_dt = torch.tensor([[2, 0]], dtype=torch.float)
    grad_dt = grad_dt.T

    for layer in layers[:4]:
        x = layer(x)

    for idx, layer in enumerate(layers[4:]):
        if idx == len(layers[4:]) - 1:
            output = layer(x)
            grad_dt = layer.weight @ grad_dt
            break

        if isinstance(layer, torch.nn.Softplus):
            continue

        # grad composes of the multiplication of all grads until now
        diagonal = torch.diag(sigmoid(layer(x)).flatten())
        grad_dt = diagonal @ layer.weight @ grad_dt

        # x contains phi_m-1 now
        x = layer(x)
        x = softplus(x)

    return output, grad_dt

def compute_gradient_dx(x, layers):
    grad_dx = torch.tensor([[0, 1]], dtype=torch.float)
    grad_dx = grad_dx.T

    for layer in layers[:4]:
        x = layer(x)

    for idx, layer in enumerate(layers[4:]):
        if idx == len(layers[4:]) - 1:
            output = layer(x)
            grad_dx = layer.weight @ grad_dx
            break

        if isinstance(layer, torch.nn.Softplus):
            continue

        # grad composes of the multiplication of all grads until now
        diagonal = torch.diag(sigmoid(layer(x)).flatten())
        grad_dx = diagonal @ layer.weight @ grad_dx

        # x contains phi_m-1 now
        x = layer(x)
        x = softplus(x)

    return output, grad_dx

def compute_torch_gradient_dt(model, X_r):
    t, x = X_r[:, 0:1], X_r[:, 1:2]
    t.requires_grad_()
    x.requires_grad_()

    u = model(torch.hstack([t, x]))
    u_t = torch.autograd.grad(u.sum(), t, create_graph=True, retain_graph=True, allow_unused=True)[0]

    return u, u_t

def compute_torch_gradient_dx(model, X_r):
    t, x = X_r[:, 0:1], X_r[:, 1:2]
    t.requires_grad_()
    x.requires_grad_()

    u = model(torch.hstack([t, x]))
    u_x = torch.autograd.grad(u.sum(), x, create_graph=True, retain_graph=True, allow_unused=True)[0]

    return u, u_x

def compute_torch_gradient_dxdx(model, X_r):
    t, x = X_r[:, 0:1], X_r[:, 1:2]
    t.requires_grad_()
    x.requires_grad_()

    u = model(torch.hstack([t, x]))
    u_x = torch.autograd.grad(u.sum(), x, create_graph=True, retain_graph=True, allow_unused=True)[0]
    u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True, retain_graph=True, allow_unused=True)[0]

    return u, u_xx

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

def compute_gradient_dxdx(x, layers):
    grad_dx = torch.tensor([[0, 1]], dtype=torch.float)
    grad_dx = grad_dx.T

    for layer in layers[:4]:
        x = layer(x)

    psi_m_2 = grad_dx
    d_psi_m_2_d_x_i = torch.tensor([[0, 0]], dtype=torch.float).T

    for idx, layer in enumerate(layers[4:]):
        if idx == len(layers[4:]) - 1:
            # last layer
            output = layer(x)
            grad_dxdx = layer.weight @ d_psi_m_2_d_x_i
            break

        if isinstance(layer, torch.nn.Softplus):
            continue

        pre_act_layer_output = layer(x).flatten()
        dd_Phi_m_1_d_x_1_d_Phi_m_2 = torch.diag(sigmoid_derivative(pre_act_layer_output) * (layer.weight @ psi_m_2).flatten()) @ layer.weight
        d_Phi_m_1_d_Phi_m_2 = torch.diag(sigmoid(pre_act_layer_output)) @ layer.weight

        d_psi_m_1_d_x_i = dd_Phi_m_1_d_x_1_d_Phi_m_2 @ psi_m_2 + d_Phi_m_1_d_Phi_m_2 @ d_psi_m_2_d_x_i
        psi_m_1 = d_Phi_m_1_d_Phi_m_2 @ psi_m_2

        psi_m_2 = psi_m_1
        d_psi_m_2_d_x_i = d_psi_m_1_d_x_i

        # x contains phi_m-1 now
        x = layer(x)
        x = softplus(x)

    return output, grad_dxdx

# idx = 550504
# output_, grad_dt_ = compute_gradient_dt(grid_points[idx:idx+1], layers)
# output, grad_dt = compute_torch_gradient_dt(model, grid_points[idx:idx+1])

# print("Output's different:", torch.abs(output - output_).item() >= 1e-6)
# print("Grad_dt's different:", torch.abs(grad_dt - grad_dt_).item() >= 1e-6)

# output_, grad_dx_ = compute_gradient_dx(grid_points[idx:idx+1], layers)
# output, grad_dx = compute_torch_gradient_dx(model, grid_points[idx:idx+1])

# print("Output's different:", torch.abs(output - output_).item() >= 1e-6)
# print("Grad_dx's different:", torch.abs(grad_dx - grad_dx_).item() >= 1e-6)

# output, grad_dxdx = compute_torch_gradient_dxdx(model, grid_points[idx:idx+1])
# output_, grad_dxdx_ = compute_gradient_dxdx(grid_points[idx:idx+1], layers)

# print("Output's different:", torch.abs(output - output_).item() >= 1e-6)
# print("Grad_dxdx's different:", torch.abs(grad_dxdx - grad_dxdx_).item() >= 1e-5)

input_pt = torch.tensor([0.5505505800247192, 0.009009003639221191])
output = -0.5788357257843018
grad_dt_value = 0.8444004058837891
grad_dx_value = -94.25582122802734
grad_dxdx_value = 17379.271484375

# output_, grad_dt_ = compute_torch_gradient_dt(model, input_pt.unsqueeze(0))
# _, grad_dx_ = compute_torch_gradient_dx(model, input_pt.unsqueeze(0))
_, grad_dxdx_ = compute_gradient_dxdx(input_pt.unsqueeze(0), layers)

import pdb
pdb.set_trace()