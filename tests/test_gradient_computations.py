"""
Test formulae used for gradient computations to make sure it is correct for FCNNs with normalizing layers.
"""

import copy
import argparse

from scipy import optimize
from tqdm import tqdm
import numpy as np
import torch
import gurobipy as grb

from pinn_verifier.utils import load_compliant_model

activation_fun_type = torch.nn.Tanh
tanh = torch.nn.Tanh()
activation_fun = tanh
activation_derivative = lambda x: 1 - tanh(x)**2
activation_derivative_derivative = lambda x: -2* tanh(x) * (1 - tanh(x)**2)


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

        if isinstance(layer, activation_fun_type):
            continue

        # grad composes of the multiplication of all grads until now
        diagonal = torch.diag(activation_derivative(layer(x)).flatten())
        grad_dt = diagonal @ layer.weight @ grad_dt

        # x contains phi_m-1 now
        x = layer(x)
        x = activation_fun(x)

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

        if isinstance(layer, activation_fun_type):
            continue

        # grad composes of the multiplication of all grads until now
        diagonal = torch.diag(activation_derivative(layer(x)).flatten())
        grad_dx = diagonal @ layer.weight @ grad_dx

        # x contains phi_m-1 now
        x = layer(x)
        x = activation_fun(x)

    return output, grad_dx

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

        if isinstance(layer, activation_fun_type):
            continue

        pre_act_layer_output = layer(x).flatten()
        dd_Phi_m_1_d_x_1_d_Phi_m_2 = torch.diag(activation_derivative_derivative(pre_act_layer_output) * (layer.weight @ psi_m_2).flatten()) @ layer.weight
        d_Phi_m_1_d_Phi_m_2 = torch.diag(activation_derivative(pre_act_layer_output)) @ layer.weight

        d_psi_m_1_d_x_i = dd_Phi_m_1_d_x_1_d_Phi_m_2 @ psi_m_2 + d_Phi_m_1_d_Phi_m_2 @ d_psi_m_2_d_x_i
        psi_m_1 = d_Phi_m_1_d_Phi_m_2 @ psi_m_2

        psi_m_2 = psi_m_1
        d_psi_m_2_d_x_i = d_psi_m_1_d_x_i

        # x contains phi_m-1 now
        x = layer(x)
        x = activation_fun(x)

    return output, grad_dxdx

def gt_compute_torch_gradient_dt(model, X_r):
    t, x = X_r[:, 0:1], X_r[:, 1:2]
    t.requires_grad_()
    x.requires_grad_()

    u = model(torch.hstack([t, x]))
    u_t = torch.autograd.grad(u.sum(), t, create_graph=True, retain_graph=True, allow_unused=True)[0]

    return u, u_t

def gt_compute_torch_gradient_dx(model, X_r):
    t, x = X_r[:, 0:1], X_r[:, 1:2]
    t.requires_grad_()
    x.requires_grad_()

    u = model(torch.hstack([t, x]))
    u_x = torch.autograd.grad(u.sum(), x, create_graph=True, retain_graph=True, allow_unused=True)[0]

    return u, u_x

def gt_compute_torch_gradient_dxdx(model, X_r):
    t, x = X_r[:, 0:1], X_r[:, 1:2]
    t.requires_grad_()
    x.requires_grad_()

    u = model(torch.hstack([t, x]))
    u_x = torch.autograd.grad(u.sum(), x, create_graph=True, retain_graph=True, allow_unused=True)[0]
    u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True, retain_graph=True, allow_unused=True)[0]

    return u, u_xx


def test_gradient_computations():
    N = 100

    torch.manual_seed(43)
    model, layers = load_compliant_model("data/burgers_tanh_lbfgs.onnx")

    # Skip all gradient computation for the weights of the Net
    for layer in layers:
        for param in layer.parameters():
            param.requires_grad = False

    ts = torch.rand(N)
    xs = torch.zeros(N).uniform_(-1, 1)
    grid_ts, grid_xs = torch.meshgrid(ts, xs, indexing='ij')
    grid_points = torch.dstack([grid_ts, grid_xs]).reshape(-1, 2)

    gt_output, gt_grad_dt = gt_compute_torch_gradient_dt(model, grid_points)
    _, gt_grad_dx = gt_compute_torch_gradient_dx(model, grid_points)
    _, gt_grad_dxdx = gt_compute_torch_gradient_dxdx(model, grid_points)

    output_grad_dt = torch.zeros_like(gt_output)
    output_grad_dx = torch.zeros_like(gt_output)
    output_grad_dxdx = torch.zeros_like(gt_output)
    grad_dt_ = torch.zeros_like(gt_grad_dt)
    grad_dx_ = torch.zeros_like(gt_grad_dx)
    grad_dxdx_ = torch.zeros_like(gt_grad_dxdx)

    for idx in range(N * N):
        output_grad_dt[idx], grad_dt_[idx] = compute_gradient_dt(grid_points[idx:idx+1], layers)
        output_grad_dx[idx], grad_dx_[idx] = compute_gradient_dx(grid_points[idx:idx+1], layers)
        output_grad_dxdx[idx], grad_dxdx_[idx] = compute_gradient_dxdx(grid_points[idx:idx+1], layers)
    
    assert (output_grad_dt - gt_output).flatten().abs().max() <= 1e-4
    assert (output_grad_dx - gt_output).flatten().abs().max() <= 1e-4
    assert (output_grad_dxdx - gt_output).flatten().abs().max() <= 1e-4

    assert (grad_dt_ - gt_grad_dt).flatten().abs().max() <= 1e-3
    assert (grad_dx_ - gt_grad_dx).flatten().abs().max() <= 1e-3
    assert (grad_dxdx_ - gt_grad_dxdx).flatten().abs().max() <= 1e-3

    return True
