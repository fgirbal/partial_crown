"""
Test the computations of partial-CROWN bounds are done correctly.
"""
from typing import List

import torch

from pinn_verifier.utils import load_compliant_model
from pinn_verifier.activations import ActivationRelaxationType
from pinn_verifier.activations.tanh import TanhRelaxation, TanhDerivativeRelaxation, TanhSecondDerivativeRelaxation
from pinn_verifier.crown import CROWNPINNSolution, CROWNPINNPartialDerivative, CROWNPINNSecondPartialDerivative
# from pinn_verifier.burgers import CROWNBurgersVerifier


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


def fn_test_at_point(model: torch.nn.Module, layers: List[torch.nn.Module], input_point: torch.Tensor) -> None:
    input_point_unsqueezed = input_point.unsqueeze(0)

    u_theta = CROWNPINNSolution(
        layers,
        activation_relaxation=TanhRelaxation(ActivationRelaxationType.SINGLE_LINE),
    )
    u_theta.domain_bounds = torch.tensor([
        [input_point[0] - 1e-3, input_point[1] - 1e-3],
        [input_point[0] + 1e-3, input_point[1] + 1e-3]
    ])
    u_theta.compute_bounds(debug=False)

    output = model(input_point_unsqueezed)
    assert u_theta.lower_bounds[-1] <= output
    assert u_theta.upper_bounds[-1] >= output
    assert torch.abs(output - u_theta.lower_bounds[-1]) <= 1e-1
    assert torch.abs(output - u_theta.upper_bounds[-1]) <= 1e-1

    u_dt_theta = CROWNPINNPartialDerivative(
        u_theta,
        component_idx=0,
        activation_derivative_relaxation=TanhDerivativeRelaxation(ActivationRelaxationType.SINGLE_LINE)
    )
    u_dt_theta.compute_bounds(debug=False)

    _, grad_dt = compute_torch_gradient_dt(model, input_point_unsqueezed)
    assert u_dt_theta.lower_bounds[-1] <= grad_dt
    assert u_dt_theta.upper_bounds[-1] >= grad_dt
    assert torch.abs(grad_dt - u_dt_theta.lower_bounds[-1]) <= 1e1
    assert torch.abs(grad_dt - u_dt_theta.upper_bounds[-1]) <= 1e1

    u_dx_theta = CROWNPINNPartialDerivative(
        u_theta,
        component_idx=1,
        activation_derivative_relaxation=TanhDerivativeRelaxation(ActivationRelaxationType.SINGLE_LINE)
    )
    u_dx_theta.compute_bounds(debug=False)

    _, grad_dx = compute_torch_gradient_dx(model, input_point_unsqueezed)
    assert u_dx_theta.lower_bounds[-1] <= grad_dx
    assert u_dx_theta.upper_bounds[-1] >= grad_dx
    assert torch.abs(grad_dx - u_dx_theta.lower_bounds[-1]) <= 1e1
    assert torch.abs(grad_dx - u_dx_theta.upper_bounds[-1]) <= 1e1

    u_dxdx_theta = CROWNPINNSecondPartialDerivative(
        u_dx_theta,
        component_idx=1,
        activation_derivative_derivative_relaxation=TanhSecondDerivativeRelaxation(ActivationRelaxationType.SINGLE_LINE)
    )
    u_dxdx_theta.compute_bounds(debug=False)

    _, grad_dxdx = compute_torch_gradient_dxdx(model, input_point_unsqueezed)
    assert u_dxdx_theta.lower_bounds[-1] <= grad_dxdx
    assert u_dxdx_theta.upper_bounds[-1] >= grad_dxdx
    assert torch.abs(grad_dxdx - u_dxdx_theta.lower_bounds[-1]) <= 1e3
    assert torch.abs(grad_dxdx - u_dxdx_theta.upper_bounds[-1]) <= 1e3


def test_random_point_inside_component_bounds():
    # torch.manual_seed(43)
    model, layers = load_compliant_model("data/burgers_tanh_lbfgs.onnx")

    # Skip all gradient computation for the weights of the Net
    for layer in layers:
        for param in layer.parameters():
            param.requires_grad = False
    
    N = 10
    ts = torch.rand(N)
    xs = torch.zeros(N).uniform_(-1, 1)
    grid_ts, grid_xs = torch.meshgrid(ts, xs, indexing='ij')
    grid_points = torch.dstack([grid_ts, grid_xs]).reshape(-1, 2)

    for grid_point in grid_points:
        fn_test_at_point(model, layers, grid_point)
