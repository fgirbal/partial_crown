import copy
from typing import List

import torch

import tools.bab_tools.vnnlib_utils as vnnlib_utils

from pinn_verifier.activation_relaxations import ActivationRelaxationType, SoftplusRelaxation
from pinn_verifier import PINNSolution, PINNPartialDerivative, PINNSecondPartialDerivative


def load_model():
    model, in_shape, out_shape, dtype, model_correctness = vnnlib_utils.onnx_to_pytorch("data/Softplus.onnx")
    model.eval()

    if not model_correctness:
        raise ValueError(
            "model has not been loaded successfully; some operations are not compatible, please check manually"
        )

    # should fail if some maxpools are actually in the network
    layers = vnnlib_utils.remove_maxpools(copy.deepcopy(list(model.children())), [], dtype=dtype)

    return model, layers

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


def test_at_point(model: torch.nn.Module, layers: List[torch.nn.Module], input_point: torch.Tensor) -> None:
    input_point_unsqueezed = input_point.unsqueeze(0)

    u_theta = PINNSolution(
        layers,
        activation_relaxation=SoftplusRelaxation(ActivationRelaxationType.MULTI_LINE),
        feasibility_tol=1e-4,
        optimality_tol=1e-4
    )
    u_theta.domain_bounds = torch.tensor([[input_point[0], input_point[1]], [input_point[0], input_point[1]]])
    u_theta.compute_bounds(debug=True)

    output = model(input_point_unsqueezed)
    assert u_theta.lower_bounds[-1] <= output
    assert u_theta.upper_bounds[-1] >= output
    assert torch.abs(output - u_theta.lower_bounds[-1]) <= 1e-2
    assert torch.abs(output - u_theta.upper_bounds[-1]) <= 1e-2

    u_dt_theta = PINNPartialDerivative(u_theta, component_idx=0)
    u_dt_theta.compute_bounds(debug=True)

    _, grad_dt = compute_torch_gradient_dt(model, input_point_unsqueezed)
    assert u_dt_theta.lower_bounds[-1] <= grad_dt
    assert u_dt_theta.upper_bounds[-1] >= grad_dt
    assert torch.abs(grad_dt - u_dt_theta.lower_bounds[-1]) <= 1e2
    assert torch.abs(grad_dt - u_dt_theta.upper_bounds[-1]) <= 1e2

    u_dx_theta = PINNPartialDerivative(u_theta, component_idx=1)
    u_dx_theta.compute_bounds(debug=True)

    _, grad_dx = compute_torch_gradient_dt(model, input_point_unsqueezed)
    assert u_dx_theta.lower_bounds[-1] <= grad_dx
    assert u_dx_theta.upper_bounds[-1] >= grad_dx
    assert torch.abs(grad_dx - u_dx_theta.lower_bounds[-1]) <= 1e2
    assert torch.abs(grad_dx - u_dx_theta.upper_bounds[-1]) <= 1e2

    u_dxdx_theta = PINNSecondPartialDerivative(u_dx_theta, component_idx=1)
    u_dxdx_theta.compute_bounds(debug=True)

    _, grad_dxdx = compute_torch_gradient_dt(model, input_point_unsqueezed)
    assert u_dxdx_theta.lower_bounds[-1] <= grad_dxdx
    assert u_dxdx_theta.upper_bounds[-1] >= grad_dxdx
    assert torch.abs(grad_dxdx - u_dxdx_theta.lower_bounds[-1]) <= 1e3
    assert torch.abs(grad_dxdx - u_dxdx_theta.upper_bounds[-1]) <= 1e3

model, layers = load_model()
test_at_point(model, layers, torch.tensor([0.5505505800247192, 0.009009003639221191]))