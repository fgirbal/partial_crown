import argparse

import numpy as np
import torch
from tools.custom_torch_modules import Mul

from pinn_verifier.activations.activation_relaxations import ActivationRelaxationType
from pinn_verifier.activations.tanh import TanhRelaxation, TanhDerivativeRelaxation, TanhSecondDerivativeRelaxation
from pinn_verifier.utils import load_compliant_model
from pinn_verifier.lp import LPPINNSolution
from pinn_verifier.schrodingers import CROWNSchrodingersVerifier
from pinn_verifier.branching_mirrored import mirrored_greedy_input_branching, VerbosityLevel

torch.manual_seed(43)

parser = argparse.ArgumentParser()
parser.add_argument(
    '-n', '--network-filename',
    required=True,
    type=str,
    help='onnx file to load.'
)
parser.add_argument(
    '-o', '--greedy-output-pieces',
    required=True,
    type=str,
    help='write the new pieces to this file'
)
parser.add_argument(
    '-m', '--maximum-computations',
    required=True,
    type=int,
    help='maximum number of total allowed computations'
)
parser.add_argument(
    '-i', '--greedy-input-pieces',
    type=str,
    help='if passed; load the pieces from this file and continue from here'
)
parser.add_argument(
    '-s', '--save-frequency',
    type=int,
    help='save frequency in terms of branches',
    default=2500
)
args = parser.parse_args()

dtype = torch.float32
model, layers = load_compliant_model(args.network_filename)

# Skip all gradient computation for the weights of the Net
for layer in layers:
    for param in layer.parameters():
        param.requires_grad = False

initial_conditions_domain = torch.tensor([[0, -5], [torch.pi / 2, -5]], dtype=dtype)

def empirical_evaluation(model, grid_points, other_grid_points):
    t, x = grid_points[:, 0:1], grid_points[:, 1:2]
    t.requires_grad_()
    x.requires_grad_()

    h_boundary_low, v_boundary_low = model(torch.hstack([t, x])).T
    partial_x_h_boundary_low = torch.autograd.grad(h_boundary_low.sum(), x, create_graph=True, retain_graph=True, allow_unused=True)[0]
    partial_x_v_boundary_low = torch.autograd.grad(v_boundary_low.sum(), x, create_graph=True, retain_graph=True, allow_unused=True)[0]

    t, x = other_grid_points[:, 0:1], other_grid_points[:, 1:2]
    t.requires_grad_()
    x.requires_grad_()
    h_boundary_high, v_boundary_high = model(torch.hstack([t, x])).T
    partial_x_h_boundary_high = torch.autograd.grad(h_boundary_high.sum(), x, create_graph=True, retain_graph=True, allow_unused=True)[0]
    partial_x_v_boundary_high = torch.autograd.grad(v_boundary_high.sum(), x, create_graph=True, retain_graph=True, allow_unused=True)[0]

    # return (partial_x_h_boundary_high - partial_x_h_boundary_low) ** 2
    # return (partial_x_v_boundary_high - partial_x_v_boundary_low) ** 2

    u_norm_squared = (partial_x_h_boundary_high - partial_x_h_boundary_low)**2 + (partial_x_v_boundary_high - partial_x_v_boundary_low)**2
    return u_norm_squared

def crown_verifier_function(layers, piece_domain, debug=True):
    schrodingers = CROWNSchrodingersVerifier(
        layers,
        activation_relaxation=TanhRelaxation(ActivationRelaxationType.SINGLE_LINE),
        activation_derivative_relaxation=TanhDerivativeRelaxation(ActivationRelaxationType.SINGLE_LINE),
        activation_second_derivative_relaxation=TanhSecondDerivativeRelaxation(ActivationRelaxationType.SINGLE_LINE),
    )
    ub, lb = schrodingers.compute_partial_x_boundary_conditions(
        piece_domain,
        debug=debug
    )

    if any(ub <= lb):
        import pdb
        pdb.set_trace()

    logs = [
        {
            'h_error_lower_bound': schrodingers.initial_conditions_intermediate_bounds['h_error_lower_bound'][i].item(),
            'h_error_upper_bound': schrodingers.initial_conditions_intermediate_bounds['h_error_upper_bound'][i].item(),
            'v_error_lower_bound': schrodingers.initial_conditions_intermediate_bounds['v_error_lower_bound'][i].item(),
            'v_error_upper_bound': schrodingers.initial_conditions_intermediate_bounds['v_error_upper_bound'][i].item(),
        }
        for i in range(ub.shape[0])
    ]

    return lb, ub, logs

# piece_domain = torch.Tensor([
#     [[ 1.2763, -5.0000], [ 1.3254, -5.0000]],
# ])

# piece_domain = torch.Tensor([
#     [[ 0.01, -2.02], [ 0.02, -2.01]],
# ])

# t_min, x_min = piece_domain[0][0]
# t_max, x_max = piece_domain[0][1]

# ts = torch.linspace(t_min, t_max, 100)
# xs = torch.linspace(x_min, x_max, 100)
# grid_ts, grid_xs = torch.meshgrid(ts, xs, indexing='ij')
# grid_points_low = torch.dstack([grid_ts, grid_xs]).reshape(-1, 2)

# ts = torch.linspace(t_min, t_max, 100)
# xs = torch.linspace(-x_min, -x_max, 100)
# grid_ts, grid_xs = torch.meshgrid(ts, xs, indexing='ij')
# grid_points_high = torch.dstack([grid_ts, grid_xs]).reshape(-1, 2)

# t, x = grid_points_low[:, 0:1], grid_points_low[:, 1:2]
# t.requires_grad_()
# x.requires_grad_()

# h_boundary_low, v_boundary_low = model(torch.hstack([t, x])).T
# partial_x_h_boundary_low = torch.autograd.grad(h_boundary_low.sum(), x, create_graph=True, retain_graph=True, allow_unused=True)[0]
# partial_x_v_boundary_low = torch.autograd.grad(v_boundary_low.sum(), x, create_graph=True, retain_graph=True, allow_unused=True)[0]

# emp_min_h = partial_x_h_boundary_low.min()
# emp_max_h = partial_x_h_boundary_low.max()
# emp_min_v = partial_x_v_boundary_low.min()
# emp_max_v = partial_x_v_boundary_low.max()

# emp_lbs = torch.Tensor([emp_min_h, emp_min_v])
# emp_ubs = torch.Tensor([emp_max_h, emp_max_v])

# model_pts = empirical_evaluation(model, grid_points_low, grid_points_high)
# emp_min, emp_max = model_pts.min(), model_pts.max()

# schrodingers = CROWNSchrodingersVerifier(
#     layers,
#     activation_relaxation=TanhRelaxation(ActivationRelaxationType.SINGLE_LINE),
#     activation_derivative_relaxation=TanhDerivativeRelaxation(ActivationRelaxationType.SINGLE_LINE),
#     activation_second_derivative_relaxation=TanhSecondDerivativeRelaxation(ActivationRelaxationType.SINGLE_LINE),
# )
# ub, lb = schrodingers.compute_partial_x_boundary_conditions(
#     piece_domain,
#     debug=False
# )

# from pinn_verifier.crown import CROWNPINNSolution, CROWNPINNPartialDerivative, BackpropMode
# u_theta = CROWNPINNSolution(
#     model,
#     activation_relaxation=TanhRelaxation(ActivationRelaxationType.SINGLE_LINE)
# )
# u_dx_theta = CROWNPINNPartialDerivative(
#     u_theta,
#     component_idx=1,
#     activation_derivative_relaxation=TanhDerivativeRelaxation(ActivationRelaxationType.SINGLE_LINE)
# )
# u_theta.domain_bounds = piece_domain
# u_theta.compute_bounds(debug=False)
# u_dx_theta.compute_bounds(debug=True, backprop_mode=BackpropMode.COMPONENT_BACKPROP)
# ub = u_dx_theta.upper_bounds[-1]
# lb = u_dx_theta.lower_bounds[-1]

# import pdb
# pdb.set_trace()

mirrored_greedy_input_branching(
    layers,
    model,
    initial_conditions_domain,
    empirical_evaluation,
    crown_verifier_function,
    args.greedy_output_pieces,
    input_filename=args.greedy_input_pieces,
    verbose=VerbosityLevel.NO_INDIVIDUAL_PROGRESS,
    maximum_computations=args.maximum_computations,
    save_frequency=args.save_frequency
)

import pdb
pdb.set_trace()
