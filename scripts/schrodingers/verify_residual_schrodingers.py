import argparse

import numpy as np
import torch
from tools.custom_torch_modules import Mul

from pinn_verifier.activations.activation_relaxations import ActivationRelaxationType
from pinn_verifier.activations.tanh import TanhRelaxation, TanhDerivativeRelaxation, TanhSecondDerivativeRelaxation
from pinn_verifier.utils import load_compliant_model
from pinn_verifier.lp import LPPINNSolution
from pinn_verifier.schrodingers import CROWNSchrodingersVerifier
from pinn_verifier.branching import greedy_input_branching, VerbosityLevel

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

residual_domain = torch.tensor([[0, -5], [torch.pi / 2, 5]], dtype=dtype)

def empirical_evaluation(model, grid_points):
    t, x = grid_points[:, 0:1], grid_points[:, 1:2]
    t.requires_grad_()
    x.requires_grad_()

    u = model(torch.hstack([t, x]))
    h, v = u[:, 0:1], u[:, 1:2]
    h_x = torch.autograd.grad(h.sum(), x, create_graph=True, retain_graph=True, allow_unused=True)[0]
    v_x = torch.autograd.grad(v.sum(), x, create_graph=True, retain_graph=True, allow_unused=True)[0]

    h_t = torch.autograd.grad(h.sum(), t, create_graph=True, retain_graph=True, allow_unused=True)[0]
    v_t = torch.autograd.grad(v.sum(), t, create_graph=True, retain_graph=True, allow_unused=True)[0]

    h_xx = torch.autograd.grad(h_x.sum(), x, create_graph=True, retain_graph=True, allow_unused=True)[0]
    v_xx = torch.autograd.grad(v_x.sum(), x, create_graph=True, retain_graph=True, allow_unused=True)[0]

    # print('h:', h.min().item(), h.max().item())
    # print('v:', v.min().item(), v.max().item())
    # print('|u|^2:', (h**2 + v**2).min().item(), (h**2 + v**2).max().item())
    # print('h_t:', h_t.min().item(), h_t.max().item())
    # print('v_t:', v_t.min().item(), v_t.max().item())
    # print('h_xx:', h_xx.min().item(), h_xx.max().item())
    # print('v_xx:', v_xx.min().item(), v_xx.max().item())

    real_f = v_t - 0.5 * h_xx - (h**2 + v**2) * h
    imag_f = h_t + 0.5 * v_xx + (h**2 + v**2) * v

    # print('real_f:', real_f.min().item(), real_f.max().item())
    # print('imag_f:', imag_f.min().item(), imag_f.max().item())

    # return real_f
    # return imag_f

    return real_f**2 + imag_f**2

def crown_verifier_function(layers, piece_domain, debug=True):
    schrodingers = CROWNSchrodingersVerifier(
        layers,
        activation_relaxation=TanhRelaxation(ActivationRelaxationType.SINGLE_LINE),
        activation_derivative_relaxation=TanhDerivativeRelaxation(ActivationRelaxationType.SINGLE_LINE),
        activation_second_derivative_relaxation=TanhSecondDerivativeRelaxation(ActivationRelaxationType.SINGLE_LINE),
    )
    ub, lb = schrodingers.compute_residual_bound(
        piece_domain,
        debug=debug
    )

    if any(ub <= lb):
        import pdb
        pdb.set_trace()

    logs = [
        {
            'h_theta_lower_bound': schrodingers.residual_intermediate_bounds['h_theta_lower_bound'][i].item(),
            'h_theta_upper_bound': schrodingers.residual_intermediate_bounds['h_theta_upper_bound'][i].item(),
            'v_theta_lower_bound': schrodingers.residual_intermediate_bounds['v_theta_lower_bound'][i].item(),
            'v_theta_upper_bound': schrodingers.residual_intermediate_bounds['v_theta_upper_bound'][i].item(),
            'u_theta_norm_lower_bound': schrodingers.residual_intermediate_bounds['u_theta_norm_lower_bound'][i].item(),
            'u_theta_norm_upper_bound': schrodingers.residual_intermediate_bounds['u_theta_norm_upper_bound'][i].item(),
            'h_dt_theta_lower_bound': schrodingers.residual_intermediate_bounds['h_dt_theta_lower_bound'][i].item(),
            'h_dt_theta_upper_bound': schrodingers.residual_intermediate_bounds['h_dt_theta_upper_bound'][i].item(),
            'v_dt_theta_lower_bound': schrodingers.residual_intermediate_bounds['v_dt_theta_lower_bound'][i].item(),
            'v_dt_theta_upper_bound': schrodingers.residual_intermediate_bounds['v_dt_theta_upper_bound'][i].item(),
            'h_dxdx_theta_lower_bound': schrodingers.residual_intermediate_bounds['h_dxdx_theta_lower_bound'][i].item(),
            'h_dxdx_theta_upper_bound': schrodingers.residual_intermediate_bounds['h_dxdx_theta_upper_bound'][i].item(),
            'v_dxdx_theta_lower_bound': schrodingers.residual_intermediate_bounds['v_dxdx_theta_lower_bound'][i].item(),
            'v_dxdx_theta_upper_bound': schrodingers.residual_intermediate_bounds['v_dxdx_theta_upper_bound'][i].item(),
            'real_residual_lower_bound': schrodingers.residual_intermediate_bounds['real_residual_lower_bound'][i].item(),
            'real_residual_upper_bound': schrodingers.residual_intermediate_bounds['real_residual_upper_bound'][i].item(),
            'imag_residual_lower_bound': schrodingers.residual_intermediate_bounds['imag_residual_lower_bound'][i].item(),
            'imag_residual_upper_bound': schrodingers.residual_intermediate_bounds['imag_residual_upper_bound'][i].item()
        }
        for i in range(ub.shape[0])
    ]

    return lb, ub, logs

# piece_domain = torch.Tensor([
#     [[ 0.8408, 0.0504], [ 0.8410, 0.0506]],
# ])

# piece_domain = torch.Tensor([
#     [[0.2332, 0.2148], [0.2347, 0.2246]],
#     [[0.2332, 0.2246], [0.2347, 0.2344]],
#     [[0.2347, 0.2148], [0.2362, 0.2246]],
#     [[0.2347, 0.2246], [0.2362, 0.2344]]
# ])

# t_min, x_min = piece_domain[0][0]
# t_max, x_max = piece_domain[0][1]

# ts = torch.linspace(t_min, t_max, 100)
# xs = torch.linspace(x_min, x_max, 100)
# grid_ts, grid_xs = torch.meshgrid(ts, xs, indexing='ij')
# grid_points = torch.dstack([grid_ts, grid_xs]).reshape(-1, 2)

# model_pts = empirical_evaluation(model, grid_points)
# emp_min, emp_max = model_pts.min(), model_pts.max()

# import time
# s = time.time()
# schrodingers = CROWNSchrodingersVerifier(
#     layers,
#     activation_relaxation=TanhRelaxation(ActivationRelaxationType.SINGLE_LINE),
#     activation_derivative_relaxation=TanhDerivativeRelaxation(ActivationRelaxationType.SINGLE_LINE),
#     activation_second_derivative_relaxation=TanhSecondDerivativeRelaxation(ActivationRelaxationType.SINGLE_LINE),
# )
# ub, lb = schrodingers.compute_residual_bound(
#     piece_domain,
#     debug=False
# )
# print(time.time()-s)

# intermediate_comps_time = schrodingers.u_theta.computation_times['total_computation_time'] +\
#     schrodingers.u_dt_theta.computation_times['total_computation_time'] +\
#     schrodingers.u_dx_theta.computation_times['total_computation_time'] +\
#     schrodingers.u_dxdx_theta.computation_times['total_computation_time']

# intermediate_relaxations_time = schrodingers.u_theta.computation_times['activation_relaxations'] +\
#     schrodingers.u_dt_theta.computation_times['activation_relaxations'] +\
#     schrodingers.u_dx_theta.computation_times['activation_relaxations'] +\
#     schrodingers.u_dxdx_theta.computation_times['activation_relaxations']

# print(intermediate_comps_time)
# print(intermediate_relaxations_time)
# print(intermediate_relaxations_time/intermediate_comps_time)

# import pdb
# pdb.set_trace()

greedy_input_branching(
    layers,
    model,
    residual_domain,
    empirical_evaluation,
    crown_verifier_function,
    args.greedy_output_pieces,
    input_filename=args.greedy_input_pieces,
    verbose=VerbosityLevel.NO_INDIVIDUAL_PROGRESS,
    maximum_computations=args.maximum_computations,
    save_frequency=args.save_frequency,
    save_history=False,
    input_v1=False
)

# import pdb
# pdb.set_trace()
