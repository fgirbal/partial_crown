import argparse

import numpy as np
import torch
from tools.custom_torch_modules import Mul

from pinn_verifier.activations.activation_relaxations import ActivationRelaxationType
from pinn_verifier.activations.tanh import TanhRelaxation, TanhDerivativeRelaxation, TanhSecondDerivativeRelaxation
from pinn_verifier.utils import load_compliant_model
from pinn_verifier.diff_sorp import CROWNDiffusionSorpionVerifier
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

residual_domain = torch.tensor([[0, 0], [1, 500]], dtype=dtype)
activation_relaxation = TanhRelaxation(ActivationRelaxationType.SINGLE_LINE)
activation_derivative_relaxation = TanhDerivativeRelaxation(ActivationRelaxationType.SINGLE_LINE)
activation_second_derivative_relaxation = TanhSecondDerivativeRelaxation(ActivationRelaxationType.SINGLE_LINE)

D: float = 5e-4
por: float = 0.29
rho_s: float = 2880
k_f: float = 3.5e-4
n_f: float = 0.874

def empirical_evaluation(model, grid_points):
    x, t = grid_points[:, 0:1], grid_points[:, 1:2]
    x.requires_grad_()
    t.requires_grad_()

    u = torch.nn.ReLU()(model(torch.hstack([x, t])))

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

    retardation_factor = 1 + ((1 - por) / por) * rho_s * k_f * n_f * (u + 1e-6) ** (
        n_f - 1
    )

    return u_t - D / retardation_factor * u_xx

def crown_verifier_function(layers, piece_domain, debug=True):
    diff_sorp = CROWNDiffusionSorpionVerifier(
        layers,
        activation_relaxation=activation_relaxation,
        activation_derivative_relaxation=activation_derivative_relaxation,
        activation_second_derivative_relaxation=activation_second_derivative_relaxation
    )
    ub, lb = diff_sorp.compute_residual_bound(
        piece_domain,
        debug=debug
    )

    if any(ub <= lb):
        import pdb
        pdb.set_trace()

    logs = [
        {}
        for i in range(ub.shape[0])
    ]

    return lb, ub, logs

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
# burgers = CROWNBurgersVerifier(
#     layers,
#     activation_relaxation=TanhRelaxation(ActivationRelaxationType.SINGLE_LINE),
#     activation_derivative_relaxation=TanhDerivativeRelaxation(ActivationRelaxationType.SINGLE_LINE),
#     activation_second_derivative_relaxation=TanhSecondDerivativeRelaxation(ActivationRelaxationType.SINGLE_LINE),
# )
# ub, lb = burgers.compute_residual_bound(
#     piece_domain,
#     debug=False
# )
# print(time.time()-s)

# intermediate_comps_time = burgers.u_theta.computation_times['total_computation_time'] +\
#     burgers.u_dt_theta.computation_times['total_computation_time'] +\
#     burgers.u_dx_theta.computation_times['total_computation_time'] +\
#     burgers.u_dxdx_theta.computation_times['total_computation_time']

# intermediate_relaxations_time = burgers.u_theta.computation_times['activation_relaxations'] +\
#     burgers.u_dt_theta.computation_times['activation_relaxations'] +\
#     burgers.u_dx_theta.computation_times['activation_relaxations'] +\
#     burgers.u_dxdx_theta.computation_times['activation_relaxations']

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
    save_history=True
)

# piece_domain = torch.tensor([[ 0.7422, -0.0078], [ 0.7441, -0.0039]])
# u_theta = CROWNPINNSolution(
#     layers,
#     activation_relaxation=activation_relaxation
# )
# u_theta.domain_bounds = piece_domain
# u_theta.compute_bounds(debug=False)

# u_dt_theta = CROWNPINNPartialDerivative(
#     u_theta,
#     component_idx=1,
#     activation_derivative_relaxation=TanhDerivativeRelaxation(ActivationRelaxationType.MULTI_LINE)
# )
# u_dt_theta.compute_bounds(debug=True)

# ts = torch.linspace(piece_domain[0, 0], piece_domain[1, 0], 250)
# xs = torch.linspace(piece_domain[0, 1], piece_domain[1, 1], 250)
# grid_ts, grid_xs = torch.meshgrid(ts, xs, indexing='ij')
# grid_points = torch.dstack([grid_ts, grid_xs]).reshape(-1, 2)

# model_pts = empirical_evaluation(model, grid_points)
# emp_min, emp_max = model_pts.min(), model_pts.max()

import pdb
pdb.set_trace()
