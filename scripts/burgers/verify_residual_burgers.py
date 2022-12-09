import argparse

import numpy as np
import torch
from tools.custom_torch_modules import Mul

from pinn_verifier.activations.activation_relaxations import ActivationRelaxationType
from pinn_verifier.activations.tanh import TanhRelaxation, TanhDerivativeRelaxation, TanhSecondDerivativeRelaxation
from pinn_verifier.utils import load_compliant_model
from pinn_verifier.lp import LPPINNSolution
from pinn_verifier.crown import CROWNPINNPartialDerivative, CROWNPINNSolution, CROWNPINNSecondPartialDerivative
from pinn_verifier.burgers import CROWNBurgersVerifier
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

boundary_conditions = torch.tensor([[0, -1], [1, 1]], dtype=dtype)
activation_relaxation = TanhRelaxation(ActivationRelaxationType.SINGLE_LINE)
activation_derivative_relaxation = TanhDerivativeRelaxation(ActivationRelaxationType.SINGLE_LINE)
activation_second_derivative_relaxation = TanhSecondDerivativeRelaxation(ActivationRelaxationType.SINGLE_LINE)

def empirical_evaluation(model, grid_points):
    # t, x = grid_points[:, 0:1], grid_points[:, 1:2]
    # t.requires_grad_()
    # x.requires_grad_()

    # u = model(torch.hstack([t, x]))
    # # u_t = torch.autograd.grad(u.sum(), t, create_graph=True, retain_graph=True, allow_unused=True)[0]
    # u_x = torch.autograd.grad(u.sum(), x, create_graph=True, retain_graph=True, allow_unused=True)[0]
    # u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True, retain_graph=True, allow_unused=True)[0]

    # return u_xx
    t, x = grid_points[:, 0:1], grid_points[:, 1:2]
    t.requires_grad_()
    x.requires_grad_()

    u = model(torch.hstack([t, x]))
    u_x = torch.autograd.grad(u.sum(), x, create_graph=True, retain_graph=True, allow_unused=True)[0]
    u_t = torch.autograd.grad(u.sum(), t, create_graph=True, retain_graph=True, allow_unused=True)[0]
    u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True, retain_graph=True, allow_unused=True)[0]

    return u_t + u * u_x - (0.01/np.pi) * u_xx

def crown_verifier_function(layers, piece_domain, debug=True):
    # u_theta = CROWNPINNSolution(
    #     layers,
    #     activation_relaxation=activation_relaxation
    # )
    # u_theta.domain_bounds = piece_domain
    # u_theta.compute_bounds(debug=debug)

    # # u_dt_theta = CROWNPINNPartialDerivative(
    # #     u_theta,
    # #     component_idx=0,
    # #     activation_derivative_relaxation=TanhDerivativeRelaxation(ActivationRelaxationType.MULTI_LINE)
    # # )
    # # u_dt_theta.compute_bounds(debug=debug)

    # u_dx_theta = CROWNPINNPartialDerivative(
    #     u_theta,
    #     component_idx=1,
    #     activation_derivative_relaxation=activation_derivative_relaxation
    # )
    # u_dx_theta.compute_bounds(debug=debug)

    # u_dxdx_theta = CROWNPINNSecondPartialDerivative(
    #     u_dx_theta,
    #     component_idx=1,
    #     activation_second_derivative_relaxation=activation_second_derivative_relaxation
    # )
    # u_dxdx_theta.compute_bounds(debug=debug)

    # all_lbs, all_ubs = u_dxdx_theta.lower_bounds, u_dxdx_theta.upper_bounds

    burgers = CROWNBurgersVerifier(
        layers,
        activation_relaxation=activation_relaxation,
        activation_derivative_relaxation=activation_derivative_relaxation,
        activation_second_derivative_relaxation=activation_second_derivative_relaxation
    )
    ub, lb = burgers.compute_residual_bound(
        piece_domain,
        debug=False
    )

    if any(ub <= lb):
        import pdb
        pdb.set_trace()

    logs = [
        {
            "u_theta": [burgers.u_theta.lower_bounds[-1][i].item(), burgers.u_theta.upper_bounds[-1][i].item()],
            "u_dt_theta": [burgers.u_dt_theta.lower_bounds[-1][i].item(), burgers.u_dt_theta.upper_bounds[-1][i].item()],
            "u_dx_theta": [burgers.u_dx_theta.lower_bounds[-1][i].item(), burgers.u_dx_theta.upper_bounds[-1][i].item()],
            "u_dxdx_theta": [burgers.u_dxdx_theta.lower_bounds[-1][i].item(), burgers.u_dxdx_theta.upper_bounds[-1][i].item()]
        }
        for i in range(ub.shape[0])
    ]

    return lb, ub, logs


greedy_input_branching(
    layers,
    model,
    boundary_conditions,
    empirical_evaluation,
    crown_verifier_function,
    args.greedy_output_pieces,
    input_filename=args.greedy_input_pieces,
    verbose=VerbosityLevel.NO_INDIVIDUAL_PROGRESS,
    maximum_computations=args.maximum_computations,
    save_frequency=args.save_frequency
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
