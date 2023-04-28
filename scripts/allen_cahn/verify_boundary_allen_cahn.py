import argparse

import numpy as np
import torch
from tools.custom_torch_modules import Mul

from pinn_verifier.activations.activation_relaxations import ActivationRelaxationType
from pinn_verifier.activations.tanh import TanhRelaxation, TanhDerivativeRelaxation, TanhSecondDerivativeRelaxation
from pinn_verifier.utils import load_compliant_model
from pinn_verifier.allen_cahn import CROWNAllenCahnVerifier
from pinn_verifier.branching_mirrored import mirrored_greedy_input_branching_allen_cahn, VerbosityLevel

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

initial_conditions_domain = torch.tensor([[-1, 0], [-1, 1]], dtype=dtype)

def empirical_evaluation(model, grid_points_low, grid_points_high):
    u_boundary_low = model(grid_points_low).T
    u_boundary_high = model(grid_points_high).T

    return u_boundary_high - u_boundary_low

def crown_verifier_function(layers, piece_domain, debug=True):
    allen_cahn = CROWNAllenCahnVerifier(
        layers,
        activation_relaxation=TanhRelaxation(ActivationRelaxationType.SINGLE_LINE),
        activation_derivative_relaxation=TanhDerivativeRelaxation(ActivationRelaxationType.SINGLE_LINE),
        activation_second_derivative_relaxation=TanhSecondDerivativeRelaxation(ActivationRelaxationType.SINGLE_LINE),
    )
    ub, lb = allen_cahn.compute_boundary_conditions(
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
#    [[1.0000,  0.5000], [1.0000,  0.7500]]
# ])

# piece_domain = torch.Tensor([
#    [[0.99 - 1e-2,  0.5 - 1e-2], [0.99 + 1e-2,  0.5 + 1e-2]]
# ])

# allen_cahn = CROWNAllenCahnVerifier(
#     layers,
#     activation_relaxation=TanhRelaxation(ActivationRelaxationType.SINGLE_LINE),
#     activation_derivative_relaxation=TanhDerivativeRelaxation(ActivationRelaxationType.SINGLE_LINE),
#     activation_second_derivative_relaxation=TanhSecondDerivativeRelaxation(ActivationRelaxationType.SINGLE_LINE),
# )
# u_theta = allen_cahn.u_theta
# u_theta.domain_bounds = piece_domain
# u_theta.compute_bounds(debug=True)

# t_min, x_min = piece_domain[0][0]
# t_max, x_max = piece_domain[0][1]

# ts = torch.linspace(t_min, t_max, 100)
# xs = torch.linspace(x_min, x_max, 100)
# grid_ts, grid_xs = torch.meshgrid(ts, xs, indexing='ij')
# grid_points = torch.dstack([grid_ts, grid_xs]).reshape(-1, 2)

# model_pts = empirical_evaluation(model, grid_points, grid_points)
# emp_min, emp_max = model_pts.min(), model_pts.max()

# import pdb
# pdb.set_trace()

mirrored_greedy_input_branching_allen_cahn(
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

# import pdb
# pdb.set_trace()
