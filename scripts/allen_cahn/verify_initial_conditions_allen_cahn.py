import argparse

import numpy as np
import torch
from tools.custom_torch_modules import Mul

from pinn_verifier.activations.activation_relaxations import ActivationRelaxationType
from pinn_verifier.activations.tanh import TanhRelaxation, TanhDerivativeRelaxation, TanhSecondDerivativeRelaxation
from pinn_verifier.utils import load_compliant_model
from pinn_verifier.allen_cahn import CROWNAllenCahnVerifier
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

initial_conditions_domain = torch.tensor([[-1, 0], [1, 0]], dtype=dtype)

def empirical_evaluation(model, grid_points):
    return model(grid_points) - (grid_points[:, 0]**2 * torch.cos(torch.pi * grid_points[:, 0])).unsqueeze(1)

def crown_verifier_function(layers, piece_domain, debug=True):
    burgers = CROWNAllenCahnVerifier(
        layers,
        activation_relaxation=TanhRelaxation(ActivationRelaxationType.SINGLE_LINE),
        activation_derivative_relaxation=TanhDerivativeRelaxation(ActivationRelaxationType.SINGLE_LINE),
        activation_second_derivative_relaxation=TanhSecondDerivativeRelaxation(ActivationRelaxationType.SINGLE_LINE),
    )
    ub, lb = burgers.compute_initial_conditions_bounds(
        piece_domain,
        debug=debug
    )

    if any(ub <= lb):
        import pdb
        pdb.set_trace()

    logs = [{} for i in range(ub.shape[0])]

    return lb, ub, logs


greedy_input_branching(
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
