import argparse
from multiprocessing.context import assert_spawning

import torch
from tools.custom_torch_modules import Mul

from pinn_verifier.activations.activation_relaxations import ActivationRelaxationType
from pinn_verifier.activations.tanh import TanhDerivativeRelaxation, TanhRelaxation
from pinn_verifier.utils import load_compliant_model
from pinn_verifier.lp import LPPINNSolution
from pinn_verifier.crown import CROWNPINNPartialDerivative, CROWNPINNSolution
from pinn_verifier.branching import greedy_input_branching, VerbosityLevel

torch.manual_seed(43)

parser = argparse.ArgumentParser()
parser.add_argument('--network-filename', required=True, type=str, help='onnx file to load.')
parser.add_argument('--greedy-output-pieces', required=True, type=str, help='write the new pieces to this file')
parser.add_argument('--greedy-input-pieces', type=str, help='if passed; load the pieces from this file and continue from here')
args = parser.parse_args()

dtype = torch.float32
model, layers = load_compliant_model(args.network_filename)

# Skip all gradient computation for the weights of the Net
for param in model.parameters():
    param.requires_grad = False

boundary_conditions = torch.tensor([[0.89-1e-2, -1-1e-2], [0.89+1e-2, -1+1e-2]])
# boundary_conditions = torch.tensor([[0, -1], [1, -1]], dtype=dtype)
activation_relaxation = TanhRelaxation(ActivationRelaxationType.SINGLE_LINE)

def empirical_evaluation(model, grid_points):
    return model(grid_points)

def ibp_verifier_function(layers, piece_domain, debug=True):
    verifier = CROWNPINNSolution(
        layers,
        activation_relaxation=activation_relaxation
    )
    verifier.domain_bounds = piece_domain
    verifier.compute_IBP_bounds(debug=debug)
    return verifier.lower_bounds[-1], verifier.upper_bounds[-1], {}


def crown_verifier_function(layers, piece_domain, debug=True):
    verifier = CROWNPINNSolution(
        layers,
        activation_relaxation=activation_relaxation
    )
    verifier.domain_bounds = piece_domain
    verifier.compute_bounds(debug=debug)
    all_lbs, all_ubs = verifier.lower_bounds, verifier.upper_bounds

    return all_lbs[-1], all_ubs[-1], {}


# greedy_input_branching(
#     layers,
#     model,
#     boundary_conditions,
#     empirical_evaluation,
#     crown_verifier_function,
#     args.greedy_output_pieces,
#     input_filename=args.greedy_input_pieces,
#     verbose=VerbosityLevel.NO_INDIVIDUAL_PROGRESS,
#     maximum_computations=1000,
#     save_frequency=100
# )

import pickle
# with open("../pinn_verifier/lp_bounds__0.89__-1.pb", "rb") as f:
#     bounds = pickle.load(f)
with open("../pinn_verifier/lp_bounds_full_domain.pb", "rb") as f:
    bounds = pickle.load(f)

u_theta = CROWNPINNSolution(
    layers,
    activation_relaxation=activation_relaxation
)
u_theta.domain_bounds = bounds["domain"]
u_theta.compute_bounds(debug=True)

assert all([all(u_theta.lower_bounds[i] <= (bounds['u_theta']['lbs'][i])+1e-5) for i in range(len(u_theta.lower_bounds))])
assert all([all(u_theta.upper_bounds[i] >= (bounds['u_theta']['ubs'][i])-1e-5) for i in range(len(u_theta.upper_bounds))])

lp_u_dt_theta_bounds = bounds['u_dt_theta']

u_dt_theta = CROWNPINNPartialDerivative(
    u_theta,
    component_idx=0,
    activation_derivative_relaxation=TanhDerivativeRelaxation(ActivationRelaxationType.MULTI_LINE)
)
u_dt_theta.compute_bounds(
    debug=True,
    lp_first_matmul_bounds=[lp_u_dt_theta_bounds['d_phi_m_d_phi_m_1_lbs'], lp_u_dt_theta_bounds['d_phi_m_d_phi_m_1_ubs']],
    lp_layer_output_bounds=[lp_u_dt_theta_bounds['lbs'], lp_u_dt_theta_bounds['ubs']]
)

import pdb
pdb.set_trace()
