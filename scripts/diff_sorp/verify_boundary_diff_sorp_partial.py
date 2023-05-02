import argparse

import torch

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

boundary_conditions = torch.tensor([[1, 1], [1, 500]], dtype=dtype)
# boundary_conditions = torch.tensor([[1, 8.796875], [1, 16.59375]], dtype=dtype)
activation_relaxation = TanhRelaxation(ActivationRelaxationType.SINGLE_LINE)
relu = torch.nn.ReLU()

def empirical_evaluation(model, grid_points):
    x, t = grid_points[:, 0:1], grid_points[:, 1:2]
    x.requires_grad_()
    t.requires_grad_()

    u = relu(model(torch.hstack([x, t])))
    u_x = torch.autograd.grad(
        u, x,
        grad_outputs=torch.ones_like(u),
        retain_graph=True,
        create_graph=True
    )[0]

    return u - 5e-4 * u_x

def crown_verifier_function(layers, piece_domain, debug=True):
    diffusion_sorpion = CROWNDiffusionSorpionVerifier(
        layers,
        activation_relaxation=TanhRelaxation(ActivationRelaxationType.SINGLE_LINE),
        activation_derivative_relaxation=TanhDerivativeRelaxation(ActivationRelaxationType.SINGLE_LINE),
        activation_second_derivative_relaxation=TanhSecondDerivativeRelaxation(ActivationRelaxationType.SINGLE_LINE),
    )
    ub, lb = diffusion_sorpion.compute_boundary_conditions_partial(
        piece_domain,
        debug=debug
    )

    if any(ub < lb):
        import pdb
        pdb.set_trace()

    logs = [{} for i in range(ub.shape[0])]

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

# piece_domain = torch.tensor([
#     [[ 1.0000,  8.7969], [ 1.0000, 16.5938]],
#     [[ 1.0000,  1.0000], [ 1.0000,  8.7969]]
# ])

# t_min, x_min = piece_domain[0][0]
# t_max, x_max = piece_domain[0][1]
# ts = torch.linspace(t_min, t_max, 100)
# xs = torch.linspace(x_min, x_max, 100)
# grid_ts, grid_xs = torch.meshgrid(ts, xs, indexing='ij')
# grid_points = torch.dstack([grid_ts, grid_xs]).reshape(-1, 2)

# model_pts = empirical_evaluation(model, grid_points)
# emp_min, emp_max = model_pts.min(), model_pts.max()

# print(emp_min, emp_max)

# diffusion_sorpion = CROWNDiffusionSorpionVerifier(
#     layers,
#     activation_relaxation=TanhRelaxation(ActivationRelaxationType.SINGLE_LINE),
#     activation_derivative_relaxation=TanhDerivativeRelaxation(ActivationRelaxationType.SINGLE_LINE),
#     activation_second_derivative_relaxation=TanhSecondDerivativeRelaxation(ActivationRelaxationType.SINGLE_LINE),
# )
# ub, lb = diffusion_sorpion.compute_boundary_conditions_partial(
#     piece_domain,
#     debug=False
# )

# import pdb
# pdb.set_trace()
