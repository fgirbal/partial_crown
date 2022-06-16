import copy
import argparse
import time
import json

from scipy import optimize
from tqdm import tqdm
import numpy as np
import torch
import gurobipy as grb

import tools.bab_tools.vnnlib_utils as vnnlib_utils
from pinn_verifier import BurgersVerifier
# from activation_relaxations import ActivationRelaxationType, SoftplusRelaxation

debug = False
torch.manual_seed(43)

parser = argparse.ArgumentParser()
parser.add_argument('--network-filename', required=True, type=str, help='onnx file to load.')
parser.add_argument('--greedy-output-pieces', required=True, type=str, help='write the new pieces to this file')
parser.add_argument('--greedy-input-pieces', type=str, help='if passed; load the pieces from this file and continue from here')
parser.add_argument('--no-debug', action="store_false", help='if true no debug printing will happen')
args = parser.parse_args()

assert args.network_filename.endswith(".onnx")
model, in_shape, out_shape, dtype, model_correctness = vnnlib_utils.onnx_to_pytorch(args.network_filename)
model.eval()

if not model_correctness:
    raise ValueError(
        "model has not been loaded successfully; some operations are not compatible, please check manually"
    )

# should fail if some maxpools are actually in the network
layers = vnnlib_utils.remove_maxpools(copy.deepcopy(list(model.children())), [], dtype=dtype)

# Skip all gradient computation for the weights of the Net
for param in model.parameters():
    param.requires_grad = False

def model_get_residual(model, X_r):
    t, x = X_r[:, 0:1], X_r[:, 1:2]
    t.requires_grad_()
    x.requires_grad_()

    u = model(torch.hstack([t, x]))

    u_x = torch.autograd.grad(u.sum(), x, create_graph=True, retain_graph=True, allow_unused=True)[0]
    u_t = torch.autograd.grad(u.sum(), t, create_graph=True, retain_graph=True, allow_unused=True)[0]

    u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True, retain_graph=True, allow_unused=True)[0]

    return u_t + u * u_x - .01/np.pi * u_xx

# ts = torch.linspace(0, 1, 1000)
# xs = torch.linspace(-1, 1, 1000)
# grid_ts, grid_xs = torch.meshgrid(ts, xs, indexing='ij')
# grid_points = torch.dstack([grid_ts, grid_xs]).reshape(-1, 2)

# grid_residuals = model_get_residual(model, grid_points)
# print(torch.abs(grid_residuals).max())

# format should be [[lb_1, ..., lb_n], [ub_1, ..., ub_n]]
domain_bounds = torch.tensor([[0, -1], [1, 1]])

# layers = layers[:6] + layers[-1:]

if not debug:
    verifier = BurgersVerifier(layers)

    domain_bounds = torch.tensor([[0.5000, 0.0000], [1.0000, 1.0000]])

    import time

    start_time = time.time()
    f_lb, f_ub = verifier.compute_residual_bound(domain_bounds)
    elapsed_time = time.time() - start_time
    print(f"Computation took {elapsed_time:2f} seconds")

    import pdb
    pdb.set_trace()

    # from pinn_verifier import PINNSolution

    # u_theta = PINNSolution(layers, activation_relaxation=SoftplusRelaxation(ActivationRelaxationType.MULTI_LINE))
    # # u_theta = PINNSolution(layers, activation_relaxation=SoftplusRelaxation(ActivationRelaxationType.PIECEWISE_LINEAR, multi_line_biases=[0.5]))
    # u_theta.domain_bounds = domain_bounds
    # u_theta.compute_bounds()

    # print("full lb: ", u_theta.lower_bounds[-1][0])
    # print("full ub: ", u_theta.upper_bounds[-1][0])

    # ----- uniform input branching -----
    # pieces_lb = []
    # pieces_ub = []

    # i = 1
    # n_pieces = 10
    # t_domains = np.linspace(0, 1, n_pieces+1)
    # x_domains = np.linspace(-1, 1, n_pieces+1)
    # for t_min, t_max in zip(t_domains[:-1], t_domains[1:]):
    #     for x_min, x_max in zip(x_domains[:-1], x_domains[1:]):
    #         new_u_theta = PINNSolution(layers, activation_relaxation=SoftplusRelaxation(ActivationRelaxationType.MULTI_LINE))
    #         new_u_theta.domain_bounds = torch.tensor([[t_min, x_min], [t_max, x_max]])
    #         new_u_theta.compute_bounds(debug=False)
    #         piece_lb, piece_ub = new_u_theta.lower_bounds[-1][0], new_u_theta.upper_bounds[-1][0]

    #         ts = torch.linspace(t_min, t_max, 250)
    #         xs = torch.linspace(x_min, x_max, 250)
    #         grid_ts, grid_xs = torch.meshgrid(ts, xs, indexing='ij')
    #         grid_points = torch.dstack([grid_ts, grid_xs]).reshape(-1, 2)

    #         model_pts = model(grid_points)
    #         emp_min, emp_max = model_pts.min(), model_pts.max()

    #         print(f"---- {i} / {n_pieces**2} ----")
    #         print(f"lb: {piece_lb}\t empirical_min: {emp_min}")
    #         print(f"ub: {piece_ub}\t empirical_max: {emp_max}")

    #         if emp_min < piece_lb or emp_max > piece_ub:
    #             import pdb
    #             pdb.set_trace()

    #         pieces_lb.append(piece_lb)
    #         pieces_ub.append(piece_ub)

    #         i += 1

    # print("pieces lb: ", min(pieces_lb))
    # print("pieces ub: ", max(pieces_ub))

    # ----- greedy input branching -----

    import time

    pieces_saved = []

    ts = torch.linspace(domain_bounds[0, 0], domain_bounds[1, 0], 1000)
    xs = torch.linspace(domain_bounds[0, 1], domain_bounds[1, 1], 1000)
    grid_ts, grid_xs = torch.meshgrid(ts, xs, indexing='ij')
    grid_points = torch.dstack([grid_ts, grid_xs]).reshape(-1, 2)

    model_pts = model_get_residual(model, grid_points)
    all_min, all_max = model_pts.min(), model_pts.max()

    max_compute = 100

    if args.greedy_input_pieces is None:
        pieces = []
        i = 1
        current_parent_piece = domain_bounds
    else:
        with open(args.greedy_input_pieces, "r") as fp:
            pieces_saved = json.load(fp)
        
        last_piece = pieces_saved[-1]
        pieces = last_piece["pieces"]
        i = last_piece["i"]

        # select the next parent piece
        current_parent_piece = max(pieces, key=lambda piece: piece[0])
        current_parent_piece_idx = pieces.index(current_parent_piece)
        pieces.pop(current_parent_piece_idx)
        current_parent_piece = np.array(current_parent_piece[-1])

    start_time = time.time()

    while True:
        t_domains = np.linspace(current_parent_piece[0, 0], current_parent_piece[1, 0], 3)
        x_domains = np.linspace(current_parent_piece[0, 1], current_parent_piece[1, 1], 3)
        for t_min, t_max in zip(t_domains[:-1], t_domains[1:]):
            for x_min, x_max in zip(x_domains[:-1], x_domains[1:]):
                current_piece_domain = torch.tensor([[t_min, x_min], [t_max, x_max]])

                print(f"---- {i} / {max_compute} ----")
                print(f"{current_piece_domain}")

                # new_u_theta = PINNSolution(layers, activation_relaxation=SoftplusRelaxation(ActivationRelaxationType.MULTI_LINE))
                # new_u_theta.domain_bounds = current_piece_domain
                # new_u_theta.compute_bounds(debug=False)
                # piece_lb, piece_ub = new_u_theta.lower_bounds[-1][0], new_u_theta.upper_bounds[-1][0]
                verifier = BurgersVerifier(layers)
                piece_lb, piece_ub = verifier.compute_residual_bound(current_piece_domain, debug=True)

                ts = torch.linspace(t_min, t_max, 250)
                xs = torch.linspace(x_min, x_max, 250)
                grid_ts, grid_xs = torch.meshgrid(ts, xs, indexing='ij')
                grid_points = torch.dstack([grid_ts, grid_xs]).reshape(-1, 2)

                model_pts = model_get_residual(model, grid_points)
                emp_min, emp_max = model_pts.min(), model_pts.max()
            
                print(f"lb: {piece_lb}\t empirical_min: {emp_min}")
                print(f"ub: {piece_ub}\t empirical_max: {emp_max}")

                delta_min = emp_min - piece_lb
                delta_max = piece_ub - emp_max
                if delta_min < 0 or delta_max < 0:
                    import pdb
                    pdb.set_trace()

                pieces.append([
                    max(all_min - piece_lb, piece_ub - all_max).item(),
                    piece_lb,
                    piece_ub,
                    {
                        "u_theta_bounds": [verifier.u_theta.lower_bounds[-1][0].item(), verifier.u_theta.upper_bounds[-1][0].item()],
                        "u_dt_theta_bounds": [verifier.u_dt_theta.lower_bounds[-1][0].item(), verifier.u_dt_theta.upper_bounds[-1][0].item()],
                        "u_dx_theta_bounds": [verifier.u_dx_theta.lower_bounds[-1][0].item(), verifier.u_dx_theta.upper_bounds[-1][0].item()],
                        "u_dxdx_theta_bounds": [verifier.u_dxdx_theta.lower_bounds[-1][0].item(), verifier.u_dxdx_theta.upper_bounds[-1][0].item()],
                    },
                    current_piece_domain.detach().numpy().tolist()
                ])
                i += 1

        if i > max_compute:
            break

        elapsed_time = time.time() - start_time
        print(f"Computation so far took {elapsed_time:2f} seconds")

        # compute min and max accross the pieces
        pieces_lbs = [piece[1] for piece in pieces]
        pieces_lb = min(pieces_lbs)

        pieces_ubs = [piece[2] for piece in pieces]
        pieces_ub = max(pieces_ubs)

        pieces_saved.append({
            "i": i-1,
            "n_pieces": len(pieces),
            "lb": pieces_lb,
            "ub": pieces_ub,
            "pieces": copy.deepcopy(pieces)
        })
        json.dump(pieces_saved, open(args.greedy_output_pieces, "w"), indent=2)

        # select the next parent piece
        current_parent_piece = max(pieces, key=lambda piece: piece[0])
        current_parent_piece_idx = pieces.index(current_parent_piece)
        pieces.pop(current_parent_piece_idx)
        current_parent_piece = np.array(current_parent_piece[-1])

    pieces_lbs = [piece[1] for piece in pieces]
    pieces_lb = min(pieces_lbs)

    pieces_ubs = [piece[2] for piece in pieces]
    pieces_ub = max(pieces_ubs)

    print("-------------------------")
    print(f"pieces lb: {pieces_lb}\t empirical min: {all_min}")
    print(f"pieces ub: {pieces_ub}\t empirical max: {all_max}")

    import pdb
    pdb.set_trace()
