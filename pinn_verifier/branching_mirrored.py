"""
Greedy branching strategy as described in the paper.
"""
import os
import copy
import time
import json
from typing import List
from enum import Enum
import itertools
from pathlib import Path

import numpy as np
import torch

class VerbosityLevel(Enum):
    NO_OUTPUT = 0
    NO_INDIVIDUAL_PROGRESS = 1
    ALL_OUTPUT = 2

class MessageType(Enum):
    INDIVIDUAL_PROGRESS = 0
    GENERAL_OUTPUT = 1


def verbose_log(level: VerbosityLevel, message: str, msg_type: MessageType):
    if level is VerbosityLevel.NO_OUTPUT:
        return
    
    if level is VerbosityLevel.ALL_OUTPUT or msg_type is MessageType.GENERAL_OUTPUT:
        print(message)


def mirrored_greedy_input_branching(
        layers: List[torch.nn.Module],
        model: torch.nn.Module,
        domain_bounds: torch.Tensor,
        empirical_evaluation_fn: callable,
        verifier_fn: callable,
        output_filename: str,
        input_filename: str = None,
        maximum_computations: int = 100,
        verbose: VerbosityLevel = VerbosityLevel.ALL_OUTPUT,
        save_frequency: int = 1,
        grid_size: int = 50
):
    """Greedly branch on the input by calling the vertifier and empirical evaluation
    functions; verification will include a mirror evaluation on the x domain.

    Args:
        layers (List[torch.nn.Module]): list of layers of the model
        model (torch.nn.Module): model as a torch.nn.Module
        domain_bounds (torch.Tensor): lower and upper bounds of the domain to verify
        empirical_evaluation_fn (callable): function that takes in a model and a
            grid of points, and returns the function evaluation on those
            points.
        verifier_fn (callable): function that takes in a list of layers and a
            domain, and returns the lower and upper bounds, along with a log
            of other recorded variables/info.
        output_filename (str): file where to write the pieces and bounds
        input_filename (str, optional): file from where to read previously computed
            pieces and bounds. Defaults to None.
        maximum_computations (int, optional): # computations allowed for this problem.
            Defaults to 100.
        verbose (VerbosityLevel, optional): level of output verbosity.
        save_frequency (int, optional): saves output every `save_frequency` branches.
        grid_size (int, optional): size of the grid in each direction; will result in a
            (grid_size)^n grid, where n is the number of input dimensions.
    """

    elapsed_time = 0
    debug_param = (verbose is VerbosityLevel.ALL_OUTPUT)
    pieces_saved = []
    Path(os.path.dirname(output_filename)).mkdir(parents=True, exist_ok=True)

    ts = torch.linspace(domain_bounds[0, 0], domain_bounds[1, 0], 100)
    xs = torch.linspace(domain_bounds[0, 1], domain_bounds[1, 1], 100)
    grid_ts, grid_xs = torch.meshgrid(ts, xs, indexing='ij')
    grid_points_low = torch.dstack([grid_ts, grid_xs]).reshape(-1, 2)

    ts = torch.linspace(domain_bounds[0, 0], domain_bounds[1, 0], 100)
    xs = torch.linspace(-domain_bounds[0, 1], -domain_bounds[1, 1], 100)
    grid_ts, grid_xs = torch.meshgrid(ts, xs, indexing='ij')
    grid_points_high = torch.dstack([grid_ts, grid_xs]).reshape(-1, 2)

    model_pts = empirical_evaluation_fn(model, grid_points_low, grid_points_high)
    all_min, all_max = model_pts.min(), model_pts.max()

    import pdb
    pdb.set_trace()

    if input_filename is None:
        pieces = []
        i = 1
        current_parent_piece = domain_bounds

        s = time.time()
        piece_lb, piece_ub, piece_logs = verifier_fn(layers, current_parent_piece.unsqueeze(0), debug=debug_param)
        elapsed_time += (time.time() - s)

        if type(piece_lb) is torch.Tensor:
            piece_lb = piece_lb.detach().item()

        if type(piece_ub) is torch.Tensor:
            piece_ub = piece_ub.detach().item()

        ts = torch.linspace(current_parent_piece[0, 0], current_parent_piece[1, 0], grid_size)
        xs = torch.linspace(current_parent_piece[0, 1], current_parent_piece[1, 1], grid_size)
        grid_ts, grid_xs = torch.meshgrid(ts, xs, indexing='ij')
        grid_points_low = torch.dstack([grid_ts, grid_xs]).reshape(-1, 2)

        ts = torch.linspace(current_parent_piece[0, 0], current_parent_piece[1, 0], grid_size)
        xs = torch.linspace(-current_parent_piece[0, 1], -current_parent_piece[1, 1], grid_size)
        grid_ts, grid_xs = torch.meshgrid(ts, xs, indexing='ij')
        grid_points_high = torch.dstack([grid_ts, grid_xs]).reshape(-1, 2)

        model_pts = empirical_evaluation_fn(model, grid_points_low, grid_points_high)
        emp_min, emp_max = model_pts.min(), model_pts.max()

        try:
            assert emp_min >= piece_lb
            assert emp_max <= piece_ub
        except:
            import pdb
            pdb.set_trace()

        verbose_log(verbose, f"lb: {piece_lb}\t empirical_min: {emp_min}", MessageType.GENERAL_OUTPUT)
        verbose_log(verbose, f"ub: {piece_ub}\t empirical_max: {emp_max}", MessageType.GENERAL_OUTPUT)

        pieces.append([
            max(all_min - piece_lb, piece_ub - all_max).item(),
            piece_lb,
            piece_ub,
            piece_logs,
            current_parent_piece.detach().numpy().tolist()
        ])

        pieces_saved.append({
            "i": i,
            "n_pieces": len(pieces),
            "lb": piece_lb,
            "ub": piece_ub,
            "pieces": copy.deepcopy(pieces),
            "elapsed_time": elapsed_time,
            "all_min": all_min.item(),
            "all_max": all_max.item()
        })
        json.dump(pieces_saved, open(output_filename, "w"))
    else:
        with open(input_filename, "r") as fp:
            pieces_saved = json.load(fp)

        last_piece = pieces_saved[-1]
        pieces = last_piece["pieces"]
        i = last_piece["i"]
        elapsed_time = last_piece["elapsed_time"]

    # select the next parent piece
    current_parent_piece = max(pieces, key=lambda piece: piece[0])
    current_parent_piece_idx = pieces.index(current_parent_piece)
    current_parent_piece_max_diff = current_parent_piece[0]
    pieces.pop(current_parent_piece_idx)
    current_parent_piece = np.array(current_parent_piece[-1])

    last_save = 0
    while True:
        verbose_log(verbose, f"Breaking down piece with maximum difference of {current_parent_piece_max_diff}", MessageType.GENERAL_OUTPUT)

        if current_parent_piece[0, 0] != current_parent_piece[1, 0]:
            t_domains = np.linspace(current_parent_piece[0, 0], current_parent_piece[1, 0], 3)
            t_intervals = [(t_min, t_max) for t_min, t_max in zip(t_domains[:-1], t_domains[1:])]
        else:
            t_intervals = [(current_parent_piece[0, 0].item(), current_parent_piece[1, 0].item())]

        if current_parent_piece[0, 1] != current_parent_piece[1, 1]:
            x_domains = np.linspace(current_parent_piece[0, 1], current_parent_piece[1, 1], 3)
            x_intervals = [(x_min, x_max) for x_min, x_max in zip(x_domains[:-1], x_domains[1:])]
        else:
            x_intervals = [(current_parent_piece[0, 1].item(), current_parent_piece[1, 1].item())]

        branches = torch.Tensor(list(itertools.product(t_intervals, x_intervals))).transpose(1, 2)

        s = time.time()
        pieces_lb, pieces_ub, pieces_logs = verifier_fn(layers, branches, debug=debug_param)
        elapsed_time += (time.time() - s)

        for k, subdomain in enumerate(branches):
            t_min, x_min = subdomain[0]
            t_max, x_max = subdomain[1]
            piece_lb = pieces_lb[k]
            piece_ub = pieces_ub[k]
            piece_logs = pieces_logs[k]

            if type(piece_lb) is torch.Tensor:
                piece_lb = piece_lb.detach().item()

            if type(piece_ub) is torch.Tensor:
                piece_ub = piece_ub.detach().item()

            verbose_log(verbose, f"---- {i+k} / {maximum_computations} ----", MessageType.GENERAL_OUTPUT)
            verbose_log(verbose, f"{subdomain}", MessageType.INDIVIDUAL_PROGRESS)

            ts = torch.linspace(t_min, t_max, grid_size)
            xs = torch.linspace(x_min, x_max, grid_size)
            grid_ts, grid_xs = torch.meshgrid(ts, xs, indexing='ij')
            grid_points_low = torch.dstack([grid_ts, grid_xs]).reshape(-1, 2)

            ts = torch.linspace(t_min, t_max, grid_size)
            xs = torch.linspace(-x_min, -x_max, grid_size)
            grid_ts, grid_xs = torch.meshgrid(ts, xs, indexing='ij')
            grid_points_high = torch.dstack([grid_ts, grid_xs]).reshape(-1, 2)

            model_pts = empirical_evaluation_fn(model, grid_points_low, grid_points_high)
            emp_min, emp_max = model_pts.min(), model_pts.max()

            try:
                assert emp_min >= piece_lb
                assert emp_max <= piece_ub
            except:
                print("----- Error!! Empirical value outside bounds -----")

                import pdb
                pdb.set_trace()

            verbose_log(verbose, f"lb: {piece_lb}\t empirical_min: {emp_min}", MessageType.GENERAL_OUTPUT)
            verbose_log(verbose, f"ub: {piece_ub}\t empirical_max: {emp_max}", MessageType.GENERAL_OUTPUT)

            pieces.append([
                max(all_min - piece_lb, piece_ub - all_max).item(),
                piece_lb,
                piece_ub,
                piece_logs,
                subdomain.detach().numpy().tolist()
            ])

        i += pieces_lb.shape[0]
        if i > maximum_computations:
            break

        verbose_log(verbose, f"Total computation so far took {elapsed_time:2f} seconds", MessageType.INDIVIDUAL_PROGRESS)

        # compute min and max accross the pieces
        pieces_lbs = [piece[1] for piece in pieces]
        pieces_lb = min(pieces_lbs)

        pieces_ubs = [piece[2] for piece in pieces]
        pieces_ub = max(pieces_ubs)

        last_save += 1
        if last_save % save_frequency == 0:
            verbose_log(verbose, "Saving to file...", MessageType.GENERAL_OUTPUT)

            pieces_saved.append({
                "i": i-1,
                "n_pieces": len(pieces),
                "lb": pieces_lb,
                "ub": pieces_ub,
                "pieces": copy.deepcopy(pieces),
                "elapsed_time": elapsed_time
            })
            json.dump(pieces_saved, open(output_filename, "w"))

            verbose_log(verbose, f"Total computation so far took {elapsed_time:2f} seconds", MessageType.GENERAL_OUTPUT)

        # select the next parent piece
        current_parent_piece = max(pieces, key=lambda piece: piece[0])
        current_parent_piece_idx = pieces.index(current_parent_piece)
        pieces.pop(current_parent_piece_idx)
        current_parent_piece_max_diff = current_parent_piece[0]
        current_parent_piece = np.array(current_parent_piece[-1])

    pieces_lbs = [piece[1] for piece in pieces]
    pieces_lb = min(pieces_lbs)

    pieces_ubs = [piece[2] for piece in pieces]
    pieces_ub = max(pieces_ubs)

    pieces_saved.append({
        "i": i-1,
        "n_pieces": len(pieces),
        "lb": pieces_lb,
        "ub": pieces_ub,
        "pieces": copy.deepcopy(pieces),
        "elapsed_time": elapsed_time
    })
    json.dump(pieces_saved, open(output_filename, "w"))

    verbose_log(verbose, "-------------------------", MessageType.GENERAL_OUTPUT)
    verbose_log(verbose, "Full results:", MessageType.GENERAL_OUTPUT)
    verbose_log(verbose, f"Total computation time: {elapsed_time:2f} seconds", MessageType.GENERAL_OUTPUT)
    verbose_log(verbose, f"pieces lb: {pieces_lb}\t empirical min: {all_min}", MessageType.GENERAL_OUTPUT)
    verbose_log(verbose, f"pieces ub: {pieces_ub}\t empirical max: {all_max}", MessageType.GENERAL_OUTPUT)

    import pdb
    pdb.set_trace()

def mirrored_greedy_input_branching_allen_cahn(
        layers: List[torch.nn.Module],
        model: torch.nn.Module,
        domain_bounds: torch.Tensor,
        empirical_evaluation_fn: callable,
        verifier_fn: callable,
        output_filename: str,
        input_filename: str = None,
        maximum_computations: int = 100,
        verbose: VerbosityLevel = VerbosityLevel.ALL_OUTPUT,
        save_frequency: int = 1,
        grid_size: int = 50
):
    """Greedly branch on the input by calling the vertifier and empirical evaluation
    functions; verification will include a mirror evaluation on the x domain.

    Args:
        layers (List[torch.nn.Module]): list of layers of the model
        model (torch.nn.Module): model as a torch.nn.Module
        domain_bounds (torch.Tensor): lower and upper bounds of the domain to verify
        empirical_evaluation_fn (callable): function that takes in a model and a
            grid of points, and returns the function evaluation on those
            points.
        verifier_fn (callable): function that takes in a list of layers and a
            domain, and returns the lower and upper bounds, along with a log
            of other recorded variables/info.
        output_filename (str): file where to write the pieces and bounds
        input_filename (str, optional): file from where to read previously computed
            pieces and bounds. Defaults to None.
        maximum_computations (int, optional): # computations allowed for this problem.
            Defaults to 100.
        verbose (VerbosityLevel, optional): level of output verbosity.
        save_frequency (int, optional): saves output every `save_frequency` branches.
        grid_size (int, optional): size of the grid in each direction; will result in a
            (grid_size)^n grid, where n is the number of input dimensions.
    """

    elapsed_time = 0
    debug_param = (verbose is VerbosityLevel.ALL_OUTPUT)
    pieces_saved = []
    Path(os.path.dirname(output_filename)).mkdir(parents=True, exist_ok=True)

    ts = torch.linspace(domain_bounds[0, 0], domain_bounds[1, 0], 100)
    xs = torch.linspace(domain_bounds[0, 1], domain_bounds[1, 1], 100)
    grid_ts, grid_xs = torch.meshgrid(ts, xs, indexing='ij')
    grid_points_low = torch.dstack([grid_ts, grid_xs]).reshape(-1, 2)

    ts = torch.linspace(-domain_bounds[0, 0], -domain_bounds[1, 0], 100)
    xs = torch.linspace(domain_bounds[0, 1], domain_bounds[1, 1], 100)
    grid_ts, grid_xs = torch.meshgrid(ts, xs, indexing='ij')
    grid_points_high = torch.dstack([grid_ts, grid_xs]).reshape(-1, 2)

    model_pts = empirical_evaluation_fn(model, grid_points_low, grid_points_high)
    all_min, all_max = model_pts.min(), model_pts.max()

    if input_filename is None:
        pieces = []
        i = 1
        current_parent_piece = domain_bounds

        s = time.time()
        piece_lb, piece_ub, piece_logs = verifier_fn(layers, current_parent_piece.unsqueeze(0), debug=debug_param)
        elapsed_time += (time.time() - s)

        if type(piece_lb) is torch.Tensor:
            piece_lb = piece_lb.detach().item()

        if type(piece_ub) is torch.Tensor:
            piece_ub = piece_ub.detach().item()

        ts = torch.linspace(current_parent_piece[0, 0], current_parent_piece[1, 0], grid_size)
        xs = torch.linspace(current_parent_piece[0, 1], current_parent_piece[1, 1], grid_size)
        grid_ts, grid_xs = torch.meshgrid(ts, xs, indexing='ij')
        grid_points_low = torch.dstack([grid_ts, grid_xs]).reshape(-1, 2)

        ts = torch.linspace(-current_parent_piece[0, 0], -current_parent_piece[1, 0], grid_size)
        xs = torch.linspace(current_parent_piece[0, 1], current_parent_piece[1, 1], grid_size)
        grid_ts, grid_xs = torch.meshgrid(ts, xs, indexing='ij')
        grid_points_high = torch.dstack([grid_ts, grid_xs]).reshape(-1, 2)

        model_pts = empirical_evaluation_fn(model, grid_points_low, grid_points_high)
        emp_min, emp_max = model_pts.min(), model_pts.max()

        try:
            assert emp_min >= piece_lb
            assert emp_max <= piece_ub
        except:
            import pdb
            pdb.set_trace()

        verbose_log(verbose, f"lb: {piece_lb}\t empirical_min: {emp_min}", MessageType.GENERAL_OUTPUT)
        verbose_log(verbose, f"ub: {piece_ub}\t empirical_max: {emp_max}", MessageType.GENERAL_OUTPUT)

        pieces.append([
            max(all_min - piece_lb, piece_ub - all_max).item(),
            piece_lb,
            piece_ub,
            piece_logs,
            current_parent_piece.detach().numpy().tolist()
        ])

        pieces_saved.append({
            "i": i,
            "n_pieces": len(pieces),
            "lb": piece_lb,
            "ub": piece_ub,
            "pieces": copy.deepcopy(pieces),
            "elapsed_time": elapsed_time,
            "all_min": all_min.item(),
            "all_max": all_max.item()
        })
        json.dump(pieces_saved, open(output_filename, "w"))
    else:
        with open(input_filename, "r") as fp:
            pieces_saved = json.load(fp)

        last_piece = pieces_saved[-1]
        pieces = last_piece["pieces"]
        i = last_piece["i"]
        elapsed_time = last_piece["elapsed_time"]

    # select the next parent piece
    current_parent_piece = max(pieces, key=lambda piece: piece[0])
    current_parent_piece_idx = pieces.index(current_parent_piece)
    current_parent_piece_max_diff = current_parent_piece[0]
    pieces.pop(current_parent_piece_idx)
    current_parent_piece = np.array(current_parent_piece[-1])

    last_save = 0
    while True:
        verbose_log(verbose, f"Breaking down piece with maximum difference of {current_parent_piece_max_diff}", MessageType.GENERAL_OUTPUT)

        if current_parent_piece[0, 0] != current_parent_piece[1, 0]:
            t_domains = np.linspace(current_parent_piece[0, 0], current_parent_piece[1, 0], 3)
            t_intervals = [(t_min, t_max) for t_min, t_max in zip(t_domains[:-1], t_domains[1:])]
        else:
            t_intervals = [(current_parent_piece[0, 0].item(), current_parent_piece[1, 0].item())]

        if current_parent_piece[0, 1] != current_parent_piece[1, 1]:
            x_domains = np.linspace(current_parent_piece[0, 1], current_parent_piece[1, 1], 3)
            x_intervals = [(x_min, x_max) for x_min, x_max in zip(x_domains[:-1], x_domains[1:])]
        else:
            x_intervals = [(current_parent_piece[0, 1].item(), current_parent_piece[1, 1].item())]

        branches = torch.Tensor(list(itertools.product(t_intervals, x_intervals))).transpose(1, 2)

        s = time.time()
        pieces_lb, pieces_ub, pieces_logs = verifier_fn(layers, branches, debug=debug_param)
        elapsed_time += (time.time() - s)

        for k, subdomain in enumerate(branches):
            t_min, x_min = subdomain[0]
            t_max, x_max = subdomain[1]
            piece_lb = pieces_lb[k]
            piece_ub = pieces_ub[k]
            piece_logs = pieces_logs[k]

            if type(piece_lb) is torch.Tensor:
                piece_lb = piece_lb.detach().item()

            if type(piece_ub) is torch.Tensor:
                piece_ub = piece_ub.detach().item()

            verbose_log(verbose, f"---- {i+k} / {maximum_computations} ----", MessageType.GENERAL_OUTPUT)
            verbose_log(verbose, f"{subdomain}", MessageType.INDIVIDUAL_PROGRESS)

            ts = torch.linspace(t_min, t_max, grid_size)
            xs = torch.linspace(x_min, x_max, grid_size)
            grid_ts, grid_xs = torch.meshgrid(ts, xs, indexing='ij')
            grid_points_low = torch.dstack([grid_ts, grid_xs]).reshape(-1, 2)

            ts = torch.linspace(-t_min, -t_max, grid_size)
            xs = torch.linspace(x_min, x_max, grid_size)
            grid_ts, grid_xs = torch.meshgrid(ts, xs, indexing='ij')
            grid_points_high = torch.dstack([grid_ts, grid_xs]).reshape(-1, 2)

            model_pts = empirical_evaluation_fn(model, grid_points_low, grid_points_high)
            emp_min, emp_max = model_pts.min(), model_pts.max()

            try:
                assert emp_min >= piece_lb
                assert emp_max <= piece_ub
            except:
                print("----- Error!! Empirical value outside bounds -----")

                import pdb
                pdb.set_trace()

            verbose_log(verbose, f"lb: {piece_lb}\t empirical_min: {emp_min}", MessageType.GENERAL_OUTPUT)
            verbose_log(verbose, f"ub: {piece_ub}\t empirical_max: {emp_max}", MessageType.GENERAL_OUTPUT)

            pieces.append([
                max(all_min - piece_lb, piece_ub - all_max).item(),
                piece_lb,
                piece_ub,
                piece_logs,
                subdomain.detach().numpy().tolist()
            ])

        i += pieces_lb.shape[0]
        if i > maximum_computations:
            break

        verbose_log(verbose, f"Total computation so far took {elapsed_time:2f} seconds", MessageType.INDIVIDUAL_PROGRESS)

        # compute min and max accross the pieces
        pieces_lbs = [piece[1] for piece in pieces]
        pieces_lb = min(pieces_lbs)

        pieces_ubs = [piece[2] for piece in pieces]
        pieces_ub = max(pieces_ubs)

        last_save += 1
        if last_save % save_frequency == 0:
            verbose_log(verbose, "Saving to file...", MessageType.GENERAL_OUTPUT)

            pieces_saved.append({
                "i": i-1,
                "n_pieces": len(pieces),
                "lb": pieces_lb,
                "ub": pieces_ub,
                "pieces": copy.deepcopy(pieces),
                "elapsed_time": elapsed_time
            })
            json.dump(pieces_saved, open(output_filename, "w"))

            verbose_log(verbose, f"Total computation so far took {elapsed_time:2f} seconds", MessageType.GENERAL_OUTPUT)

        # select the next parent piece
        current_parent_piece = max(pieces, key=lambda piece: piece[0])
        current_parent_piece_idx = pieces.index(current_parent_piece)
        pieces.pop(current_parent_piece_idx)
        current_parent_piece_max_diff = current_parent_piece[0]
        current_parent_piece = np.array(current_parent_piece[-1])

    pieces_lbs = [piece[1] for piece in pieces]
    pieces_lb = min(pieces_lbs)

    pieces_ubs = [piece[2] for piece in pieces]
    pieces_ub = max(pieces_ubs)

    pieces_saved.append({
        "i": i-1,
        "n_pieces": len(pieces),
        "lb": pieces_lb,
        "ub": pieces_ub,
        "pieces": copy.deepcopy(pieces),
        "elapsed_time": elapsed_time
    })
    json.dump(pieces_saved, open(output_filename, "w"))

    verbose_log(verbose, "-------------------------", MessageType.GENERAL_OUTPUT)
    verbose_log(verbose, "Full results:", MessageType.GENERAL_OUTPUT)
    verbose_log(verbose, f"Total computation time: {elapsed_time:2f} seconds", MessageType.GENERAL_OUTPUT)
    verbose_log(verbose, f"pieces lb: {pieces_lb}\t empirical min: {all_min}", MessageType.GENERAL_OUTPUT)
    verbose_log(verbose, f"pieces ub: {pieces_ub}\t empirical max: {all_max}", MessageType.GENERAL_OUTPUT)

    import pdb
    pdb.set_trace()