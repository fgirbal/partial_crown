"""
Test the activation relaxations are valid lower and upper bound lines of the underlying functions.
"""
from typing import Tuple

import torch

from pinn_verifier.activations import ActivationRelaxation, ActivationRelaxationType
from pinn_verifier.activations.tanh import TanhRelaxation, TanhDerivativeRelaxation, TanhSecondDerivativeRelaxation
from pinn_verifier.activations.sin import SinRelaxation

N_INTERVALS = 1000
N_INTERVALS_LB_UB_IN_INTERVAL = 10000

def activation_relaxation_bounds_included(activation_relaxation: ActivationRelaxation, lb: torch.Tensor, ub: torch.Tensor):
    lb_line, ub_line = activation_relaxation.get_bounds(lb, ub)

    x_vals = torch.linspace(lb.item(), ub.item(), 1000, dtype=torch.float64)
    actual_y_vals = activation_relaxation.evaluate(x_vals)

    ret_val = True
    if isinstance(lb_line[0], Tuple):
        lb_lines, ub_lines = lb_line, ub_line
    else:
        lb_lines, ub_lines = [lb_line], [ub_line]

    for lb_line, ub_line in zip(lb_lines, ub_lines):
        lb_y_vals = lb_line[0] * x_vals + lb_line[1]
        ub_y_vals = ub_line[0] * x_vals + ub_line[1]

        ret_val &= (all(actual_y_vals >= lb_y_vals - 1e-3) and all(actual_y_vals <= ub_y_vals + 1e-3))

        if not ret_val:
            import pdb
            pdb.set_trace()
    
    return ret_val

def test_random_tanh_relaxation():
    activation = TanhRelaxation(ActivationRelaxationType.SINGLE_LINE)

    lbs = torch.FloatTensor(N_INTERVALS).uniform_(-5, 5)
    ubs = lbs + torch.FloatTensor(N_INTERVALS).uniform_(1e-2, 10)

    for i in range(N_INTERVALS):
        assert activation_relaxation_bounds_included(activation, lbs[i], ubs[i])

    return True

def test_random_tanh_derivative_relaxation():
    activation = TanhDerivativeRelaxation(ActivationRelaxationType.MULTI_LINE)

    lbs = torch.FloatTensor(N_INTERVALS).uniform_(-5, 5)
    ubs = lbs + torch.FloatTensor(N_INTERVALS).uniform_(1e-2, 4)

    for i in range(N_INTERVALS):
        assert activation_relaxation_bounds_included(activation, lbs[i], ubs[i])

    return True

def test_random_tanh_second_derivative_relaxation():
    activation = TanhSecondDerivativeRelaxation(ActivationRelaxationType.MULTI_LINE)

    lbs = torch.FloatTensor(N_INTERVALS).uniform_(-5, 5)
    ubs = lbs + torch.FloatTensor(N_INTERVALS).uniform_(1e-2, 10)

    for i in range(N_INTERVALS):
        assert activation_relaxation_bounds_included(activation, lbs[i], ubs[i])

    return True

def test_random_sin_relaxation():
    activation = SinRelaxation(ActivationRelaxationType.SINGLE_LINE)

    lbs = torch.FloatTensor(N_INTERVALS).uniform_(-1, 1-1e-3)
    ubs = torch.clamp(lbs + torch.FloatTensor(N_INTERVALS).uniform_(1e-2, 2), max=1)

    for i in range(N_INTERVALS):
        assert activation_relaxation_bounds_included(activation, lbs[i], ubs[i])

    return True

def lb_ub_in_interval_valid(activation_relaxation: ActivationRelaxation, lb: torch.Tensor, ub: torch.Tensor):
    interval_lb, interval_ub = activation_relaxation.get_lb_ub_in_interval(lb, ub)

    x_vals = torch.linspace(lb.item(), ub.item(), 1000, dtype=torch.float64)
    actual_y_vals = activation_relaxation.evaluate(x_vals)

    ret_val = True
    ret_val = (actual_y_vals.min() >= interval_lb - 1e-5) & (actual_y_vals.max() <= interval_ub + 1e-5)

    if not ret_val:
        import pdb
        pdb.set_trace()
    
    return ret_val

def test_random_tanh_second_derivative_get_lb_ub_in_interval():
    activation = TanhSecondDerivativeRelaxation(ActivationRelaxationType.SINGLE_LINE)

    lbs = torch.FloatTensor(N_INTERVALS_LB_UB_IN_INTERVAL).uniform_(-5, 5)
    ubs = lbs + torch.FloatTensor(N_INTERVALS_LB_UB_IN_INTERVAL).uniform_(1e-2, 10)

    for i in range(N_INTERVALS_LB_UB_IN_INTERVAL):
        assert lb_ub_in_interval_valid(activation, lbs[i], ubs[i])

    return True