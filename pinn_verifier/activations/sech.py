from typing import List, Tuple

import torch
import numpy as np
import gurobipy as grb
from scipy import optimize

from .activation_relaxations import ActivationRelaxation, ActivationRelaxationType, line_type

tanh = torch.nn.Tanh()

# brenth_xtol = 2e-12
# brenth_rtol = 8.881784197001252e-16
brenth_xtol = 1e-8
brenth_rtol = 1e-8

def get_lb_line_lb_in_convex_ub_in_concave(lb: torch.tensor, ub: torch.tensor, split_point: float, np_fn: callable, np_fn_derivative: callable) -> Tuple[line_type, line_type]:
    def fn_derivative_bound_d(x, bound):
        return ((np_fn_derivative(x)) - (np_fn(x) - np_fn(bound)) / (x - bound))

    lb = lb.numpy()
    ub = ub.numpy()

    try:
        d_lb = optimize.root_scalar(lambda d: fn_derivative_bound_d(d, ub), bracket=[lb, split_point], method='brentq', xtol=brenth_xtol, rtol=brenth_rtol).root
    except:
        d_lb = split_point + 1
    
    if d_lb <= split_point:
    # tangent line at point d_lb
        lb_line_m = np_fn_derivative(d_lb)
        lb_line_b = np_fn(ub) - lb_line_m * ub
    else:
        # lb line attempts to connect upper and lower bound points
        if (np_fn(ub) - np_fn(lb)) / (ub - lb) <= np_fn_derivative(lb):
            lb_line_m = (np_fn(ub) - np_fn(lb)) / (ub - lb)
            lb_line_b = np_fn(ub) - lb_line_m * ub
        else:
            d_1 = (lb + ub) / 2
            lb_line_m = np_fn_derivative(d_1)
            lb_line_b = np_fn(d_1) - lb_line_m * d_1

    return (torch.tensor(lb_line_m), torch.tensor(lb_line_b))

def get_ub_line_lb_in_convex_ub_in_concave(lb: torch.tensor, ub: torch.tensor, split_point: float, np_fn: callable, np_fn_derivative: callable) -> Tuple[line_type, line_type]:
    def fn_derivative_bound_d(x, bound):
        return ((np_fn_derivative(x)) - (np_fn(x) - np_fn(bound)) / (x - bound))

    lb = lb.numpy()
    ub = ub.numpy()

    try:
        d_ub = optimize.root_scalar(lambda d: fn_derivative_bound_d(d, lb), bracket=[split_point, ub], method='brentq', xtol=brenth_xtol, rtol=brenth_rtol).root
    except:
        d_ub = split_point - 1

    if d_ub >= split_point:
        # tangent line at point d_ub
        ub_line_m = np_fn_derivative(d_ub)
        ub_line_b = np_fn(lb) - ub_line_m * lb
    else:
        # ub line just connects upper and lower bound points
        ub_line_m = min((np_fn(ub) - np_fn(lb)) / (ub - lb), np_fn_derivative(ub))
        ub_line_b = np_fn(ub) - ub_line_m * ub
    
    return (torch.tensor(ub_line_m), torch.tensor(ub_line_b))

def get_lines_lb_in_convex_ub_in_concave(lb: torch.tensor, ub: torch.tensor, split_point: float, np_fn: callable, np_fn_derivative: callable) -> Tuple[line_type, line_type]:
    return (
        get_lb_line_lb_in_convex_ub_in_concave(lb, ub, split_point, np_fn, np_fn_derivative),
        get_ub_line_lb_in_convex_ub_in_concave(lb, ub, split_point, np_fn, np_fn_derivative)
    )


def get_lb_line_lb_in_concave_ub_in_convex(lb: torch.tensor, ub: torch.tensor, split_point: float, np_fn: callable, np_fn_derivative: callable) -> Tuple[line_type, line_type]:
    def fn_derivative_bound_d(x, bound):
        return ((np_fn_derivative(x)) - (np_fn(x) - np_fn(bound)) / (x - bound))
    
    lb = lb.numpy()
    ub = ub.numpy()

    try:
        d_lb = optimize.root_scalar(lambda d: fn_derivative_bound_d(d, lb), bracket=[split_point, ub], method='brentq', xtol=brenth_xtol, rtol=brenth_rtol).root
    except:
        d_lb = split_point - 1

    if d_lb >= split_point:
        # tangent line at point d_lb
        lb_line_m = np_fn_derivative(d_lb)
        lb_line_b = np_fn(lb) - lb_line_m * lb
    else:
        # lb line attempts to connect upper and lower bound points
        if (np_fn(ub) - np_fn(lb)) / (ub - lb) <= np_fn_derivative(lb):
            lb_line_m = (np_fn(ub) - np_fn(lb)) / (ub - lb)
            lb_line_b = np_fn(ub) - lb_line_m * ub
        else:
            d_1 = (lb + ub) / 2
            lb_line_m = np_fn_derivative(d_1)
            lb_line_b = np_fn(d_1) - lb_line_m * d_1
    
    return (torch.tensor(lb_line_m), torch.tensor(lb_line_b))

def get_ub_line_lb_in_concave_ub_in_convex(lb: torch.tensor, ub: torch.tensor, split_point: float, np_fn: callable, np_fn_derivative: callable) -> Tuple[line_type, line_type]:
    def fn_derivative_bound_d(x, bound):
        return ((np_fn_derivative(x)) - (np_fn(x) - np_fn(bound)) / (x - bound))
    
    lb = lb.numpy()
    ub = ub.numpy()

    try:
        d_ub = optimize.root_scalar(lambda d: fn_derivative_bound_d(d, ub), bracket=[lb, split_point], method='brentq', xtol=brenth_xtol, rtol=brenth_rtol).root
    except:
        d_ub = split_point + 1

    if d_ub <= split_point:
        # tangent line at point d_ub
        ub_line_m = np_fn_derivative(d_ub)
        ub_line_b = np_fn(ub) - ub_line_m * ub
    else:
        # ub line just connects upper and lower bound points
        ub_line_m = (np_fn(ub) - np_fn(lb)) / (ub - lb)
        ub_line_b = np_fn(ub) - ub_line_m * ub

    # try:
    #     d_lb = optimize.root_scalar(lambda d: fn_derivative_bound_d(d, lb), bracket=[split_point, ub], method='brentq', xtol=brenth_xtol, rtol=brenth_rtol).root
    # except:
    #     d_lb = split_point - 1
    
    return (torch.tensor(ub_line_m), torch.tensor(ub_line_b))

def get_lines_lb_in_concave_ub_in_convex(lb: torch.tensor, ub: torch.tensor, split_point: float, np_fn: callable, np_fn_derivative: callable) -> Tuple[line_type, line_type]:
    return (
        get_lb_line_lb_in_concave_ub_in_convex(lb, ub, split_point, np_fn, np_fn_derivative),
        get_ub_line_lb_in_concave_ub_in_convex(lb, ub, split_point, np_fn, np_fn_derivative)
    )



class SechRelaxation(ActivationRelaxation):
    def __init__(self, type: ActivationRelaxationType, single_line_lb_line_bias: float = 0.4, single_line_ub_line_bias: float = 0.5) -> None:
        super().__init__(type)
        self.single_line_lb_line_bias = single_line_lb_line_bias
        self.single_line_ub_line_bias = single_line_ub_line_bias

        result = optimize.minimize(lambda x: -self.np_sech_derivative(x), x0=-1.5, bounds=[(-5, 0)])
        x_val = result.x.item()

        self.first_convex_region = [-np.Inf, x_val]
        self.concave_region = [x_val, -x_val]
        self.second_convex_region = [-x_val, np.Inf]

        # the minimum intersection should be the one that touches the inflextion point
        self.b_intersect = self.np_sech(result.x) - self.np_sech_derivative(result.x) * result.x

    @staticmethod
    def evaluate(x: torch.tensor):
        return 2 / torch.cosh(x)

    @staticmethod
    def sech_derivative(x: torch.tensor):
        return - 2 * np.tanh(x) / np.cosh(x)

    @staticmethod
    def in_region(p: float, region: Tuple[float, float]):
        return p >= region[0] and p <= region[1]
    
    @staticmethod
    def np_sech(x: float):
        return 2 / np.cosh(x)

    @staticmethod
    def np_sech_derivative(x: float):        
        return - 2 * np.tanh(x) / np.cosh(x)
    
    def multi_line_relaxation(self, lb: torch.tensor, ub: torch.tensor) -> Tuple[List[line_type], List[line_type]]:
        lb_lines = []
        ub_lines = []

        lb = lb.to(torch.float64)
        ub = ub.to(torch.float64)

        if (self.in_region(lb, self.first_convex_region) and self.in_region(ub, self.first_convex_region)) or (self.in_region(lb, self.second_convex_region) and self.in_region(ub, self.second_convex_region)):
            # in this location, the function is convex, use the same bounds as in softplus case
        
            # tangent line at point d_1
            d_1 = self.single_line_lb_line_bias * ub + (1 - self.single_line_lb_line_bias) * lb
            lb_m = self.sech_derivative(d_1)
            lb_b = self.evaluate(d_1) - lb_m * d_1
            lb_lines.append((lb_m, lb_b))

            # ub line just connects upper and lower bound points
            ub_m = (self.evaluate(ub) - self.evaluate(lb)) / (ub - lb)
            ub_b = self.evaluate(ub) - ub_m * ub
            ub_lines.append((ub_m, ub_b))
        elif self.in_region(lb, self.concave_region) and self.in_region(ub, self.concave_region):
            # in this location, the function is concave, use the inverted bounds from softplus case

            # lb line just connects upper and lower bound points
            lb_m = (self.evaluate(ub) - self.evaluate(lb)) / (ub - lb)
            lb_b = self.evaluate(ub) - lb_m * ub
            lb_lines.append((lb_m, lb_b))

            # tangent line at point d_1
            d_1 = self.single_line_ub_line_bias * ub + (1 - self.single_line_ub_line_bias) * lb
            ub_m = self.sech_derivative(d_1)
            ub_b = self.evaluate(d_1) - ub_m * d_1
            ub_lines.append((ub_m, ub_b))
        else:
            # points are in different regions; 
            # are they in the first convex and the concave regions?
            if self.in_region(lb, self.first_convex_region) and self.in_region(ub, self.concave_region):
                lb_line, ub_line = get_lines_lb_in_convex_ub_in_concave(lb, ub, self.concave_region[0], self.np_sech, self.np_sech_derivative)

                lb_lines.append(lb_line)
                ub_lines.append(ub_line)
            elif self.in_region(lb, self.concave_region) and self.in_region(ub, self.second_convex_region):
                lb_line, ub_line = get_lines_lb_in_concave_ub_in_convex(lb, ub, self.concave_region[1], self.np_sech, self.np_sech_derivative)

                lb_lines.append(lb_line)
                ub_lines.append(ub_line)

            # are they in the two convex regions?
            elif self.in_region(lb, self.first_convex_region) and self.in_region(ub, self.second_convex_region):
                # lb should be a single line, no benefit of more than one
                if -lb >= ub:
                    lb_line_left = get_lb_line_lb_in_convex_ub_in_concave(lb, ub, self.concave_region[0], self.np_sech, self.np_sech_derivative)
                    lb_lines.append(lb_line_left)
                else:
                    lb_line_right = get_lb_line_lb_in_concave_ub_in_convex(lb, ub, self.concave_region[1], self.np_sech, self.np_sech_derivative)
                    lb_lines.append(lb_line_right)

                ub_line_left = get_ub_line_lb_in_convex_ub_in_concave(lb, torch.tensor([0]), self.concave_region[0], self.np_sech, self.np_sech_derivative)
                ub_line_right = get_ub_line_lb_in_concave_ub_in_convex(torch.tensor([0]), ub, self.concave_region[1], self.np_sech, self.np_sech_derivative)

                ub_lines.append(ub_line_left)
                ub_lines.append(ub_line_right)

        return lb_lines, ub_lines
    
    def single_line_relaxation(self, lb: torch.tensor, ub: torch.tensor) -> line_type:
        lb_lines, ub_lines = self.multi_line_relaxation(lb, ub)

        lb_line = []
        ub_line = []

        if len(lb_lines) == 1:
            lb_line = lb_lines[0]
        else:
            lb_bias = -lb/(-lb + ub)
            biases = [lb_bias, 1 - lb_bias]

            lb_line.append(sum([bias*m for (m, _), bias in zip(lb_lines, biases)]))
            lb_line.append(sum([bias*b for (_, b), bias in zip(lb_lines, biases)]))

        if len(ub_lines) == 1:
            ub_line = ub_lines[0]
        else:
            lb_bias = -lb/(-lb + ub)
            biases = [lb_bias, 1 - lb_bias]

            ub_line.append(sum([bias*m for (m, _), bias in zip(ub_lines, biases)]))
            ub_line.append(sum([bias*b for (_, b), bias in zip(ub_lines, biases)]))

        return lb_line, ub_line
    
    def get_lb_ub_in_interval(self, lb: torch.tensor, ub: torch.tensor) -> Tuple[torch.tensor, torch.tensor]:
        raise NotImplementedError

        out_lb = min(self.evaluate(lb), self.evaluate(ub))
        if lb == ub:
            return out_lb, out_lb

        out_ub = -optimize.minimize(lambda x: -self.evaluate(torch.tensor(x)).item(), bounds=[(lb, ub)], x0=(lb+ub)/2).fun + 1e-3

        return out_lb, out_ub
