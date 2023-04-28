from typing import List, Tuple

import torch
import numpy as np
import gurobipy as grb
from scipy import optimize

from .activation_relaxations import ActivationRelaxation, ActivationRelaxationType, line_type

tanh = torch.nn.Tanh()

relaxation_tolerance = 1e-7
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


class TanhRelaxation(ActivationRelaxation):
    def __init__(self, type: ActivationRelaxationType, single_line_lb_line_bias: float = 0.9, single_line_ub_line_bias: float = 0.2) -> None:
        super().__init__(type)
        self.single_line_lb_line_bias = single_line_lb_line_bias
        self.single_line_ub_line_bias = single_line_ub_line_bias

    @staticmethod
    def evaluate(x: torch.tensor):
        return tanh(x)

    @staticmethod
    def tanh_derivative(x: torch.tensor):
        return 1 - tanh(x)**2

    def single_line_relaxation(self, lb: torch.tensor, ub: torch.tensor) -> line_type:
        def np_tanh(x):
            return np.tanh(x)

        def tanh_bound_d(x, bound):
            return (1 - np_tanh(x)**2) - (np_tanh(x) - np_tanh(bound)) / (x - bound)

        if lb < 0 and ub < 0:
            # in this location, the function is convex, use the same bounds as in softplus case
        
            # tangent line at point d_1
            d_1 = self.single_line_lb_line_bias * ub + (1 - self.single_line_lb_line_bias) * lb
            lb_m = self.tanh_derivative(d_1)
            lb_b = tanh(d_1) - lb_m * d_1

            # ub line just connects upper and lower bound points
            ub_m = (tanh(ub) - tanh(lb)) / (ub - lb)
            ub_b = tanh(ub) - ub_m * ub
        elif lb > 0 and ub > 0:
            # in this location, the function is concave, use the inverted bounds from softplus case

            # lb line just connects upper and lower bound points
            lb_m = (tanh(ub) - tanh(lb)) / (ub - lb)
            lb_b = tanh(ub) - lb_m * ub

            # tangent line at point d_1
            d_1 = self.single_line_ub_line_bias * ub + (1 - self.single_line_ub_line_bias) * lb
            ub_m = self.tanh_derivative(d_1)
            ub_b = tanh(d_1) - ub_m * d_1
        else:
            try:
                d_ub = optimize.root_scalar(lambda d: tanh_bound_d(d, lb), bracket=[0, ub], method='brentq', xtol=brenth_xtol, rtol=brenth_rtol).root
            except:
                d_ub = -1

            try:
                d_lb = optimize.root_scalar(lambda d: tanh_bound_d(d, ub), bracket=[lb, 0], method='brentq', xtol=brenth_xtol, rtol=brenth_rtol).root
            except:
                d_lb = 1

            d_ub, d_lb = torch.tensor(d_ub), torch.tensor(d_lb)

            if d_lb <= 0.0:
                # tangent line at point d_lb
                lb_m = self.tanh_derivative(d_lb)
                lb_b = tanh(ub) - lb_m * ub
            else:
                # lb line just connects upper and lower bound points
                lb_m = (tanh(ub) - tanh(lb)) / (ub - lb)
                lb_b = tanh(ub) - lb_m * ub

            if d_ub >= 0:
                # tangent line at point d_ub
                ub_m = self.tanh_derivative(d_ub)
                ub_b = tanh(lb) - ub_m * lb
            else:
                # ub line just connects upper and lower bound points
                ub_m = (tanh(ub) - tanh(lb)) / (ub - lb)
                ub_b = tanh(ub) - ub_m * ub

        # lb_b -= relaxation_tolerance
        # ub_b += relaxation_tolerance

        return (lb_m, lb_b), (ub_m, ub_b)
    
    def get_lb_ub_in_interval(self, lb: torch.tensor, ub: torch.tensor) -> Tuple[torch.tensor, torch.tensor]:
        return self.evaluate(lb), self.evaluate(ub)


class TanhDerivativeRelaxation(ActivationRelaxation):
    def __init__(self, type: ActivationRelaxationType, single_line_lb_line_bias: float = 0.9, single_line_ub_line_bias: float = 0.2) -> None:
        super().__init__(type)
        self.single_line_lb_line_bias = single_line_lb_line_bias
        self.single_line_ub_line_bias = single_line_ub_line_bias

        result = optimize.minimize(lambda x: -self.np_tanh_second_derivative(x), x0=-1.5, bounds=[(-5, 0)])
        x_val = result.x.item()

        self.first_convex_region = [-np.Inf, x_val]
        self.concave_region = [x_val, -x_val]
        self.second_convex_region = [-x_val, np.Inf]

        # the minimum intersection should be the one that touches the inflextion point
        self.b_intersect = self.np_tanh_derivative(result.x) - self.np_tanh_second_derivative(result.x) * result.x

    @staticmethod
    def evaluate(x: torch.tensor):
        return 1 - tanh(x)**2

    @staticmethod
    def tanh_second_derivative(x: torch.tensor):
        return -2 * tanh(x) * (1 - tanh(x)**2)

    @staticmethod
    def in_region(p: float, region: Tuple[float, float]):
        return p >= region[0] and p <= region[1]
    
    @staticmethod
    def np_tanh_derivative(x: float):
        return 1 - np.tanh(x)**2

    @staticmethod
    def np_tanh_second_derivative(x: float):        
        return -2 * np.tanh(x) * (1 - np.tanh(x)**2)
    
    def multi_line_relaxation(self, lb: torch.tensor, ub: torch.tensor) -> Tuple[List[line_type], List[line_type]]:
        lb_lines = []
        ub_lines = []

        lb = lb.to(torch.float64)
        ub = ub.to(torch.float64)

        if (self.in_region(lb, self.first_convex_region) and self.in_region(ub, self.first_convex_region)) or (self.in_region(lb, self.second_convex_region) and self.in_region(ub, self.second_convex_region)):
            # in this location, the function is convex, use the same bounds as in softplus case
        
            # tangent line at point d_1
            d_1 = self.single_line_lb_line_bias * ub + (1 - self.single_line_lb_line_bias) * lb
            if d_1 == 0:
                d_1 = (self.single_line_ub_line_bias + 1e-5) * ub + (1 - self.single_line_ub_line_bias + 1e-5) * lb
            
            lb_m = self.tanh_second_derivative(d_1)
            lb_b = self.evaluate(d_1) - lb_m * d_1
            lb_lines.append((lb_m, lb_b))

            # ub line just connects upper and lower bound points
            ub_m = (self.evaluate(ub) - self.evaluate(lb)) / (ub - lb)
            if lb_m == 0:
                # loosen the bound a bit
                lb_m = (self.evaluate(ub + 1e-5) - self.evaluate(lb)) / (ub + 1e-5 - lb)
    
            ub_b = self.evaluate(ub) - ub_m * ub
            ub_lines.append((ub_m, ub_b))
        elif self.in_region(lb, self.concave_region) and self.in_region(ub, self.concave_region):
            # in this location, the function is concave, use the inverted bounds from softplus case

            # lb line just connects upper and lower bound points
            lb_m = (self.evaluate(ub) - self.evaluate(lb)) / (ub - lb)
            if lb_m == 0:
                # loosen the bound a bit
                lb_m = (self.evaluate(ub + 1e-5) - self.evaluate(lb)) / (ub + 1e-5 - lb)

            lb_b = self.evaluate(ub) - lb_m * ub
            lb_lines.append((lb_m, lb_b))

            # tangent line at point d_1
            d_1 = self.single_line_ub_line_bias * ub + (1 - self.single_line_ub_line_bias) * lb
            if d_1 == 0:
                d_1 = (self.single_line_ub_line_bias + 1e-5) * ub + (1 - self.single_line_ub_line_bias + 1e-5) * lb
            
            ub_m = self.tanh_second_derivative(d_1)
            ub_b = self.evaluate(d_1) - ub_m * d_1
            ub_lines.append((ub_m, ub_b))
        else:
            # points are in different regions; 
            # are they in the first convex and the concave regions?
            if self.in_region(lb, self.first_convex_region) and self.in_region(ub, self.concave_region):
                lb_line, ub_line = get_lines_lb_in_convex_ub_in_concave(lb, ub, self.concave_region[0], self.np_tanh_derivative, self.np_tanh_second_derivative)

                lb_lines.append(lb_line)
                ub_lines.append(ub_line)
            elif self.in_region(lb, self.concave_region) and self.in_region(ub, self.second_convex_region):
                lb_line, ub_line = get_lines_lb_in_concave_ub_in_convex(lb, ub, self.concave_region[1], self.np_tanh_derivative, self.np_tanh_second_derivative)

                lb_lines.append(lb_line)
                ub_lines.append(ub_line)

            # are they in the two convex regions?
            elif self.in_region(lb, self.first_convex_region) and self.in_region(ub, self.second_convex_region):
                # lb should be a single line, no benefit of more than one
                if -lb >= ub:
                    lb_line_left = get_lb_line_lb_in_convex_ub_in_concave(lb, ub, self.concave_region[0], self.np_tanh_derivative, self.np_tanh_second_derivative)
                    lb_lines.append(lb_line_left)
                else:
                    lb_line_right = get_lb_line_lb_in_concave_ub_in_convex(lb, ub, self.concave_region[1], self.np_tanh_derivative, self.np_tanh_second_derivative)
                    lb_lines.append(lb_line_right)

                ub_line_left = get_ub_line_lb_in_convex_ub_in_concave(lb, torch.tensor([0]), self.concave_region[0], self.np_tanh_derivative, self.np_tanh_second_derivative)
                ub_line_right = get_ub_line_lb_in_concave_ub_in_convex(torch.tensor([0]), ub, self.concave_region[1], self.np_tanh_derivative, self.np_tanh_second_derivative)

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

        # lb_line[1] -= relaxation_tolerance
        # ub_line[1] += relaxation_tolerance

        return lb_line, ub_line
    
    def get_lb_ub_in_interval(self, lb: torch.tensor, ub: torch.tensor) -> Tuple[torch.tensor, torch.tensor]:
        out_lb = min(self.evaluate(lb), self.evaluate(ub))
        if lb == ub:
            return out_lb, out_lb

        out_ub = -optimize.minimize(lambda x: -self.evaluate(torch.tensor(x)).item(), bounds=[(lb, ub)], x0=(lb+ub)/2).fun + 1e-3

        return out_lb, out_ub


class TanhSecondDerivativeRelaxation(ActivationRelaxation):
    def __init__(self, type: ActivationRelaxationType, single_line_lb_line_bias: float = 0.9, single_line_ub_line_bias: float = 0.2) -> None:
        super().__init__(type)
        self.single_line_lb_line_bias = single_line_lb_line_bias
        self.single_line_ub_line_bias = single_line_ub_line_bias

        result = optimize.minimize(lambda x: -self.np_tanh_third_derivative(x), x0=-1.146, bounds=[(-5, -1)])
        x_val = result.x.item()
        self.first_convex_region = [-np.Inf, x_val]
        self.first_concave_region = [x_val, 0]

        # the minimum intersection should be the one that touches the inflextion point
        b_intersect = self.np_tanh_second_derivative(x_val) - self.np_tanh_third_derivative(x_val) * x_val
        result = optimize.minimize(lambda x: -self.np_tanh_second_derivative(x), x0=-0.66, bounds=[(-5, 0)])
        self.first_mound_lhs = self.np_tanh_third_derivative(x_val) * result.x + b_intersect
        self.first_mound_rhs = self.np_tanh_third_derivative(0) * result.x
        self.first_mound_x = result.x.item()

        result = optimize.minimize(lambda x: -self.np_tanh_third_derivative(x), x0=1.146, bounds=[(1, 5)])
        x_val = result.x.item()
        self.second_convex_region = [0, x_val]
        self.second_concave_region = [x_val, np.Inf]

        b_intersect = self.np_tanh_second_derivative(x_val) - self.np_tanh_third_derivative(x_val) * x_val
        result = optimize.minimize(lambda x: -self.np_tanh_second_derivative(x), x0=0.66, bounds=[(0, 5)])
        self.second_mound_lhs = -self.first_mound_rhs
        self.second_mound_rhs = -self.first_mound_lhs
        self.second_mound_x = -self.first_mound_x

    @staticmethod
    def evaluate(x: torch.tensor):
        return -2 * tanh(x) * (1 - tanh(x)**2)

    @staticmethod
    def evaluate_np(x: np.array):
        return -2 * np.tanh(x) * (1 - np.tanh(x)**2)

    @staticmethod
    def tanh_derivative(x: torch.tensor):
        return 1 - tanh(x)**2

    @staticmethod
    def in_region(p: torch.tensor, region: Tuple[float, float]):
        return p >= region[0] and p <= region[1]

    @staticmethod
    def np_tanh_derivative(x: float):
        return 1 - np.tanh(x)**2

    @staticmethod
    def np_tanh_second_derivative(x: float):
        return -2* np.tanh(x) * (1 - np.tanh(x)**2)

    @staticmethod
    def np_tanh_third_derivative(x: float):
        return -2 + 8 * np.tanh(x)**2 - 6 * np.tanh(x)**4

    def multi_line_relaxation(self, lb: torch.tensor, ub: torch.tensor) -> Tuple[List[line_type], List[line_type]]:
        lb_lines = []
        ub_lines = []

        if (self.in_region(lb, self.first_convex_region) and self.in_region(ub, self.first_convex_region)) or (self.in_region(lb, self.second_convex_region) and self.in_region(ub, self.second_convex_region)):
            # in this location, the function is convex, use the same bounds as in softplus case
        
            # tangent line at point d_1
            d_1 = self.single_line_lb_line_bias * ub + (1 - self.single_line_lb_line_bias) * lb
            lb_m = self.np_tanh_third_derivative(d_1)
            lb_b = self.evaluate(d_1) - lb_m * d_1
            lb_lines.append((lb_m, lb_b))

            # ub line just connects upper and lower bound points
            ub_m = (self.evaluate(ub) - self.evaluate(lb)) / (ub - lb)
            ub_b = self.evaluate(ub) - ub_m * ub
            ub_lines.append((ub_m, ub_b))
        elif (self.in_region(lb, self.first_concave_region) and self.in_region(ub, self.first_concave_region)) or (self.in_region(lb, self.second_concave_region) and self.in_region(ub, self.second_concave_region)):
            # in this location, the function is concave, use the inverted bounds from softplus case

            # lb line just connects upper and lower bound points
            lb_m = (self.evaluate(ub) - self.evaluate(lb)) / (ub - lb)
            lb_b = self.evaluate(ub) - lb_m * ub
            lb_lines.append((lb_m, lb_b))

            # tangent line at point d_1
            d_1 = self.single_line_ub_line_bias * ub + (1 - self.single_line_ub_line_bias) * lb
            ub_m = self.np_tanh_third_derivative(d_1)
            ub_b = self.evaluate(d_1) - ub_m * d_1
            ub_lines.append((ub_m, ub_b))
        else:
            # points are in different regions; 
            # they are in adjoint regions (3 cases)
            if self.in_region(lb, self.first_convex_region) and self.in_region(ub, self.first_concave_region):
                lb_line, ub_line = get_lines_lb_in_convex_ub_in_concave(lb, ub, self.first_concave_region[0], self.np_tanh_second_derivative, self.np_tanh_third_derivative)

                lb_lines.append(lb_line)
                ub_lines.append(ub_line)
            elif self.in_region(lb, self.second_convex_region) and self.in_region(ub, self.second_concave_region):
                lb_line, ub_line = get_lines_lb_in_convex_ub_in_concave(lb, ub, self.second_concave_region[0], self.np_tanh_second_derivative, self.np_tanh_third_derivative)

                lb_lines.append(lb_line)
                ub_lines.append(ub_line)
            elif self.in_region(lb, self.first_concave_region) and self.in_region(ub, self.second_convex_region):
                lb_line, ub_line = get_lines_lb_in_concave_ub_in_convex(lb, ub, self.second_convex_region[0], self.np_tanh_second_derivative, self.np_tanh_third_derivative)

                lb_lines.append(lb_line)
                ub_lines.append(ub_line)

            # they are in non-adjoint regions (3 cases)
            elif self.in_region(lb, self.first_convex_region) and self.in_region(ub, self.second_convex_region):
                lb_line_right = get_lb_line_lb_in_concave_ub_in_convex(lb, ub, self.first_concave_region[1], self.np_tanh_second_derivative, self.np_tanh_third_derivative)
                lb_lines.append(lb_line_right)

                ub_line_left = get_ub_line_lb_in_convex_ub_in_concave(lb, torch.tensor([self.first_mound_x]), self.first_concave_region[0], self.np_tanh_second_derivative, self.np_tanh_third_derivative)
                ub_line_right = get_ub_line_lb_in_concave_ub_in_convex(torch.tensor([self.first_mound_x]), ub, self.first_concave_region[1], self.np_tanh_second_derivative, self.np_tanh_third_derivative)

                ub_lines.append(ub_line_left)
                ub_lines.append(ub_line_right)

            elif self.in_region(lb, self.first_concave_region) and self.in_region(ub, self.second_concave_region):
                lb_line_left = get_lb_line_lb_in_concave_ub_in_convex(lb, ub, self.second_convex_region[0], self.np_tanh_second_derivative, self.np_tanh_third_derivative)
                lb_line_right = get_lb_line_lb_in_convex_ub_in_concave(torch.tensor([self.second_mound_x]), ub, self.second_convex_region[1], self.np_tanh_second_derivative, self.np_tanh_third_derivative)

                lb_lines.append(lb_line_left)
                lb_lines.append(lb_line_right)

                ub_line = get_ub_line_lb_in_concave_ub_in_convex(lb, ub, self.second_convex_region[0], self.np_tanh_second_derivative, self.np_tanh_third_derivative)
                ub_lines.append(ub_line)
            elif self.in_region(lb, self.first_convex_region) and self.in_region(ub, self.second_concave_region):
                lb_line_left = get_lb_line_lb_in_concave_ub_in_convex(lb, ub, self.second_convex_region[0], self.np_tanh_second_derivative, self.np_tanh_third_derivative)
                lb_line_right = get_lb_line_lb_in_convex_ub_in_concave(torch.tensor([self.second_mound_x]), ub, self.second_convex_region[1], self.np_tanh_second_derivative, self.np_tanh_third_derivative)

                lb_lines.append(lb_line_left)
                lb_lines.append(lb_line_right)
                
                ub_line_left = get_ub_line_lb_in_convex_ub_in_concave(lb, torch.tensor([self.first_mound_x]), self.first_concave_region[0], self.np_tanh_second_derivative, self.np_tanh_third_derivative)
                ub_line_right = get_ub_line_lb_in_concave_ub_in_convex(torch.tensor([self.first_mound_x]), ub, self.first_concave_region[1], self.np_tanh_second_derivative, self.np_tanh_third_derivative)

                ub_lines.append(ub_line_left)
                ub_lines.append(ub_line_right)
            else:
                raise ValueError("should not happen")

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
        
        # lb_line[1] -= relaxation_tolerance
        # ub_line[1] += relaxation_tolerance

        return lb_line, ub_line

    def get_lb_ub_in_interval(self, lb: torch.tensor, ub: torch.tensor) -> Tuple[torch.tensor, torch.tensor]:
        if lb == ub:
            return self.evaluate(lb), self.evaluate(ub)

        abs_min = -0.7699003589195009 - 1e-4
        abs_max = 0.7699003589195009 + 1e-4

        # 10 cases in total
        # 4 cases of being in the same region
        # 2 cases - if in the first convex or second concave region, min and max are just evaluation of function
        if (self.in_region(lb, self.first_convex_region) and self.in_region(ub, self.first_convex_region)) or\
                (self.in_region(lb, self.second_concave_region) and self.in_region(ub, self.second_concave_region)):
            out_lb = self.evaluate(lb)
            out_ub = self.evaluate(ub)
        # 2 cases - if in the first concave region or second convex region, check whether first_mound_x is between them
        elif (self.in_region(lb, self.first_concave_region) and self.in_region(ub, self.first_concave_region)):
            out_lb = min(self.evaluate(lb), self.evaluate(ub))

            # are the points on either side of the first mound?
            if (lb <= self.first_mound_x and self.first_mound_x <= ub):
                out_ub = abs_max
            else:
                out_ub = max(self.evaluate(lb), self.evaluate(ub))
        elif (self.in_region(lb, self.second_convex_region) and self.in_region(ub, self.second_convex_region)):
            out_ub = max(self.evaluate(lb), self.evaluate(ub))
            
            # are the points on either side of the first mound?
            if (lb <= self.second_mound_x and self.second_mound_x <= ub):
                out_lb = abs_min
            else:
                out_lb = min(self.evaluate(lb), self.evaluate(ub))
        
        # 3 cases for adjacent regions
        elif (self.in_region(lb, self.first_convex_region) and self.in_region(ub, self.first_concave_region)):
            out_lb = min(self.evaluate(lb), self.evaluate(ub))

            if (ub <= self.first_mound_x):
                out_ub = self.evaluate(ub)
            else:
                out_ub = abs_max
        elif (self.in_region(lb, self.first_concave_region) and self.in_region(ub, self.second_convex_region)):
            if (lb <= self.first_mound_x):
                out_ub = abs_max
            else:
                out_ub = self.evaluate(lb)

            if (ub >= self.second_mound_x):
                out_lb = abs_min
            else:
                out_lb = self.evaluate(ub)
        elif (self.in_region(lb, self.second_convex_region) and self.in_region(ub, self.second_concave_region)):
            out_ub = max(self.evaluate(lb), self.evaluate(ub))

            if (lb >= self.second_mound_x):
                out_lb = self.evaluate(lb)
            else:
                out_lb = abs_min

        # 2 cases for regions that are separated by another region
        # 2 cases - if a mound is in between lb and ub, take that min/max depending on which it is
        elif (self.in_region(lb, self.first_convex_region) and self.in_region(ub, self.second_convex_region)):
            out_ub = abs_max

            if (ub <= self.second_mound_x):
                out_lb = self.evaluate(ub)
            else:
                out_lb = abs_min
        elif (self.in_region(lb, self.first_concave_region) and self.in_region(ub, self.second_concave_region)):
            out_lb = abs_min

            if (lb >= self.first_mound_x):
                out_ub = self.evaluate(lb)
            else:
                out_ub = abs_max
        

        # 1 case for extremeties - if in the extremeties, it's abs_min and abs_max
        elif (self.in_region(lb, self.first_convex_region) and self.in_region(ub, self.second_concave_region)):
            return abs_min, abs_max
        else:
            import pdb
            pdb.set_trace()

        # if not min_determined:
        # out_lb_ = optimize.minimize(self.evaluate_np, bounds=[(lb, ub)], x0=(lb+ub)/2, tol=1e-8).fun
        
        # def neg_evaluate_np(x):
        #     return -self.evaluate_np(x)

        # # if not max_determined:
        # # out_ub_ = -optimize.minimize(neg_evaluate_np, bounds=[(lb, ub)], x0=ub, tol=1e-8).fun

        # try:
        #     assert np.abs(out_lb_ - out_lb) <= 1e-3
        #     # assert np.abs(out_ub_ - out_ub) <= 1e-3
        # except:
        #     import pdb
        #     pdb.set_trace()

        return out_lb, out_ub
