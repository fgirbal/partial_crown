from typing import List, Tuple

import torch
import numpy as np
import gurobipy as grb
from scipy import optimize

from .activation_relaxations import ActivationRelaxation, ActivationRelaxationType, line_type, plot_upper_lower_bounds

tanh = torch.nn.Tanh()

def get_lines_lb_in_convex_ub_in_concave(lb: torch.tensor, ub: torch.tensor, split_point: torch.tensor, fn: callable, fn_derivative: callable) -> Tuple[line_type, line_type]:
    def fn_derivative_bound_d(x, bound):
        x_torch = torch.Tensor([x])
        return ((fn_derivative(x_torch)) - (fn(x_torch) - fn(bound)) / (x_torch - bound))

    try:
        d_ub = optimize.root_scalar(lambda d: fn_derivative_bound_d(d, lb), bracket=[split_point, ub], method='brentq').root
    except:
        print("here 1")
        d_ub = split_point - 1

    try:
        d_lb = optimize.root_scalar(lambda d: fn_derivative_bound_d(d, ub), bracket=[lb, split_point], method='brentq').root
    except:
        print("here 2")
        d_lb = split_point + 1
    
    d_ub, d_lb = torch.tensor(d_ub), torch.tensor(d_lb)

    if d_lb <= split_point:
    # tangent line at point d_lb
        lb_line_m = fn_derivative(d_lb)
        lb_line_b = fn(ub) - lb_line_m * ub
    else:
        # lb line attempts to connect upper and lower bound points
        if (fn(ub) - fn(lb)) / (ub - lb) <= fn_derivative(lb):
            lb_line_m = (fn(ub) - fn(lb)) / (ub - lb)
            lb_line_b = fn(ub) - lb_line_m * ub
        else:
            d_1 = (lb + ub) / 2
            lb_line_m = fn_derivative(d_1)
            lb_line_b = fn(d_1) - lb_line_m * d_1

    if d_ub >= split_point:
        # tangent line at point d_ub
        ub_line_m = fn_derivative(d_ub)
        ub_line_b = fn(lb) - ub_line_m * lb
    else:
        # ub line just connects upper and lower bound points
        ub_line_m = min((fn(ub) - fn(lb)) / (ub - lb), fn_derivative(ub))
        ub_line_b = fn(ub) - ub_line_m * ub

    return (lb_line_m, lb_line_b), (ub_line_m, ub_line_b)


def get_lines_lb_in_concave_ub_in_convex(lb: torch.tensor, ub: torch.tensor, split_point: torch.tensor, fn: callable, fn_derivative: callable) -> Tuple[line_type, line_type]:
    def fn_derivative_bound_d(x, bound):
        x_torch = torch.Tensor([x])
        return ((fn_derivative(x_torch)) - (fn(x_torch) - fn(bound)) / (x_torch - bound))

    try:
        d_ub = optimize.root_scalar(lambda d: fn_derivative_bound_d(d, ub), bracket=[lb, split_point], method='brentq').root
    except:
        print("here 1")
        d_ub = split_point + 1

    try:
        d_lb = optimize.root_scalar(lambda d: fn_derivative_bound_d(d, lb), bracket=[split_point, ub], method='brentq').root
    except:
        print("here 2")
        d_lb = split_point - 1

    d_ub, d_lb = torch.tensor(d_ub), torch.tensor(d_lb)

    if d_lb >= split_point:
        # tangent line at point d_lb
        lb_line_m = fn_derivative(d_lb)
        lb_line_b = fn(lb) - lb_line_m * lb
    else:
        # lb line attempts to connect upper and lower bound points
        lb_line_m = min((fn(ub) - fn(lb)) / (ub - lb), fn_derivative(lb))
        lb_line_b = fn(lb) - lb_line_m * lb

    if d_ub <= split_point:
        # tangent line at point d_ub
        ub_line_m = fn_derivative(d_ub)
        ub_line_b = fn(ub) - ub_line_m * ub
    else:
        # ub line just connects upper and lower bound points
        ub_line_m = (fn(ub) - fn(lb)) / (ub - lb)
        ub_line_b = fn(ub) - ub_line_m * ub

    return (lb_line_m, lb_line_b), (ub_line_m, ub_line_b)


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

        lb_line = [0, 0]
        ub_line = [0, 0]

        assert lb <= ub

        if lb < 0 and ub < 0:
            # in this location, the function is convex, use the same bounds as in softplus case
        
            # tangent line at point d_1
            d_1 = self.single_line_lb_line_bias * ub + (1 - self.single_line_lb_line_bias) * lb
            lb_line[0] = self.tanh_derivative(d_1)
            lb_line[1] = tanh(d_1) - lb_line[0] * d_1

            # ub line just connects upper and lower bound points
            ub_line[0] = (tanh(ub) - tanh(lb)) / (ub - lb)
            ub_line[1] = tanh(ub) - ub_line[0] * ub
        elif lb > 0 and ub > 0:
            # in this location, the function is concave, use the inverted bounds from softplus case

            # lb line just connects upper and lower bound points
            lb_line[0] = (tanh(ub) - tanh(lb)) / (ub - lb)
            lb_line[1] = tanh(ub) - lb_line[0] * ub

            # tangent line at point d_1
            d_1 = self.single_line_ub_line_bias * ub + (1 - self.single_line_ub_line_bias) * lb
            ub_line[0] = self.tanh_derivative(d_1)
            ub_line[1] = tanh(d_1) - ub_line[0] * d_1
        else:
            try:
                d_ub = optimize.root_scalar(lambda d: tanh_bound_d(d, lb), bracket=[0, ub], method='brentq').root
            except:
                d_ub = -1

            try:
                d_lb = optimize.root_scalar(lambda d: tanh_bound_d(d, ub), bracket=[lb, 0], method='brentq').root
            except:
                d_lb = 1

            d_ub, d_lb = torch.tensor(d_ub), torch.tensor(d_lb)

            if d_lb <= 0.0:
                # tangent line at point d_lb
                lb_line[0] = self.tanh_derivative(d_lb)
                lb_line[1] = tanh(ub) - lb_line[0] * ub
            else:
                # lb line just connects upper and lower bound points
                lb_line[0] = (tanh(ub) - tanh(lb)) / (ub - lb)
                lb_line[1] = tanh(ub) - lb_line[0] * ub

            if d_ub >= 0:
                # tangent line at point d_ub
                ub_line[0] = self.tanh_derivative(d_ub)
                ub_line[1] = tanh(lb) - ub_line[0] * lb
            else:
                # ub line just connects upper and lower bound points
                ub_line[0] = (tanh(ub) - tanh(lb)) / (ub - lb)
                ub_line[1] = tanh(ub) - ub_line[0] * ub

        return lb_line, ub_line
    
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
    def np_tanh_derivative(x):
        return 1 - np.tanh(x)**2

    @staticmethod
    def np_tanh_second_derivative(x: float):        
        return -2 * np.tanh(x) * (1 - np.tanh(x)**2)
    
    def multi_line_relaxation(self, lb: torch.tensor, ub: torch.tensor) -> Tuple[List[line_type], List[line_type]]:
        lb_lines = []
        ub_lines = []

        assert lb <= ub
        lb = lb.to(torch.float64)
        ub = ub.to(torch.float64)

        if (self.in_region(lb, self.first_convex_region) and self.in_region(ub, self.first_convex_region)) or (self.in_region(lb, self.second_convex_region) and self.in_region(ub, self.second_convex_region)):
            # in this location, the function is convex, use the same bounds as in softplus case
        
            # tangent line at point d_1
            d_1 = self.single_line_lb_line_bias * ub + (1 - self.single_line_lb_line_bias) * lb
            lb_m = self.tanh_second_derivative(d_1)
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
            ub_m = self.tanh_second_derivative(d_1)
            ub_b = self.evaluate(d_1) - ub_m * d_1
            ub_lines.append((ub_m, ub_b))
        else:
            # points are in different regions; 
            # are they in the first convex and the concave regions?
            if self.in_region(lb, self.first_convex_region) and self.in_region(ub, self.concave_region):
                # if -lb >= ub:
                #     lb_m = min(self.tanh_second_derivative(lb), (self.evaluate(ub) - self.evaluate(lb)) / (ub - lb))
                #     lb_b = self.evaluate(lb) - lb_m * lb
                #     lb_lines.append((lb_m, lb_b))
                # else:
                #     lb_m = max(self.tanh_second_derivative(ub), (self.evaluate(ub) - self.evaluate(lb)) / (ub - lb))
                #     lb_b = self.evaluate(ub) - lb_m * ub
                #     lb_lines.append((lb_m, lb_b))

                # # ub_lines
                # ub_m = ((self.evaluate(lb) - self.b_intersect)/lb)
                # ub_b = self.evaluate(lb) - ub_m * lb
                # ub_lines.append((ub_m, ub_b))

                # if ub > 0:
                #     ub_m = ((self.evaluate(ub) - self.b_intersect)/ub)
                #     ub_b = self.evaluate(ub) - ub_m * ub
                #     ub_lines.append((ub_m, ub_b))

                lb_line, ub_line = get_lines_lb_in_convex_ub_in_concave(lb, ub, self.concave_region[0], self.evaluate, self.tanh_second_derivative)

                lb_lines.append(lb_line)
                ub_lines.append(ub_line)
            elif self.in_region(lb, self.concave_region) and self.in_region(ub, self.second_convex_region):
                # if -lb >= ub:
                #     lb_m = min(self.tanh_second_derivative(lb), (self.evaluate(ub) - self.evaluate(lb)) / (ub - lb))
                #     lb_b = self.evaluate(lb) - lb_m * lb
                #     lb_lines.append((lb_m, lb_b))
                # else:
                #     lb_m = max(self.tanh_second_derivative(ub), (self.evaluate(ub) - self.evaluate(lb)) / (ub - lb))
                #     lb_b = self.evaluate(ub) - lb_m * ub
                #     lb_lines.append((lb_m, lb_b))

                # # ub_lines
                # if lb < 0:
                #     ub_m = ((self.evaluate(lb) - self.b_intersect)/lb)
                #     ub_b = self.evaluate(lb) - ub_m * lb
                #     ub_lines.append((ub_m, ub_b))

                # ub_m = ((self.evaluate(ub) - self.b_intersect)/ub)
                # ub_b = self.evaluate(ub) - ub_m * ub
                # ub_lines.append((ub_m, ub_b))

                lb_line, ub_line = get_lines_lb_in_concave_ub_in_convex(lb, ub, self.concave_region[1], self.evaluate, self.tanh_second_derivative)

                lb_lines.append(lb_line)
                ub_lines.append(ub_line)

            # are they in the two convex regions?
            elif self.in_region(lb, self.first_convex_region) and self.in_region(ub, self.second_convex_region):
                # lb should be a single line, no benefit of more than one
                if -lb >= ub:
                    lb_line_left, _ = get_lines_lb_in_convex_ub_in_concave(lb, ub, self.concave_region[0], self.evaluate, self.tanh_second_derivative)
                    lb_lines.append(lb_line_left)
                else:
                    lb_line_right, _ = get_lines_lb_in_concave_ub_in_convex(lb, ub, self.concave_region[1], self.evaluate, self.tanh_second_derivative)
                    lb_lines.append(lb_line_right)

                _, ub_line_left = get_lines_lb_in_convex_ub_in_concave(lb, torch.tensor([0]), self.concave_region[0], self.evaluate, self.tanh_second_derivative)
                _, ub_line_right = get_lines_lb_in_concave_ub_in_convex(torch.tensor([0]), ub, self.concave_region[1], self.evaluate, self.tanh_second_derivative)

                ub_lines.append(ub_line_left)
                ub_lines.append(ub_line_right)

        return lb_lines, ub_lines
    
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

        assert lb <= ub

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
                lb_m = min(self.np_tanh_third_derivative(lb), (self.evaluate(ub) - self.evaluate(lb)) / (ub - lb))
                lb_b = self.evaluate(lb) - lb_m * lb
                lb_lines.append((lb_m, lb_b))

                # ub_lines
                ub_m = ((self.evaluate(lb) - self.first_mound_lhs)/(lb - self.first_mound_x))
                ub_b = self.evaluate(lb) - ub_m * lb
                ub_lines.append((ub_m, ub_b))

                if ub > self.first_mound_x:
                    ub_m = ((self.evaluate(ub) - self.first_mound_rhs)/(ub - self.first_mound_x))
                    ub_b = self.evaluate(ub) - ub_m * ub
                    ub_lines.append((ub_m, ub_b))
            elif self.in_region(lb, self.second_convex_region) and self.in_region(ub, self.second_concave_region):
                ub_m = min(self.np_tanh_third_derivative(ub), (self.evaluate(ub) - self.evaluate(lb)) / (ub - lb))
                ub_b = self.evaluate(ub) - ub_m * ub
                ub_lines.append((ub_m, ub_b))

                # lb_lines
                lb_m = ((self.evaluate(ub) - self.second_mound_rhs)/(ub - self.second_mound_x))
                lb_b = self.evaluate(ub) - lb_m * ub
                lb_lines.append((lb_m, lb_b))
                
                if lb < self.second_mound_x:
                    lb_m = ((self.evaluate(lb) - self.second_mound_lhs)/(lb - self.second_mound_x))
                    lb_b = self.evaluate(lb) - lb_m * lb
                    lb_lines.append((lb_m, lb_b))
            elif self.in_region(lb, self.first_concave_region) and self.in_region(ub, self.second_convex_region):
                if lb < self.first_mound_x:
                    ub_m = ((self.evaluate(lb) - self.first_mound_lhs)/(lb - self.first_mound_x))
                    ub_b = self.evaluate(lb) - ub_m * lb
                    ub_lines.append((ub_m, ub_b))
                
                ub_m = ((self.evaluate(ub) - self.first_mound_rhs)/(ub - self.first_mound_x))
                ub_b = self.evaluate(ub) - ub_m * ub
                ub_lines.append((ub_m, ub_b))

                lb_m = ((self.evaluate(lb) - self.second_mound_lhs)/(lb - self.second_mound_x))
                lb_b = self.evaluate(lb) - lb_m * lb
                lb_lines.append((lb_m, lb_b))

                if ub > self.second_mound_x:
                    lb_m = ((self.evaluate(ub) - self.second_mound_rhs)/(ub - self.second_mound_x))
                    lb_b = self.evaluate(ub) - lb_m * ub
                    lb_lines.append((lb_m, lb_b))

            # they are in non-adjoint regions (3 cases)
            elif self.in_region(lb, self.first_convex_region) and self.in_region(ub, self.second_convex_region):
                # ub_lines
                ub_m = ((self.evaluate(lb) - self.first_mound_lhs)/(lb - self.first_mound_x))
                ub_b = self.evaluate(lb) - ub_m * lb
                ub_lines.append((ub_m, ub_b))

                ub_m = ((self.evaluate(ub) - self.first_mound_rhs)/(ub - self.first_mound_x))
                ub_b = self.evaluate(ub) - ub_m * ub
                ub_lines.append((ub_m, ub_b))
                
                # lb_lines
                lb_m = ((self.evaluate(lb) - self.second_mound_lhs)/(lb - self.second_mound_x))
                lb_b = self.evaluate(lb) - lb_m * lb
                lb_lines.append((lb_m, lb_b))

                if ub > self.second_mound_x:
                    lb_m = ((self.evaluate(ub) - self.second_mound_rhs)/(ub - self.second_mound_x))
                    lb_b = self.evaluate(ub) - lb_m * ub
                    lb_lines.append((lb_m, lb_b))

            elif self.in_region(lb, self.first_concave_region) and self.in_region(ub, self.second_concave_region):
                if lb < self.first_mound_x:
                    ub_m = ((self.evaluate(lb) - self.first_mound_lhs)/(lb - self.first_mound_x))
                    ub_b = self.evaluate(lb) - ub_m * lb
                    ub_lines.append((ub_m, ub_b))
                
                ub_m = ((self.evaluate(ub) - self.first_mound_rhs)/(ub - self.first_mound_x))
                ub_b = self.evaluate(ub) - ub_m * ub
                ub_lines.append((ub_m, ub_b))

                # lb_lines
                lb_m = ((self.evaluate(ub) - self.second_mound_rhs)/(ub - self.second_mound_x))
                lb_b = self.evaluate(ub) - lb_m * ub
                lb_lines.append((lb_m, lb_b))
                
                lb_m = ((self.evaluate(lb) - self.second_mound_lhs)/(lb - self.second_mound_x))
                lb_b = self.evaluate(lb) - lb_m * lb
                lb_lines.append((lb_m, lb_b))
                
            elif self.in_region(lb, self.first_convex_region) and self.in_region(ub, self.second_concave_region):
                ub_m = ((self.evaluate(lb) - self.first_mound_lhs)/(lb - self.first_mound_x))
                ub_b = self.evaluate(lb) - ub_m * lb
                ub_lines.append((ub_m, ub_b))
                
                ub_m = ((self.evaluate(ub) - self.first_mound_rhs)/(ub - self.first_mound_x))
                ub_b = self.evaluate(ub) - ub_m * ub
                ub_lines.append((ub_m, ub_b))

                # lb_lines
                lb_m = ((self.evaluate(ub) - self.second_mound_rhs)/(ub - self.second_mound_x))
                lb_b = self.evaluate(ub) - lb_m * ub
                lb_lines.append((lb_m, lb_b))
                
                lb_m = ((self.evaluate(lb) - self.second_mound_lhs)/(lb - self.second_mound_x))
                lb_b = self.evaluate(lb) - lb_m * lb
                lb_lines.append((lb_m, lb_b))
            else:
                raise ValueError("should not happen")

        return lb_lines, ub_lines

    def get_lb_ub_in_interval(self, lb: torch.tensor, ub: torch.tensor) -> Tuple[torch.tensor, torch.tensor]:
        if lb == ub:
            return self.evaluate(lb), self.evaluate(ub)

        out_lb = optimize.minimize(lambda x: self.evaluate(torch.tensor(x)).item(), bounds=[(lb, ub)], x0=(lb+ub)/2).fun - 1e-3
        out_ub = -optimize.minimize(lambda x: -self.evaluate(torch.tensor(x)).item(), bounds=[(lb, ub)], x0=(lb+ub)/2).fun + 1e-3

        return out_lb, out_ub
