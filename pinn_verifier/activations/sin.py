from typing import List, Tuple

import torch
import numpy as np
import gurobipy as grb
from scipy import optimize

from .activation_relaxations import ActivationRelaxation, ActivationRelaxationType, line_type

sin = lambda x: torch.sin(torch.pi * x)

brenth_xtol = 1e-8
brenth_rtol = 1e-8

class SinRelaxation(ActivationRelaxation):
    def __init__(self, type: ActivationRelaxationType, single_line_lb_line_bias: float = 0.9, single_line_ub_line_bias: float = 0.2) -> None:
        super().__init__(type)
        self.single_line_lb_line_bias = single_line_lb_line_bias
        self.single_line_ub_line_bias = single_line_ub_line_bias

    @staticmethod
    def evaluate(x: torch.tensor):
        return sin(x)

    @staticmethod
    def sin_derivative(x: torch.tensor):
        return torch.pi * torch.cos(torch.pi * x)

    def single_line_relaxation(self, lb: torch.tensor, ub: torch.tensor) -> line_type:
        if lb < -1 or ub > 1:
            raise NotImplementedError('relaxation obtained specifically for bounds [-1, 1]')

        def np_sin(x):
            return np.sin(np.pi * x)

        def sin_bound_d(x, bound):
            return (np.pi * np.cos(np.pi * x)) - (np_sin(x) - np_sin(bound)) / (x - bound)

        lb_line = [0, 0]
        ub_line = [0, 0]

        if lb < 0 and ub < 0:
            # in this location, the function is convex, use the same bounds as in softplus case
        
            # tangent line at point d_1
            d_1 = self.single_line_lb_line_bias * ub + (1 - self.single_line_lb_line_bias) * lb
            lb_line[0] = self.sin_derivative(d_1)
            lb_line[1] = sin(d_1) - lb_line[0] * d_1

            # ub line just connects upper and lower bound points
            ub_line[0] = (sin(ub) - sin(lb)) / (ub - lb)
            ub_line[1] = sin(ub) - ub_line[0] * ub
        elif lb > 0 and ub > 0:
            # in this location, the function is concave, use the inverted bounds from softplus case

            # lb line just connects upper and lower bound points
            lb_line[0] = (sin(ub) - sin(lb)) / (ub - lb)
            lb_line[1] = sin(ub) - lb_line[0] * ub

            # tangent line at point d_1
            d_1 = self.single_line_ub_line_bias * ub + (1 - self.single_line_ub_line_bias) * lb
            ub_line[0] = self.sin_derivative(d_1)
            ub_line[1] = sin(d_1) - ub_line[0] * d_1
        else:
            try:
                d_ub = optimize.root_scalar(lambda d: sin_bound_d(d, lb), bracket=[0, ub], method='brentq', xtol=1e-8, rtol=1e-8).root
            except:
                d_ub = -1

            try:
                d_lb = optimize.root_scalar(lambda d: sin_bound_d(d, ub), bracket=[lb, 0], method='brentq', xtol=1e-8, rtol=1e-8).root
            except:
                d_lb = 1

            d_ub, d_lb = torch.tensor(d_ub), torch.tensor(d_lb)

            if d_lb <= 0.0:
                # tangent line at point d_lb
                lb_line[0] = self.sin_derivative(d_lb)
                lb_line[1] = sin(ub) - lb_line[0] * ub
            else:
                # lb line just connects upper and lower bound points
                lb_line[0] = (sin(ub) - sin(lb)) / (ub - lb)
                lb_line[1] = sin(ub) - lb_line[0] * ub

            if d_ub >= 0:
                # tangent line at point d_ub
                ub_line[0] = self.sin_derivative(d_ub)
                ub_line[1] = sin(lb) - ub_line[0] * lb
            else:
                # ub line just connects upper and lower bound points
                ub_line[0] = (sin(ub) - sin(lb)) / (ub - lb)
                ub_line[1] = sin(ub) - ub_line[0] * ub
        
        lb_line[1] -= 1e-5
        ub_line[1] += 1e-5

        return lb_line, ub_line
    
    def get_lb_ub_in_interval(self, lb: torch.tensor, ub: torch.tensor) -> Tuple[torch.tensor, torch.tensor]:
        raise NotImplementedError
