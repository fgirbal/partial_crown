from typing import List, Tuple

import torch
import numpy as np
import gurobipy as grb
from scipy import optimize

from .activation_relaxations import ActivationRelaxation, ActivationRelaxationType, line_type

relu = torch.nn.ReLU()

relaxation_tolerance = 1e-7

class ReLURelaxation(ActivationRelaxation):
    def __init__(self, type: ActivationRelaxationType) -> None:
        super().__init__(type)

    @staticmethod
    def evaluate(x: torch.tensor):
        return relu(x)

    def single_line_relaxation(self, lb: torch.tensor, ub: torch.tensor) -> line_type:
        if lb < 0 and ub < 0:
            lb_m = 0
            lb_b = -relaxation_tolerance

            ub_m = 0
            ub_b = relaxation_tolerance
        elif lb > 0 and ub > 0:
            lb_m = 1
            lb_b = ub - lb_m * ub - relaxation_tolerance

            ub_m = lb_m
            ub_b = lb_b + 2 * relaxation_tolerance
        else:
            # lower bound is any tangent line, take the one that passes through zero
            lb_m = 1
            lb_b = -relaxation_tolerance

            # connect both points for the upper bound
            ub_m = (relu(ub) - relu(lb)) / (ub - lb)
            ub_b = relu(ub) - ub_m * ub

        return (lb_m, lb_b), (ub_m, ub_b)
    
    def get_lb_ub_in_interval(self, lb: torch.tensor, ub: torch.tensor) -> Tuple[torch.tensor, torch.tensor]:
        return self.evaluate(lb), self.evaluate(ub)

class ReLUFirstDerivativeRelaxation(ActivationRelaxation):
    def __init__(self, type: ActivationRelaxationType) -> None:
        super().__init__(type)

    @staticmethod
    def evaluate(x: torch.tensor):
        return torch.heaviside(x, values=torch.ones_like(x))

    def single_line_relaxation(self, lb: torch.tensor, ub: torch.tensor) -> line_type:
        if lb < 0 and ub < 0:
            lb_m = 0
            lb_b = -relaxation_tolerance

            ub_m = 0
            ub_b = relaxation_tolerance
        elif lb > 0 and ub > 0:
            lb_m = 0
            lb_b = 1 - relaxation_tolerance

            ub_m = 0
            ub_b = 1 + relaxation_tolerance
        else:
            lb_m = 0
            lb_b = -relaxation_tolerance

            ub_m = 0
            ub_b = 1 + relaxation_tolerance

        return (lb_m, lb_b), (ub_m, ub_b)
    
    def get_lb_ub_in_interval(self, lb: torch.tensor, ub: torch.tensor) -> Tuple[torch.tensor, torch.tensor]:
        return self.evaluate(lb), self.evaluate(ub)
