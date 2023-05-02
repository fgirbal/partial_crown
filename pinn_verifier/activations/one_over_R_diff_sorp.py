from typing import List, Tuple

import torch

from .activation_relaxations import ActivationRelaxation, ActivationRelaxationType, line_type

sin = lambda x: torch.sin(torch.pi * x)

brenth_xtol = 1e-8
brenth_rtol = 1e-8

class OneOverRDiffSorpRelaxation(ActivationRelaxation):
    def __init__(self, type: ActivationRelaxationType, single_line_ub_line_bias: float = 0.9, por: float = 0.29, rho_s: float = 2880, k_f: float = 3.5e-4, n_f: float = 0.874) -> None:
        super().__init__(type)
        self.single_line_ub_line_bias = single_line_ub_line_bias

        self.u_coefficient = ((1 - por) / por) * rho_s * k_f * n_f
        self.u_power = n_f - 1

    def evaluate(self, x: torch.tensor):
        return 1 / (1 + self.u_coefficient * (x + 1e-6) ** self.u_power)

    def one_over_R_derivative(self, x: torch.tensor):
        return -(self.u_coefficient * self.u_power * (1e-6 + x)**(-1 + self.u_power))/(1 + self.u_coefficient * (1e-6 + x)**self.u_power)**2

    def single_line_relaxation(self, lb: torch.tensor, ub: torch.tensor) -> line_type:
        if lb < 0.0:
            raise NotImplementedError('relaxation obtained specifically for bounds [0.0, \infty)')

        lb_line = [0, 0]
        ub_line = [0, 0]

        lb_line[0] = (self.evaluate(ub) - self.evaluate(lb)) / (ub - lb)
        lb_line[1] = self.evaluate(ub) - lb_line[0] * ub

        # tangent line at point d_1
        d_1 = self.single_line_ub_line_bias * ub + (1 - self.single_line_ub_line_bias) * lb
        ub_line[0] = self.one_over_R_derivative(d_1)
        ub_line[1] = self.evaluate(d_1) - ub_line[0] * d_1

        lb_line[1] -= 1e-7
        ub_line[1] += 1e-7

        return lb_line, ub_line
    
    def get_lb_ub_in_interval(self, lb: torch.tensor, ub: torch.tensor) -> Tuple[torch.tensor, torch.tensor]:
        raise NotImplementedError
