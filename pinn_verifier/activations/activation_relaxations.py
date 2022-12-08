from enum import Enum
from typing import List, Tuple

import torch
import gurobipy as grb

line_type = Tuple[torch.tensor, torch.tensor]
quadratic_type = Tuple[torch.tensor, torch.tensor, torch.tensor]


class ActivationRelaxationType(Enum):
    SINGLE_LINE = 0
    MULTI_LINE = 1
    QUADRATIC = 2
    PIECEWISE_LINEAR = 3


class ActivationRelaxation():
    def __init__(self, type: ActivationRelaxationType) -> None:
        self.type = type
        self.high_precision = True
    
    @staticmethod
    def add_linear_lb(grb_model: grb.Model, input_var: grb.Var, output_var: grb.Var, lb_line: line_type):
        grb_model.addConstr(output_var >= lb_line[0].item() * input_var + lb_line[1].item() - grb_model.params.FeasibilityTol)

    @staticmethod
    def add_linear_ub(grb_model: grb.Model, input_var: grb.Var, output_var: grb.Var, ub_line: line_type):
        try:
            grb_model.addConstr(output_var <= ub_line[0].item() * input_var + ub_line[1].item() + grb_model.params.FeasibilityTol)
        except:
            import pdb
            pdb.set_trace()

    def single_line_relaxation(self, lb: torch.tensor, ub: torch.tensor) -> line_type:
        raise NotImplementedError
    
    def multi_line_relaxation(self, lb: torch.tensor, ub: torch.tensor)  -> Tuple[List[line_type], List[line_type]]:
        raise NotImplementedError
    
    def quadratic_relaxation(self, lb: torch.tensor, ub: torch.tensor) -> quadratic_type:
        raise NotImplementedError

    def piecewise_linear_relaxation(self, lb: torch.tensor, ub: torch.tensor):
        raise NotImplementedError

    def add_single_line_bounds(self, grb_model: grb.Model, input_var: grb.Var, output_var: grb.Var, lb_fns: line_type, ub_fns: line_type):
        self.add_linear_lb(grb_model, input_var, output_var, lb_fns)
        self.add_linear_ub(grb_model, input_var, output_var, ub_fns)

        return lb_fns, ub_fns

    def add_multi_line_bounds(self, grb_model: grb.Model, input_var: grb.Var, output_var: grb.Var, lb_fns: List[line_type], ub_fns: List[line_type]):
        for lb_line in lb_fns:
            self.add_linear_lb(grb_model, input_var, output_var, lb_line)
        
        for ub_line in ub_fns:
            self.add_linear_ub(grb_model, input_var, output_var, ub_line)

        return lb_fns, ub_fns

    def add_quadratic_bounds(self, grb_model: grb.Model, input_var: grb.Var, output_var: grb.Var, lb_fns: quadratic_type, ub_fns: quadratic_type):
        lb_quad_expr = lb_fns[1].item() * input_var + lb_fns[2].item()
        ub_quad_expr = lb_fns[0].item() * input_var * input_var + lb_fns[1].item() * input_var + lb_fns[2].item()

        grb_model.addConstr(output_var >= lb_quad_expr)
        grb_model.addConstr(output_var <= ub_quad_expr)

        return lb_fns, ub_fns
    
    def add_piecewise_linear_bounds(self, grb_model: grb.Model, input_var: grb.Var, output_var: grb.Var, lb_fns: quadratic_type, ub_fns: quadratic_type):
        raise NotImplementedError

    def get_bounds(self, lb: torch.tensor, ub: torch.tensor) -> List[List[float]]:
        if lb == ub:
            # no need to know the underlying function, any single line will do
            y_val = self.evaluate(lb)
            lb_line = [1, y_val - lb - 1e-5]
            ub_line = [1, y_val - ub + 1e-5]

            if self.type == ActivationRelaxationType.SINGLE_LINE:
                return lb_line, ub_line
            elif self.type == ActivationRelaxationType.MULTI_LINE:
                return [lb_line], [ub_line]
            else:
                raise NotImplementedError('other')

        assert ub > lb
        lb = lb.detach()
        ub = ub.detach()

        if self.high_precision:
            lb = lb.to(torch.float64)
            ub = ub.to(torch.float64)

        if self.type == ActivationRelaxationType.SINGLE_LINE:
            return self.single_line_relaxation(lb, ub)
        if self.type == ActivationRelaxationType.MULTI_LINE:
            return self.multi_line_relaxation(lb, ub)
        if self.type == ActivationRelaxationType.QUADRATIC:
            return self.quadratic_relaxation(lb, ub)
        if self.type == ActivationRelaxationType.PIECEWISE_LINEAR:
            return self.piecewise_linear_relaxation(lb, ub)

    def relax_lp(self, grb_model: grb.Model, input_var: grb.Var, output_var: grb.Var, lb: torch.tensor, ub: torch.tensor) -> List[List[float]]:
        out = self.get_bounds(lb, ub)
        if out is None:
            return None

        lb_fns, ub_fns = out

        if self.type == ActivationRelaxationType.SINGLE_LINE:
            return self.add_single_line_bounds(grb_model, input_var, output_var, lb_fns, ub_fns)
        if self.type == ActivationRelaxationType.MULTI_LINE:
            return self.add_multi_line_bounds(grb_model, input_var, output_var, lb_fns, ub_fns)
        if self.type == ActivationRelaxationType.QUADRATIC:
            return self.add_quadratic_bounds(grb_model, input_var, output_var, lb_fns, ub_fns)
        if self.type == ActivationRelaxationType.PIECEWISE_LINEAR:
            return self.add_piecewise_linear_bounds(grb_model, input_var, output_var, lb_fns, ub_fns)


def plot_upper_lower_bounds(activation_relax: ActivationRelaxation, lb: float, ub: float, lb_lines: List[line_type], ub_lines: List[line_type]):
    import matplotlib.pyplot as plt

    x = torch.linspace(min([-5, lb]), max([5, ub]), 1000)
    y = activation_relax.evaluate(x)

    plt.plot(x, y, c="b")

    x_1 = torch.linspace(lb, ub, 250)
    for lb_line in lb_lines:
        y_lb = lb_line[0] * x_1 + lb_line[1]
        plt.plot(x_1, y_lb, c="g")

    for ub_line in ub_lines:
        y_ub = ub_line[0] * x_1 + ub_line[1]
        plt.plot(x_1, y_ub, c="r")

    plt.show()
