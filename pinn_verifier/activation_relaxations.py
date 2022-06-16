from enum import Enum
from typing import List, Tuple

import torch
import numpy as np
import gurobipy as grb
from scipy import optimize

softplus = torch.nn.Softplus()
sigmoid = torch.nn.Sigmoid()

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
    
    @staticmethod
    def add_linear_lb(grb_model: grb.Model, input_var: grb.Var, output_var: grb.Var, lb_line: line_type):
        grb_model.addConstr(output_var >= lb_line[0].item() * input_var + lb_line[1].item() - grb_model.params.FeasibilityTol)

    @staticmethod
    def add_linear_ub(grb_model: grb.Model, input_var: grb.Var, output_var: grb.Var, ub_line: line_type):
        grb_model.addConstr(output_var <= ub_line[0].item() * input_var + ub_line[1].item() + grb_model.params.FeasibilityTol)

    def single_line_relaxation(self, grb_model: grb.Model, input_var: grb.Var, output_var: grb.Var, lb: torch.tensor, ub: torch.tensor) -> line_type:
        raise NotImplementedError
    
    def multi_line_relaxation(self, grb_model: grb.Model, input_var: grb.Var, output_var: grb.Var, lb: torch.tensor, ub: torch.tensor)  -> Tuple[List[line_type], List[line_type]]:
        raise NotImplementedError
    
    def quadratic_relaxation(self, grb_model: grb.Model, input_var: grb.Var, output_var: grb.Var, lb: torch.tensor, ub: torch.tensor):
        raise NotImplementedError

    def piecewise_linear_relaxation(self, grb_model: grb.Model, input_var: grb.Var, output_var: grb.Var, lb: torch.tensor, ub: torch.tensor):
        raise NotImplementedError

    def relax(self, grb_model: grb.Model, input_var: grb.Var, output_var: grb.Var, lb: torch.tensor, ub: torch.tensor) -> List[List[float]]:        
        if lb == ub:
            return None

        assert ub > lb

        if self.type == ActivationRelaxationType.SINGLE_LINE:
            return self.single_line_relaxation(grb_model, input_var, output_var, lb, ub)
        if self.type == ActivationRelaxationType.MULTI_LINE:
            return self.multi_line_relaxation(grb_model, input_var, output_var, lb, ub)
        if self.type == ActivationRelaxationType.QUADRATIC:
            return self.quadratic_relaxation(grb_model, input_var, output_var, lb, ub)
        if self.type == ActivationRelaxationType.PIECEWISE_LINEAR:
            return self.piecewise_linear_relaxation(grb_model, input_var, output_var, lb, ub)


class SoftplusRelaxation(ActivationRelaxation):
    def __init__(self, type: ActivationRelaxationType, single_line_bias: float = 0.65, multi_line_biases: List[float] = [0.1, 0.25, 0.5, 0.75, 0.9], piecewise_linear_bias: float = 0.5) -> None:
        super().__init__(type)
        self.single_line_bias = single_line_bias
        self.multi_line_biases = multi_line_biases
        self.piecewise_linear_bias = piecewise_linear_bias

    @staticmethod
    def evaluate(x: torch.tensor):
        return softplus(x)

    @staticmethod
    def softplus_tangent_at_point_d(d: torch.tensor) -> line_type:
        m = sigmoid(d)
        b = softplus(d) - m * d
        return m, b

    def single_line_relaxation(self, grb_model: grb.Model, input_var: grb.Var, output_var: grb.Var, lb: torch.tensor, ub: torch.tensor) -> line_type:
        # softplus is a convex function, so upper bound is just the connection of the two ends of the interval
        # lower bound is at a given bias point
        ub_line_m = (softplus(ub) - softplus(lb)) / (ub - lb)
        ub_line_b = softplus(ub) - ub_line_m * ub
        ub_line = (ub_line_m, ub_line_b)

        # tangent line at point d
        d = self.single_line_bias * ub + (1 - self.single_line_bias) * lb
        lb_line = self.softplus_tangent_at_point_d(d)

        self.add_linear_lb(grb_model, input_var, output_var, lb_line)
        self.add_linear_ub(grb_model, input_var, output_var, ub_line)

        return lb_line, ub_line
    
    def multi_line_relaxation(self, grb_model: grb.Model, input_var: grb.Var, output_var: grb.Var, lb: torch.tensor, ub: torch.tensor) -> Tuple[List[line_type], List[line_type]]:
        # relax softplus with multiple linear lower bounds
        # returns an array of lower bounds and an array of upper bounds
        ub_line_m = (softplus(ub) - softplus(lb)) / (ub - lb)
        ub_line_b = softplus(ub) - ub_line_m * ub
        ub_line = (ub_line_m, ub_line_b)
        ub_lines = [ub_line]

        lb_lines = []
        for bias in self.multi_line_biases:
            d = bias * ub + (1 - bias) * lb
            lb_lines.append(self.softplus_tangent_at_point_d(d))

        for lb_line in lb_lines:
            self.add_linear_lb(grb_model, input_var, output_var, lb_line)
        
        for ub_line in ub_lines:
            self.add_linear_ub(grb_model, input_var, output_var, ub_line)

        # import matplotlib.pyplot as plt

        # x = torch.linspace(-5, 5, 1000)
        # y = softplus(x)
        # plt.plot(x, y, c="b")

        # lb = torch.tensor(lb, dtype=torch.float)
        # ub = torch.tensor(ub, dtype=torch.float)
        # x_1 = torch.linspace(lb, ub, 250)

        # for lb_line in lb_lines:
        #     y_lb = lb_line[0] * x_1 + lb_line[1]
        #     plt.plot(x_1, y_lb, c='r')

        # for ub_line in ub_lines:
        #     y_ub = ub_line[0] * x_1 + ub_line[1]
        #     plt.plot(x_1, y_ub, c='g')

        # plt.show()

        return lb_lines, ub_lines
    
    def quadratic_relaxation(self, grb_model: grb.Model, input_var: grb.Var, output_var: grb.Var, lb: torch.tensor, ub: torch.tensor) -> quadratic_type:
        # linear lower bound and universal upper bound
        d = self.single_line_bias * ub + (1 - self.single_line_bias) * lb
        lb_line = self.softplus_tangent_at_point_d(d)

        ub_quad = (0.1, 0.5, 0.85)

        lb_quad_expr = lb_line[1].item() * input_var + lb_line[2].item()
        ub_quad_expr = ub_quad[0].item() * input_var * input_var + ub_quad[1].item() * input_var + ub_quad[2].item()

        grb_model.addConstr(output_var >= lb_quad_expr)
        grb_model.addConstr(output_var <= ub_quad_expr)

        return (0, lb_line[0], lb_line[1]), ub_quad

    def piecewise_linear_relaxation(self, grb_model: grb.Model, input_var: grb.Var, output_var: grb.Var, lb: torch.tensor, ub: torch.tensor):
        # approximate the lower bound the multi-line pieces]
        lb_lines = []
        for bias in self.multi_line_biases:
            d = bias * ub + (1 - bias) * lb
            lb_lines.append(self.softplus_tangent_at_point_d(d))

        for lb_line in lb_lines:
            self.add_linear_lb(grb_model, input_var, output_var, lb_line)

        # two pieces for the upper bound
        d = self.piecewise_linear_bias * ub + (1 - self.piecewise_linear_bias) * lb
        
        max_v = grb_model.addVar(
            lb=softplus(lb), ub=softplus(ub),
            obj=0, vtype=grb.GRB.CONTINUOUS
        )

        # xs = [lb.item(), d.item(), ub.item()]
        # ys = [softplus(lb).item(), softplus(d).item(), softplus(ub).item()]
        # grb_model.addGenConstrPWL(input_var, max_v, xs, ys)
        
        mid_point = (d, softplus(d))
        first_half_m = (mid_point[1] - softplus(lb)) / (mid_point[0] - lb)
        first_half_b = mid_point[1] - first_half_m * mid_point[0]
        first_half = (first_half_m, first_half_b)

        second_half_m = (softplus(ub) - mid_point[1]) / (ub - mid_point[0])
        second_half_b = mid_point[1] - second_half_m * mid_point[0]
        second_half = (second_half_m, second_half_b)

        max_v = grb_model.addVar(
            lb=softplus(lb), ub=softplus(ub),
            obj=0, vtype=grb.GRB.CONTINUOUS,
            name="u_theta_softplus_max_{layer_idx}_{neuron_idx}"
        )
        first_half_line = grb_model.addVar(
            lb=softplus(lb), ub=softplus(ub),
            obj=0, vtype=grb.GRB.CONTINUOUS,
            name="u_theta_softplus_first_half_{layer_idx}_{neuron_idx}"
        )
        second_half_line = grb_model.addVar(
            lb=softplus(lb), ub=softplus(ub),
            obj=0, vtype=grb.GRB.CONTINUOUS,
            name="u_theta_softplus_second_half_{layer_idx}_{neuron_idx}"
        )
        grb_model.addConstr(first_half_line == first_half[0].item() * input_var + first_half[1].item())
        grb_model.addConstr(second_half_line == second_half[0].item() * input_var + second_half[1].item())
        grb_model.addConstr(max_v == grb.max_(first_half_line, second_half_line))
        grb_model.addConstr(output_var <= max_v)

        # import numpy as np
        # import matplotlib.pyplot as plt
        # x = torch.linspace(-5, 5, 1000)
        # y = torch.nn.Softplus()(x)
        # plt.plot(x, y)

        # x_1 = torch.linspace(lb, ub, 250)
        # for lb_line in lb_lines:
        #     y_lb = lb_line[0] * x_1 + lb_line[1]
        #     plt.plot(x_1, y_lb, c="g")

        
        # y_ub_0 = first_half[0] * x_1 + first_half[1]
        # y_ub_1 = second_half[0] * x_1 + second_half[1]
        # plt.plot(x_1, np.maximum(y_ub_0, y_ub_1))
        # plt.show()

        # grb_model.update()

        # grb_model.setObjective(output_var, grb.GRB.MINIMIZE)
        # grb_model.update()
        # grb_model.reset()
        # grb_model.optimize()

        # import pdb
        # pdb.set_trace()

        return lb_lines, [first_half, second_half]


class SigmoidRelaxation(ActivationRelaxation):
    def __init__(self, type: ActivationRelaxationType, single_line_lb_line_bias: float = 0.9, single_line_ub_line_bias: float = 0.2) -> None:
        super().__init__(type)
        self.single_line_lb_line_bias = single_line_lb_line_bias
        self.single_line_ub_line_bias = single_line_ub_line_bias

    @staticmethod
    def evaluate(x: torch.tensor):
        return sigmoid(x)

    @staticmethod
    def sigmoid_derivative(x):
        return sigmoid(x) * (1 - sigmoid(x))

    def single_line_relaxation(self, grb_model: grb.Model, input_var: grb.Var, output_var: grb.Var, lb: torch.tensor, ub: torch.tensor) -> line_type:
        def np_sigmoid(x):
            return 1 / (1 + np.exp(-x))

        def sigmoid_bound_d(x, bound):
            return np_sigmoid(x) * (1 - np_sigmoid(x)) - (np_sigmoid(x) - np_sigmoid(bound)) / (x - bound) 
        
        lb_line = [0, 0]
        ub_line = [0, 0]

        if lb < 0 and ub < 0:
            # in this location, the function is convex, use the same bounds as in softplus case
        
            # tangent line at point d_1
            d_1 = self.single_line_lb_line_bias * ub + (1 - self.single_line_lb_line_bias) * lb
            lb_line_m = self.sigmoid_derivative(d_1)
            lb_line_b = sigmoid(d_1) - lb_line_m * d_1

            # ub line just connects upper and lower bound points
            ub_line_m = (sigmoid(ub) - sigmoid(lb)) / (ub - lb)
            ub_line_b = sigmoid(ub) - ub_line_m * ub
        elif lb > 0 and ub > 0:
            # in this location, the function is concave, use the inverted bounds from softplus case

            # lb line just connects upper and lower bound points
            lb_line_m = (sigmoid(ub) - sigmoid(lb)) / (ub - lb)
            lb_line_b = sigmoid(ub) - lb_line_m * ub

            # tangent line at point d_1
            d_1 = self.single_line_ub_line_bias * ub + (1 - self.single_line_ub_line_bias) * lb
            ub_line_m = self.sigmoid_derivative(d_1)
            ub_line_b = sigmoid(d_1) - ub_line_m * d_1
        else:
            try:
                d_ub = optimize.root_scalar(lambda d: sigmoid_bound_d(d, lb), bracket=[0, ub], method='brentq').root
            except:
                d_ub = -1

            try:
                d_lb = optimize.root_scalar(lambda d: sigmoid_bound_d(d, ub), bracket=[lb, 0], method='brentq').root
            except:
                d_lb = 1

            d_ub, d_lb = torch.tensor(d_ub), torch.tensor(d_lb)

            if d_lb <= 0.0:
                # tangent line at point d_lb
                lb_line_m = self.sigmoid_derivative(d_lb)
                lb_line_b = sigmoid(ub) - lb_line_m * ub
            else:
                # lb line just connects upper and lower bound points
                lb_line_m = (sigmoid(ub) - sigmoid(lb)) / (ub - lb)
                lb_line_b = sigmoid(ub) - lb_line_m * ub

            if d_ub >= 0:
                # tangent line at point d_ub
                ub_line_m = self.sigmoid_derivative(d_ub)
                ub_line_b = sigmoid(lb) - ub_line_m * lb
            else:
                # ub line just connects upper and lower bound points
                ub_line_m = (sigmoid(ub) - sigmoid(lb)) / (ub - lb)
                ub_line_b = sigmoid(ub) - ub_line_m * ub

        lb_line = (lb_line_m, lb_line_b)
        ub_line = (ub_line_m, ub_line_b)

        self.add_linear_lb(grb_model, input_var, output_var, lb_line)
        self.add_linear_ub(grb_model, input_var, output_var, ub_line)

        return lb_line, ub_line
    
    def multi_line_relaxation(self, grb_model: grb.Model, input_var: grb.Var, output_var: grb.Var, lb: torch.tensor, ub: torch.tensor) -> Tuple[List[line_type], List[line_type]]:
        raise NotImplementedError
    
    def quadratic_relaxation(self, grb_model: grb.Model, input_var: grb.Var, output_var: grb.Var, lb: torch.tensor, ub: torch.tensor) -> quadratic_type:
        raise NotImplementedError

    def piecewise_linear_relaxation(self, grb_model: grb.Model, input_var: grb.Var, output_var: grb.Var, lb: torch.tensor, ub: torch.tensor):
        raise NotImplementedError


class SigmoidDerivativeRelaxation(ActivationRelaxation):
    def __init__(self, type: ActivationRelaxationType, single_line_lb_line_bias: float = 0.9, single_line_ub_line_bias: float = 0.2) -> None:
        super().__init__(type)
        self.single_line_lb_line_bias = single_line_lb_line_bias
        self.single_line_ub_line_bias = single_line_ub_line_bias

    @staticmethod
    def evaluate(x: torch.tensor):
        return sigmoid(x)

    @staticmethod
    def sigmoid_derivative(x):
        return sigmoid(x) * (1 - sigmoid(x))

    @staticmethod
    def sigmoid_derivative_derivative(x):
        return sigmoid(x) * (1 - sigmoid(x)) * (1 - 2 * sigmoid(x))

    def multi_line_relaxation(self, grb_model: grb.Model, input_var: grb.Var, output_var: grb.Var, lb: torch.tensor, ub: torch.tensor) -> Tuple[List[line_type], List[line_type]]:
        ub_lines = []
        lb_lines = []

        assert lb <= ub

        first_convex_region = [-np.Inf, -1.3169559128480408]
        concave_region = [-1.3169559128480408, 1.3169559080930038]
        second_convex_region = [1.3169559080930038, np.Inf]

        b_intersect = 0.3
        min_lb_m = -(self.sigmoid_derivative(torch.tensor(first_convex_region[1])).item() + b_intersect)/(first_convex_region[1])

        def in_region(p, region):
            return p >= region[0] and p <= region[1]

        if (in_region(lb, first_convex_region) and in_region(ub, first_convex_region)) or (in_region(lb, second_convex_region) and in_region(ub, second_convex_region)):
            # in this location, the function is convex, use the same bounds as in softplus case
        
            # tangent line at point d_1
            d_1 = self.single_line_lb_line_bias * ub + (1 - self.single_line_lb_line_bias) * lb
            lb_m = self.sigmoid_derivative_derivative(d_1)
            lb_b = self.sigmoid_derivative(d_1) - lb_m * d_1
            lb_lines.append((lb_m, lb_b))

            # ub line just connects upper and lower bound points
            ub_m = (self.sigmoid_derivative(ub) - self.sigmoid_derivative(lb)) / (ub - lb)
            ub_b = self.sigmoid_derivative(ub) - ub_m * ub
            ub_lines.append((ub_m, ub_b))
        elif in_region(lb, concave_region) and in_region(ub, concave_region):
            # in this location, the function is concave, use the inverted bounds from softplus case

            # lb line just connects upper and lower bound points
            lb_m = (self.sigmoid_derivative(ub) - self.sigmoid_derivative(lb)) / (ub - lb)
            lb_b = self.sigmoid_derivative(ub) - lb_m * ub
            lb_lines.append((lb_m, lb_b))

            # tangent line at point d_1
            d_1 = self.single_line_ub_line_bias * ub + (1 - self.single_line_ub_line_bias) * lb
            ub_m = self.sigmoid_derivative_derivative(d_1)
            ub_b = self.sigmoid_derivative(d_1) - ub_m * d_1
            ub_lines.append((ub_m, ub_b))
        else:
            # points are in different regions; 
            # are they in the first convex and the concave regions?
            if in_region(lb, first_convex_region) and in_region(ub, concave_region):
                if -lb >= ub:
                    lb_m = min(self.sigmoid_derivative_derivative(lb), (self.sigmoid_derivative(ub) - self.sigmoid_derivative(lb)) / (ub - lb))
                    lb_b = self.sigmoid_derivative(lb) - lb_m * lb
                    lb_lines.append((lb_m, lb_b))
                else:
                    lb_m = max(self.sigmoid_derivative_derivative(ub), (self.sigmoid_derivative(ub) - self.sigmoid_derivative(lb)) / (ub - lb))
                    lb_b = self.sigmoid_derivative(ub) - lb_m * ub
                    lb_lines.append((lb_m, lb_b))

                # ub_lines
                ub_m = (self.sigmoid_derivative(lb) - 0.3)/lb
                ub_b = self.sigmoid_derivative(lb) - ub_m * lb
                ub_lines.append((ub_m, ub_b))

                if ub > 0:
                    ub_m = (self.sigmoid_derivative(ub) - 0.3)/ub
                    ub_b = self.sigmoid_derivative(ub) - ub_m * ub
                    ub_lines.append((ub_m, ub_b))
            elif in_region(lb, concave_region) and in_region(ub, second_convex_region):
                if -lb >= ub:
                    lb_m = min(self.sigmoid_derivative_derivative(lb), (self.sigmoid_derivative(ub) - self.sigmoid_derivative(lb)) / (ub - lb))
                    lb_b = self.sigmoid_derivative(lb) - lb_m * lb
                    lb_lines.append((lb_m, lb_b))
                else:
                    lb_m = max(self.sigmoid_derivative_derivative(ub), (self.sigmoid_derivative(ub) - self.sigmoid_derivative(lb)) / (ub - lb))
                    lb_b = self.sigmoid_derivative(ub) - lb_m * ub
                    lb_lines.append((lb_m, lb_b))

                # ub_lines
                if lb < 0:
                    ub_m = (self.sigmoid_derivative(lb) - 0.3)/lb
                    ub_b = self.sigmoid_derivative(lb) - ub_m * lb
                    ub_lines.append((ub_m, ub_b))

                ub_m = (self.sigmoid_derivative(ub) - 0.3)/ub
                ub_b = self.sigmoid_derivative(ub) - ub_m * ub
                ub_lines.append((ub_m, ub_b))
            # are they in the two convex regions?
            elif in_region(lb, first_convex_region) and in_region(ub, second_convex_region):
                # lb should be a single line, no benefit of more than one
                if -lb >= ub:
                    lb_m = min(self.sigmoid_derivative_derivative(lb), (self.sigmoid_derivative(ub) - self.sigmoid_derivative(lb)) / (ub - lb))
                    lb_b = self.sigmoid_derivative(lb) - lb_m * lb
                    lb_lines.append((lb_m, lb_b))
                else:
                    lb_m = max(self.sigmoid_derivative_derivative(ub), (self.sigmoid_derivative(ub) - self.sigmoid_derivative(lb)) / (ub - lb))
                    lb_b = self.sigmoid_derivative(ub) - lb_m * ub
                    lb_lines.append((lb_m, lb_b))

                # ub_lines
                ub_m = (self.sigmoid_derivative(lb) - 0.3)/lb
                ub_b = self.sigmoid_derivative(lb) - ub_m * lb
                ub_lines.append((ub_m, ub_b))

                ub_m = (self.sigmoid_derivative(ub) - 0.3)/ub
                ub_b = self.sigmoid_derivative(ub) - ub_m * ub
                ub_lines.append((ub_m, ub_b))
        
        # import matplotlib.pyplot as plt

        # sigmoid = torch.nn.Sigmoid()

        # def sigmoid_derivative(x):
        #     return sigmoid(x) * (1 - sigmoid(x))

        # def sigmoid_derivative_derivative(x):
        #     return sigmoid(x) * (1 - sigmoid(x)) * (1 - 2 * sigmoid(x))

        # x = torch.linspace(-5, 5, 1000)
        # y = sigmoid_derivative(x)
        # plt.plot(x, y, c="b")

        # lb = torch.tensor(lb, dtype=torch.float)
        # ub = torch.tensor(ub, dtype=torch.float)
        # x_1 = torch.linspace(lb, ub, 250)

        # for lb_line in lb_lines:
        #     y_lb = lb_line[0] * x_1 + lb_line[1]
        #     plt.plot(x_1, y_lb, c='r')

        # for ub_line in ub_lines:
        #     y_ub = ub_line[0] * x_1 + ub_line[1]
        #     plt.plot(x_1, y_ub, c='g')

        # plt.show()

        # import pdb
        # pdb.set_trace()

        for lb_line in lb_lines:
            self.add_linear_lb(grb_model, input_var, output_var, lb_line)
        
        for ub_line in ub_lines:
            self.add_linear_ub(grb_model, input_var, output_var, ub_line)

        return lb_lines, ub_lines
    
    def single_line_relaxation(self, grb_model: grb.Model, input_var: grb.Var, output_var: grb.Var, lb: torch.tensor, ub: torch.tensor) -> Tuple[List[line_type], List[line_type]]:
        raise NotImplementedError
    
    def quadratic_relaxation(self, grb_model: grb.Model, input_var: grb.Var, output_var: grb.Var, lb: torch.tensor, ub: torch.tensor) -> quadratic_type:
        raise NotImplementedError

    def piecewise_linear_relaxation(self, grb_model: grb.Model, input_var: grb.Var, output_var: grb.Var, lb: torch.tensor, ub: torch.tensor):
        raise NotImplementedError

    def get_lb_ub_in_interval(self, lb: torch.tensor, ub: torch.tensor) -> Tuple[torch.tensor, torch.tensor]:
        out_lb = min(self.sigmoid_derivative(lb), self.sigmoid_derivative(ub))
        if lb == ub:
            return out_lb, out_lb

        out_ub = -optimize.minimize(lambda x: -self.sigmoid_derivative(torch.tensor(x)).item(), bounds=[(lb, ub)], x0=(lb+ub)/2).fun + 1e-3

        return out_lb, out_ub


def plot_upper_lower_bounds(activation_relax: ActivationRelaxation, lb: float, ub: float, lb_lines: List[line_type], ub_lines: List[line_type]):
    import matplotlib.pyplot as plt

    x = torch.linspace(min([-5, lb]), max([5, ub]), 1000)
    if isinstance(activation_relax, SoftplusRelaxation): 
        y = softplus(x)
    elif isinstance(activation_relax, SigmoidRelaxation):
        y = sigmoid(x)

    plt.plot(x, y, c="b")

    x_1 = torch.linspace(lb, ub, 250)
    for lb_line in lb_lines:
        y_lb = lb_line[0] * x_1 + lb_line[1]
        plt.plot(x_1, y_lb, c="g")

    for ub_line in ub_lines:
        y_ub = ub_line[0] * x_1 + ub_line[1]
        plt.plot(x_1, y_ub, c="r")

    plt.show()
