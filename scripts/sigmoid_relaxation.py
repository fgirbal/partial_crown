from mailbox import _singlefileMailbox
import matplotlib.pyplot as plt
import torch.nn
from scipy import optimize
import numpy as np

softplus = torch.nn.Softplus()
sigmoid = torch.nn.Sigmoid()
def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

def compute_lower_upper_bounds_lines(lb, ub, ub_line_ub_bias=0.9, lb_line_ub_bias=0.2):
    def np_sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_bound_d(x, bound):
        return np_sigmoid(x) * (1 - np_sigmoid(x)) - (np_sigmoid(x) - np_sigmoid(bound)) / (x - bound) 

    lb_line = [0, 0]
    ub_line = [0, 0]

    assert lb <= ub

    if lb < 0 and ub < 0:
        # in this location, the function is convex, use the same bounds as in softplus case
    
        # tangent line at point d_1
        d_1 = lb_line_ub_bias * ub + (1 - lb_line_ub_bias) * lb
        lb_line[0] = sigmoid_derivative(d_1)
        lb_line[1] = sigmoid(d_1) - lb_line[0] * d_1

        # ub line just connects upper and lower bound points
        ub_line[0] = (sigmoid(ub) - sigmoid(lb)) / (ub - lb)
        ub_line[1] = sigmoid(ub) - ub_line[0] * ub
    elif lb > 0 and ub > 0:
        # in this location, the function is concave, use the inverted bounds from softplus case

        # lb line just connects upper and lower bound points
        lb_line[0] = (sigmoid(ub) - sigmoid(lb)) / (ub - lb)
        lb_line[1] = sigmoid(ub) - lb_line[0] * ub

        # tangent line at point d_1
        d_1 = ub_line_ub_bias * ub + (1 - ub_line_ub_bias) * lb
        ub_line[0] = sigmoid_derivative(d_1)
        ub_line[1] = sigmoid(d_1) - ub_line[0] * d_1
    else:
        try:
            d_ub = optimize.root_scalar(lambda d: sigmoid_bound_d(d, lb), bracket=[0, ub], method='brentq').root
        except:
            print("here 1")
            d_ub = -1

        try:
            d_lb = optimize.root_scalar(lambda d: sigmoid_bound_d(d, ub), bracket=[lb, 0], method='brentq').root
        except:
            print("here 2")
            d_lb = 1

        d_ub, d_lb = torch.tensor(d_ub), torch.tensor(d_lb)

        if d_lb <= 0.0:
            # tangent line at point d_lb
            lb_line[0] = sigmoid_derivative(d_lb)
            lb_line[1] = sigmoid(ub) - lb_line[0] * ub
        else:
            # lb line just connects upper and lower bound points
            lb_line[0] = (sigmoid(ub) - sigmoid(lb)) / (ub - lb)
            lb_line[1] = sigmoid(ub) - lb_line[0] * ub

        if d_ub >= 0:
            # tangent line at point d_ub
            ub_line[0] = sigmoid_derivative(d_ub)
            ub_line[1] = sigmoid(lb) - ub_line[0] * lb
        else:
            # ub line just connects upper and lower bound points
            ub_line[0] = (sigmoid(ub) - sigmoid(lb)) / (ub - lb)
            ub_line[1] = sigmoid(ub) - ub_line[0] * ub

    return lb_line, ub_line


x = torch.linspace(-5, 5, 1000)
y = torch.nn.Sigmoid()(x)
plt.plot(x, y)

lb = torch.tensor(-2.5, dtype=torch.float)
ub = torch.tensor(1.9, dtype=torch.float)
x_1 = torch.linspace(lb, ub, 250)

lb_line, ub_line = compute_lower_upper_bounds_lines(lb, ub, lb_line_ub_bias=0.4, ub_line_ub_bias=0.5)
y_lb = lb_line[0] * x_1 + lb_line[1]
y_ub = ub_line[0] * x_1 + ub_line[1]

plt.plot(x_1, y_lb)
plt.plot(x_1, y_ub)

plt.tight_layout()
plt.show()

# import pdb
# pdb.set_trace()