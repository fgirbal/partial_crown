import argparse

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

    return [lb_line], [ub_line]


from matplotlib.widgets import Slider, Button

parser = argparse.ArgumentParser()
parser.add_argument('-lb', type=float, default=-3.5)
parser.add_argument('-ub', type=float, default=-0.5)
args = parser.parse_args()

fn_min = -0.1
fn_max = 1.1

fig, ax = plt.subplots()

x = torch.linspace(-5, 5, 1000)
y = sigmoid(x)
plt.plot(x, y, c="b")

lb = torch.tensor(args.lb, dtype=torch.float)
ub = torch.tensor(args.ub, dtype=torch.float)
x_1 = torch.linspace(lb, ub, 250)

lb_lines, ub_lines = compute_lower_upper_bounds_lines(lb, ub, lb_line_ub_bias=0.4, ub_line_ub_bias=0.5)

for lb_line in lb_lines:
    y_lb = lb_line[0] * x_1 + lb_line[1]
    plt.plot(x_1, y_lb, c='r')

for ub_line in ub_lines:
    y_ub = ub_line[0] * x_1 + ub_line[1]
    plt.plot(x_1, y_ub, c='g')

plt.subplots_adjust(left=0.1, bottom=0.35, right=0.9)

ax_ub = plt.axes([0.1, 0.15, 0.8, 0.03])
ub_slider = Slider(
    ax=ax_ub,
    label='ub',
    valmin=-5,
    valmax=5,
    valinit=args.ub,
)

ax_lb = plt.axes([0.1, 0.05, 0.8, 0.03])
lb_slider = Slider(
    ax=ax_lb,
    label='lb',
    valmin=-5,
    valmax=5,
    valinit=args.lb,
)

# The function to be called anytime a slider's value changes
def update(val):
    for _ in enumerate(ax.lines):
        ax.lines.pop(0)

    # plt.clf()

    x = torch.linspace(-5, 5, 1000)
    y = sigmoid(x)
    ax.plot(x, y, c="b")

    x_1 = torch.linspace(torch.tensor(lb_slider.val), torch.tensor(ub_slider.val), 250)

    lb_lines, ub_lines = compute_lower_upper_bounds_lines(torch.tensor(lb_slider.val), torch.tensor(ub_slider.val), lb_line_ub_bias=0.4, ub_line_ub_bias=0.5)

    for lb_line in lb_lines:
        y_lb = lb_line[0] * x_1 + lb_line[1]
        ax.plot(x_1, y_lb, c='r')

    for ub_line in ub_lines:
        y_ub = ub_line[0] * x_1 + ub_line[1]
        ax.plot(x_1, y_ub, c='g')

    ax.set_xlim([-5.5, 5.5])
    ax.set_ylim([fn_min, fn_max])

    # fig.canvas.draw_idle()


# register the update function with each slider
ub_slider.on_changed(update)
lb_slider.on_changed(update)

# plt.plot(x_1, y_lb)
# plt.plot(x_1, y_ub)

ax.set_xlim([-5.5, 5.5])
ax.set_ylim([fn_min, fn_max])

# plt.tight_layout()
plt.show()

# import pdb
# pdb.set_trace()