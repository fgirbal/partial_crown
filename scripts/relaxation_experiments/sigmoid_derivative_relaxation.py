from mailbox import _singlefileMailbox
import matplotlib.pyplot as plt
import torch.nn
from scipy import optimize
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-lb', type=float, default=-3.5)
parser.add_argument('-ub', type=float, default=-0.5)
args = parser.parse_args()

softplus = torch.nn.Softplus()
sigmoid = torch.nn.Sigmoid()

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

def sigmoid_derivative_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x)) * (1 - 2 * sigmoid(x))

def compute_lower_upper_bounds_lines(lb, ub, ub_line_ub_bias=0.9, lb_line_ub_bias=0.2):
    # def np_sigmoid(x):
    #     return 1 / (1 + np.exp(-x))

    # def np_sigmoid_derivative(x):
    #     return np_sigmoid(x) * (1 - np_sigmoid(x))

    # def sigmoid_derivative_bound_d(x, bound):
    #     return np_sigmoid_derivative(x) * (1 - np_sigmoid_derivative(x)) - (np_sigmoid_derivative(x) - np_sigmoid_derivative(bound)) / (x - bound) 

    ub_lines = []
    lb_lines = []

    assert lb <= ub

    first_convex_region = [-np.Inf, -1.3169559128480408]
    concave_region = [-1.3169559128480408, 1.3169559080930038]
    second_convex_region = [1.3169559080930038, np.Inf]

    b_intersect = 0.3
    min_lb_m = -(sigmoid_derivative(torch.tensor(first_convex_region[1])).item() + b_intersect)/(first_convex_region[1])

    def in_region(p, region):
        return p >= region[0] and p <= region[1]

    if (in_region(lb, first_convex_region) and in_region(ub, first_convex_region)) or (in_region(lb, second_convex_region) and in_region(ub, second_convex_region)):
        # in this location, the function is convex, use the same bounds as in softplus case
    
        # tangent line at point d_1
        d_1 = lb_line_ub_bias * ub + (1 - lb_line_ub_bias) * lb
        lb_m = sigmoid_derivative_derivative(d_1)
        lb_b = sigmoid_derivative(d_1) - lb_m * d_1
        lb_lines.append((lb_m, lb_b))

        # ub line just connects upper and lower bound points
        ub_m = (sigmoid_derivative(ub) - sigmoid_derivative(lb)) / (ub - lb)
        ub_b = sigmoid_derivative(ub) - ub_m * ub
        ub_lines.append((ub_m, ub_b))
    elif in_region(lb, concave_region) and in_region(ub, concave_region):
        # in this location, the function is concave, use the inverted bounds from softplus case

        # lb line just connects upper and lower bound points
        lb_m = (sigmoid_derivative(ub) - sigmoid_derivative(lb)) / (ub - lb)
        lb_b = sigmoid_derivative(ub) - lb_m * ub
        lb_lines.append((lb_m, lb_b))

        # tangent line at point d_1
        d_1 = ub_line_ub_bias * ub + (1 - ub_line_ub_bias) * lb
        ub_m = sigmoid_derivative_derivative(d_1)
        ub_b = sigmoid_derivative(d_1) - ub_m * d_1
        ub_lines.append((ub_m, ub_b))
    else:
        # points are in different regions; 
        # are they in the first convex and the concave regions?
        if in_region(lb, first_convex_region) and in_region(ub, concave_region):
            if -lb >= ub:
                lb_m = min(sigmoid_derivative_derivative(lb), (sigmoid_derivative(ub) - sigmoid_derivative(lb)) / (ub - lb))
                lb_b = sigmoid_derivative(lb) - lb_m * lb
                lb_lines.append((lb_m, lb_b))
            else:
                lb_m = max(sigmoid_derivative_derivative(ub), (sigmoid_derivative(ub) - sigmoid_derivative(lb)) / (ub - lb))
                lb_b = sigmoid_derivative(ub) - lb_m * ub
                lb_lines.append((lb_m, lb_b))

            # ub_lines
            ub_m = ((sigmoid_derivative(lb) - b_intersect)/lb).item()
            ub_b = sigmoid_derivative(lb) - ub_m * lb
            ub_lines.append((ub_m, ub_b))

            if ub > 0:
                ub_m = ((sigmoid_derivative(ub) - b_intersect)/ub).item()
                ub_b = sigmoid_derivative(ub) - ub_m * ub
                ub_lines.append((ub_m, ub_b))
        elif in_region(lb, concave_region) and in_region(ub, second_convex_region):
            if -lb >= ub:
                lb_m = min(sigmoid_derivative_derivative(lb), (sigmoid_derivative(ub) - sigmoid_derivative(lb)) / (ub - lb))
                lb_b = sigmoid_derivative(lb) - lb_m * lb
                lb_lines.append((lb_m, lb_b))
            else:
                lb_m = max(sigmoid_derivative_derivative(ub), (sigmoid_derivative(ub) - sigmoid_derivative(lb)) / (ub - lb))
                lb_b = sigmoid_derivative(ub) - lb_m * ub
                lb_lines.append((lb_m, lb_b))

            # ub_lines
            if lb < 0:
                ub_m = ((sigmoid_derivative(lb) - b_intersect)/lb).item()
                ub_b = sigmoid_derivative(lb) - ub_m * lb
                ub_lines.append((ub_m, ub_b))

            ub_m = ((sigmoid_derivative(ub) - b_intersect)/ub).item()
            ub_b = sigmoid_derivative(ub) - ub_m * ub
            ub_lines.append((ub_m, ub_b))
        # are they in the two convex regions?
        elif in_region(lb, first_convex_region) and in_region(ub, second_convex_region):
            # lb should be a single line, no benefit of more than one
            if -lb >= ub:
                lb_m = min(sigmoid_derivative_derivative(lb), (sigmoid_derivative(ub) - sigmoid_derivative(lb)) / (ub - lb))
                lb_b = sigmoid_derivative(lb) - lb_m * lb
                lb_lines.append((lb_m, lb_b))
            else:
                lb_m = max(sigmoid_derivative_derivative(ub), (sigmoid_derivative(ub) - sigmoid_derivative(lb)) / (ub - lb))
                lb_b = sigmoid_derivative(ub) - lb_m * ub
                lb_lines.append((lb_m, lb_b))

            # ub_lines
            ub_m = ((sigmoid_derivative(lb) - b_intersect)/lb).item()
            ub_b = sigmoid_derivative(lb) - ub_m * lb
            ub_lines.append((ub_m, ub_b))

            ub_m = ((sigmoid_derivative(ub) - b_intersect)/ub).item()
            ub_b = sigmoid_derivative(ub) - ub_m * ub
            ub_lines.append((ub_m, ub_b))

    return lb_lines, ub_lines

from matplotlib.widgets import Slider, Button

fig, ax = plt.subplots()

x = torch.linspace(-5, 5, 1000)
y = sigmoid_derivative(x)
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
    y = sigmoid_derivative(x)
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
    ax.set_ylim([-0.05, 0.6])

    # fig.canvas.draw_idle()


# register the update function with each slider
ub_slider.on_changed(update)
lb_slider.on_changed(update)

# plt.plot(x_1, y_lb)
# plt.plot(x_1, y_ub)

ax.set_xlim([-5.5, 5.5])
ax.set_ylim([-0.05, 0.6])

# plt.tight_layout()
plt.show()

# import pdb
# pdb.set_trace()