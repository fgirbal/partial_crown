import argparse
from multiprocessing.sharedctypes import Value

import matplotlib.pyplot as plt
import torch.nn
from scipy import optimize
import numpy as np

tanh = torch.nn.Tanh()

def tanh_derivative(x):
    return 1 - tanh(x)**2

def tanh_second_derivative(x):
    return -2 * tanh(x) * (1 - tanh(x)**2)

def tanh_third_derivative(x):
    return -2 + 8 * tanh(x)**2 - 6 * tanh(x)**4

def in_region(p, region):
    return p >= region[0] and p <= region[1]

def np_tanh(x):
    return np.tanh(x)

def np_tanh_derivative(x):
    return 1 - np_tanh(x)**2

def np_tanh_second_derivative(x):
    return -2* np_tanh(x) * (1 - np_tanh(x)**2)

def np_tanh_third_derivative(x):
    return -2 + 8 * np_tanh(x)**2 - 6 * np_tanh(x)**4

result = optimize.minimize(lambda x: -np_tanh_third_derivative(x), x0=-1.146, bounds=[(-5, -1)])
x_val = result.x.item()
first_convex_region = [-np.Inf, x_val]
first_concave_region = [x_val, 0]

# the minimum intersection should be the one that touches the inflextion point
b_intersect = np_tanh_second_derivative(x_val) - np_tanh_third_derivative(x_val) * x_val
result = optimize.minimize(lambda x: -np_tanh_second_derivative(x), x0=-0.66, bounds=[(-5, 0)])
first_mound_lhs = np_tanh_third_derivative(x_val) * result.x + b_intersect
first_mound_rhs = np_tanh_third_derivative(0) * result.x
first_mound_x = result.x.item()

result = optimize.minimize(lambda x: -np_tanh_third_derivative(x), x0=1.146, bounds=[(1, 5)])
x_val = result.x.item()
second_convex_region = [0, x_val]
second_concave_region = [x_val, np.Inf]

b_intersect = np_tanh_second_derivative(x_val) - np_tanh_third_derivative(x_val) * x_val
result = optimize.minimize(lambda x: -np_tanh_second_derivative(x), x0=0.66, bounds=[(0, 5)])
second_mound_lhs = -first_mound_rhs
second_mound_rhs = -first_mound_lhs
second_mound_x = -first_mound_x

def lb_in_convex_ub_in_concave(lb, ub, split_point, fn, fn_derivative):
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
        # print(split_point)
        ub_line_m = min((fn(ub) - fn(lb)) / (ub - lb), fn_derivative(ub))
        ub_line_b = fn(ub) - ub_line_m * ub

        # if ub <= split_point:
        #     print('ub <= split_point')
        #     ub_line_m = (fn(ub) - fn(lb)) / (ub - lb)
        #     ub_line_b = fn(ub) - ub_line_m * ub
        # else:
        #     print('ub > split_point')
            # d_1 = (lb + ub) / 2
            # ub_line_m = fn_derivative(d_1)
            # ub_line_b = fn(d_1) - ub_line_m * d_1

    return (lb_line_m, lb_line_b), (ub_line_m, ub_line_b)


def lb_in_concave_ub_in_convex(lb, ub, split_point, fn, fn_derivative):
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

        # if lb <= split_point - 1e-2:
        #     lb_line_m = (fn(ub) - fn(lb)) / (ub - lb)
        #     lb_line_b = fn(ub) - lb_line_m * ub
        # else:
        #     d_1 = (lb + ub) / 2
        #     lb_line_m = fn_derivative(d_1)
        #     lb_line_b = fn(d_1) - lb_line_m * d_1

    if d_ub <= split_point:
        # tangent line at point d_ub
        ub_line_m = fn_derivative(d_ub)
        ub_line_b = fn(ub) - ub_line_m * ub
    else:
        # ub line just connects upper and lower bound points
        ub_line_m = (fn(ub) - fn(lb)) / (ub - lb)
        ub_line_b = fn(ub) - ub_line_m * ub

    return (lb_line_m, lb_line_b), (ub_line_m, ub_line_b)







def compute_lower_upper_bounds_lines(lb, ub, ub_line_ub_bias=0.9, lb_line_ub_bias=0.2):
    lb_lines = []
    ub_lines = []

    assert lb <= ub

    if (in_region(lb, first_convex_region) and in_region(ub, first_convex_region)) or (in_region(lb, second_convex_region) and in_region(ub, second_convex_region)):
        # in this location, the function is convex, use the same bounds as in softplus case
    
        # tangent line at point d_1
        d_1 = lb_line_ub_bias * ub + (1 - lb_line_ub_bias) * lb
        lb_m = np_tanh_third_derivative(d_1)
        lb_b = tanh_second_derivative(d_1) - lb_m * d_1
        lb_lines.append((lb_m, lb_b))

        # ub line just connects upper and lower bound points
        ub_m = (tanh_second_derivative(ub) - tanh_second_derivative(lb)) / (ub - lb)
        ub_b = tanh_second_derivative(ub) - ub_m * ub
        ub_lines.append((ub_m, ub_b))
    elif (in_region(lb, first_concave_region) and in_region(ub, first_concave_region)) or (in_region(lb, second_concave_region) and in_region(ub, second_concave_region)):
        # in this location, the function is concave, use the inverted bounds from softplus case

        # lb line just connects upper and lower bound points
        lb_m = (tanh_second_derivative(ub) - tanh_second_derivative(lb)) / (ub - lb)
        lb_b = tanh_second_derivative(ub) - lb_m * ub
        lb_lines.append((lb_m, lb_b))

        # tangent line at point d_1
        d_1 = ub_line_ub_bias * ub + (1 - ub_line_ub_bias) * lb
        ub_m = np_tanh_third_derivative(d_1)
        ub_b = tanh_second_derivative(d_1) - ub_m * d_1
        ub_lines.append((ub_m, ub_b))
    else:
        # points are in different regions; 
        # they are in adjacent regions (3 cases)
        if in_region(lb, first_convex_region) and in_region(ub, first_concave_region):
            lb_line, ub_line = lb_in_convex_ub_in_concave(lb, ub, first_concave_region[0], tanh_second_derivative, tanh_third_derivative)

            lb_lines.append(lb_line)
            ub_lines.append(ub_line)
        elif in_region(lb, second_convex_region) and in_region(ub, second_concave_region):
            lb_line, ub_line = lb_in_convex_ub_in_concave(lb, ub, second_concave_region[0], tanh_second_derivative, tanh_third_derivative)

            lb_lines.append(lb_line)
            ub_lines.append(ub_line)
        elif in_region(lb, first_concave_region) and in_region(ub, second_convex_region):
            lb_line, ub_line = lb_in_concave_ub_in_convex(lb, ub, second_convex_region[0], tanh_second_derivative, tanh_third_derivative)

            lb_lines.append(lb_line)
            ub_lines.append(ub_line)
        # they are in non-adjoint regions (3 cases)
        elif in_region(lb, first_convex_region) and in_region(ub, second_convex_region):
            lb_line_right, _ = lb_in_concave_ub_in_convex(lb, ub, first_concave_region[1], tanh_second_derivative, tanh_third_derivative)
            lb_lines.append(lb_line_right)

            _, ub_line_left = lb_in_convex_ub_in_concave(lb, torch.tensor([first_mound_x]), first_concave_region[0], tanh_second_derivative, tanh_third_derivative)
            _, ub_line_right = lb_in_concave_ub_in_convex(torch.tensor([first_mound_x]), ub, first_concave_region[1], tanh_second_derivative, tanh_third_derivative)

            ub_lines.append(ub_line_left)
            ub_lines.append(ub_line_right)
        elif in_region(lb, first_concave_region) and in_region(ub, second_concave_region):
            lb_line_left, _ = lb_in_concave_ub_in_convex(lb, ub, second_convex_region[0], tanh_second_derivative, tanh_third_derivative)
            lb_line_right, _ = lb_in_convex_ub_in_concave(torch.tensor([second_mound_x]), ub, second_convex_region[1], tanh_second_derivative, tanh_third_derivative)

            lb_lines.append(lb_line_left)
            lb_lines.append(lb_line_right)

            _, ub_line = lb_in_concave_ub_in_convex(lb, ub, second_convex_region[0], tanh_second_derivative, tanh_third_derivative)
            ub_lines.append(ub_line)
        elif in_region(lb, first_convex_region) and in_region(ub, second_concave_region):
            lb_line_left, _ = lb_in_concave_ub_in_convex(lb, ub, second_convex_region[0], tanh_second_derivative, tanh_third_derivative)
            lb_line_right, _ = lb_in_convex_ub_in_concave(torch.tensor([second_mound_x]), ub, second_convex_region[1], tanh_second_derivative, tanh_third_derivative)

            lb_lines.append(lb_line_left)
            lb_lines.append(lb_line_right)
            
            _, ub_line_left = lb_in_convex_ub_in_concave(lb, torch.tensor([first_mound_x]), first_concave_region[0], tanh_second_derivative, tanh_third_derivative)
            _, ub_line_right = lb_in_concave_ub_in_convex(torch.tensor([first_mound_x]), ub, first_concave_region[1], tanh_second_derivative, tanh_third_derivative)

            ub_lines.append(ub_line_left)
            ub_lines.append(ub_line_right)
        else:
            raise ValueError("should not happen")

    return lb_lines, ub_lines


from matplotlib.widgets import Slider, Button

parser = argparse.ArgumentParser()
parser.add_argument('-lb', type=float, default=-0.9)
parser.add_argument('-ub', type=float, default=2.8)
args = parser.parse_args()

paper_mode = True
fn_min = -1
fn_max = 1
x_min = -5
x_max = 5
fn_call = tanh_second_derivative

if not paper_mode:
    fig, ax = plt.subplots()

    x = torch.linspace(x_min, x_max, 1000)
    y = fn_call(x)
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

        x = torch.linspace(x_min, x_max, 1000)
        y = fn_call(x)
        ax.plot(x, y, c="b")

        x_1 = torch.linspace(torch.tensor(lb_slider.val), torch.tensor(ub_slider.val), 250)

        lb_lines, ub_lines = compute_lower_upper_bounds_lines(torch.tensor(lb_slider.val), torch.tensor(ub_slider.val), lb_line_ub_bias=0.4, ub_line_ub_bias=0.5)

        for lb_line in lb_lines:
            y_lb = lb_line[0] * x_1 + lb_line[1]
            ax.plot(x_1, y_lb, c='r')

        for ub_line in ub_lines:
            y_ub = ub_line[0] * x_1 + ub_line[1]
            ax.plot(x_1, y_ub, c='g')

        if len(ub_lines) >= 2:
            y_ub = sum([m for m, b in ub_lines])/len(ub_lines) * x_1 + sum([b for m, b in ub_lines])/len(ub_lines)
            ax.plot(x_1, y_ub, c='y')

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

else:
    # paper mode
    import seaborn as sns
    sns.set_theme()
    sns.set(font_scale=1.7, rc={'text.usetex' : True})
    # sns.set(font_scale=1.6)

    lw = 2
    cp = sns.color_palette()

    fig, ax = plt.subplots(figsize=(6, 4))

    x = torch.linspace(x_min, x_max, 1000)
    y = fn_call(x)
    ax.plot(x, y, c=cp[0], lw=lw, label=r"$\sigma''$", zorder=2)

    lb = torch.tensor(args.lb, dtype=torch.float64)
    ub = torch.tensor(args.ub, dtype=torch.float64)
    x_1 = torch.linspace(lb, ub, 250).to(torch.float64)

    lb_lines, ub_lines = compute_lower_upper_bounds_lines(lb, ub, lb_line_ub_bias=0.4, ub_line_ub_bias=0.5)

    for ith, lb_line in enumerate(lb_lines):
        y_lb = lb_line[0] * x_1 + lb_line[1] - 1e-4
        if ith == 0:
            ax.plot(x_1, y_lb, c=cp[1], lw=lw, label=r"$h^L$", zorder=2)
        else:
            ax.plot(x_1, y_lb, c=cp[1], lw=lw, zorder=2)

    for ith, ub_line in enumerate(ub_lines):
        y_ub = ub_line[0] * x_1 + ub_line[1] + 1e-4
        if ith == 0:
            ax.plot(x_1, y_ub, c=cp[2], lw=lw, label=r"$h^U$", zorder=2)
        else:
            ax.plot(x_1, y_ub, c=cp[2], lw=lw, zorder=2)

    lb_bias = (first_mound_x - lb)/(-lb + ub)
    biases = [lb_bias, 1-lb_bias]
    if len(ub_lines) >= 2:
        y_ub = sum([bias*m for (m, b), bias in zip(ub_lines, biases)]) * x_1 + sum([bias*b for (m, b), bias in zip(ub_lines, biases)])
        ax.plot(x_1, y_ub, c=cp[5], lw=lw, label=r"$h^{U, \alpha}$", zorder=2)

    lb_bias = (second_mound_x - lb)/(-lb + ub)
    biases = [lb_bias, 1-lb_bias]
    if len(lb_lines) >= 2:
        y_lb = sum([bias*m for (m, b), bias in zip(lb_lines, biases)]) * x_1 + sum([bias*b for (m, b), bias in zip(lb_lines, biases)])
        ax.plot(x_1, y_lb, c=cp[6], lw=lw, label=r"$h^{L, \alpha}$", zorder=2)

    ax.fill_between(x_1, y_lb, y_ub, facecolor=cp[3], alpha=0.1, zorder=1)
    ax.legend(loc='upper right')
    ax.set_xlabel(r"$y$")

    def plot_regions(ax, list_of_limits, plot_limits):
        for i, (x_min, x_max) in enumerate(list_of_limits):
            if x_min == -np.infty:
                x_min = plot_limits[0]
            
            if x_max == np.infty:
                x_max = plot_limits[1]
            else:
                ax.plot([x_max, x_max], [fn_min-0.5, fn_max+0.5], c=cp[7], linestyle='dashed', lw=1.3, zorder=1)
            
            ax.text((x_min + x_max)/2, fn_min + 0.1, r"$\mathcal{{R}}_{}$".format(i+1), fontsize=16, ha='center', va='center', c=cp[7])

    plot_regions(ax, [first_convex_region, first_concave_region, second_convex_region, second_concave_region], [x_min, x_max])

    ax.set_xlim([x_min - (x_max - x_min)*0.05, x_max + (x_max - x_min)*0.05])
    ax.set_ylim([fn_min, fn_max])

    fig.tight_layout()
    plt.subplots_adjust(left=0.10, top=0.95, bottom=0.18, right=0.97)
    plt.show()

# import pdb
# pdb.set_trace()