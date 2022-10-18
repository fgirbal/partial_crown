import matplotlib.pyplot as plt
import torch.nn

softplus = torch.nn.Softplus()
def softplus_derivative(x):
    return 1/ (1 + torch.exp(-x))

def compute_lower_upper_bounds_lines(lb, ub, ub_bias=0.65):
    lb_line = [0, 0]
    ub_line = [0, 0]

    ub_line[0] = (softplus(ub) - softplus(lb)) / (ub - lb)
    ub_line[1] = softplus(ub) - ub_line[0] * ub

    # tangent line at point d
    d = ub_bias * ub + (1 - ub_bias) * lb
    lb_line[0] = softplus_derivative(d)
    lb_line[1] = softplus(d) - lb_line[0] * d

    return lb_line, ub_line

def compute_quadratic_upper_bound(lb, ub, ub_bias=0.5):
    x_1, y_1 = lb, softplus(lb)
    d = ub_bias * ub + (1 - ub_bias) * lb
    x_2, y_2 = d, softplus(d)
    x_3, y_3 = ub, softplus(ub)

    a = (y_2 - y_3 - (x_2 - x_3) * (y_1 - y_2) / (x_1 - x_2)) / (x_2**2 - x_3**2 - (x_2 - x_3) * (x_1**2 - x_2**2) / (x_1 - x_2))
    b = (y_1 - y_2 - (x_1**2 - x_2**2) * a) / (x_1 - x_2)
    c = y_1 - a * x_1**2 - b * x_1

    return a, b, c

def compute_upper_bound_pwl(lb, ub, ub_bias=0.5):
    first_half = [0, 0]
    second_half = [0, 0]

    d = ub_bias * ub + (1 - ub_bias) * lb
    first_half[0] = (softplus(d) - softplus(lb)) / (d - lb)
    first_half[1] = softplus(d) - first_half[0] * d

    second_half[0] = (softplus(ub) - softplus(d)) / (ub - d)
    second_half[1] = softplus(d) - second_half[0] * d

    return first_half, second_half


x = torch.linspace(-5, 5, 1000)
y = torch.nn.Softplus()(x)

plt.plot(x, y)

lb = torch.tensor(-3, dtype=torch.float)
ub = torch.tensor(2.0, dtype=torch.float)
x_1 = torch.linspace(lb, ub, 250)

lb_line, ub_line = compute_lower_upper_bounds_lines(lb, ub, ub_bias=0.1)
y_lb = lb_line[0] * x_1 + lb_line[1]

ub_bias = 0.5
first_ub_half, second_ub_half = compute_upper_bound_pwl(lb, ub, ub_bias=ub_bias)
d = ub_bias * ub + (1 - ub_bias) * lb
y_ub_1 = first_ub_half[0] * torch.linspace(lb, d, 125) + first_ub_half[1]
y_ub_2 = second_ub_half[0] * torch.linspace(d, ub, 125) + second_ub_half[1]

# ub_quadratic = compute_quadratic_upper_bound(lb, ub, ub_bias=0.5)
# y_ub = ub_quadratic[0] * x_1**2 + ub_quadratic[1] * x_1 + ub_quadratic[2]

# plt.plot(x_1, y_lb)
plt.plot(torch.linspace(lb, d, 125), y_ub_1)
plt.plot(torch.linspace(d, ub, 125), y_ub_2)

lb_line, ub_line = compute_lower_upper_bounds_lines(lb, ub, ub_bias=0.5)
y_lb = lb_line[0] * x_1 + lb_line[1]
# plt.plot(x_1, y_lb)

lb_line, ub_line = compute_lower_upper_bounds_lines(lb, ub, ub_bias=0.9)
y_lb = lb_line[0] * x_1 + lb_line[1]
# plt.plot(x_1, y_lb)

plt.tight_layout()
plt.show()

# import pdb
# pdb.set_trace()