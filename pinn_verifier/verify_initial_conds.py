import copy
import argparse

from tqdm import tqdm
import numpy as np
import torch

import gurobipy as grb
import tools.bab_tools.vnnlib_utils as vnnlib_utils


torch.manual_seed(43)

parser = argparse.ArgumentParser()
parser.add_argument('--network-filename', required=True, type=str, help='onnx file to load.')
args = parser.parse_args()

assert args.network_filename.endswith(".onnx")
model, in_shape, out_shape, dtype, model_correctness = vnnlib_utils.onnx_to_pytorch(args.network_filename)
model.eval()

if not model_correctness:
    raise ValueError(
        "model has not been loaded successfully; some operations are not compatible, please check manually"
    )

# should fail if some maxpools are actually in the network
layers = vnnlib_utils.remove_maxpools(copy.deepcopy(list(model.children())), [], dtype=dtype)

# Skip all gradient computation for the weights of the Net
for param in model.parameters():
    param.requires_grad = False

def interval_propagation(layers, bounds):
    # interval analysis
    current_lb = bounds[0]
    current_ub = bounds[1]
    for layer in layers:
        if isinstance(layer, torch.nn.Linear):
            # bound it from the top and bottom
            pos_weights = torch.clamp(layer.weight, min=0)
            neg_weights = torch.clamp(layer.weight, max=0)

            new_layer_lb = pos_weights @ current_lb + neg_weights @ current_ub + layer.bias
            new_layer_ub = pos_weights @ current_ub + neg_weights @ current_lb + layer.bias
            
            current_lb = new_layer_lb
            current_ub = new_layer_ub
        elif isinstance(layer, torch.nn.Softplus):
            # monotonicity of softplus means we can simply apply the operation to the bounds
            current_lb = layer(current_lb.unsqueeze(0)).squeeze(0)
            current_ub = layer(current_ub.unsqueeze(0)).squeeze(0)
        else:
            # it's an Add, Mul, or Div: apply directly
            current_lb = layer(current_lb.unsqueeze(0)).squeeze(0)
            current_ub = layer(current_ub.unsqueeze(0)).squeeze(0)
    
    return current_lb, current_ub

boundary_conditions = torch.tensor([[0, -1], [1, -1]])
print("using interval propagation (boundary x=-1): ", interval_propagation(layers, boundary_conditions))

boundary_conditions = torch.tensor([[0, 1], [1, 1]])
print("using interval propagation (boundary x=1): ", interval_propagation(layers, boundary_conditions))




softplus = torch.nn.Softplus()
def softplus_derivative(x):
    return 1/ (1 + torch.exp(-x))

def compute_lower_upper_bounds_lines(lb, ub, lb_line_ub_bias=0.65):
    lb_line = [0, 0]
    ub_line = [0, 0]

    ub_line[0] = (softplus(ub) - softplus(lb)) / (ub - lb)
    ub_line[1] = softplus(ub) - ub_line[0] * ub

    # tangent line at point d
    d = lb_line_ub_bias * ub + (1 - lb_line_ub_bias) * lb
    lb_line[0] = softplus_derivative(d)
    lb_line[1] = softplus(d) - lb_line[0] * d

    return lb_line, ub_line

def compute_upper_bound_pwl(lb, ub, ub_bias=0.5):
    first_half = [0, 0]
    second_half = [0, 0]

    d = ub_bias * ub + (1 - ub_bias) * lb
    first_half[0] = (softplus(d) - softplus(lb)) / (d - lb)
    first_half[1] = softplus(d) - first_half[0] * d

    second_half[0] = (softplus(ub) - softplus(d)) / (ub - d)
    second_half[1] = softplus(d) - second_half[0] * d

    return first_half, second_half


boundary_conditions = torch.tensor([[0, -1], [1, -1]])

# apply first few layers on the bounds
for layer in layers[:4]:
    boundary_conditions = layer(boundary_conditions)
boundary_conditions = boundary_conditions.T

gurobi_vars = []
lower_bounds = []
upper_bounds = []

gurobi_env = grb.Env()
grb_model = grb.Model(env=gurobi_env)
grb_model.setParam('OutputFlag', False)
grb_model.setParam('Threads', 1)
grb_model.setParam('NonConvex', 2)

# initial conditions
input_vars = []
input_lbs = []
input_ubs = []
for idx, (lb, ub) in enumerate(boundary_conditions):
    v = grb_model.addVar(lb=lb, ub=ub, obj=0, vtype=grb.GRB.CONTINUOUS, name=f'inp_{idx}')
    input_vars.append(v)
    input_lbs.append(lb)
    input_ubs.append(ub)

grb_model.update()

gurobi_vars.append(input_vars)
lower_bounds.append(torch.tensor(input_lbs, dtype=dtype))
upper_bounds.append(torch.tensor(input_ubs, dtype=dtype))

# Other layers
for layer_idx, layer in enumerate(tqdm(layers[4:])):
    is_final = (layer_idx == len(layers) - 1)
    new_layer_lb = []
    new_layer_ub = []
    new_layer_gurobi_vars = []

    pre_vars = gurobi_vars[-1]
    pre_lb = lower_bounds[-1]
    pre_ub = upper_bounds[-1]
    if isinstance(layer, torch.nn.Linear):
        pos_w = torch.clamp(layer.weight, min=0)
        neg_w = torch.clamp(layer.weight, max=0)

        out_lbs = pos_w @ pre_lb + neg_w @ pre_ub + layer.bias
        out_ubs = pos_w @ pre_ub + neg_w @ pre_lb + layer.bias

        for neuron_idx in range(layer.weight.size(0)):
            lin_expr = layer.bias[neuron_idx].item()
            coeffs = layer.weight[neuron_idx, :]
            lin_expr += grb.LinExpr(coeffs, pre_vars)

            out_lb = out_lbs[neuron_idx].item()
            out_ub = out_ubs[neuron_idx].item()

            v = grb_model.addVar(
                lb=out_lb, ub=out_ub,
                obj=0, vtype=grb.GRB.CONTINUOUS,
                name=f'lay_{layer_idx}_{neuron_idx}'
            )
            grb_model.addConstr(v == lin_expr)

            if layer_idx >= 1:
                grb_model.setObjective(v, grb.GRB.MAXIMIZE)
                grb_model.update()
                grb_model.reset()
                grb_model.optimize()

                out_ub = v.X

                grb_model.setObjective(v, grb.GRB.MINIMIZE)
                grb_model.update()
                grb_model.reset()
                grb_model.optimize()

                if grb_model.status == 3:
                    import pdb
                    pdb.set_trace()

                out_lb = v.X

            new_layer_lb.append(out_lb)
            new_layer_ub.append(out_ub)
            new_layer_gurobi_vars.append(v)
    elif isinstance(layer, torch.nn.Softplus):
        for neuron_idx, pre_var in enumerate(pre_vars):
            pre_lb_neuron = pre_lb[neuron_idx]
            pre_ub_neuron = pre_ub[neuron_idx]
            
            v = grb_model.addVar(
                lb=softplus(pre_lb_neuron), ub=softplus(pre_ub_neuron),
                obj=0, vtype=grb.GRB.CONTINUOUS,
                name=f'Softplus_{layer_idx}_{neuron_idx}'
            )

            lb_line, ub_line = compute_lower_upper_bounds_lines(pre_lb_neuron, pre_ub_neuron, lb_line_ub_bias=0.1)
            grb_model.addConstr(v >= lb_line[0].item() * pre_var + lb_line[1].item())

            lb_line, ub_line = compute_lower_upper_bounds_lines(pre_lb_neuron, pre_ub_neuron, lb_line_ub_bias=0.25)
            grb_model.addConstr(v >= lb_line[0].item() * pre_var + lb_line[1].item())

            lb_line, ub_line = compute_lower_upper_bounds_lines(pre_lb_neuron, pre_ub_neuron, lb_line_ub_bias=0.5)
            grb_model.addConstr(v >= lb_line[0].item() * pre_var + lb_line[1].item())

            lb_line, ub_line = compute_lower_upper_bounds_lines(pre_lb_neuron, pre_ub_neuron, lb_line_ub_bias=0.75)
            grb_model.addConstr(v >= lb_line[0].item() * pre_var + lb_line[1].item())

            lb_line, ub_line = compute_lower_upper_bounds_lines(pre_lb_neuron, pre_ub_neuron, lb_line_ub_bias=0.9)
            grb_model.addConstr(v >= lb_line[0].item() * pre_var + lb_line[1].item())

            # line = ub_line[0].item() * pre_var + ub_line[1].item()
            # grb_model.addConstr(v <= line)

            # quad = 0.1 * pre_var * pre_var + 0.5 * pre_var + 0.85
            # grb_model.addConstr(v <= quad)

            # piecewise linear constraints
            first_ub_half, second_ub_half = compute_upper_bound_pwl(pre_lb_neuron, pre_ub_neuron, ub_bias=0.5)
            max_v = grb_model.addVar(
                lb=softplus(pre_lb_neuron), ub=softplus(pre_ub_neuron),
                obj=0, vtype=grb.GRB.CONTINUOUS,
                name=f'Softplus_max_{layer_idx}_{neuron_idx}'
            )
            first_half_line = grb_model.addVar(
                lb=softplus(pre_lb_neuron), ub=softplus(pre_ub_neuron),
                obj=0, vtype=grb.GRB.CONTINUOUS,
                name=f'Softplus_pwl_first_{layer_idx}_{neuron_idx}'
            )
            second_half_line = grb_model.addVar(
                lb=softplus(pre_lb_neuron), ub=softplus(pre_ub_neuron),
                obj=0, vtype=grb.GRB.CONTINUOUS,
                name=f'Softplus_pwl_second_{layer_idx}_{neuron_idx}'
            )
            grb_model.addConstr(first_half_line == first_ub_half[0].item() * pre_var + first_ub_half[1].item())
            grb_model.addConstr(second_half_line == second_ub_half[0].item() * pre_var + second_ub_half[1].item())
            grb_model.addConstr(max_v == grb.max_([first_half_line, second_half_line]))
            grb_model.addConstr(v <= max_v)

            # if not (pre_lb_neuron < -2 and pre_ub_neuron < -2) and not (pre_lb_neuron > 2 and pre_ub_neuron > 2):
            #     quad = 0.065 * pre_var * pre_var + 0.5 * pre_var + 1.10
            #     grb_model.addConstr(v <= quad)

            # import pdb
            # pdb.set_trace()

            # compute the new lower bound and upper bound using the exact function
            out_lb = softplus(pre_lb_neuron)
            out_ub = softplus(pre_ub_neuron)

            new_layer_lb.append(out_lb)
            new_layer_ub.append(out_ub)
            new_layer_gurobi_vars.append(v)

    grb_model.update()
    lower_bounds.append(torch.tensor(new_layer_lb))
    upper_bounds.append(torch.tensor(new_layer_ub))
    gurobi_vars.append(new_layer_gurobi_vars)

# setup the rest of the optimization; maximize and minimize: take the max absolute value
grb_model.setObjective(gurobi_vars[-1][0], grb.GRB.MAXIMIZE)
grb_model.update()
grb_model.reset()

grb_model.optimize()
delta_b_max = gurobi_vars[-1][0].X

grb_model.setObjective(gurobi_vars[-1][0], grb.GRB.MINIMIZE)
grb_model.update()
grb_model.reset()

grb_model.optimize()
delta_b_min = gurobi_vars[-1][0].X

delta_b = max(np.abs(delta_b_max), np.abs(delta_b_min))

print(delta_b)

import pdb
pdb.set_trace()
