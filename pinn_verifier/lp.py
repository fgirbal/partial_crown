import copy
from typing import List, Tuple

from scipy import optimize
from tqdm import tqdm
import numpy as np
import torch
import gurobipy as grb

from tools.custom_torch_modules import Add, Mul
from pinn_verifier.activations.activation_relaxations import ActivationRelaxationType, ActivationRelaxation

N_THREADS = 8
supported_normalization_ops = [Add, Mul]
supported_activations = [torch.nn.Softplus, torch.nn.Tanh]


def new_gurobi_model(name, feasibility_tol=1e-6, optimality_tol=1e-6):
    grb_model = grb.Model(name)
    grb_model.setParam('OutputFlag', False)
    grb_model.setParam('Threads', N_THREADS)
    grb_model.setParam('FeasibilityTol', feasibility_tol)
    grb_model.setParam('OptimalityTol', optimality_tol)

    return grb_model


def get_multiplication_bounds(first_term_lb, first_term_ub, second_term_lb, second_term_ub):
    ll_b = first_term_lb * second_term_lb
    lu_b = first_term_lb * second_term_ub
    ul_b = first_term_ub * second_term_lb
    uu_b = first_term_ub * second_term_ub

    return min([ll_b, uu_b, lu_b, ul_b]), max([ll_b, uu_b, lu_b, ul_b])

def mccormick_relaxation(grb_model: grb.Model, input_var_1: grb.Var, input_var_2: grb.Var, output_var: grb.Var, lb_var_1: torch.tensor, ub_var_1: torch.tensor, lb_var_2: torch.tensor, ub_var_2: torch.tensor):
    margin = grb_model.params.FeasibilityTol

    grb_model.addConstr(output_var >= lb_var_1 * input_var_2 + input_var_1 * lb_var_2 - lb_var_1 * lb_var_2 - margin)
    grb_model.addConstr(output_var >= ub_var_1 * input_var_2 + input_var_1 * ub_var_2 - ub_var_1 * ub_var_2 - margin)

    grb_model.addConstr(output_var <= ub_var_1 * input_var_2 + input_var_1 * lb_var_2 - ub_var_1 * lb_var_2 + margin)
    grb_model.addConstr(output_var <= lb_var_1 * input_var_2 + input_var_1 * ub_var_2 - lb_var_1 * ub_var_2 + margin)

def add_var_to_model(grb_model: grb.Model, lb: torch.Tensor, ub: torch.Tensor, name: str):
    margin = grb_model.params.FeasibilityTol

    return grb_model.addVar(
        lb=lb - margin,
        ub=ub + margin,
        obj=0,
        vtype=grb.GRB.CONTINUOUS,
        name=name
    )

def solve_for_lower_upper_bound_var(grb_model: grb.Model, variable: grb.Var):
    grb_model.setObjective(variable, grb.GRB.MINIMIZE)
    grb_model.update()
    grb_model.reset()
    grb_model.optimize()

    while grb_model.status != 2:
        print(f"Warning: min optimization failed for variable '{variable.VarName}' and {N_THREADS} threads...")
        print(f"----- Increasing FeasibilityTol to {grb_model.params.FeasibilityTol * 10}")
        grb_model.setParam("FeasibilityTol", grb_model.params.FeasibilityTol * 10)

        grb_model.reset()
        grb_model.optimize()

        if grb_model.params.FeasibilityTol >= 1e-3:
            import pdb
            pdb.set_trace()

    out_lb = variable.X

    grb_model.setObjective(variable, grb.GRB.MAXIMIZE)
    grb_model.update()
    grb_model.reset()
    grb_model.optimize()

    while grb_model.status != 2:
        print(f"Warning: max optimization failed for variable '{variable.VarName}' and {N_THREADS} threads...")
        print(f"----- Increasing FeasibilityTol to {grb_model.params.FeasibilityTol * 10}")
        grb_model.setParam("FeasibilityTol", grb_model.params.FeasibilityTol * 10)

        grb_model.reset()
        grb_model.optimize()

        if grb_model.params.FeasibilityTol >= 1e-3:
            import pdb
            pdb.set_trace()

    out_ub = variable.X

    variable.lb = out_lb
    variable.ub = out_ub

    return out_lb, out_ub


class LPPINNSolution():
    def __init__(self, model: List[torch.nn.Module], activation_relaxation: ActivationRelaxation, feasibility_tol: float = 1e-6, optimality_tol: float = 1e-6) -> None:
        # class used to represent a LPPINNSolution
        # currently supported model is a fully connected network with a few normalization layers (Add and Mul only)
        if not self.is_model_supported(model):
            raise ValueError(
                "model passed to LPPINNSolution is not supported in current implementation"
            )
        
        self.layers = model
        self.norm_layers, self.fc_layers = self.separate_norm_and_fc_layers(model)

        self._domain_bounds = None
        self.computed_bounds = False
        self.gurobi_vars = []
        self.lower_bounds = []
        self.upper_bounds = []

        # save in object for future computations
        self.feasibility_tol = feasibility_tol
        self.optimality_tol = optimality_tol
        self.grb_model = new_gurobi_model(
            "u_theta",
            feasibility_tol=feasibility_tol,
            optimality_tol=optimality_tol
        )

        self.activation_relaxation = activation_relaxation

    def clear_bound_computations(self):
        self.computed_bounds = False
        self.gurobi_vars = []
        self.lower_bounds = []
        self.upper_bounds = []

    @property
    def domain_bounds(self):
        return self._domain_bounds

    @domain_bounds.setter
    def domain_bounds(self, domain_bounds: torch.tensor):
        # set the domain bounds and clear previous computation ahead of use in PINN expression
        assert domain_bounds.shape[0] == 2

        self._domain_bounds = domain_bounds
        self.clear_bound_computations()
    
    @staticmethod
    def is_model_supported(model: List[torch.nn.Module]) -> bool:
        norm_layers = True
        for idx, layer in enumerate(model):
            if type(layer) in supported_normalization_ops:
                # normalization layers are allowed at the beginning of the network, but not after the first linear layer
                if norm_layers:
                    continue
                else:
                    return False
            else:
                if type(layer) == torch.nn.Linear:
                    # it's a linear layer

                    if norm_layers:
                        # it's the first linear layer
                        norm_layers = False
                        continue
                    
                    # it's not the first linear layer, must check the preceding one is a supported activation
                    if type(model[idx-1]) not in supported_activations:
                        return False
                elif type(layer) in supported_activations:
                    # it's an activation layer, must check the preceding one is a linear one
                    if type(model[idx-1]) != torch.nn.Linear:
                        return False
                else:
                    return False
        
        # current computation expects a single output PINN
        if type(model[-1]) != torch.nn.Linear or model[-1].weight.shape[0] != 1:
            return False

        # all conditions are satisfied, so this PINN is supported
        return True

    @staticmethod
    def separate_norm_and_fc_layers(layers: List[torch.nn.Module]) -> Tuple[List[torch.nn.Module], List[torch.nn.Module]]:
        # returns the normalization layers first and then fully connected
        norm_layers = [layer for layer in layers if type(layer) in supported_normalization_ops]
        fc_layers = layers[len(norm_layers):]

        return norm_layers, fc_layers
    
    def compute_IBP_bounds(self, debug: bool = True):
        domain_bounds = self.domain_bounds

        # apply normalization layers on the bounds
        for layer in self.norm_layers:
            domain_bounds = layer(domain_bounds)
            
            if type(layer) == Mul:
                # a multiplication can change the direction of the bounds, sort them accordingly
                domain_bounds = domain_bounds.sort(dim=0).values

        # interval analysis
        current_lb = domain_bounds[0]
        current_ub = domain_bounds[1]

        self.lower_bounds.append(current_lb)
        self.upper_bounds.append(current_ub)

        it_object = self.fc_layers
        if debug:
            print("Propagating IBP bounds through u_theta...")
            it_object = tqdm(self.fc_layers)

        for layer in it_object:
            if isinstance(layer, torch.nn.Linear):
                # bound it from the top and bottom
                pos_weights = torch.clamp(layer.weight, min=0)
                neg_weights = torch.clamp(layer.weight, max=0)

                new_layer_lb = pos_weights @ current_lb + neg_weights @ current_ub + layer.bias
                new_layer_ub = pos_weights @ current_ub + neg_weights @ current_lb + layer.bias
                
                current_lb = new_layer_lb
                current_ub = new_layer_ub
            elif isinstance(layer, torch.nn.Tanh):
                # monotonicity of tanh means we can simply apply the operation to the bounds
                current_lb = layer(current_lb)
                current_ub = layer(current_ub)

            assert all(current_lb <= current_ub)

            self.lower_bounds.append(current_lb)
            self.upper_bounds.append(current_ub)

    def compute_bounds(self, debug: bool = True):
        # if the bounds are already computed, simply add the variables to the problem without re-computing, otherwise compute the bounds along the way 
        domain_bounds = self.domain_bounds

        # apply normalization layers on the bounds
        for layer in self.norm_layers:
            domain_bounds = layer(domain_bounds)
            
            if type(layer) == Mul:
                # a multiplication can change the direction of the bounds, sort them accordingly
                domain_bounds = domain_bounds.sort(dim=0).values

        domain_bounds = domain_bounds.T

        grb_model = self.grb_model
        gurobi_vars = self.gurobi_vars
        lower_bounds = self.lower_bounds
        upper_bounds = self.upper_bounds

        # initial conditions
        input_vars = []
        input_lbs = []
        input_ubs = []
        for idx, (lb, ub) in enumerate(domain_bounds):
            v = grb_model.addVar(lb=lb, ub=ub, obj=0, vtype=grb.GRB.CONTINUOUS, name=f'inp_{idx}')
            input_vars.append(v)
            input_lbs.append(lb)
            input_ubs.append(ub)

        grb_model.update()

        gurobi_vars.append(input_vars)
        lower_bounds.append(torch.tensor(input_lbs, dtype=torch.float))
        upper_bounds.append(torch.tensor(input_ubs, dtype=torch.float))

        it_object = self.fc_layers
        if debug:
            print("Propagating LP bounds through u_theta...")
            it_object = tqdm(self.fc_layers)

        # apply bounds on the fully connected layers
        for layer_idx, layer in enumerate(it_object):
            new_layer_lb = []
            new_layer_ub = []
            new_layer_gurobi_vars = []

            pre_vars = gurobi_vars[layer_idx]
            pre_lb = lower_bounds[layer_idx]
            pre_ub = upper_bounds[layer_idx]
            if type(layer) == torch.nn.Linear:
                # it's a linear layer
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

                    v = add_var_to_model(grb_model, out_lb, out_ub, f'u_theta_linear_{layer_idx}_{neuron_idx}')
                    grb_model.addConstr(v == lin_expr)

                    if layer_idx >= 1:
                        out_lb, out_ub = solve_for_lower_upper_bound_var(grb_model, v)

                    new_layer_lb.append(out_lb)
                    new_layer_ub.append(out_ub)
                    new_layer_gurobi_vars.append(v)
            elif type(layer) in supported_activations:
                for neuron_idx, pre_var in enumerate(pre_vars):
                    pre_lb_neuron = pre_lb[neuron_idx]
                    pre_ub_neuron = pre_ub[neuron_idx]

                    # compute the new lower bound and upper bound using the exact function
                    # out_lb = layer(pre_lb_neuron)
                    # out_ub = layer(pre_ub_neuron)

                    out_lb, out_ub = self.activation_relaxation.get_lb_ub_in_interval(
                        pre_lb_neuron, pre_ub_neuron
                    )

                    v = add_var_to_model(grb_model, out_lb, out_ub, f'u_theta_sigma_{layer_idx}_{neuron_idx}')
                    self.activation_relaxation.relax_lp(grb_model, pre_var, v, pre_lb_neuron, pre_ub_neuron)

                    new_layer_lb.append(out_lb)
                    new_layer_ub.append(out_ub)
                    new_layer_gurobi_vars.append(v)

            lower_bounds.append(torch.tensor(new_layer_lb))
            upper_bounds.append(torch.tensor(new_layer_ub))
            gurobi_vars.append(new_layer_gurobi_vars)

        self.computed_bounds = True
    
    def add_vars_and_constraints(self, grb_model: grb.Model):
        if not self.computed_bounds:
            self.compute_bounds()
        
        domain_bounds = self.domain_bounds

        # apply normalization layers on the bounds
        for layer in self.norm_layers:
            domain_bounds = layer(domain_bounds)
            
            if type(layer) == Mul:
                # a multiplication can change the direction of the bounds, sort them accordingly
                domain_bounds = domain_bounds.sort(dim=0).values

        domain_bounds = domain_bounds.T

        # lower and upper bounds are computed at this point, just use them
        gurobi_vars = []
        lower_bounds = self.lower_bounds
        upper_bounds = self.upper_bounds

        input_vars = []
        for idx, (lb, ub) in enumerate(domain_bounds):
            v = grb_model.addVar(lb=lb, ub=ub, obj=0, vtype=grb.GRB.CONTINUOUS, name=f'inp_{idx}')
            input_vars.append(v)

        gurobi_vars.append(input_vars)

        # apply bounds on the fully connected layers
        for layer_idx, layer in enumerate(self.fc_layers):
            new_layer_gurobi_vars = []

            pre_vars = gurobi_vars[layer_idx]
            pre_lb = lower_bounds[layer_idx]
            pre_ub = upper_bounds[layer_idx]
            if type(layer) == torch.nn.Linear:
                post_lb = lower_bounds[layer_idx+1]
                post_ub = upper_bounds[layer_idx+1]

                for neuron_idx in range(layer.weight.size(0)):
                    lin_expr = layer.bias[neuron_idx].item()
                    coeffs = layer.weight[neuron_idx, :]
                    lin_expr += grb.LinExpr(coeffs, pre_vars)

                    out_lb = post_lb[neuron_idx]
                    out_ub = post_ub[neuron_idx]

                    v = add_var_to_model(grb_model, out_lb, out_ub, f'u_theta_linear_{layer_idx}_{neuron_idx}')
                    grb_model.addConstr(v == lin_expr)

                    new_layer_gurobi_vars.append(v)
            elif type(layer) in supported_activations:
                for neuron_idx, pre_var in enumerate(pre_vars):
                    pre_lb_neuron = pre_lb[neuron_idx]
                    pre_ub_neuron = pre_ub[neuron_idx]

                    # compute the new lower bound and upper bound using the exact function
                    # out_lb = layer(pre_lb_neuron)
                    # out_ub = layer(pre_ub_neuron)

                    out_lb, out_ub = self.activation_relaxation.get_lb_ub_in_interval(
                        pre_lb_neuron, pre_ub_neuron
                    )
                    
                    v = add_var_to_model(grb_model, out_lb, out_ub, f'u_theta_sigma_{layer_idx}_{neuron_idx}')
                    self.activation_relaxation.relax_lp(grb_model, pre_var, v, pre_lb_neuron, pre_ub_neuron)

                    new_layer_gurobi_vars.append(v)
            
            gurobi_vars.append(new_layer_gurobi_vars)
        
        grb_model.update()

        return gurobi_vars


class LPPINNPartialDerivative():
    def __init__(self, pinn_solution: LPPINNSolution, component_idx: int, activation_derivative_relaxation: ActivationRelaxation) -> None:
        # class used to compute the bounds of the derivative of the PINN; it operates over the model defined in pinn.grb_model
        self.u_theta = pinn_solution

        # the component should be one of the input dimensions
        input_shape = self.u_theta.fc_layers[0].weight.shape[1]
        assert component_idx >= 0
        assert component_idx < input_shape

        self.component_idx = component_idx
        self.initial_partial_grad = torch.zeros([1, input_shape])
        self.initial_partial_grad[0][component_idx] = 1

        self.computed_bounds = False
        self.gurobi_vars = []
        self.lower_bounds = []
        self.upper_bounds = []
        self.d_phi_m_d_phi_m_1_vars = []
        self.d_phi_m_d_phi_m_1_lbs = []
        self.d_phi_m_d_phi_m_1_ubs = []

        self.activation_derivative_relaxation = activation_derivative_relaxation    
    
    def compute_bounds(self, debug: bool = True):
        # if we haven't computed the intermediate bounds on u_theta, do it now
        if not self.u_theta.computed_bounds:
            self.u_theta.compute_bounds(debug=debug)

        # proceed if the computation is successful and these variables are now populated
        assert self.u_theta.computed_bounds

        self.grb_model = new_gurobi_model(
            f"u_d{self.component_idx}_theta",
            feasibility_tol=self.u_theta.feasibility_tol,
            optimality_tol=self.u_theta.optimality_tol
        )

        u_theta_gurobi_vars = self.u_theta.add_vars_and_constraints(self.grb_model)
        grb_model = self.grb_model

        # computing a bound on $\partial_t u_theta$ using the previously computed bounds
        u_dxi_gurobi_vars = self.gurobi_vars
        u_dxi_lower_bounds = self.lower_bounds
        u_dxi_upper_bounds = self.upper_bounds

        # compute the initial gradient with 
        norm_layer_partial_grad = copy.deepcopy(self.initial_partial_grad)
        for norm_layer in self.u_theta.norm_layers:
            if type(norm_layer) == Add:
                continue  
            elif type(norm_layer) == Mul:
                derivative = norm_layer(norm_layer_partial_grad) / norm_layer_partial_grad
                derivative[torch.isnan(derivative)] = 0.0

                norm_layer_partial_grad *= derivative
            else:
                raise NotImplementedError

        # input variables
        input_dxi_vars = []
        input_dxi_lbs = []
        input_dxi_ubs = []
        for idx, val in enumerate(norm_layer_partial_grad[0]):
            v = grb_model.addVar(lb=val, ub=val, obj=0, vtype=grb.GRB.CONTINUOUS, name=f'inp_d{self.component_idx}_{idx}')
            input_dxi_vars.append(v)
            input_dxi_lbs.append(val)
            input_dxi_ubs.append(val)

        grb_model.update()

        u_dxi_gurobi_vars.append(input_dxi_vars)
        u_dxi_lower_bounds.append(torch.tensor(input_dxi_lbs, dtype=torch.float))
        u_dxi_upper_bounds.append(torch.tensor(input_dxi_ubs, dtype=torch.float))

        it_object = self.u_theta.fc_layers
        if debug:
            print(f"Propagating LP bounds through du_theta/dx{self.component_idx}...")
            it_object = tqdm(self.u_theta.fc_layers)

        for layer_idx, layer in enumerate(it_object):
            is_final = (layer_idx == len(self.u_theta.fc_layers) - 1)
            new_layer_lb = []
            new_layer_ub = []
            new_layer_gurobi_vars = []

            # activations are subsumed in the gradient of the linear layers
            if type(layer) in supported_activations:
                continue

            assert isinstance(layer, torch.nn.Linear)

            if not is_final:
                u_theta_pre_activation_lbs = self.u_theta.lower_bounds[layer_idx+1]
                u_theta_pre_activation_ubs = self.u_theta.upper_bounds[layer_idx+1]
                u_theta_pre_activation_vars = u_theta_gurobi_vars[layer_idx+1]
                u_dxi_lbs = u_dxi_lower_bounds[-1]
                u_dxi_ubs = u_dxi_upper_bounds[-1]
                u_dxi_pre_vars = u_dxi_gurobi_vars[-1]

                pos_w = torch.clamp(layer.weight, min=0)
                neg_w = torch.clamp(layer.weight, max=0)

                # first step is to bound the output of \sigma'(pre_activation_output)
                sigma_prime_output_vars = []
                sigma_prime_output_lbs = []
                sigma_prime_output_ubs = []
                for neuron_idx, u_theta_pre_act_var in enumerate(u_theta_pre_activation_vars):
                    u_theta_pre_act_neuron_lb = u_theta_pre_activation_lbs[neuron_idx]
                    u_theta_pre_act_neuron_ub = u_theta_pre_activation_ubs[neuron_idx]
                    
                    # neuron_sigma_prime_lb = self.activation_derivative_relaxation.evaluate(u_theta_pre_act_neuron_lb)
                    # neuron_sigma_prime_ub = self.activation_derivative_relaxation.evaluate(u_theta_pre_act_neuron_ub)

                    neuron_sigma_prime_lb, neuron_sigma_prime_ub = self.activation_derivative_relaxation.get_lb_ub_in_interval(
                        u_theta_pre_act_neuron_lb, u_theta_pre_act_neuron_ub
                    )

                    neuron_sigma_prime = add_var_to_model(
                        grb_model,
                        neuron_sigma_prime_lb,
                        neuron_sigma_prime_ub,
                        f'u_d{self.component_idx}_{layer_idx}_sigma_prime_{neuron_idx}'
                    )

                    # relax the derivative of the relaxation
                    self.activation_derivative_relaxation.relax_lp(
                        grb_model,
                        u_theta_pre_act_var,
                        neuron_sigma_prime,
                        u_theta_pre_act_neuron_lb,
                        u_theta_pre_act_neuron_ub
                    )

                    sigma_prime_output_vars.append(neuron_sigma_prime)
                    sigma_prime_output_lbs.append(neuron_sigma_prime_lb)
                    sigma_prime_output_ubs.append(neuron_sigma_prime_ub)
                
                # compute matmul(diag(\sigma'(...)), W) using the previous variables
                layer_weight = layer.weight.detach().numpy()
                first_matmul_vars = []
                first_matmul_lbs = []
                first_matmul_ubs = []
                for i in range(len(sigma_prime_output_vars)):
                    sigma_prime_output_var = sigma_prime_output_vars[i]
                    sigma_prime_output_lb = sigma_prime_output_lbs[i]
                    sigma_prime_output_ub = sigma_prime_output_ubs[i]

                    row_vars = []
                    row_lbs = []
                    row_ubs = []
                    for j in range(layer_weight.shape[1]):
                        weight = layer_weight[i, j]
                        lb = min(weight*sigma_prime_output_lb, weight*sigma_prime_output_ub)
                        ub = max(weight*sigma_prime_output_lb, weight*sigma_prime_output_ub)

                        first_matmul_var = add_var_to_model(
                            grb_model,
                            lb,
                            ub,
                            f'u_d{self.component_idx}_{layer_idx}_first_matmul_{i}_{j}'
                        )
                        grb_model.addConstr(first_matmul_var == weight * sigma_prime_output_var)

                        row_vars.append(first_matmul_var)
                        row_lbs.append(lb)
                        row_ubs.append(ub)
                    
                    first_matmul_vars.append(row_vars)
                    first_matmul_lbs.append(row_lbs)
                    first_matmul_ubs.append(row_ubs)

                self.d_phi_m_d_phi_m_1_vars.append(first_matmul_vars)
                self.d_phi_m_d_phi_m_1_lbs.append(torch.tensor(first_matmul_lbs))
                self.d_phi_m_d_phi_m_1_ubs.append(torch.tensor(first_matmul_ubs))

                # compute matmul(first_matmul, u_dxi_pre_vars)
                for out_neuron_idx in range(len(first_matmul_vars)):
                    each_multiplication_vars = []
                    each_multiplication_lbs = []
                    each_multiplication_ubs = []

                    for j in range(len(u_dxi_pre_vars)):
                        jth_out_lb, jth_out_ub = get_multiplication_bounds(
                            first_matmul_lbs[out_neuron_idx][j], first_matmul_ubs[out_neuron_idx][j],
                            u_dxi_lbs[j], u_dxi_ubs[j]
                        )

                        jth_out_var = add_var_to_model(
                            grb_model,
                            jth_out_lb,
                            jth_out_ub,
                            f'u_d{self.component_idx}_{layer_idx}_output_{out_neuron_idx}_{j}th_term'
                        )

                        # exact modeling - non-convex
                        # grb_model.addConstr(jth_out_var == first_matmul_vars[out_neuron_idx][j] * u_dxi_pre_vars[j])
                        mccormick_relaxation(
                            grb_model,
                            input_var_1=first_matmul_vars[out_neuron_idx][j],
                            lb_var_1=first_matmul_lbs[out_neuron_idx][j],
                            ub_var_1=first_matmul_ubs[out_neuron_idx][j],
                            input_var_2=u_dxi_pre_vars[j],
                            lb_var_2=u_dxi_lbs[j],
                            ub_var_2=u_dxi_ubs[j],
                            output_var=jth_out_var
                        )

                        each_multiplication_vars.append(jth_out_var)
                        each_multiplication_lbs.append(jth_out_lb)
                        each_multiplication_ubs.append(jth_out_ub)

                    out_lb = sum(each_multiplication_lbs)
                    out_ub = sum(each_multiplication_ubs)

                    out_var = add_var_to_model(
                        grb_model,
                        out_lb,
                        out_ub,
                        f'u_d{self.component_idx}_{layer_idx}_out_neuron_{out_neuron_idx}'
                    )
                    grb_model.addConstr(out_var == sum(each_multiplication_vars))

                    out_lb, out_ub = solve_for_lower_upper_bound_var(grb_model, out_var)

                    new_layer_gurobi_vars.append(out_var)
                    new_layer_lb.append(out_lb)
                    new_layer_ub.append(out_ub)
            else:
                # it's the last layer, there's no activations, just the linear part
                u_dxi_lbs = u_dxi_lower_bounds[-1]
                u_dxi_ubs = u_dxi_upper_bounds[-1]
                u_dxi_pre_vars = u_dxi_gurobi_vars[-1]

                pos_w = torch.clamp(layer.weight, min=0)
                neg_w = torch.clamp(layer.weight, max=0)

                u_dxi_out_lbs = pos_w @ u_dxi_lbs + neg_w @ u_dxi_ubs
                u_dxi_out_ubs = pos_w @ u_dxi_ubs + neg_w @ u_dxi_lbs

                for neuron_idx in range(len(u_dxi_out_lbs)):
                    final_var = add_var_to_model(
                        grb_model,
                        u_dxi_out_lbs[neuron_idx],
                        u_dxi_out_ubs[neuron_idx],
                        f'u_d{self.component_idx}_{layer_idx}_{neuron_idx}_final_var'
                    )

                    coeffs = layer.weight[neuron_idx, :]
                    lin_expr = grb.LinExpr(coeffs, u_dxi_pre_vars)
                    grb_model.addConstr(final_var == lin_expr)

                    out_lb, out_ub = solve_for_lower_upper_bound_var(grb_model, final_var)

                    new_layer_lb.append(out_lb)
                    new_layer_ub.append(out_ub)
                    new_layer_gurobi_vars.append(final_var)

            u_dxi_gurobi_vars.append(new_layer_gurobi_vars)
            u_dxi_lower_bounds.append(torch.Tensor(new_layer_lb))
            u_dxi_upper_bounds.append(torch.Tensor(new_layer_ub))

        self.computed_bounds = True

    def add_vars_and_constraints(self, grb_model: grb.Model, u_theta_gurobi_vars: List[List[grb.Var]] = None):
        # if we haven't computed the intermediate bounds on u_theta, do it now
        if not self.u_theta.computed_bounds:
            self.u_theta.compute_bounds()

        # proceed if the computation is successful and these variables are now populated
        assert self.u_theta.computed_bounds

        if u_theta_gurobi_vars is None:
            u_theta_gurobi_vars = self.u_theta.add_vars_and_constraints(grb_model)

        # computing a bound on $\partial_t u_theta$ using the previously computed bounds
        u_dxi_gurobi_vars = []
        d_phi_m_d_phi_m_1_gurobi_vars = []
        u_dxi_lower_bounds = self.lower_bounds
        u_dxi_upper_bounds = self.upper_bounds

        # compute the initial gradient with 
        norm_layer_partial_grad = copy.deepcopy(self.initial_partial_grad)
        for norm_layer in self.u_theta.norm_layers:
            if type(norm_layer) == Add:
                continue  
            elif type(norm_layer) == Mul:
                derivative = norm_layer(norm_layer_partial_grad) / norm_layer_partial_grad
                derivative[torch.isnan(derivative)] = 0.0

                norm_layer_partial_grad *= derivative
            else:
                raise NotImplementedError

        # input variables
        input_dxi_vars = []
        for idx, val in enumerate(norm_layer_partial_grad[0]):
            v = grb_model.addVar(lb=val, ub=val, obj=0, vtype=grb.GRB.CONTINUOUS, name=f'inp_d{self.component_idx}_{idx}')
            input_dxi_vars.append(v)

        grb_model.update()

        u_dxi_gurobi_vars.append(input_dxi_vars)

        for layer_idx, layer in enumerate(self.u_theta.fc_layers):
            is_final = (layer_idx == len(self.u_theta.fc_layers) - 1)
            new_layer_gurobi_vars = []

            # activations are subsumed in the gradient of the linear layers
            if type(layer) in supported_activations:
                continue

            assert isinstance(layer, torch.nn.Linear)

            if not is_final:
                u_theta_pre_activation_lbs = self.u_theta.lower_bounds[layer_idx+1]
                u_theta_pre_activation_ubs = self.u_theta.upper_bounds[layer_idx+1]
                u_theta_pre_activation_vars = u_theta_gurobi_vars[layer_idx+1]
                u_dxi_lbs = u_dxi_lower_bounds[int(layer_idx / 2)]
                u_dxi_ubs = u_dxi_upper_bounds[int(layer_idx / 2)]
                u_dxi_pre_vars = u_dxi_gurobi_vars[int(layer_idx / 2)]

                # first step is to bound the output of \sigma'(pre_activation_output)
                sigma_prime_output_vars = []
                sigma_prime_output_lbs = []
                sigma_prime_output_ubs = []
                for neuron_idx, u_theta_pre_act_var in enumerate(u_theta_pre_activation_vars):
                    u_theta_pre_act_neuron_lb = u_theta_pre_activation_lbs[neuron_idx]
                    u_theta_pre_act_neuron_ub = u_theta_pre_activation_ubs[neuron_idx]
                    
                    # neuron_sigma_prime_lb = self.activation_derivative_relaxation.evaluate(u_theta_pre_act_neuron_lb)
                    # neuron_sigma_prime_ub = self.activation_derivative_relaxation.evaluate(u_theta_pre_act_neuron_ub)

                    neuron_sigma_prime_lb, neuron_sigma_prime_ub = self.activation_derivative_relaxation.get_lb_ub_in_interval(
                        u_theta_pre_act_neuron_lb, u_theta_pre_act_neuron_ub
                    )

                    neuron_sigma_prime = add_var_to_model(
                        grb_model,
                        neuron_sigma_prime_lb,
                        neuron_sigma_prime_ub,
                        f'u_d{self.component_idx}_{layer_idx}_sigma_prime_{neuron_idx}'
                    )

                    # relax the derivative of the relaxation
                    self.activation_derivative_relaxation.relax_lp(
                        grb_model,
                        u_theta_pre_act_var,
                        neuron_sigma_prime,
                        u_theta_pre_act_neuron_lb,
                        u_theta_pre_act_neuron_ub
                    )

                    sigma_prime_output_vars.append(neuron_sigma_prime)
                    sigma_prime_output_lbs.append(neuron_sigma_prime_lb)
                    sigma_prime_output_ubs.append(neuron_sigma_prime_ub)
                
                # compute matmul(diag(\sigma'(...)), W) using the previous variables
                layer_weight = layer.weight.detach().numpy()
                first_matmul_vars = []
                first_matmul_lbs = []
                first_matmul_ubs = []
                for i in range(len(sigma_prime_output_vars)):
                    sigma_prime_output_var = sigma_prime_output_vars[i]
                    sigma_prime_output_lb = sigma_prime_output_lbs[i]
                    sigma_prime_output_ub = sigma_prime_output_ubs[i]

                    row_vars = []
                    row_lbs = []
                    row_ubs = []
                    for j in range(layer_weight.shape[1]):
                        weight = layer_weight[i, j]
                        lb = min(weight*sigma_prime_output_lb, weight*sigma_prime_output_ub)
                        ub = max(weight*sigma_prime_output_lb, weight*sigma_prime_output_ub)

                        first_matmul_var = add_var_to_model(
                            grb_model,
                            lb,
                            ub,
                            f'u_d{self.component_idx}_{layer_idx}_first_matmul_{i}_{j}'
                        )
                        grb_model.addConstr(first_matmul_var == weight * sigma_prime_output_var)
                        row_vars.append(first_matmul_var)
                        row_lbs.append(lb)
                        row_ubs.append(ub)
                    
                    first_matmul_vars.append(row_vars)
                    first_matmul_lbs.append(row_lbs)
                    first_matmul_ubs.append(row_ubs)

                d_phi_m_d_phi_m_1_gurobi_vars.append(first_matmul_vars)

                out_lbs = u_dxi_lower_bounds[int(layer_idx / 2)+1]
                out_ubs = u_dxi_upper_bounds[int(layer_idx / 2)+1]

                # compute matmul(first_matmul, u_dxi_pre_vars)
                for out_neuron_idx in range(len(first_matmul_vars)):
                    each_multiplication_vars = []
                    each_multiplication_lbs = []
                    each_multiplication_ubs = []

                    for j in range(len(u_dxi_pre_vars)):
                        jth_out_lb, jth_out_ub = get_multiplication_bounds(
                            first_matmul_lbs[out_neuron_idx][j], first_matmul_ubs[out_neuron_idx][j],
                            u_dxi_lbs[j], u_dxi_ubs[j]
                        )

                        jth_out_var = add_var_to_model(
                            grb_model,
                            jth_out_lb,
                            jth_out_ub,
                            f'u_d{self.component_idx}_{layer_idx}_output_{out_neuron_idx}_{j}th_term'
                        )

                        # exact modeling - non-convex
                        # grb_model.addConstr(jth_out_var == first_matmul_vars[out_neuron_idx][j] * u_dxi_pre_vars[j])

                        mccormick_relaxation(
                            grb_model,
                            input_var_1=first_matmul_vars[out_neuron_idx][j],
                            lb_var_1=first_matmul_lbs[out_neuron_idx][j],
                            ub_var_1=first_matmul_ubs[out_neuron_idx][j],
                            input_var_2=u_dxi_pre_vars[j],
                            lb_var_2=u_dxi_lbs[j],
                            ub_var_2=u_dxi_ubs[j],
                            output_var=jth_out_var
                        )

                        each_multiplication_vars.append(jth_out_var)
                        each_multiplication_lbs.append(jth_out_lb)
                        each_multiplication_ubs.append(jth_out_ub)

                    grb_model.update()

                    out_var = add_var_to_model(
                        grb_model,
                        out_lbs[out_neuron_idx],
                        out_ubs[out_neuron_idx],
                        f'u_d{self.component_idx}_{layer_idx}_out_neuron_{out_neuron_idx}'
                    )
                    grb_model.addConstr(out_var == sum(each_multiplication_vars))

                    new_layer_gurobi_vars.append(out_var)
            else:
                # it's the last layer, there's no activations, just the linear part
                u_dxi_out_lbs = u_dxi_lower_bounds[-1]
                u_dxi_out_ubs = u_dxi_upper_bounds[-1]
                u_dxi_pre_vars = u_dxi_gurobi_vars[int(layer_idx / 2)]

                for neuron_idx in range(len(u_dxi_out_lbs)):
                    final_var = add_var_to_model(
                        grb_model,
                        u_dxi_out_lbs[neuron_idx],
                        u_dxi_out_ubs[neuron_idx],
                        f'u_d{self.component_idx}_{layer_idx}_{neuron_idx}_final_var'
                    )

                    coeffs = layer.weight[neuron_idx, :]
                    lin_expr = grb.LinExpr(coeffs, u_dxi_pre_vars)
                    grb_model.addConstr(final_var == lin_expr)

                    new_layer_gurobi_vars.append(final_var)

            u_dxi_gurobi_vars.append(new_layer_gurobi_vars)

        return [u_theta_gurobi_vars], [u_dxi_gurobi_vars, d_phi_m_d_phi_m_1_gurobi_vars]


class LPPINNSecondPartialDerivative():
    def __init__(self, pinn_partial_derivative: LPPINNPartialDerivative, component_idx: int, activation_derivative_derivative_relaxation: ActivationRelaxation) -> None:
        # class used to compute the bounds of the second derivative of the PINN; it operates over the model defined in pinn_partial_derivative.u_theta.grb_model
        self.partial_derivative = pinn_partial_derivative
        self.u_theta = pinn_partial_derivative.u_theta

        # the component should be one of the input dimensions
        input_shape = self.u_theta.fc_layers[0].weight.shape[1]
        assert component_idx >= 0
        assert component_idx < input_shape

        self.component_idx = component_idx

        if component_idx != pinn_partial_derivative.component_idx:
            # this implementation assumes the second derivative is with respect to the same derivative
            raise NotImplementedError

        self.initial_partial_grad = torch.zeros([1, input_shape])

        self.computed_bounds = False
        self.gurobi_vars = []
        self.lower_bounds = []
        self.upper_bounds = []

        self.activation_derivative_derivative_relaxation = activation_derivative_derivative_relaxation    
    
    def compute_bounds(self, debug: bool = True):
        # if we haven't computed the intermediate bounds on u_theta, do it now
        if not self.partial_derivative.computed_bounds:
            self.partial_derivative.compute_bounds(debug=debug)

        # proceed if the computation is successful and these variables are now populated
        assert self.u_theta.computed_bounds
        assert self.partial_derivative.computed_bounds

        self.grb_model = new_gurobi_model(
            f"u_d{self.partial_derivative.component_idx}_{self.component_idx}_theta",
            feasibility_tol=self.u_theta.feasibility_tol,
            optimality_tol=self.u_theta.optimality_tol
        )

        u_theta_all_vars, u_theta_dxi_all_vars = self.partial_derivative.add_vars_and_constraints(self.grb_model)

        u_theta_gurobi_vars = u_theta_all_vars[0]
        u_dxi_gurobi_vars, d_phi_m_d_phi_m_1_gurobi_vars = u_theta_dxi_all_vars

        grb_model = self.grb_model

        # computing a bound on $\partial_t u_theta$ using the previously computed bounds
        u_dxixi_gurobi_vars = self.gurobi_vars
        u_dxixi_lower_bounds = self.lower_bounds
        u_dxixi_upper_bounds = self.upper_bounds
        self.first_sum_term_lbs = []
        self.first_sum_term_ubs = []
        self.second_sum_term_lbs = []
        self.second_sum_term_ubs = []

        # d_psi_0_dx_i is equal to 0 (second derivative of x with respect to x_i) and it'll remain the
        # same through the normalization layers
        zero_vec = [0 for _ in range(self.u_theta.fc_layers[0].weight.shape[1])]
        u_dxixi_gurobi_vars.append(zero_vec)
        u_dxixi_lower_bounds.append(torch.tensor(zero_vec, dtype=torch.float))
        u_dxixi_upper_bounds.append(torch.tensor(zero_vec, dtype=torch.float))

        it_object = self.u_theta.fc_layers
        if debug:
            print(f"Propagating LP bounds through d^2u_theta/dx{self.component_idx}^2...")
            it_object = tqdm(self.u_theta.fc_layers)

        for layer_idx, layer in enumerate(it_object):
            is_final = (layer_idx == len(self.u_theta.fc_layers) - 1)
            new_layer_lb = []
            new_layer_ub = []
            new_layer_gurobi_vars = []

            # activations are subsumed in the gradient of the linear layers
            if type(layer) in supported_activations:
                continue

            assert isinstance(layer, torch.nn.Linear)

            if not is_final:
                # load all the previous variables required for the computation of this block
                u_theta_pre_activation_lbs = self.u_theta.lower_bounds[layer_idx+1]
                u_theta_pre_activation_ubs = self.u_theta.upper_bounds[layer_idx+1]
                u_theta_pre_activation_vars = u_theta_gurobi_vars[layer_idx+1]

                u_dxi_lbs = self.partial_derivative.lower_bounds[int(layer_idx / 2)]
                u_dxi_ubs = self.partial_derivative.upper_bounds[int(layer_idx / 2)]
                u_dxi_pre_vars = u_dxi_gurobi_vars[int(layer_idx / 2)]

                d_phi_m_d_phi_m_1_ubs = self.partial_derivative.d_phi_m_d_phi_m_1_ubs[int(layer_idx / 2)]
                d_phi_m_d_phi_m_1_lbs = self.partial_derivative.d_phi_m_d_phi_m_1_lbs[int(layer_idx / 2)]
                d_phi_m_d_phi_m_1_vars = d_phi_m_d_phi_m_1_gurobi_vars[int(layer_idx / 2)]
                
                u_dxixi_lbs = u_dxixi_lower_bounds[-1]
                u_dxixi_ubs = u_dxixi_upper_bounds[-1]
                u_dxixi_pre_vars = u_dxixi_gurobi_vars[-1]

                # ----- compute first dpsi_1/dx_i -----

                pos_w = torch.clamp(layer.weight, min=0)
                neg_w = torch.clamp(layer.weight, max=0)

                first_matmul_lbs = pos_w @ u_dxi_lbs + neg_w @ u_dxi_ubs
                first_matmul_ubs = pos_w @ u_dxi_ubs + neg_w @ u_dxi_lbs

                first_matmul_vars = []
                first_matmul_lin_expressions = layer.weight.detach().numpy() @ np.array(u_dxi_pre_vars)
                for i, expression in enumerate(first_matmul_lin_expressions):
                    first_matmul_i = add_var_to_model(
                        grb_model,
                        first_matmul_lbs[i],
                        first_matmul_ubs[i],
                        f'u_d{self.partial_derivative.component_idx}_{self.component_idx}_{layer_idx}_first_matmul_{i}'
                    )
                    grb_model.addConstr(first_matmul_i == expression)

                    first_matmul_vars.append(first_matmul_i)

                # first step is to bound the output of \sigma''(pre_activation_output)
                sigma_prime_prime_output_vars = []
                sigma_prime_prime_output_lbs = []
                sigma_prime_prime_output_ubs = []
                for neuron_idx, (u_theta_pre_act_var, u_theta_pre_act_neuron_lb, u_theta_pre_act_neuron_ub) in \
                        enumerate(zip(u_theta_pre_activation_vars, u_theta_pre_activation_lbs, u_theta_pre_activation_ubs)):
                    # sigma'' = sigmoid' = sigmoid * (1 - sigmoid) is not a monotonic function, so need to be careful about that
                    neuron_sigma_prime_prime_lb, neuron_sigma_prime_prime_ub = self.activation_derivative_derivative_relaxation.get_lb_ub_in_interval(
                        u_theta_pre_act_neuron_lb, u_theta_pre_act_neuron_ub
                    )

                    neuron_sigma_prime_prime = add_var_to_model(
                        grb_model,
                        neuron_sigma_prime_prime_lb,
                        neuron_sigma_prime_prime_ub,
                        f'u_d{self.partial_derivative.component_idx}_{self.component_idx}_{layer_idx}_sigma_prime_prime_{neuron_idx}'
                    )

                    # relax the derivative of the relaxation
                    self.activation_derivative_derivative_relaxation.relax_lp(
                        grb_model,
                        u_theta_pre_act_var,
                        neuron_sigma_prime_prime,
                        u_theta_pre_act_neuron_lb,
                        u_theta_pre_act_neuron_ub
                    )

                    sigma_prime_prime_output_vars.append(neuron_sigma_prime_prime)
                    sigma_prime_prime_output_lbs.append(neuron_sigma_prime_prime_lb)
                    sigma_prime_prime_output_ubs.append(neuron_sigma_prime_prime_ub)


                # compute elementwise_multiply(sigma_prime_output_vars, first_matmul_vars)
                elem_multiply_vars = []
                elem_multiply_lbs = []
                elem_multiply_ubs = []
                for i in range(len(sigma_prime_prime_output_vars)):
                    elem_multiply_lb_i, elem_multiply_ub_i = get_multiplication_bounds(
                        first_matmul_lbs[i], first_matmul_ubs[i],
                        sigma_prime_prime_output_lbs[i], sigma_prime_prime_output_ubs[i]
                    )

                    elem_multiply_var_i = add_var_to_model(
                        grb_model,
                        elem_multiply_lb_i,
                        elem_multiply_ub_i,
                        f'u_d{self.partial_derivative.component_idx}_{self.component_idx}_{layer_idx}_elemwise_mul_{i}'
                    )

                    # exact modeling - non-convex
                    # grb_model.addConstr(elem_multiply_var_i == first_matmul_vars[i] * sigma_prime_prime_output_vars[i])

                    mccormick_relaxation(
                        grb_model,
                        input_var_1=first_matmul_vars[i],
                        lb_var_1=first_matmul_lbs[i],
                        ub_var_1=first_matmul_ubs[i],
                        input_var_2=sigma_prime_prime_output_vars[i],
                        lb_var_2=sigma_prime_prime_output_lbs[i],
                        ub_var_2=sigma_prime_prime_output_ubs[i],
                        output_var=elem_multiply_var_i
                    )

                    elem_multiply_vars.append(elem_multiply_var_i)
                    elem_multiply_lbs.append(elem_multiply_lb_i)
                    elem_multiply_ubs.append(elem_multiply_ub_i)

                # finally, compute matmul(diag(elem_multiply_vars), W)
                layer_weight = layer.weight.detach().numpy()
                first_chain_terms = []
                first_chain_term_lbs = []
                first_chain_term_ubs = []
                for i in range(len(elem_multiply_vars)):
                    elem_multiply_var = elem_multiply_vars[i]
                    elem_multiply_lb = elem_multiply_lbs[i]
                    elem_multiply_ub = elem_multiply_ubs[i]

                    row_vars = []
                    row_lbs = []
                    row_ubs = []
                    for j in range(layer_weight.shape[1]):
                        weight = layer_weight[i, j]
                        lb = min(weight*elem_multiply_lb, weight*elem_multiply_ub)
                        ub = max(weight*elem_multiply_lb, weight*elem_multiply_ub)

                        first_chain_term = add_var_to_model(
                            grb_model,
                            lb,
                            ub,
                            f'u_d{self.partial_derivative.component_idx}_{self.component_idx}__{layer_idx}_first_chain_term_{i}_{j}'
                        )
                        grb_model.addConstr(first_chain_term == weight * elem_multiply_var)
                        row_vars.append(first_chain_term)
                        row_lbs.append(lb)
                        row_ubs.append(ub)

                    first_chain_terms.append(row_vars)
                    first_chain_term_lbs.append(row_lbs)
                    first_chain_term_ubs.append(row_ubs)

                grb_model.update()

                # ----- end of computation of dpsi_1/dx_i -----

                # ----- first chain multiplication, it's matmul(first_chain_terms, u_dxi_pre_vars) ----- 
                first_sum_terms = []
                first_sum_term_lbs = []
                first_sum_term_ubs = []
                for out_neuron_idx in range(len(first_chain_terms)):
                    each_multiplication_vars = []
                    each_multiplication_lbs = []
                    each_multiplication_ubs = []

                    for j in range(len(u_dxi_pre_vars)):
                        jth_out_lb, jth_out_ub = get_multiplication_bounds(
                            first_chain_term_lbs[out_neuron_idx][j], first_chain_term_ubs[out_neuron_idx][j],
                            u_dxi_lbs[j], u_dxi_ubs[j]
                        )

                        jth_out_var = add_var_to_model(
                            grb_model,
                            jth_out_lb,
                            jth_out_ub,
                            f'u_d{self.partial_derivative.component_idx}_{self.component_idx}_{layer_idx}_first_sum_term_intermediate_{out_neuron_idx}_{j}th_term'
                        )

                        # exact modeling - non-convex
                        # grb_model.addConstr(jth_out_var == first_chain_terms[out_neuron_idx][j] * u_dxi_pre_vars[j])

                        mccormick_relaxation(
                            grb_model,
                            input_var_1=first_chain_terms[out_neuron_idx][j],
                            lb_var_1=first_chain_term_lbs[out_neuron_idx][j],
                            ub_var_1=first_chain_term_ubs[out_neuron_idx][j],
                            input_var_2=u_dxi_pre_vars[j],
                            lb_var_2=u_dxi_lbs[j],
                            ub_var_2=u_dxi_ubs[j],
                            output_var=jth_out_var
                        )

                        each_multiplication_vars.append(jth_out_var)
                        each_multiplication_lbs.append(jth_out_lb)
                        each_multiplication_ubs.append(jth_out_ub)

                    grb_model.update()

                    first_sum_term_lb = sum(each_multiplication_lbs)
                    first_sum_term_ub = sum(each_multiplication_ubs)

                    first_sum_term = add_var_to_model(
                        grb_model,
                        first_sum_term_lb,
                        first_sum_term_ub,
                        f'u_d{self.partial_derivative.component_idx}_{self.component_idx}_{layer_idx}_first_sum_term_{out_neuron_idx}'
                    )
                    grb_model.addConstr(first_sum_term == sum(each_multiplication_vars))

                    first_sum_term_lb, first_sum_term_ub = solve_for_lower_upper_bound_var(grb_model, first_sum_term)

                    first_sum_terms.append(first_sum_term)
                    first_sum_term_lbs.append(first_sum_term_lb)
                    first_sum_term_ubs.append(first_sum_term_ub)

                # save to object for use in other models
                self.first_sum_term_lbs.append(first_sum_term_lbs)
                self.first_sum_term_ubs.append(first_sum_term_ubs)

                # ----- second chain multiplication, it's matmul(d_phi_m_d_phi_m_1_vars, u_dxixi_pre_vars) -----
                second_sum_terms = []
                second_sum_term_lbs = []
                second_sum_term_ubs = []
                for out_neuron_idx in range(len(d_phi_m_d_phi_m_1_vars)):
                    each_multiplication_vars = []
                    each_multiplication_lbs = []
                    each_multiplication_ubs = []

                    for j in range(len(u_dxixi_pre_vars)):
                        jth_out_lb, jth_out_ub = get_multiplication_bounds(
                            d_phi_m_d_phi_m_1_lbs[out_neuron_idx][j], d_phi_m_d_phi_m_1_ubs[out_neuron_idx][j],
                            u_dxixi_lbs[j], u_dxixi_ubs[j]
                        )

                        jth_out_var = add_var_to_model(
                            grb_model,
                            jth_out_lb,
                            jth_out_ub,
                            f'u_d{self.partial_derivative.component_idx}_{self.component_idx}_{layer_idx}_second_sum_term_intermediate_{out_neuron_idx}_{j}th_term'
                        )

                        # exact modeling - non-convex
                        # grb_model.addConstr(jth_out_var == d_phi_m_d_phi_m_1_vars[out_neuron_idx][j] * u_dxixi_pre_vars[j])

                        mccormick_relaxation(
                            grb_model,
                            input_var_1=u_dxixi_pre_vars[j],
                            lb_var_1=u_dxixi_lbs[j],
                            ub_var_1=u_dxixi_ubs[j],
                            input_var_2=d_phi_m_d_phi_m_1_vars[out_neuron_idx][j],
                            lb_var_2=d_phi_m_d_phi_m_1_lbs[out_neuron_idx][j],
                            ub_var_2=d_phi_m_d_phi_m_1_ubs[out_neuron_idx][j],
                            output_var=jth_out_var
                        )

                        each_multiplication_vars.append(jth_out_var)
                        each_multiplication_lbs.append(jth_out_lb)
                        each_multiplication_ubs.append(jth_out_ub)

                    grb_model.update()

                    second_sum_term_lb = sum(each_multiplication_lbs)
                    second_sum_term_ub = sum(each_multiplication_ubs)

                    second_sum_term = add_var_to_model(
                        grb_model,
                        second_sum_term_lb,
                        second_sum_term_ub,
                        f'u_d{self.partial_derivative.component_idx}_{self.component_idx}_{layer_idx}_second_sum_term_{neuron_idx}'
                    )
                    grb_model.addConstr(second_sum_term == sum(each_multiplication_vars))

                    if layer_idx >= 1:
                        second_sum_term_lb, second_sum_term_ub = solve_for_lower_upper_bound_var(grb_model, second_sum_term)

                    second_sum_terms.append(second_sum_term)
                    second_sum_term_lbs.append(second_sum_term_lb)
                    second_sum_term_ubs.append(second_sum_term_ub)
                
                # save to object for use in other models
                self.second_sum_term_lbs.append(second_sum_term_lbs)
                self.second_sum_term_ubs.append(second_sum_term_ubs)

                # actual sum output of the layer
                for i in range(len(first_sum_terms)):
                    out_lb = first_sum_term_lbs[i] + second_sum_term_lbs[i]
                    out_ub = first_sum_term_ubs[i] + second_sum_term_ubs[i]

                    out_i = add_var_to_model(
                        grb_model,
                        out_lb,
                        out_ub,
                        f'u_d{self.partial_derivative.component_idx}_{self.component_idx}_{layer_idx}_out_{i}'
                    )
                    grb_model.addConstr(out_i == first_sum_terms[i] + second_sum_terms[i])

                    if layer_idx >= 1:
                        out_lb, out_ub = solve_for_lower_upper_bound_var(grb_model, out_i)

                    new_layer_lb.append(out_lb)
                    new_layer_ub.append(out_ub)
                    new_layer_gurobi_vars.append(out_i)
            else:
                # it's the last layer, there's no activations, just the linear part
                u_dxixi_lbs = u_dxixi_lower_bounds[-1]
                u_dxixi_ubs = u_dxixi_upper_bounds[-1]
                u_dxixi_pre_vars = u_dxixi_gurobi_vars[-1]

                pos_w = torch.clamp(layer.weight, min=0)
                neg_w = torch.clamp(layer.weight, max=0)

                u_dxixi_out_lbs = pos_w @ u_dxixi_lbs + neg_w @ u_dxixi_ubs
                u_dxixi_out_ubs = pos_w @ u_dxixi_ubs + neg_w @ u_dxixi_lbs

                for neuron_idx in range(len(u_dxixi_out_lbs)):
                    final_var = add_var_to_model(
                        grb_model,
                        u_dxixi_out_lbs[neuron_idx],
                        u_dxixi_out_ubs[neuron_idx],
                        f'u_d{self.partial_derivative.component_idx}_{self.component_idx}_{layer_idx}_{neuron_idx}_final_var'
                    )

                    coeffs = layer.weight[neuron_idx, :]
                    lin_expr = grb.LinExpr(coeffs, u_dxixi_pre_vars)
                    grb_model.addConstr(final_var == lin_expr)

                    out_lb, out_ub = solve_for_lower_upper_bound_var(grb_model, final_var)

                    new_layer_lb.append(out_lb)
                    new_layer_ub.append(out_ub)
                    new_layer_gurobi_vars.append(final_var)

            u_dxixi_gurobi_vars.append(new_layer_gurobi_vars)
            u_dxixi_lower_bounds.append(torch.Tensor(new_layer_lb))
            u_dxixi_upper_bounds.append(torch.Tensor(new_layer_ub))

        self.computed_bounds = True

    def add_vars_and_constraints(self, grb_model: grb.Model, u_theta_gurobi_vars: List[List[grb.Var]] = None, u_dxi_theta_all_vars: List[List[grb.Var]] = None):
        # if we haven't computed the intermediate bounds on u_theta, do it now
        if not self.partial_derivative.computed_bounds:
            self.partial_derivative.compute_bounds()

        # proceed if the computation is successful and these variables are now populated
        assert self.u_theta.computed_bounds
        assert self.partial_derivative.computed_bounds

        if u_theta_gurobi_vars is None:
            u_theta_all_vars = self.u_theta.add_vars_and_constraints(grb_model)
            u_theta_gurobi_vars = u_theta_all_vars[0]

        if u_dxi_theta_all_vars is None:
            _, u_dxi_theta_all_vars = self.partial_derivative.add_vars_and_constraints(
                self.grb_model, u_theta_gurobi_vars=u_theta_gurobi_vars
            )

        u_dxi_gurobi_vars, d_phi_m_d_phi_m_1_gurobi_vars = u_dxi_theta_all_vars

        # computing a bound on $\partial_t u_theta$ using the previously computed bounds
        u_dxixi_gurobi_vars = []
        u_dxixi_lower_bounds = self.lower_bounds
        u_dxixi_upper_bounds = self.upper_bounds

        # d_psi_0_dx_i is equal to 0 (second derivative of x with respect to x_i) and it'll remain the
        # same through the normalization layers
        zero_vec = [0 for _ in range(self.u_theta.fc_layers[0].weight.shape[1])]
        u_dxixi_gurobi_vars.append(zero_vec)

        for layer_idx, layer in enumerate(self.u_theta.fc_layers):
            is_final = (layer_idx == len(self.u_theta.fc_layers) - 1)
            new_layer_gurobi_vars = []

            # activations are subsumed in the gradient of the linear layers
            if type(layer) in supported_activations:
                continue

            assert isinstance(layer, torch.nn.Linear)

            if not is_final:
                # load all the previous variables required for the computation of this block
                u_theta_pre_activation_lbs = self.u_theta.lower_bounds[layer_idx+1]
                u_theta_pre_activation_ubs = self.u_theta.upper_bounds[layer_idx+1]
                u_theta_pre_activation_vars = u_theta_gurobi_vars[layer_idx+1]

                u_dxi_lbs = self.partial_derivative.lower_bounds[int(layer_idx / 2)]
                u_dxi_ubs = self.partial_derivative.upper_bounds[int(layer_idx / 2)]
                u_dxi_pre_vars = u_dxi_gurobi_vars[int(layer_idx / 2)]

                d_phi_m_d_phi_m_1_ubs = self.partial_derivative.d_phi_m_d_phi_m_1_ubs[int(layer_idx / 2)]
                d_phi_m_d_phi_m_1_lbs = self.partial_derivative.d_phi_m_d_phi_m_1_lbs[int(layer_idx / 2)]
                d_phi_m_d_phi_m_1_vars = d_phi_m_d_phi_m_1_gurobi_vars[int(layer_idx / 2)]
                
                u_dxixi_lbs = u_dxixi_lower_bounds[int(layer_idx / 2)]
                u_dxixi_ubs = u_dxixi_upper_bounds[int(layer_idx / 2)]
                u_dxixi_pre_vars = u_dxixi_gurobi_vars[int(layer_idx / 2)]

                # ----- compute first dpsi_1/dx_i -----

                pos_w = torch.clamp(layer.weight, min=0)
                neg_w = torch.clamp(layer.weight, max=0)

                first_matmul_lbs = pos_w @ u_dxi_lbs + neg_w @ u_dxi_ubs
                first_matmul_ubs = pos_w @ u_dxi_ubs + neg_w @ u_dxi_lbs

                first_matmul_vars = []
                first_matmul_lin_expressions = layer.weight.detach().numpy() @ np.array(u_dxi_pre_vars)
                for i, expression in enumerate(first_matmul_lin_expressions):
                    first_matmul_i = add_var_to_model(
                        grb_model,
                        first_matmul_lbs[i],
                        first_matmul_ubs[i],
                        f'u_d{self.partial_derivative.component_idx}_{self.component_idx}_{layer_idx}_first_matmul_{i}'
                    )
                    
                    grb_model.addConstr(first_matmul_i == expression)

                    first_matmul_vars.append(first_matmul_i)

                # first step is to bound the output of \sigma''(pre_activation_output)
                sigma_prime_prime_output_vars = []
                sigma_prime_prime_output_lbs = []
                sigma_prime_prime_output_ubs = []
                for neuron_idx, (u_theta_pre_act_var, u_theta_pre_act_neuron_lb, u_theta_pre_act_neuron_ub) in \
                        enumerate(zip(u_theta_pre_activation_vars, u_theta_pre_activation_lbs, u_theta_pre_activation_ubs)):
                    # sigma'' = sigmoid' = sigmoid * (1 - sigmoid) is not a monotonic function, so need to be careful about that
                    neuron_sigma_prime_prime_lb, neuron_sigma_prime_prime_ub = self.activation_derivative_derivative_relaxation.get_lb_ub_in_interval(
                        u_theta_pre_act_neuron_lb, u_theta_pre_act_neuron_ub
                    )

                    neuron_sigma_prime_prime = add_var_to_model(
                        grb_model,
                        neuron_sigma_prime_prime_lb,
                        neuron_sigma_prime_prime_ub,
                        f'u_d{self.partial_derivative.component_idx}_{self.component_idx}_{layer_idx}_sigma_prime_prime_{neuron_idx}'
                    )

                    # relax the derivative of the relaxation
                    self.activation_derivative_derivative_relaxation.relax_lp(
                        grb_model,
                        u_theta_pre_act_var,
                        neuron_sigma_prime_prime,
                        u_theta_pre_act_neuron_lb,
                        u_theta_pre_act_neuron_ub
                    )

                    sigma_prime_prime_output_vars.append(neuron_sigma_prime_prime)
                    sigma_prime_prime_output_lbs.append(neuron_sigma_prime_prime_lb)
                    sigma_prime_prime_output_ubs.append(neuron_sigma_prime_prime_ub)


                # compute elementwise_multiply(sigma_prime_output_vars, first_matmul_vars)
                elem_multiply_vars = []
                elem_multiply_lbs = []
                elem_multiply_ubs = []
                for i in range(len(sigma_prime_prime_output_vars)):
                    elem_multiply_lb_i, elem_multiply_ub_i = get_multiplication_bounds(
                        first_matmul_lbs[i], first_matmul_ubs[i],
                        sigma_prime_prime_output_lbs[i], sigma_prime_prime_output_ubs[i]
                    )

                    elem_multiply_var_i = add_var_to_model(
                        grb_model,
                        elem_multiply_lb_i,
                        elem_multiply_ub_i,
                        f'u_d{self.partial_derivative.component_idx}_{self.component_idx}_{layer_idx}_elemwise_mul_{i}'
                    )

                    # exact modeling - non-convex
                    # grb_model.addConstr(elem_multiply_var_i == first_matmul_vars[i] * sigma_prime_prime_output_vars[i])

                    mccormick_relaxation(
                        grb_model,
                        input_var_1=first_matmul_vars[i],
                        lb_var_1=first_matmul_lbs[i],
                        ub_var_1=first_matmul_ubs[i],
                        input_var_2=sigma_prime_prime_output_vars[i],
                        lb_var_2=sigma_prime_prime_output_lbs[i],
                        ub_var_2=sigma_prime_prime_output_ubs[i],
                        output_var=elem_multiply_var_i
                    )

                    elem_multiply_vars.append(elem_multiply_var_i)
                    elem_multiply_lbs.append(elem_multiply_lb_i)
                    elem_multiply_ubs.append(elem_multiply_ub_i)

                # finally, compute matmul(diag(elem_multiply_vars), W)
                layer_weight = layer.weight.detach().numpy()
                first_chain_terms = []
                first_chain_term_lbs = []
                first_chain_term_ubs = []
                for i in range(len(elem_multiply_vars)):
                    elem_multiply_var = elem_multiply_vars[i]
                    elem_multiply_lb = elem_multiply_lbs[i]
                    elem_multiply_ub = elem_multiply_ubs[i]

                    row_vars = []
                    row_lbs = []
                    row_ubs = []
                    for j in range(layer_weight.shape[1]):
                        weight = layer_weight[i, j]
                        lb = min(weight*elem_multiply_lb, weight*elem_multiply_ub)
                        ub = max(weight*elem_multiply_lb, weight*elem_multiply_ub)

                        first_chain_term = add_var_to_model(
                            grb_model,
                            lb,
                            ub,
                            f'u_d{self.partial_derivative.component_idx}_{self.component_idx}__{layer_idx}_first_chain_term_{i}_{j}'
                        )
                        
                        grb_model.addConstr(first_chain_term == weight * elem_multiply_var)
                        row_vars.append(first_chain_term)
                        row_lbs.append(lb)
                        row_ubs.append(ub)
                    
                    first_chain_terms.append(row_vars)
                    first_chain_term_lbs.append(row_lbs)
                    first_chain_term_ubs.append(row_ubs)

                # ----- end of computation of dpsi_1/dx_i -----

                # ----- first chain multiplication, it's matmul(first_chain_terms, u_dxi_pre_vars) ----- 
                first_sum_terms = []
                first_sum_term_lbs = self.first_sum_term_lbs[int(layer_idx / 2)]
                first_sum_term_ubs = self.first_sum_term_ubs[int(layer_idx / 2)]
                for out_neuron_idx in range(len(first_chain_terms)):
                    each_multiplication_vars = []
                    each_multiplication_lbs = []
                    each_multiplication_ubs = []

                    for j in range(len(u_dxi_pre_vars)):
                        jth_out_lb, jth_out_ub = get_multiplication_bounds(
                            first_chain_term_lbs[out_neuron_idx][j], first_chain_term_ubs[out_neuron_idx][j],
                            u_dxi_lbs[j], u_dxi_ubs[j]
                        )

                        jth_out_var = add_var_to_model(
                            grb_model,
                            jth_out_lb,
                            jth_out_ub,
                            f'u_d{self.partial_derivative.component_idx}_{self.component_idx}_{layer_idx}_first_sum_term_intermediate_{out_neuron_idx}_{j}th_term'
                        )

                        # exact modeling - non-convex
                        # grb_model.addConstr(jth_out_var == first_chain_terms[out_neuron_idx][j] * u_dxi_pre_vars[j])

                        mccormick_relaxation(
                            grb_model,
                            input_var_1=first_chain_terms[out_neuron_idx][j],
                            lb_var_1=first_chain_term_lbs[out_neuron_idx][j],
                            ub_var_1=first_chain_term_ubs[out_neuron_idx][j],
                            input_var_2=u_dxi_pre_vars[j],
                            lb_var_2=u_dxi_lbs[j],
                            ub_var_2=u_dxi_ubs[j],
                            output_var=jth_out_var
                        )

                        each_multiplication_vars.append(jth_out_var)
                        each_multiplication_lbs.append(jth_out_lb)
                        each_multiplication_ubs.append(jth_out_ub)

                    first_sum_term = grb_model.addVar(
                        lb=first_sum_term_lbs[out_neuron_idx], ub=first_sum_term_ubs[out_neuron_idx],
                        obj=0, vtype=grb.GRB.CONTINUOUS,
                        name=f'u_d{self.partial_derivative.component_idx}_{self.component_idx}_{layer_idx}_first_sum_term_{out_neuron_idx}'
                    )
                    grb_model.addConstr(first_sum_term == sum(each_multiplication_vars))

                    first_sum_terms.append(first_sum_term)

                # ----- second chain multiplication, it's matmul(d_phi_m_d_phi_m_1_vars, u_dxixi_pre_vars) -----
                second_sum_terms = []
                second_sum_term_lbs = self.second_sum_term_lbs[int(layer_idx / 2)]
                second_sum_term_ubs = self.second_sum_term_ubs[int(layer_idx / 2)]
                for out_neuron_idx in range(len(d_phi_m_d_phi_m_1_vars)):
                    each_multiplication_vars = []
                    each_multiplication_lbs = []
                    each_multiplication_ubs = []

                    for j in range(len(u_dxixi_pre_vars)):
                        jth_out_lb, jth_out_ub = get_multiplication_bounds(
                            d_phi_m_d_phi_m_1_lbs[out_neuron_idx][j], d_phi_m_d_phi_m_1_ubs[out_neuron_idx][j],
                            u_dxixi_lbs[j], u_dxixi_ubs[j]
                        )

                        jth_out_var = add_var_to_model(
                            grb_model,
                            jth_out_lb,
                            jth_out_ub,
                            f'u_d{self.partial_derivative.component_idx}_{self.component_idx}_{layer_idx}_second_sum_term_intermediate_{out_neuron_idx}_{j}th_term'
                        )

                        # exact modeling - non-convex
                        # grb_model.addConstr(jth_out_var == d_phi_m_d_phi_m_1_vars[out_neuron_idx][j] * u_dxixi_pre_vars[j])

                        mccormick_relaxation(
                            grb_model,
                            input_var_1=u_dxixi_pre_vars[j],
                            lb_var_1=u_dxixi_lbs[j],
                            ub_var_1=u_dxixi_ubs[j],
                            input_var_2=d_phi_m_d_phi_m_1_vars[out_neuron_idx][j],
                            lb_var_2=d_phi_m_d_phi_m_1_lbs[out_neuron_idx][j],
                            ub_var_2=d_phi_m_d_phi_m_1_ubs[out_neuron_idx][j],
                            output_var=jth_out_var
                        )

                        each_multiplication_vars.append(jth_out_var)
                        each_multiplication_lbs.append(jth_out_lb)
                        each_multiplication_ubs.append(jth_out_ub)

                    second_sum_term = add_var_to_model(
                        grb_model,
                        second_sum_term_lbs[out_neuron_idx],
                        second_sum_term_ubs[out_neuron_idx],
                        f'u_d{self.partial_derivative.component_idx}_{self.component_idx}_{layer_idx}_second_sum_term_{neuron_idx}'
                    )
                    grb_model.addConstr(second_sum_term == sum(each_multiplication_vars))

                    second_sum_terms.append(second_sum_term)

                # actual sum output of the layer
                u_dxixi_lower_bounds_layer = u_dxixi_lower_bounds[int(layer_idx / 2) + 1]
                u_dxixi_upper_bounds_layer = u_dxixi_upper_bounds[int(layer_idx / 2) + 1]
                for i in range(len(first_sum_terms)):
                    out_i = add_var_to_model(
                        grb_model,
                        u_dxixi_lower_bounds_layer[i],
                        u_dxixi_upper_bounds_layer[i],
                        f'u_d{self.partial_derivative.component_idx}_{self.component_idx}_{layer_idx}_out_{i}'
                    )
                    grb_model.addConstr(out_i == first_sum_terms[i] + second_sum_terms[i])

                    new_layer_gurobi_vars.append(out_i)
            else:
                # it's the last layer, there's no activations, just the linear part
                u_dxixi_pre_vars = u_dxixi_gurobi_vars[-1]

                u_dxixi_lower_bounds_layer = u_dxixi_lower_bounds[-1]
                u_dxixi_upper_bounds_layer = u_dxixi_upper_bounds[-1]
                for neuron_idx in range(len(u_dxixi_lower_bounds_layer)):
                    final_var = add_var_to_model(
                        grb_model,
                        u_dxixi_lower_bounds_layer[neuron_idx],
                        u_dxixi_upper_bounds_layer[neuron_idx],
                        f'u_d{self.partial_derivative.component_idx}_{self.component_idx}_{layer_idx}_{neuron_idx}_final_var'
                    )

                    coeffs = layer.weight[neuron_idx, :]
                    lin_expr = grb.LinExpr(coeffs, u_dxixi_pre_vars)
                    grb_model.addConstr(final_var == lin_expr)

                    new_layer_gurobi_vars.append(final_var)

            u_dxixi_gurobi_vars.append(new_layer_gurobi_vars)

        grb_model.update()
        return [u_theta_gurobi_vars], [u_dxi_gurobi_vars, d_phi_m_d_phi_m_1_gurobi_vars], [u_dxixi_gurobi_vars]


class LPBurgersVerifier():
    def __init__(
            self, model: List[torch.nn.Module], activation_relaxation: ActivationRelaxationType, 
            activation_derivative_relaxation: ActivationRelaxationType,
            activation_second_derivative_relaxation: ActivationRelaxationType,
            feasibility_tol: float = 1e-6, optimality_tol: float = 1e-6
    ) -> None:
        self.u_theta = LPPINNSolution(
            model,
            activation_relaxation=activation_relaxation,
            feasibility_tol=feasibility_tol,
            optimality_tol=optimality_tol
        )

        self.u_dt_theta = LPPINNPartialDerivative(
            self.u_theta,
            component_idx=0,
            activation_derivative_relaxation=activation_derivative_relaxation
        )
        self.u_dx_theta = LPPINNPartialDerivative(
            self.u_theta,
            component_idx=1,
            activation_derivative_relaxation=activation_derivative_relaxation
        )
        self.u_dxdx_theta = LPPINNSecondPartialDerivative(
            self.u_dx_theta,
            component_idx=1,
            activation_derivative_derivative_relaxation=activation_second_derivative_relaxation
        )

    def compute_residual_bound(self, domain_bounds: torch.tensor, debug: bool = True):
        # compute all the intermediate bounds of the components
        u_theta = self.u_theta
        u_theta.domain_bounds = domain_bounds
        u_theta.compute_bounds(debug=debug)

        if debug:
            print("u_theta bounds:", (u_theta.lower_bounds[-1], u_theta.upper_bounds[-1]))

        u_dt_theta = self.u_dt_theta
        u_dt_theta.compute_bounds(debug=debug)

        if debug:
            print("u_dt_theta bounds:", (u_dt_theta.lower_bounds[-1], u_dt_theta.upper_bounds[-1]))

        u_dx_theta = self.u_dx_theta
        u_dx_theta.compute_bounds(debug=debug)

        if debug:
            print("u_dx_theta bounds:", (u_dx_theta.lower_bounds[-1], u_dx_theta.upper_bounds[-1]))

        u_dxdx_theta = self.u_dxdx_theta
        u_dxdx_theta.compute_bounds(debug=debug)

        if debug:
            print("u_dxdx_theta bounds:", (u_dxdx_theta.lower_bounds[-1], u_dxdx_theta.upper_bounds[-1]))

        # create a problem with all intermediate bounds to solve for the residual
        grb_model = new_gurobi_model(
            f"f",
            feasibility_tol=u_theta.feasibility_tol,
            optimality_tol=u_theta.optimality_tol
        )

        u_theta_gurobi_vars = u_theta.add_vars_and_constraints(grb_model)

        _, u_dt_theta_all_vars = u_dt_theta.add_vars_and_constraints(
            grb_model,
            u_theta_gurobi_vars=u_theta_gurobi_vars
        )
        u_dt_theta_gurobi_vars = u_dt_theta_all_vars[0]

        _, u_dx_theta_all_vars = u_dx_theta.add_vars_and_constraints(
            grb_model,
            u_theta_gurobi_vars=u_theta_gurobi_vars
        )
        u_dx_theta_gurobi_vars = u_dx_theta_all_vars[0]

        _, _, u_dxdx_theta_all_vars = u_dxdx_theta.add_vars_and_constraints(
            grb_model,
            u_theta_gurobi_vars=u_theta_gurobi_vars,
            u_dxi_theta_all_vars=u_dx_theta_all_vars
        )
        u_dxdx_theta_gurobi_vars = u_dxdx_theta_all_vars[0]

        # the PINN is given by u_dt_theta + u_theta * u_dx_theta - 0.01/np.pi * u_dxdx_theta
        u_theta_u_dx_theta_mul_lb, u_theta_u_dx_theta_mul_ub = get_multiplication_bounds(
            u_theta.lower_bounds[-1][0], u_theta.upper_bounds[-1][0],
            u_dx_theta.lower_bounds[-1][0], u_dx_theta.upper_bounds[-1][0]
        )

        u_theta_u_dx_theta_mul = grb_model.addVar(
            lb=u_theta_u_dx_theta_mul_lb, ub=u_theta_u_dx_theta_mul_ub,
            obj=0, vtype=grb.GRB.CONTINUOUS,
            name=f'u_theta_u_dx_theta_mul'
        )
        # exact modeling - non-convex
        # grb_model.addConstr(u_theta_u_dx_theta_mul == u_theta_gurobi_vars[-1][0] * u_dx_theta_gurobi_vars[-1][0])

        if (u_theta_u_dx_theta_mul_ub - u_theta_u_dx_theta_mul_lb >= 1e-4):
            # mccormick relaxations; can be subject to numerical instability, so only do it if the interval is decent
            grb_model.addConstr(u_theta_u_dx_theta_mul >= u_theta_gurobi_vars[-1][0] * u_dx_theta.lower_bounds[-1][0] - u_theta.lower_bounds[-1][0] * u_dx_theta.lower_bounds[-1][0] + u_theta.lower_bounds[-1][0] * u_dx_theta_gurobi_vars[-1][0])
            grb_model.addConstr(u_theta_u_dx_theta_mul >= u_theta_gurobi_vars[-1][0] * u_dx_theta.upper_bounds[-1][0] - u_theta.upper_bounds[-1][0] * u_dx_theta.upper_bounds[-1][0] + u_theta.upper_bounds[-1][0] * u_dx_theta_gurobi_vars[-1][0])

            grb_model.addConstr(u_theta_u_dx_theta_mul <= u_theta_gurobi_vars[-1][0] * u_dx_theta.lower_bounds[-1][0] - u_theta.upper_bounds[-1][0] * u_dx_theta.lower_bounds[-1][0] + u_theta.upper_bounds[-1][0] * u_dx_theta_gurobi_vars[-1][0])
            grb_model.addConstr(u_theta_u_dx_theta_mul <= u_theta_gurobi_vars[-1][0] * u_dx_theta.upper_bounds[-1][0] - u_theta.lower_bounds[-1][0] * u_dx_theta.upper_bounds[-1][0] + u_theta.lower_bounds[-1][0] * u_dx_theta_gurobi_vars[-1][0])

        residual_lb = u_dt_theta.lower_bounds[-1][0] + u_theta_u_dx_theta_mul_lb - 0.01/np.pi * u_dxdx_theta.upper_bounds[-1][0]
        residual_ub = u_dt_theta.upper_bounds[-1][0] + u_theta_u_dx_theta_mul_ub - 0.01/np.pi * u_dxdx_theta.lower_bounds[-1][0]

        residual = grb_model.addVar(
            lb=residual_lb, ub=residual_ub,
            obj=0, vtype=grb.GRB.CONTINUOUS,
            name=f'residual'
        )
        grb_model.addConstr(residual == u_dt_theta_gurobi_vars[-1][0] + u_theta_u_dx_theta_mul - 0.01/np.pi * u_dxdx_theta_gurobi_vars[-1][0])

        if debug:
            print("Computing residual min and max...")

        grb_model.setObjective(residual, grb.GRB.MINIMIZE)
        grb_model.update()
        grb_model.reset()
        grb_model.optimize()

        if grb_model.status == 3:
            import pdb
            pdb.set_trace()

        out_lb = residual.X

        grb_model.setObjective(residual, grb.GRB.MAXIMIZE)
        grb_model.update()
        grb_model.reset()
        grb_model.optimize()

        if grb_model.status == 3:
            import pdb
            pdb.set_trace()

        out_ub = residual.X

        return out_lb, out_ub
