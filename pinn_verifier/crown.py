from enum import Enum
import copy
from typing import List, Tuple

from scipy import optimize
from tqdm import tqdm
import numpy as np
import torch
import gurobipy as grb

from tools.custom_torch_modules import Add, Mul
from pinn_verifier.activations.activation_relaxations import ActivationRelaxationType, ActivationRelaxation

supported_normalization_ops = [Add, Mul]
supported_activations = [torch.nn.Softplus, torch.nn.Tanh]


class BackpropMode(Enum):
    BLOCK_BACKPROP = 0
    COMPONENT_BACKPROP = 1
    FULL_BACKPROP = 2


class CROWNPINNSolution():
    def __init__(self, model: List[torch.nn.Module], activation_relaxation: ActivationRelaxation) -> None:
        # class used to represent a CROWNPINNSolution
        # currently supported model is a fully connected network with a few normalization layers (Add and Mul only)
        if not self.is_model_supported(model):
            raise ValueError(
                "model passed to CROWNPINNSolution is not supported in current implementation"
            )
        
        self.layers = model
        self.norm_layers, self.fc_layers = self.separate_norm_and_fc_layers(model)

        self._domain_bounds = None
        self.computed_bounds = False
        self.lower_bounds = []
        self.upper_bounds = []
        self.layer_CROWN_coefficients = []

        self.activation_relaxation = activation_relaxation

    def clear_bound_computations(self):
        self.computed_bounds = False
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

    @staticmethod
    def optimized_backward_crown(Ws, bs, pre_act_UBs, pre_act_LBs, bound_lines):
        with torch.no_grad():
            nlayer = len(Ws)

            Delta = [None] * (nlayer + 1)
            Delta[nlayer] = torch.zeros(len(bs[-1]), len(bs[-1]))
            Delta[0] = torch.zeros(len(bs[-1]), len(bs[-1]))

            Theta = [None] * (nlayer + 1)
            Theta[nlayer] = torch.zeros(len(bs[-1]), len(bs[-1]))
            Theta[0] = torch.zeros(len(bs[-1]), len(bs[-1]))

            Lambda = [None] * (nlayer + 1)
            Lambda[nlayer] = torch.eye(len(bs[-1]))

            Omega = [None] * (nlayer + 1)
            Omega[nlayer] = torch.eye(len(bs[-1]))

            for k in range(nlayer-1, -1, -1):
                Lambda_W_multiply = (Lambda[k+1] @ Ws[k])
                Omega_W_multiply = (Omega[k+1] @ Ws[k])

                bound_lines_k = bound_lines[k-1]
                alpha_L = bound_lines_k[:,0]
                beta_L = bound_lines_k[:,1]
                alpha_U = bound_lines_k[:,2]
                beta_U = bound_lines_k[:,3]

                if k == 0:
                    lambda_ = torch.ones(len(bs[-1]), Ws[0].shape[1])
                    omega = torch.ones(len(bs[-1]), Ws[0].shape[1])
                else:
                    lambda_ = (Lambda_W_multiply >= 0) * alpha_U + (Lambda_W_multiply < 0) * alpha_L
                    omega = (Omega_W_multiply >= 0) * alpha_L + (Omega_W_multiply < 0) * alpha_U

                if k != 0:
                    Delta[k] = (Lambda_W_multiply >= 0) * beta_U + (Lambda_W_multiply < 0) * beta_L
                    Theta[k] = (Omega_W_multiply >= 0) * beta_L + (Omega_W_multiply < 0) * beta_U

                Lambda[k] = Lambda_W_multiply * lambda_
                Omega[k] = Omega_W_multiply * omega

            # compute the end expressions
            bs_Delta = [bs_k + Delta_k for (bs_k, Delta_k) in zip(bs, Delta[1:])]
            bs_Theta = [bs_k + Theta_k for (bs_k, Theta_k) in zip(bs, Theta[1:])]

            f_U_A_0 = Lambda[0]
            f_U_constant = sum([torch.sum(Lambda_k * bs_Delta_k, dim=1) for (Lambda_k, bs_Delta_k) in zip(Lambda[1:], bs_Delta)])

            f_L_A_0 = Omega[0]
            f_L_constant = sum([torch.sum(Omega_k * bs_Theta_k, dim=1) for (Omega_k, bs_Theta_k) in zip(Omega[1:], bs_Theta)])

            # take lower bound of x[i] if f_U_j_A_0[i] is negative, upper bound of x[i] if f_U_j_A_0[i] is positive
            UB_final = torch.sum(f_U_A_0 * ((f_U_A_0 >= 0) * pre_act_UBs[0] + (f_U_A_0 < 0) * pre_act_LBs[0]), dim=1) + f_U_constant

            # take upper bound of x[i] if f_L_j_A_0[i] is negative, lower bound of x[i] if f_L_j_A_0[i] is positive
            LB_final = torch.sum(f_L_A_0 * ((f_L_A_0 >= 0) * pre_act_LBs[0] + (f_L_A_0 < 0) * pre_act_UBs[0]), dim=1) + f_L_constant

        assert all(UB_final >= LB_final)

        return UB_final, LB_final, (f_U_A_0, f_U_constant, f_L_A_0, f_L_constant)

    def compute_bounds(self, debug: bool = True, backprop_mode: BackpropMode = BackpropMode.FULL_BACKPROP):
        domain_bounds = self.domain_bounds

        point = self.domain_bounds.mean(dim=0).unsqueeze(0)

        # apply normalization layers on the bounds
        for layer in self.norm_layers:
            domain_bounds = layer(domain_bounds)
            point = layer(point)
            
            if type(layer) == Mul:
                # a multiplication can change the direction of the bounds, sort them accordingly
                domain_bounds = domain_bounds.sort(dim=0).values
        
        current_lb = domain_bounds[0]
        current_ub = domain_bounds[1]

        self.x_lb = current_lb
        self.x_ub = current_ub

        Ws = []
        bs = []
        pre_act_LBs = [current_lb]
        pre_act_UBs = [current_ub]
        layers_output_alpha_betas = []
        all_crown_lbs = self.lower_bounds
        all_crown_ubs = self.upper_bounds

        all_crown_lbs.append(current_lb)
        all_crown_ubs.append(current_ub)

        lin_layers = [layer for layer in self.fc_layers if isinstance(layer, torch.nn.Linear)]
        for n_layer, layer in enumerate(lin_layers):
            Ws.append(layer.weight)
            bs.append(layer.bias)

            if n_layer == 0:
                # first layer, compute the bounds in close form easily
                pos_weights = torch.clamp(layer.weight, min=0)
                neg_weights = torch.clamp(layer.weight, max=0)

                pre_act_layer_lb = pos_weights @ current_lb + neg_weights @ current_ub + layer.bias
                pre_act_layer_ub = pos_weights @ current_ub + neg_weights @ current_lb + layer.bias

                pre_act_LBs.append(pre_act_layer_lb)
                pre_act_UBs.append(pre_act_layer_ub)

                all_crown_lbs.append(pre_act_layer_lb)
                all_crown_ubs.append(pre_act_layer_ub)

                self.layer_CROWN_coefficients.append((layer.weight, layer.bias, layer.weight, layer.bias))

                point = layer(point)
            else:
                layer_output_lines = torch.zeros(len(pre_act_UBs[-1]), 4)
                post_act_lbs = torch.zeros(len(pre_act_UBs[-1]))
                post_act_ubs = torch.zeros(len(pre_act_UBs[-1]))
                for i in range(len(pre_act_UBs[-1])):
                    lb_line, ub_line = self.activation_relaxation.get_bounds(pre_act_LBs[-1][i], pre_act_UBs[-1][i])
                    layer_output_lines[i, :] = torch.tensor([lb_line[0], (lb_line[1]-1e-6) / lb_line[0], ub_line[0], (ub_line[1]+1e-6) / ub_line[0]])

                    # add the post activation bounds to all bounds
                    if ub_line[0] >= 0:
                        post_act_ubs[i] = ub_line[0] * pre_act_UBs[-1][i] + ub_line[1]
                    else:
                        post_act_ubs[i] = ub_line[0] * pre_act_LBs[-1][i] + ub_line[1]
                    
                    if lb_line[0] >= 0:
                        post_act_lbs[i] = lb_line[0] * pre_act_LBs[-1][i] + lb_line[1]
                    else:
                        post_act_lbs[i] = lb_line[0] * pre_act_UBs[-1][i] + lb_line[1]
                    
                all_crown_lbs.append(post_act_lbs)
                all_crown_ubs.append(post_act_ubs)
                
                layers_output_alpha_betas.append(layer_output_lines)

                # hybrid mode
                if backprop_mode == backprop_mode.BLOCK_BACKPROP:
                    f_U_A_0, f_U_constant, f_L_A_0, f_L_constant = self.layer_CROWN_coefficients[-1]

                    # if n_layer != len(lin_layers) - 1:
                    # ---- DO NOT EDIT: THIS WORKS ----
                    alpha_L, beta_L, alpha_U, beta_U = layer_output_lines.T

                    z_k_1_U_coefficient = alpha_U.clamp(min=0).unsqueeze(1) * f_U_A_0 + alpha_U.clamp(max=0).unsqueeze(1) * f_L_A_0
                    z_k_1_U_constant = alpha_U.clamp(min=0) * f_U_constant + alpha_U.clamp(max=0) * f_L_constant + alpha_U * beta_U

                    z_k_1_L_coefficient = alpha_L.clamp(min=0).unsqueeze(1) * f_L_A_0 + alpha_L.clamp(max=0).unsqueeze(1) * f_U_A_0
                    z_k_1_L_constant = alpha_L.clamp(min=0) * f_L_constant + alpha_L.clamp(max=0) * f_U_constant + alpha_L * beta_L

                    # z_k_1_upper_bounds = torch.sum(z_k_1_U_coefficient.clamp(min=0) * self.x_ub + z_k_1_U_coefficient.clamp(max=0) * self.x_lb, dim=1) + z_k_1_U_constant
                    # z_k_1_lower_bounds = torch.sum(z_k_1_L_coefficient.clamp(min=0) * self.x_lb + z_k_1_L_coefficient.clamp(max=0) * self.x_ub, dim=1) + z_k_1_L_constant

                    z_k_point = self.activation_relaxation.evaluate(point)

                    # assert all(z_k_point.flatten() <= z_k_1_upper_bounds.flatten())
                    # assert all(z_k_point.flatten() >= z_k_1_lower_bounds.flatten())

                    output_U_coefficient = layer.weight.clamp(min=0) @ z_k_1_U_coefficient + layer.weight.clamp(max=0) @ z_k_1_L_coefficient
                    output_U_constant = layer.weight.clamp(min=0) @ z_k_1_U_constant + layer.weight.clamp(max=0) @ z_k_1_L_constant + layer.bias

                    output_L_coefficient = layer.weight.clamp(min=0) @ z_k_1_L_coefficient + layer.weight.clamp(max=0) @ z_k_1_U_coefficient
                    output_L_constant = layer.weight.clamp(min=0) @ z_k_1_L_constant + layer.weight.clamp(max=0) @ z_k_1_U_constant + layer.bias
                    # ---- DO NOT EDIT ----

                    layer_output_upper_bounds = torch.sum(output_U_coefficient.clamp(min=0) * self.x_ub + output_U_coefficient.clamp(max=0) * self.x_lb, dim=1) + output_U_constant
                    layer_output_lower_bounds = torch.sum(output_L_coefficient.clamp(min=0) * self.x_lb + output_L_coefficient.clamp(max=0) * self.x_ub, dim=1) + output_L_constant

                    # DEBUG - sanity checks
                    point = layer(z_k_point)
                    try:
                        assert all(layer_output_lower_bounds.flatten() <= layer_output_upper_bounds.flatten())
                        assert all(point.flatten() <= layer_output_upper_bounds.flatten())
                        assert all(point.flatten() >= layer_output_lower_bounds.flatten())
                    except:
                        import pdb
                        pdb.set_trace()

                    UB = layer_output_upper_bounds
                    LB = layer_output_lower_bounds
                    CROWN_coefficients = output_U_coefficient, output_U_constant, output_L_coefficient, output_L_constant
                else:
                    # ful`ly backward mode
                    UB, LB, CROWN_coefficients = self.optimized_backward_crown(Ws, bs, pre_act_UBs, pre_act_LBs, layers_output_alpha_betas)

                pre_act_LBs.append(LB)
                pre_act_UBs.append(UB)

                all_crown_lbs.append(LB)
                all_crown_ubs.append(UB)

                self.layer_CROWN_coefficients.append(CROWN_coefficients)
        
        self.pre_act_LBs = pre_act_LBs
        self.pre_act_UBs = pre_act_UBs
        self.layers_output_alpha_betas = layers_output_alpha_betas
        self.computed_bounds = True


class CROWNPINNPartialDerivative():
    def __init__(self, pinn_solution: CROWNPINNSolution, component_idx: int, activation_derivative_relaxation: ActivationRelaxation) -> None:
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
        self.lower_bounds = []
        self.upper_bounds = []
        self.partial_z_k_z_k_1_ubs = []
        self.partial_z_k_z_k_1_lbs = []

        self.activation_derivative_relaxation = activation_derivative_relaxation
    
    def backward_component_propagation(
            self,
            layers: List[torch.nn.Module],
            prev_layers_activation_derivative_output_bounds: List[torch.Tensor],
            prev_layers_u_theta_coefficients: List[List[torch.Tensor]],
            prev_layers_u_dxi_upper_bound: List[torch.Tensor],
            prev_layers_u_dxi_lower_bound: List[torch.Tensor],
            x_lb: torch.Tensor,
            x_ub:  torch.Tensor,
            is_final: bool = False,
            xi_i_j: float = 0.5,
            psi_i_j: float = 0.5
    ):
        rho_0_U = None
        rho_1_U = None
        rho_2_U = None
        rho_0_L = None
        rho_1_L = None
        rho_2_L = None

        n_layers = len(layers)
        if is_final:
            # remove the last couple of layers, as the last one is explicitly considered below when propagating the last hidden layer
            n_layers -= 2
        
        with torch.no_grad():
            for n_layer in range(n_layers-1, -1, -1):
                layer = layers[n_layer]
                if not isinstance(layer, torch.nn.Linear):
                    continue

                # improves efficiency by caching computation of alphas and betas when doing it only at the component level (since there's no dependency on previous layers)
                # has this layer been cached already? if so, use those values
                if n_layer in self.component_non_backward_dependencies:
                    alpha_k_0 = self.component_non_backward_dependencies[n_layer]["alpha_k_0"]
                    alpha_k_3 = self.component_non_backward_dependencies[n_layer]["alpha_k_3"]
                    alpha_k_4 = self.component_non_backward_dependencies[n_layer]["alpha_k_4"]

                    beta_k_0 = self.component_non_backward_dependencies[n_layer]["beta_k_0"]
                    beta_k_3 = self.component_non_backward_dependencies[n_layer]["beta_k_3"]
                    beta_k_4 = self.component_non_backward_dependencies[n_layer]["beta_k_4"]
                else:
                    # this is a new layer, must compute alphas and betas
                    layer_output_lines = prev_layers_activation_derivative_output_bounds[n_layer // 2]

                    u_theta_CROWN_coefficients = prev_layers_u_theta_coefficients[n_layer // 2]
                    f_U_A_0, f_U_constant, f_L_A_0, f_L_constant = u_theta_CROWN_coefficients

                    u_dxi_lbs = prev_layers_u_dxi_lower_bound[n_layer // 2]
                    u_dxi_ubs = prev_layers_u_dxi_upper_bound[n_layer // 2]
                    weight = layer.weight

                    W_pos = torch.clamp(weight, min=0)
                    W_neg = torch.clamp(weight, max=0)

                    gamma_L, delta_L, gamma_U, delta_U = layer_output_lines.T
                    gamma_L, delta_L = gamma_L.unsqueeze(1), delta_L.unsqueeze(1)
                    gamma_U, delta_U = gamma_U.unsqueeze(1), delta_U.unsqueeze(1)

                    u_dxi_layer_U_coeff = gamma_U * W_pos + gamma_L * W_neg
                    u_dxi_layer_U_coeff_pos = u_dxi_layer_U_coeff.clamp(min=0)
                    u_dxi_layer_U_coeff_neg = u_dxi_layer_U_coeff.clamp(max=0)
                    u_dxi_layer_U_const = gamma_U * delta_U * W_pos + gamma_L * delta_L * W_neg

                    u_dxi_layer_L_coeff = gamma_L * W_pos + gamma_U * W_neg
                    u_dxi_layer_L_coeff_pos = u_dxi_layer_L_coeff.clamp(min=0)
                    u_dxi_layer_L_coeff_neg = u_dxi_layer_L_coeff.clamp(max=0)
                    u_dxi_layer_L_const = gamma_L * delta_L * W_pos + gamma_U * delta_U * W_neg

                    new_U_coefficients = u_dxi_layer_U_coeff_pos.reshape(-1, 1) * f_U_A_0.repeat_interleave(weight.shape[1], dim=0) + u_dxi_layer_U_coeff_neg.reshape(-1, 1) * f_L_A_0.repeat_interleave(weight.shape[1], dim=0)
                    new_U_coefficients = new_U_coefficients.reshape(weight.shape[0], weight.shape[1], -1)
                    new_U_constants = u_dxi_layer_U_const + u_dxi_layer_U_coeff_pos * f_U_constant.unsqueeze(1) + u_dxi_layer_U_coeff_neg * f_L_constant.unsqueeze(1)

                    new_L_coefficients = u_dxi_layer_L_coeff_pos.reshape(-1, 1) * f_L_A_0.repeat_interleave(weight.shape[1], dim=0) + u_dxi_layer_L_coeff_neg.reshape(-1, 1) * f_U_A_0.repeat_interleave(weight.shape[1], dim=0)
                    new_L_coefficients = new_L_coefficients.reshape(weight.shape[0], weight.shape[1], -1)
                    new_L_constants = u_dxi_layer_L_const + u_dxi_layer_L_coeff_pos * f_L_constant.unsqueeze(1) + u_dxi_layer_L_coeff_neg * f_U_constant.unsqueeze(1)

                    first_matmul_ubs = torch.sum(new_U_coefficients.clamp(min=0) * x_ub + new_U_coefficients.clamp(max=0) * x_lb, dim=2) + new_U_constants
                    first_matmul_lbs = torch.sum(new_L_coefficients.clamp(min=0) * x_lb + new_L_coefficients.clamp(max=0) * x_ub, dim=2) + new_L_constants

                    # save for future computations
                    new_partial_z_k_z_k_1_coeffs_ubs = new_U_coefficients
                    new_partial_z_k_z_k_1_consts_ubs = new_U_constants
                    new_partial_z_k_z_k_1_coeffs_lbs = new_L_coefficients
                    new_partial_z_k_z_k_1_consts_lbs = new_L_constants

                    alpha_k_0 = xi_i_j * first_matmul_ubs + (1 - xi_i_j) * first_matmul_lbs
                    alpha_k_1 = (xi_i_j * u_dxi_lbs + (1 - xi_i_j) * u_dxi_ubs).repeat(weight.shape[0], 1)
                    alpha_k_2 = -xi_i_j * first_matmul_ubs * u_dxi_lbs - (1 - xi_i_j) * first_matmul_lbs * u_dxi_ubs

                    alpha_k_3 = alpha_k_1.unsqueeze(2).clamp(min=0) * new_U_coefficients + alpha_k_1.unsqueeze(2).clamp(max=0) * new_L_coefficients
                    alpha_k_4 = alpha_k_1.clamp(min=0) * new_U_constants + alpha_k_1.clamp(max=0) * new_L_constants + alpha_k_2

                    beta_k_0 = psi_i_j * first_matmul_lbs + (1 - psi_i_j) * first_matmul_ubs
                    beta_k_1 = (psi_i_j * u_dxi_lbs + (1 - psi_i_j) * u_dxi_ubs).repeat(weight.shape[0], 1)
                    beta_k_2 = -psi_i_j * first_matmul_lbs * u_dxi_lbs - (1 - psi_i_j) * first_matmul_ubs * u_dxi_ubs

                    beta_k_3 = beta_k_1.unsqueeze(2).clamp(min=0) * new_L_coefficients + beta_k_1.unsqueeze(2).clamp(max=0) * new_U_coefficients
                    beta_k_4 = beta_k_1.clamp(min=0) * new_L_constants + beta_k_1.clamp(max=0) * new_U_constants + beta_k_2

                    # cache these computations because at the component level back-prop they are constant
                    self.component_non_backward_dependencies[n_layer] = {}
                    self.component_non_backward_dependencies[n_layer]["alpha_k_0"] = alpha_k_0
                    self.component_non_backward_dependencies[n_layer]["alpha_k_3"] = alpha_k_3
                    self.component_non_backward_dependencies[n_layer]["alpha_k_4"] = alpha_k_4

                    self.component_non_backward_dependencies[n_layer]["beta_k_0"] = beta_k_0
                    self.component_non_backward_dependencies[n_layer]["beta_k_3"] = beta_k_3
                    self.component_non_backward_dependencies[n_layer]["beta_k_4"] = beta_k_4

                # backward propagation through the partial derivative network
                if n_layer == n_layers - 1:
                    if is_final:
                        # penultimate layer of the full NN, take into account the weight of the final layer
                        last_W = layers[n_layers+1].weight
                        last_W_pos = last_W.clamp(min=0)
                        last_W_neg = last_W.clamp(max=0)

                        rho_0_U = last_W_pos @ alpha_k_0 + last_W_neg @ beta_k_0
                        rho_1_U = (last_W_pos @ alpha_k_3.reshape(alpha_k_3.shape[0], -1) + last_W_neg @ beta_k_3.reshape(beta_k_3.shape[0], -1)).reshape(last_W_pos.shape[0], *alpha_k_3.shape[1:])
                        rho_2_U = last_W_pos @ alpha_k_4 + last_W_neg @ beta_k_4

                        rho_0_L = last_W_pos @ beta_k_0 + last_W_neg @ alpha_k_0
                        rho_1_L = (last_W_pos @ beta_k_3.reshape(beta_k_3.shape[0], -1) + last_W_neg @ alpha_k_3.reshape(alpha_k_3.shape[0], -1)).reshape(last_W_pos.shape[0], *alpha_k_3.shape[1:])
                        rho_2_L = last_W_pos @ beta_k_4 + last_W_neg @ alpha_k_4
                    else:
                        # it's just a middle layer, we want to compute it's output directly
                        rho_0_U = alpha_k_0
                        rho_1_U = alpha_k_3
                        rho_2_U = alpha_k_4
                        rho_0_L = beta_k_0
                        rho_1_L = beta_k_3
                        rho_2_L = beta_k_4

                        # it's the first time we're exploring this layer, add intermediate coefficients, constants and bounds for further computations
                        self.partial_z_k_z_k_1_coefficients_lbs.append(new_partial_z_k_z_k_1_coeffs_lbs)
                        self.partial_z_k_z_k_1_constants_lbs.append(new_partial_z_k_z_k_1_consts_lbs)
                        self.partial_z_k_z_k_1_coefficients_ubs.append(new_partial_z_k_z_k_1_coeffs_ubs)
                        self.partial_z_k_z_k_1_constants_ubs.append(new_partial_z_k_z_k_1_consts_ubs)

                        self.partial_z_k_z_k_1_ubs.append(first_matmul_ubs)
                        self.partial_z_k_z_k_1_lbs.append(first_matmul_lbs)
                else:
                    new_rho_0_U = (rho_0_U.unsqueeze(2).clamp(min=0) * alpha_k_0.unsqueeze(0) + rho_0_U.unsqueeze(2).clamp(max=0) * beta_k_0.unsqueeze(0)).sum(dim=1)
                    new_rho_1_U = (rho_0_U.unsqueeze(2).unsqueeze(3).clamp(min=0) * alpha_k_3.unsqueeze(0) + rho_0_U.unsqueeze(2).unsqueeze(3).clamp(max=0) * beta_k_3.unsqueeze(0)).sum(dim=1) + (1/layer.weight.shape[1]) * rho_1_U.sum(dim=1).unsqueeze(1)
                    new_rho_2_U = (rho_0_U.unsqueeze(2).clamp(min=0) * alpha_k_4.unsqueeze(0) + rho_0_U.unsqueeze(2).clamp(max=0) * beta_k_4.unsqueeze(0)).sum(dim=1) + (1/layer.weight.shape[1]) * rho_2_U.sum(dim=1).unsqueeze(1)

                    new_rho_0_L = (rho_0_L.unsqueeze(2).clamp(min=0) * beta_k_0.unsqueeze(0) + rho_0_L.unsqueeze(2).clamp(max=0) * alpha_k_0.unsqueeze(0)).sum(dim=1)
                    new_rho_1_L = (rho_0_L.unsqueeze(2).unsqueeze(3).clamp(min=0) * beta_k_3.unsqueeze(0) + rho_0_L.unsqueeze(2).unsqueeze(3).clamp(max=0) * alpha_k_3.unsqueeze(0)).sum(dim=1) + (1/layer.weight.shape[1]) * rho_1_L.sum(dim=1).unsqueeze(1)
                    new_rho_2_L = (rho_0_L.unsqueeze(2).clamp(min=0) * beta_k_4.unsqueeze(0) + rho_0_L.unsqueeze(2).clamp(max=0) * alpha_k_4.unsqueeze(0)).sum(dim=1) + (1/layer.weight.shape[1]) * rho_2_L.sum(dim=1).unsqueeze(1)

                    rho_0_U = new_rho_0_U
                    rho_1_U = new_rho_1_U
                    rho_2_U = new_rho_2_U
                    rho_0_L = new_rho_0_L
                    rho_1_L = new_rho_1_L
                    rho_2_L = new_rho_2_L

        layer_crown_U_coefficients = rho_1_U.sum(dim=1)
        layer_crown_U_constants = rho_0_U[:, self.component_idx] + rho_2_U.sum(dim=1)
        layer_crown_L_coefficients = rho_1_L.sum(dim=1)
        layer_crown_L_constants = rho_0_L[:, self.component_idx] + rho_2_L.sum(dim=1)

        layer_output_upper_bounds = torch.sum(layer_crown_U_coefficients.clamp(min=0) * x_ub + layer_crown_U_coefficients.clamp(max=0) * x_lb, dim=1) + layer_crown_U_constants
        layer_output_lower_bounds = torch.sum(layer_crown_L_coefficients.clamp(min=0) * x_lb + layer_crown_L_coefficients.clamp(max=0) * x_ub, dim=1) + layer_crown_L_constants

        return layer_output_upper_bounds, layer_output_lower_bounds, (layer_crown_U_coefficients, layer_crown_U_constants, layer_crown_L_coefficients, layer_crown_L_constants)
    
    def compute_bounds(self, debug: bool = True, lp_first_matmul_bounds=None, lp_layer_output_bounds=None, backprop_mode: BackpropMode = BackpropMode.BLOCK_BACKPROP):
        self.x_lb = self.u_theta.x_lb
        self.x_ub = self.u_theta.x_ub

        with torch.no_grad():
            # if we haven't computed the intermediate bounds on u_theta, do it now
            if not self.u_theta.computed_bounds:
                self.u_theta.compute_bounds(debug=debug)

            # proceed if the computation is successful and these variables are now populated
            assert self.u_theta.computed_bounds

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

            u_dxi_lower_bounds = self.lower_bounds
            u_dxi_upper_bounds = self.upper_bounds
            u_dxi_lower_bounds.append(norm_layer_partial_grad.flatten())
            u_dxi_upper_bounds.append(norm_layer_partial_grad.flatten())
            self.partial_z_k_z_k_1_ubs = []
            self.partial_z_k_z_k_1_lbs = []

            xi_i_j = 0.5
            psi_i_j = 0.5

            self.u_dxi_crown_coefficients_lbs = [torch.zeros(u_dxi_lower_bounds[-1].shape[0], u_dxi_lower_bounds[-1].shape[0])]
            self.u_dxi_crown_constants_lbs = [u_dxi_lower_bounds[-1]]
            self.u_dxi_crown_coefficients_ubs = [torch.zeros(u_dxi_upper_bounds[-1].shape[0], u_dxi_upper_bounds[-1].shape[0])]
            self.u_dxi_crown_constants_ubs = [u_dxi_upper_bounds[-1]]
            self.partial_z_k_z_k_1_coefficients_lbs = []
            self.partial_z_k_z_k_1_constants_lbs = []
            self.partial_z_k_z_k_1_coefficients_ubs = []
            self.partial_z_k_z_k_1_constants_ubs = []

            # debug computation with interval midpoint
            if debug:
                norm_point = self.u_theta.domain_bounds.mean(dim=0)
                norm_grad = copy.deepcopy(norm_layer_partial_grad).T
                for norm_layer in self.u_theta.norm_layers:
                    norm_point = norm_layer(norm_point)

            it_object = self.u_theta.fc_layers
            if debug:
                print(f"Propagating CROWN bounds through du_theta/dx{self.component_idx}...")
                it_object = tqdm(self.u_theta.fc_layers)

            layers_activation_derivative_output_bounds = []
            if backprop_mode is BackpropMode.COMPONENT_BACKPROP:
                self.component_non_backward_dependencies = {}

            for n_layer, layer in enumerate(it_object):
                if debug:
                    norm_point = layer(norm_point)

                if not isinstance(layer, torch.nn.Linear):
                    continue
            
                is_final = (n_layer == len(self.u_theta.fc_layers) - 1)
                u_dxi_lbs = u_dxi_lower_bounds[-1]
                u_dxi_ubs = u_dxi_upper_bounds[-1]
                weight = layer.weight
                u_dxi_crown_coeffs_lbs = self.u_dxi_crown_coefficients_lbs[-1]
                u_dxi_crown_consts_lbs = self.u_dxi_crown_constants_lbs[-1]
                u_dxi_crown_coeffs_ubs = self.u_dxi_crown_coefficients_ubs[-1]
                u_dxi_crown_consts_ubs = self.u_dxi_crown_constants_ubs[-1]

                W_pos = torch.clamp(weight, min=0)
                W_neg = torch.clamp(weight, max=0)
                layer_output_upper_bounds = torch.zeros(weight.shape[0])
                layer_output_lower_bounds = torch.zeros(weight.shape[0])

                if not is_final:
                    # if it's not the last layer, there's an activation to relax before doing IBP on the multiplication
                    u_theta_CROWN_coefficients = self.u_theta.layer_CROWN_coefficients[n_layer // 2]
                    f_U_A_0, f_U_constant, f_L_A_0, f_L_constant = u_theta_CROWN_coefficients

                    layer_output_lines = torch.zeros(len(self.u_theta.lower_bounds[n_layer+1]), 4)
                    # use the bounds to relax sigma'(y^k_i)
                    for i in range(len(self.u_theta.lower_bounds[n_layer+1])):
                        n_layer_pre_act_lb, n_layer_pre_act_ub = self.u_theta.lower_bounds[n_layer+1][i], self.u_theta.upper_bounds[n_layer+1][i]
                        lb_lines, ub_lines = self.activation_derivative_relaxation.get_bounds(
                            n_layer_pre_act_lb,
                            n_layer_pre_act_ub
                        )

                        # can only have a single bound, take a convex combination of them
                        # TODO: move convex combination code to a SINGLE_LINE relaxation inside the ActivationRelaxation classes
                        m_combination_lb = torch.mean(torch.Tensor([m for m, b in lb_lines]))
                        b_combination_lb = torch.mean(torch.Tensor([b for m, b in lb_lines]))

                        m_combination_ub = torch.mean(torch.Tensor([m for m, b in ub_lines]))
                        b_combination_ub = torch.mean(torch.Tensor([b for m, b in ub_lines]))

                        # CROWN line definition
                        if m_combination_lb != 0:
                            b_combination_lb /= m_combination_lb
                        
                        if m_combination_ub != 0:
                            b_combination_ub /= m_combination_ub

                        layer_output_lines[i, :] = torch.tensor([
                            m_combination_lb,
                            b_combination_lb,
                            m_combination_ub,
                            b_combination_ub
                        ])

                    layers_activation_derivative_output_bounds.append(layer_output_lines)

                    if backprop_mode == BackpropMode.BLOCK_BACKPROP:
                        # compute the coefficients based on the ones from u_theta, the relaxation of sigma' and the multiplication with W^(n_layer)
                        gamma_L, delta_L, gamma_U, delta_U = layer_output_lines.T
                        gamma_L, delta_L = gamma_L.unsqueeze(1), delta_L.unsqueeze(1)
                        gamma_U, delta_U = gamma_U.unsqueeze(1), delta_U.unsqueeze(1)

                        u_dxi_layer_U_coeff = gamma_U * W_pos + gamma_L * W_neg
                        u_dxi_layer_U_coeff_pos = u_dxi_layer_U_coeff.clamp(min=0)
                        u_dxi_layer_U_coeff_neg = u_dxi_layer_U_coeff.clamp(max=0)
                        u_dxi_layer_U_const = gamma_U * delta_U * W_pos + gamma_L * delta_L * W_neg

                        u_dxi_layer_L_coeff = gamma_L * W_pos + gamma_U * W_neg
                        u_dxi_layer_L_coeff_pos = u_dxi_layer_L_coeff.clamp(min=0)
                        u_dxi_layer_L_coeff_neg = u_dxi_layer_L_coeff.clamp(max=0)
                        u_dxi_layer_L_const = gamma_L * delta_L * W_pos + gamma_U * delta_U * W_neg

                        new_U_coefficients = u_dxi_layer_U_coeff_pos.reshape(-1, 1) * f_U_A_0.repeat_interleave(weight.shape[1], dim=0) + u_dxi_layer_U_coeff_neg.reshape(-1, 1) * f_L_A_0.repeat_interleave(weight.shape[1], dim=0)
                        new_U_coefficients = new_U_coefficients.reshape(weight.shape[0], weight.shape[1], -1)
                        new_U_constants = u_dxi_layer_U_const + u_dxi_layer_U_coeff_pos * f_U_constant.unsqueeze(1) + u_dxi_layer_U_coeff_neg * f_L_constant.unsqueeze(1)

                        new_L_coefficients = u_dxi_layer_L_coeff_pos.reshape(-1, 1) * f_L_A_0.repeat_interleave(weight.shape[1], dim=0) + u_dxi_layer_L_coeff_neg.reshape(-1, 1) * f_U_A_0.repeat_interleave(weight.shape[1], dim=0)
                        new_L_coefficients = new_L_coefficients.reshape(weight.shape[0], weight.shape[1], -1)
                        new_L_constants = u_dxi_layer_L_const + u_dxi_layer_L_coeff_pos * f_L_constant.unsqueeze(1) + u_dxi_layer_L_coeff_neg * f_U_constant.unsqueeze(1)

                        first_matmul_ubs = torch.sum(new_U_coefficients.clamp(min=0) * self.x_ub + new_U_coefficients.clamp(max=0) * self.x_lb, dim=2) + new_U_constants
                        first_matmul_lbs = torch.sum(new_L_coefficients.clamp(min=0) * self.x_lb + new_L_coefficients.clamp(max=0) * self.x_ub, dim=2) + new_L_constants

                        # save for future computations
                        new_partial_z_k_z_k_1_coeffs_ubs = new_U_coefficients
                        new_partial_z_k_z_k_1_consts_ubs = new_U_constants
                        new_partial_z_k_z_k_1_coeffs_lbs = new_L_coefficients
                        new_partial_z_k_z_k_1_consts_lbs = new_L_constants

                        # upper bounds
                        alpha_k_0 = xi_i_j * first_matmul_ubs + (1 - xi_i_j) * first_matmul_lbs
                        alpha_k_1 = (xi_i_j * u_dxi_lbs + (1 - xi_i_j) * u_dxi_ubs).repeat(weight.shape[0], 1)
                        alpha_k_2 = -xi_i_j * first_matmul_ubs * u_dxi_lbs - (1 - xi_i_j) * first_matmul_lbs * u_dxi_ubs

                        alpha_k_0_unsqueezed = alpha_k_0.unsqueeze(2)
                        alpha_k_1_unsqueezed = alpha_k_1.unsqueeze(2)

                        alpha_k_5 = (
                            alpha_k_0_unsqueezed.clamp(min=0) * u_dxi_crown_coeffs_ubs + alpha_k_0_unsqueezed.clamp(max=0) * u_dxi_crown_coeffs_lbs +
                            alpha_k_1_unsqueezed.clamp(min=0) * new_U_coefficients + alpha_k_1_unsqueezed.clamp(max=0) * new_L_coefficients
                        ).sum(dim=1)
                        alpha_k_6 = (
                            alpha_k_0.clamp(min=0) * u_dxi_crown_consts_ubs.flatten() + alpha_k_0.clamp(max=0) * u_dxi_crown_consts_lbs.flatten() +
                            alpha_k_1.clamp(min=0) * new_U_constants + alpha_k_1.clamp(max=0) * new_L_constants +
                            alpha_k_2
                        ).sum(dim=1)

                        layer_output_upper_bounds = torch.sum(alpha_k_5.clamp(min=0) * self.x_ub + alpha_k_5.clamp(max=0) * self.x_lb, dim=1) + alpha_k_6
                        new_u_dxi_crown_coeffs_ubs = alpha_k_5
                        new_u_dxi_crown_consts_ubs = alpha_k_6

                        # lower bounds
                        beta_k_0 = psi_i_j * first_matmul_lbs + (1 - psi_i_j) * first_matmul_ubs
                        beta_k_1 = (psi_i_j * u_dxi_lbs + (1 - psi_i_j) * u_dxi_ubs).repeat(weight.shape[0], 1)
                        beta_k_2 = -psi_i_j * first_matmul_lbs * u_dxi_lbs - (1 - psi_i_j) * first_matmul_ubs * u_dxi_ubs

                        beta_k_0_unsqueezed = beta_k_0.unsqueeze(2)
                        beta_k_1_unsqueezed = beta_k_1.unsqueeze(2)

                        beta_k_5 = (
                            beta_k_0_unsqueezed.clamp(min=0) * u_dxi_crown_coeffs_lbs + beta_k_0_unsqueezed.clamp(max=0) * u_dxi_crown_coeffs_ubs +
                            beta_k_1_unsqueezed.clamp(min=0) * new_L_coefficients + beta_k_1_unsqueezed.clamp(max=0) * new_U_coefficients
                        ).sum(dim=1)
                        beta_k_6 = (
                            beta_k_0.clamp(min=0) * u_dxi_crown_consts_lbs.flatten() + beta_k_0.clamp(max=0) * u_dxi_crown_consts_ubs.flatten() +
                            beta_k_1.clamp(min=0) * new_L_constants + beta_k_1.clamp(max=0) * new_U_constants +
                            beta_k_2
                        ).sum(dim=1)

                        layer_output_lower_bounds = torch.sum(beta_k_5.clamp(min=0) * self.x_lb + beta_k_5.clamp(max=0) * self.x_ub, dim=1) + beta_k_6
                        new_u_dxi_crown_coeffs_lbs = beta_k_5
                        new_u_dxi_crown_consts_lbs = beta_k_6

                        # for j in range(weight.shape[0]):
                            # upper bound
                            # u_dxi_layer_U_coeff_pos_j = u_dxi_layer_U_coeff_pos[j]
                            # u_dxi_layer_U_coeff_neg_j = u_dxi_layer_U_coeff_neg[j]
                            # u_dxi_layer_U_const_j = u_dxi_layer_U_const[j]

                            # new_U_coefficient = u_dxi_layer_U_coeff_pos_j.unsqueeze(1) @ f_U_A_0[j].unsqueeze(0) + u_dxi_layer_U_coeff_neg_j.unsqueeze(1) @ f_L_A_0[j].unsqueeze(0)
                            # new_U_constant = u_dxi_layer_U_const_j + u_dxi_layer_U_coeff_pos_j * f_U_constant[j] + u_dxi_layer_U_coeff_neg_j * f_L_constant[j]

                            # first_matmul_ubs[j, :] = torch.sum(new_U_coefficient * ((new_U_coefficient >= 0) * self.x_ub + (new_U_coefficient < 0) * self.x_lb), dim=1) + new_U_constant

                            # assert torch.linalg.norm(new_U_coefficients[j] - new_U_coefficient) <= 1e-6
                            # assert torch.linalg.norm(new_U_constants[j] - new_U_constant) <= 1e-6
                            # assert torch.linalg.norm(new_first_matmul_ubs[j, :] - first_matmul_ubs[j, :]) <= 1e-6

                            # save for higher order derivative computations
                            # new_partial_z_k_z_k_1_coeffs_ubs.append(new_U_coefficient)
                            # new_partial_z_k_z_k_1_consts_ubs.append(new_U_constant)

                            # lower bound
                            # u_dxi_layer_L_coeff_pos_j = u_dxi_layer_L_coeff_pos[j]
                            # u_dxi_layer_L_coeff_neg_j = u_dxi_layer_L_coeff_neg[j]
                            # u_dxi_layer_L_const_j = u_dxi_layer_L_const[j]

                            # new_L_coefficient = u_dxi_layer_L_coeff_pos_j.unsqueeze(1) @ f_L_A_0[j].unsqueeze(0) + u_dxi_layer_L_coeff_neg_j.unsqueeze(1) @ f_U_A_0[j].unsqueeze(0)
                            # new_L_constant = u_dxi_layer_L_const_j + u_dxi_layer_L_coeff_pos_j * f_L_constant[j] + u_dxi_layer_L_coeff_neg_j * f_U_constant[j]

                            # first_matmul_lbs[j, :] = torch.sum(new_L_coefficient * ((new_L_coefficient >= 0) * self.x_lb + (new_L_coefficient < 0) * self.x_ub), dim=1) + new_L_constant

                            # assert torch.linalg.norm(new_L_coefficients[j] - new_L_coefficient) <= 1e-6
                            # assert torch.linalg.norm(new_L_constants[j] - new_L_constant) <= 1e-6
                            # assert torch.linalg.norm(new_first_matmul_lbs[j, :] - first_matmul_lbs[j, :]) <= 1e-6

                            # new_partial_z_k_z_k_1_coeffs_lbs.append(new_L_coefficient)
                            # new_partial_z_k_z_k_1_consts_lbs.append(new_L_constant)

                            # new_U_coefficient = new_U_coefficients[j]
                            # new_U_constant = new_U_constants[j]
                            # new_L_coefficient = new_L_coefficients[j]
                            # new_L_constant = new_L_constants[j]

                            # compute upper bound using the McCormick relaxation of the UBs
                            # e_0_j = xi_i_j * first_matmul_ubs[j, :] + (1 - xi_i_j) * first_matmul_lbs[j, :]
                            # e_1_j = xi_i_j * u_dxi_lbs + (1 - xi_i_j) * u_dxi_ubs
                            # e_2_j = -xi_i_j * first_matmul_ubs[j, :] * u_dxi_lbs - (1 - xi_i_j) * first_matmul_lbs[j, :] * u_dxi_ubs

                            # assert torch.linalg.norm(e_0[j] - e_0_j) <= 1e-6
                            # assert torch.linalg.norm(e_1[j] - e_1_j) <= 1e-6
                            # assert torch.linalg.norm(e_2[j] - e_2_j) <= 1e-6

                            # e_0_j = e_0[j]
                            # e_1_j = e_1[j]
                            # e_2_j = e_2[j]

                            # if n_layer == 0:
                            #     full_U_coefficients = sum([
                            #         (e_0_j[i].clamp(min=0) * u_dxi_crown_coeffs_ubs[i] + e_0_j[i].clamp(max=0) * u_dxi_crown_coeffs_lbs[i]) + 
                            #         (e_1_j[i].clamp(min=0) * new_U_coefficient[i, :] + e_1_j[i].clamp(max=0) * new_L_coefficient[i, :])
                            #         for i in range(e_0_j.shape[0])
                            #     ]).unsqueeze(0)
                            #     full_U_constant = sum([
                            #         (e_0_j[i].clamp(min=0) * u_dxi_crown_consts_ubs[i] + e_0_j[i].clamp(max=0) * u_dxi_crown_consts_lbs[i]) + 
                            #         (e_1_j[i].clamp(min=0) * new_U_constant[i] + e_1_j[i].clamp(max=0) * new_L_constant[i]) + 
                            #         e_2_j[i]
                            #         for i in range(e_0_j.shape[0])
                            #     ])
                            # else:
                            #     full_U_coefficients = sum([
                            #         (e_0_j[i].clamp(min=0) * u_dxi_crown_coeffs_ubs[i] + e_0_j[i].clamp(max=0) * u_dxi_crown_coeffs_lbs[i]) + 
                            #         (e_1_j[i].clamp(min=0) * new_U_coefficient[i, :] + e_1_j[i].clamp(max=0) * new_L_coefficient[i, :])
                            #         for i in range(e_0_j.shape[0])
                            #     ]).unsqueeze(0)
                            #     full_U_constant = sum([
                            #         (e_0_j[i].clamp(min=0) * u_dxi_crown_consts_ubs[i] + e_0_j[i].clamp(max=0) * u_dxi_crown_consts_lbs[i]) + 
                            #         (e_1_j[i].clamp(min=0) * new_U_constant[i] + e_1_j[i].clamp(max=0) * new_L_constant[i]) + 
                            #         e_2_j[i]
                            #         for i in range(e_0_j.shape[0])
                            #     ])
                            
                            # try:
                            #     assert torch.linalg.norm(new_full_U_coefficients[j] - full_U_coefficients) <= 1e-6
                            #     assert torch.linalg.norm(new_full_U_constants[j] - full_U_constant) <= 1e-6
                            # except:
                            #     import pdb
                            #     pdb.set_trace()

                            # layer_output_upper_bounds[j] = torch.sum(full_U_coefficients * ((full_U_coefficients >= 0) * self.x_ub + (full_U_coefficients < 0) * self.x_lb), dim=1) + full_U_constant

                            # assert torch.linalg.norm(new_layer_output_upper_bounds[j] - layer_output_upper_bounds[j]) <= 1e-6

                            # new_u_dxi_crown_coeffs_ubs.append(full_U_coefficients)
                            # new_u_dxi_crown_consts_ubs.append(full_U_constant)

                            # compute lower bound using the McCormick relaxation of the LBs
                            # d_0_j = psi_i_j * first_matmul_lbs[j, :] + (1 - psi_i_j) * first_matmul_ubs[j, :]
                            # d_1_j = psi_i_j * u_dxi_lbs + (1 - psi_i_j) * u_dxi_ubs
                            # d_2_j = -psi_i_j * first_matmul_lbs[j, :] * u_dxi_lbs - (1 - psi_i_j) * first_matmul_ubs[j, :] * u_dxi_ubs

                            # assert torch.linalg.norm(d_0[j] - d_0_j) <= 1e-6
                            # assert torch.linalg.norm(d_1[j] - d_1_j) <= 1e-6
                            # assert torch.linalg.norm(d_2[j] - d_2_j) <= 1e-6

                            # d_0_j = d_0[j]
                            # d_1_j = d_1[j]
                            # d_2_j = d_2[j]

                            # if n_layer == 0:
                            #     full_L_coefficients = sum([
                            #         (d_0_j[i].clamp(min=0) * u_dxi_crown_coeffs_lbs[i] + d_0_j[i].clamp(max=0) * u_dxi_crown_coeffs_ubs[i]) +
                            #         (d_1_j[i].clamp(min=0) * new_L_coefficient[i, :] + d_1_j[i].clamp(max=0) * new_U_coefficient[i, :])
                            #         for i in range(d_0_j.shape[0])
                            #     ]).unsqueeze(0)
                            #     full_L_constant = sum([
                            #         (d_0_j[i].clamp(min=0) * u_dxi_crown_consts_lbs[i] + d_0_j[i].clamp(max=0) * u_dxi_crown_consts_ubs[i]) + 
                            #         (d_1_j[i].clamp(min=0) * new_L_constant[i] + d_1_j[i].clamp(max=0) * new_U_constant[i]) + 
                            #         d_2_j[i]
                            #         for i in range(d_0_j.shape[0])
                            #     ])
                            # else:
                            #     full_L_coefficients = sum([
                            #         (d_0_j[i].clamp(min=0) * u_dxi_crown_coeffs_lbs[i] + d_0_j[i].clamp(max=0) * u_dxi_crown_coeffs_ubs[i]) +
                            #         (d_1_j[i].clamp(min=0) * new_L_coefficient[i, :] + d_1_j[i].clamp(max=0) * new_U_coefficient[i, :])
                            #         for i in range(d_0_j.shape[0])
                            #     ]).unsqueeze(0)
                            #     full_L_constant = sum([
                            #         (d_0_j[i].clamp(min=0) * u_dxi_crown_consts_lbs[i] + d_0_j[i].clamp(max=0) * u_dxi_crown_consts_ubs[i]) + 
                            #         (d_1_j[i].clamp(min=0) * new_L_constant[i] + d_1_j[i].clamp(max=0) * new_U_constant[i]) + 
                            #         d_2_j[i]
                            #         for i in range(d_0_j.shape[0])
                            #     ])

                            # try:
                            #     assert torch.linalg.norm(new_full_L_coefficients[j] - full_L_coefficients) <= 1e-5
                            #     assert torch.linalg.norm(new_full_L_constants[j] - full_L_constant) <= 1e-6
                            # except:
                            #     import pdb
                            #     pdb.set_trace()

                            # layer_output_lower_bounds[j] = torch.sum(full_L_coefficients * ((full_L_coefficients >= 0) * self.x_lb + (full_L_coefficients < 0) * self.x_ub), dim=1) + full_L_constant

                            # assert torch.linalg.norm(new_layer_output_lower_bounds[j] - layer_output_lower_bounds[j]) <= 1e-6

                            # new_u_dxi_crown_coeffs_lbs.append(full_L_coefficients)
                            # new_u_dxi_crown_consts_lbs.append(full_L_constant)

                        # keep track of the intermediate bounds
                        self.partial_z_k_z_k_1_coefficients_lbs.append(new_partial_z_k_z_k_1_coeffs_lbs)
                        self.partial_z_k_z_k_1_constants_lbs.append(new_partial_z_k_z_k_1_consts_lbs)
                        self.partial_z_k_z_k_1_coefficients_ubs.append(new_partial_z_k_z_k_1_coeffs_ubs)
                        self.partial_z_k_z_k_1_constants_ubs.append(new_partial_z_k_z_k_1_consts_ubs)

                        self.partial_z_k_z_k_1_ubs.append(first_matmul_ubs)
                        self.partial_z_k_z_k_1_lbs.append(first_matmul_lbs)
                    elif backprop_mode == BackpropMode.COMPONENT_BACKPROP:
                        layer_output_upper_bounds, layer_output_lower_bounds, CROWN_coefficients = self.backward_component_propagation(
                            self.u_theta.fc_layers[:n_layer+1],
                            layers_activation_derivative_output_bounds,
                            self.u_theta.layer_CROWN_coefficients,
                            self.upper_bounds,
                            self.lower_bounds,
                            self.x_lb,
                            self.x_ub,
                            is_final=False
                        )

                        new_u_dxi_crown_coeffs_ubs, new_u_dxi_crown_consts_ubs = CROWN_coefficients[0], CROWN_coefficients[1]
                        new_u_dxi_crown_coeffs_lbs, new_u_dxi_crown_consts_lbs = CROWN_coefficients[2], CROWN_coefficients[3]
                    else:
                        raise NotImplementedError

                    if debug:
                        # not the last layer
                        first_matmul_ubs, first_matmul_lbs = self.partial_z_k_z_k_1_ubs[-1], self.partial_z_k_z_k_1_lbs[-1]
                    
                        try:
                            # 1. all lower bounds should be smaller than upper bounds
                            assert all((first_matmul_lbs <= first_matmul_ubs).flatten())
                            assert all((layer_output_lower_bounds <= layer_output_upper_bounds).flatten())

                            # 2. a random point in the interval needs to be inside the bounds
                            point_vals_first_mul = torch.diag(self.activation_derivative_relaxation.evaluate(norm_point)) @ weight
                            norm_grad = point_vals_first_mul @ norm_grad

                            assert all((point_vals_first_mul <= first_matmul_ubs + 1e-2).flatten())
                            assert all(norm_grad.flatten() <= layer_output_upper_bounds)
                            assert all((point_vals_first_mul >= first_matmul_lbs - 1e-2).flatten())
                            assert all(norm_grad.flatten() >= layer_output_lower_bounds)

                            # 3. the CROWN bounds must be looser than the LP ones
                            if lp_first_matmul_bounds:
                                lp_first_matmul_ubs = lp_first_matmul_bounds[1][n_layer // 2]
                                lp_first_matmul_lbs = lp_first_matmul_bounds[0][n_layer // 2]

                                assert all((first_matmul_ubs >= lp_first_matmul_ubs - 1e-2).flatten())
                                assert all(layer_output_upper_bounds >= lp_layer_output_bounds[1][n_layer // 2 + 1] - 1e-3)
                                assert all((first_matmul_lbs <= lp_first_matmul_lbs + 1e-2).flatten())
                                assert all(layer_output_lower_bounds <= lp_layer_output_bounds[0][n_layer // 2 + 1] + 1e-2)
                        except:
                            print('--- exception ---')
                            import pdb
                            pdb.set_trace()
                else:
                    # it is the final layer
                    if backprop_mode == BackpropMode.BLOCK_BACKPROP:
                        new_u_dxi_crown_coeffs_ubs = W_pos @ u_dxi_crown_coeffs_ubs + W_neg @ u_dxi_crown_coeffs_lbs
                        new_u_dxi_crown_consts_ubs = W_pos @ u_dxi_crown_consts_ubs + W_neg @ u_dxi_crown_consts_lbs

                        layer_output_upper_bounds = torch.sum(new_u_dxi_crown_coeffs_ubs.clamp(min=0) * self.x_ub + new_u_dxi_crown_coeffs_ubs.clamp(max=0) * self.x_lb, dim=1) + new_u_dxi_crown_consts_ubs

                        new_u_dxi_crown_coeffs_lbs = W_pos @ u_dxi_crown_coeffs_lbs + W_neg @ u_dxi_crown_coeffs_ubs
                        new_u_dxi_crown_consts_lbs = W_pos @ u_dxi_crown_consts_lbs + W_neg @ u_dxi_crown_consts_ubs

                        layer_output_lower_bounds = torch.sum(new_u_dxi_crown_coeffs_lbs.clamp(min=0) * self.x_lb + new_u_dxi_crown_coeffs_lbs.clamp(max=0) * self.x_ub, dim=1) + new_u_dxi_crown_consts_lbs

                    elif backprop_mode == BackpropMode.COMPONENT_BACKPROP:
                        layer_output_upper_bounds, layer_output_lower_bounds, CROWN_coefficients = self.backward_component_propagation(
                            self.u_theta.fc_layers[:n_layer+1],
                            layers_activation_derivative_output_bounds,
                            self.u_theta.layer_CROWN_coefficients,
                            self.upper_bounds,
                            self.lower_bounds,
                            self.x_lb,
                            self.x_ub,
                            is_final=True
                        )

                        new_u_dxi_crown_coeffs_ubs, new_u_dxi_crown_consts_ubs = CROWN_coefficients[0], CROWN_coefficients[1]
                        new_u_dxi_crown_coeffs_lbs, new_u_dxi_crown_consts_lbs = CROWN_coefficients[2], CROWN_coefficients[3]
                    else:
                        raise NotImplementedError
                    
                    if debug:
                        try:
                            # 1. all lower bounds should be smaller than upper bounds
                            assert all((layer_output_lower_bounds <= layer_output_upper_bounds).flatten())

                            # 2. a random point in the interval needs to be inside the bounds
                            norm_grad = weight @ norm_grad
                            assert all(norm_grad.flatten() <= layer_output_upper_bounds)
                            assert all(norm_grad.flatten() >= layer_output_lower_bounds)

                            # 3. the CROWN bounds must be looser than the LP ones
                            if lp_layer_output_bounds:
                                assert all(layer_output_upper_bounds >= lp_layer_output_bounds[1][n_layer // 2 + 1] - 1e-3)
                                assert all(layer_output_lower_bounds <= lp_layer_output_bounds[0][n_layer // 2 + 1] + 1e-2)
                        except:
                            print('--- exception ---')
                            import pdb
                            pdb.set_trace()

                u_dxi_lower_bounds.append(layer_output_lower_bounds.detach())
                u_dxi_upper_bounds.append(layer_output_upper_bounds.detach())

                self.u_dxi_crown_coefficients_lbs.append(new_u_dxi_crown_coeffs_lbs)
                self.u_dxi_crown_constants_lbs.append(new_u_dxi_crown_consts_lbs)
                self.u_dxi_crown_coefficients_ubs.append(new_u_dxi_crown_coeffs_ubs)
                self.u_dxi_crown_constants_ubs.append(new_u_dxi_crown_consts_ubs)

        self.computed_bounds = True


class CROWNPINNSecondPartialDerivative():
    def __init__(self, pinn_partial_derivative: CROWNPINNPartialDerivative, component_idx: int, activation_derivative_derivative_relaxation: ActivationRelaxation) -> None:
        # class used to compute the bounds of the second derivative of the PINN; it operates over the model defined in pinn_partial_derivative.u_theta.grb_model
        self.u_dxi_theta = pinn_partial_derivative
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
        self.lower_bounds = []
        self.upper_bounds = []

        self.activation_derivative_derivative_relaxation = activation_derivative_derivative_relaxation
    
    def backward_component_propagation(
            self,
            layers: List[torch.nn.Module],
            prev_layers_activation_derivative_output_bounds: List[torch.Tensor],
            prev_layers_layer_actual_bounds: List[torch.Tensor],
            prev_layers_u_dxidxi_upper_bound: List[torch.Tensor],
            prev_layers_u_dxidxi_lower_bound: List[torch.Tensor],
            x_lb: torch.Tensor,
            x_ub:  torch.Tensor,
            is_final: bool = False,
            rho_k: float = 0.5,
            tau_k: float = 0.5,
            eta_k: float = 0.5,
            gamma_k: float = 0.5,
            zeta_k: float = 0.5,
            delta_k: float = 0.5,
    ):
        n_layers = len(layers)
        if is_final:
            # remove the last couple of layers, as the last one is explicitly considered below when propagating the last hidden layer
            n_layers -= 2
        
        with torch.no_grad():
            for n_layer in range(n_layers-1, -1, -1):
                layer = layers[n_layer]
                if not isinstance(layer, torch.nn.Linear):
                    continue

                # improves efficiency by caching computation of alphas and betas when doing it only at the component level (since there's no dependency on previous layers)
                # has this layer been cached already? if so, use those values
                if n_layer in self.component_non_backward_dependencies:
                    alpha_k_2 = self.component_non_backward_dependencies[n_layer]["alpha_k_2"]
                    alpha_k_5 = self.component_non_backward_dependencies[n_layer]["alpha_k_5"]
                    alpha_k_6 = self.component_non_backward_dependencies[n_layer]["alpha_k_6"]

                    beta_k_2 = self.component_non_backward_dependencies[n_layer]["beta_k_2"]
                    beta_k_5 = self.component_non_backward_dependencies[n_layer]["beta_k_5"]
                    beta_k_6 = self.component_non_backward_dependencies[n_layer]["beta_k_6"]
                else:
                    # this is a new layer, must compute alphas and betas
                    layer_output_lines = prev_layers_activation_derivative_output_bounds[n_layer // 2]
                    layer_actual_bounds = prev_layers_layer_actual_bounds[n_layer // 2]

                    A_k_U, a_k_U, A_k_L, a_k_L = self.u_theta.layer_CROWN_coefficients[n_layer // 2]

                    # from u_dxi_theta: \partial_{x_i} z^{(k-1)} lower and upper bounds (layers in u_dxi_theta)
                    partial_dxi_z_k_1_lbs, partial_dxi_z_k_1_ubs = self.u_dxi_theta.lower_bounds[n_layer // 2], self.u_dxi_theta.upper_bounds[n_layer // 2]
                    C_k_L, c_k_L = self.u_dxi_theta.u_dxi_crown_coefficients_lbs[n_layer // 2], self.u_dxi_theta.u_dxi_crown_constants_lbs[n_layer // 2]
                    C_k_U, c_k_U = self.u_dxi_theta.u_dxi_crown_coefficients_ubs[n_layer // 2], self.u_dxi_theta.u_dxi_crown_constants_ubs[n_layer // 2]

                    # from u_dxi_theta: \partial_{z^{k-1}} z^k lower and upper bounds
                    partial_dz_k_1_z_k_lbs, partial_dz_k_1_z_k_ubs = self.u_dxi_theta.partial_z_k_z_k_1_lbs[n_layer // 2], self.u_dxi_theta.partial_z_k_z_k_1_ubs[n_layer // 2]
                    D_k_L, d_k_L = self.u_dxi_theta.partial_z_k_z_k_1_coefficients_lbs[n_layer // 2], self.u_dxi_theta.partial_z_k_z_k_1_constants_lbs[n_layer // 2]
                    D_k_U, d_k_U = self.u_dxi_theta.partial_z_k_z_k_1_coefficients_ubs[n_layer // 2], self.u_dxi_theta.partial_z_k_z_k_1_constants_ubs[n_layer // 2]

                    u_dxixi_lbs = prev_layers_u_dxidxi_lower_bound[n_layer // 2]
                    u_dxixi_ubs = prev_layers_u_dxidxi_upper_bound[n_layer // 2]

                    weight = layer.weight
                    W_pos = torch.clamp(weight, min=0)
                    W_neg = torch.clamp(weight, max=0)

                    # define E, e, and H, h as per the paper notation
                    E_k_U = W_pos @ C_k_U + W_neg @ C_k_L
                    e_k_U = W_pos @ c_k_U + W_neg @ c_k_L
                    E_k_L = W_pos @ C_k_L + W_neg @ C_k_U
                    e_k_L = W_pos @ c_k_L + W_neg @ c_k_U

                    lambda_k_L, mu_k_L, lambda_k_U, mu_k_U = layer_output_lines.T
                    lambda_k_L, mu_k_L = lambda_k_L.unsqueeze(1), mu_k_L.unsqueeze(1)
                    lambda_k_U, mu_k_U = lambda_k_U.unsqueeze(1), mu_k_U.unsqueeze(1)

                    H_k_U = lambda_k_U.clamp(min=0) * A_k_U + lambda_k_U.clamp(max=0) * A_k_L
                    h_k_U = lambda_k_U.clamp(min=0) * a_k_U.unsqueeze(1) + lambda_k_U.clamp(max=0) * a_k_L.unsqueeze(1) + lambda_k_U * mu_k_U
                    H_k_L = lambda_k_L.clamp(min=0) * A_k_L + lambda_k_L.clamp(max=0) * A_k_U
                    h_k_L = lambda_k_L.clamp(min=0) * a_k_L.unsqueeze(1) + lambda_k_L.clamp(max=0) * a_k_U.unsqueeze(1) + lambda_k_L * mu_k_L

                    # compute \theta_j^{(k),U} and \theta_j^{(k),L} using E_k_U_j, e_k_U_j, E_k_L_j, e_k_L_j to use in following McCormick relaxation
                    theta_k_U = torch.sum(E_k_U.clamp(min=0) * self.x_ub + E_k_U.clamp(max=0) * self.x_lb, dim=1) + e_k_U
                    theta_k_L = torch.sum(E_k_L.clamp(min=0) * self.x_lb + E_k_L.clamp(max=0) * self.x_ub, dim=1) + e_k_L
                    theta_k_L, theta_k_U = theta_k_L.unsqueeze(1), theta_k_U.unsqueeze(1)

                    # define \nu's for McCormick relaxation of first, element-wise multiplication
                    # TODO: make rho_k_j and tau_k_j depend on actual bounds somehow
                    iota_k_L, iota_k_U = layer_actual_bounds.T
                    iota_k_L, iota_k_U = iota_k_L.unsqueeze(1), iota_k_U.unsqueeze(1)
                    nu_k_U_0 = rho_k * iota_k_U + (1 - rho_k) * iota_k_L
                    nu_k_U_1 = rho_k * theta_k_L + (1 - rho_k) * theta_k_U
                    nu_k_U_2 = -rho_k * iota_k_U * theta_k_L - (1 - rho_k) * iota_k_L * theta_k_U

                    nu_k_L_0 = tau_k * iota_k_L + (1 - tau_k) * iota_k_U
                    nu_k_L_1 = tau_k * theta_k_L + (1 - tau_k) * theta_k_U
                    nu_k_L_2 = -tau_k * iota_k_L * theta_k_L - (1 - tau_k) * iota_k_U * theta_k_U

                    # define \upsilon's for the linear layer that comes after that
                    upsilon_k_U_0 = nu_k_U_0 * W_pos + nu_k_L_0 * W_neg
                    upsilon_k_U_1 = nu_k_U_1 * W_pos + nu_k_L_1 * W_neg
                    upsilon_k_U_2 = nu_k_U_2 * W_pos + nu_k_L_2 * W_neg
                    upsilon_k_U_0_pos, upsilon_k_U_0_neg = upsilon_k_U_0.clamp(min=0), upsilon_k_U_0.clamp(max=0)
                    upsilon_k_U_1_pos, upsilon_k_U_1_neg = upsilon_k_U_1.clamp(min=0), upsilon_k_U_1.clamp(max=0)

                    upsilon_k_L_0 = nu_k_L_0 * W_pos + nu_k_U_0 * W_neg
                    upsilon_k_L_1 = nu_k_L_1 * W_pos + nu_k_U_1 * W_neg
                    upsilon_k_L_2 = nu_k_L_2 * W_pos + nu_k_U_2 * W_neg
                    upsilon_k_L_0_pos, upsilon_k_L_0_neg = upsilon_k_L_0.clamp(min=0), upsilon_k_L_0.clamp(max=0)
                    upsilon_k_L_1_pos, upsilon_k_L_1_neg = upsilon_k_L_1.clamp(min=0), upsilon_k_L_1.clamp(max=0)

                    # define M and m from the upsilon values
                    M_k_U = upsilon_k_U_0_pos.unsqueeze(2) * E_k_U.unsqueeze(1).repeat(1, weight.shape[1], 1) +\
                        upsilon_k_U_0_neg.unsqueeze(2) * E_k_L.unsqueeze(1).repeat(1, weight.shape[1], 1) +\
                        upsilon_k_U_1_pos.unsqueeze(2) * H_k_U.unsqueeze(1).repeat(1, weight.shape[1], 1) +\
                        upsilon_k_U_1_neg.unsqueeze(2) * H_k_L.unsqueeze(1).repeat(1, weight.shape[1], 1)
                    m_k_U = upsilon_k_U_0_pos * e_k_U.unsqueeze(1).repeat(1, weight.shape[1]) +\
                        upsilon_k_U_0_neg * e_k_L.unsqueeze(1).repeat(1, weight.shape[1]) +\
                        upsilon_k_U_1_pos * h_k_U.repeat(1, weight.shape[1]) +\
                        upsilon_k_U_1_neg * h_k_L.repeat(1, weight.shape[1]) +\
                        upsilon_k_U_2
                    
                    M_k_L = upsilon_k_L_0_pos.unsqueeze(2) * E_k_L.unsqueeze(1).repeat(1, weight.shape[1], 1) +\
                        upsilon_k_L_0_neg.unsqueeze(2) * E_k_U.unsqueeze(1).repeat(1, weight.shape[1], 1) +\
                        upsilon_k_L_1_pos.unsqueeze(2) * H_k_L.unsqueeze(1).repeat(1, weight.shape[1], 1) +\
                        upsilon_k_L_1_neg.unsqueeze(2) * H_k_U.unsqueeze(1).repeat(1, weight.shape[1], 1)
                    m_k_L = upsilon_k_L_0_pos * e_k_L.unsqueeze(1).repeat(1, weight.shape[1]) +\
                        upsilon_k_L_0_neg * e_k_U.unsqueeze(1).repeat(1, weight.shape[1]) +\
                        upsilon_k_L_1_pos * h_k_L.repeat(1, weight.shape[1]) +\
                        upsilon_k_L_1_neg * h_k_U.repeat(1, weight.shape[1]) +\
                        upsilon_k_L_2
                    
                    partial_dxiz_k_1_z_k_ubs = torch.sum(M_k_U * ((M_k_U >= 0) * self.x_ub + (M_k_U < 0) * self.x_lb), dim=2) + m_k_U
                    partial_dxiz_k_1_z_k_lbs = torch.sum(M_k_L * ((M_k_L >= 0) * self.x_lb + (M_k_L < 0) * self.x_ub), dim=2) + m_k_L

                    # define all alphas and betas starting with _0, _1, _2, _3 and _4
                    # TODO: eta_k_j_n and zeta_k_j_n depend on actual bounds somehow
                    alpha_k_0 = eta_k * partial_dxiz_k_1_z_k_ubs + (1 - eta_k) * partial_dxiz_k_1_z_k_lbs
                    alpha_k_1 = eta_k * partial_dxi_z_k_1_lbs + (1 - eta_k) * partial_dxi_z_k_1_ubs
                    alpha_k_2 = gamma_k * partial_dz_k_1_z_k_ubs + (1 - gamma_k) * partial_dz_k_1_z_k_lbs
                    alpha_k_3 = gamma_k * u_dxixi_lbs + (1 - gamma_k) * u_dxixi_ubs
                    alpha_k_4 = - eta_k * partial_dxiz_k_1_z_k_ubs * partial_dxi_z_k_1_lbs - (1 - eta_k) * partial_dxiz_k_1_z_k_lbs * partial_dxi_z_k_1_ubs + - gamma_k * partial_dz_k_1_z_k_ubs * u_dxixi_lbs - (1 - gamma_k) * partial_dz_k_1_z_k_lbs * u_dxixi_ubs

                    alpha_k_0_pos, alpha_k_0_neg = alpha_k_0.clamp(min=0), alpha_k_0.clamp(max=0)
                    alpha_k_1_pos, alpha_k_1_neg = alpha_k_1.clamp(min=0), alpha_k_1.clamp(max=0)
                    alpha_k_3_pos, alpha_k_3_neg = alpha_k_3.clamp(min=0), alpha_k_3.clamp(max=0)

                    beta_k_0 = zeta_k * partial_dxiz_k_1_z_k_lbs + (1 - zeta_k) * partial_dxiz_k_1_z_k_ubs
                    beta_k_1 = zeta_k * partial_dxi_z_k_1_lbs + (1 - zeta_k) * partial_dxi_z_k_1_ubs
                    beta_k_2 = delta_k * partial_dz_k_1_z_k_lbs + (1 - delta_k) * partial_dz_k_1_z_k_ubs
                    beta_k_3 = delta_k * u_dxixi_lbs + (1 - delta_k) * u_dxixi_ubs
                    beta_k_4 = - zeta_k * partial_dxiz_k_1_z_k_lbs * partial_dxi_z_k_1_lbs - (1 - zeta_k) * partial_dxiz_k_1_z_k_ubs * partial_dxi_z_k_1_ubs +\
                        - delta_k * partial_dz_k_1_z_k_lbs * u_dxixi_lbs - (1 - delta_k) * partial_dz_k_1_z_k_ubs * u_dxixi_ubs

                    beta_k_0_pos, beta_k_0_neg = beta_k_0.clamp(min=0), beta_k_0.clamp(max=0)
                    beta_k_1_pos, beta_k_1_neg = beta_k_1.clamp(min=0), beta_k_1.clamp(max=0)
                    beta_k_3_pos, beta_k_3_neg = beta_k_3.clamp(min=0), beta_k_3.clamp(max=0)

                    # define the alpha_5, alpha_6, beta_5, and beta_6 from Equation 19
                    alpha_k_5 = alpha_k_0_pos.unsqueeze(2) * C_k_U.unsqueeze(0).repeat(weight.shape[0], 1, 1) + alpha_k_0_neg.unsqueeze(2) * C_k_L.unsqueeze(0).repeat(weight.shape[0], 1, 1) +\
                        alpha_k_1_pos.unsqueeze(0).repeat(weight.shape[0], 1).unsqueeze(2) * M_k_U + alpha_k_1_neg.unsqueeze(0).repeat(weight.shape[0], 1).unsqueeze(2) * M_k_L +\
                        alpha_k_3_pos.unsqueeze(0).repeat(weight.shape[0], 1).unsqueeze(2) * D_k_U + alpha_k_3_neg.unsqueeze(0).repeat(weight.shape[0], 1).unsqueeze(2) * D_k_L
                    alpha_k_6 = alpha_k_0_pos * c_k_U.unsqueeze(0).repeat(weight.shape[0], 1) + alpha_k_0_neg * c_k_L.unsqueeze(0).repeat(weight.shape[0], 1) +\
                        alpha_k_1_pos.unsqueeze(0).repeat(weight.shape[0], 1) * m_k_U + alpha_k_1_neg.unsqueeze(0).repeat(weight.shape[0], 1) * m_k_L +\
                        alpha_k_3_pos.unsqueeze(0).repeat(weight.shape[0], 1) * d_k_U + alpha_k_3_neg.unsqueeze(0).repeat(weight.shape[0], 1) * d_k_L +\
                        alpha_k_4
                    
                    beta_k_5 = beta_k_0_pos.unsqueeze(2) * C_k_L.unsqueeze(0).repeat(weight.shape[0], 1, 1) + beta_k_0_neg.unsqueeze(2) * C_k_U.unsqueeze(0).repeat(weight.shape[0], 1, 1) +\
                        beta_k_1_pos.unsqueeze(0).repeat(weight.shape[0], 1).unsqueeze(2) * M_k_L + beta_k_1_neg.unsqueeze(0).repeat(weight.shape[0], 1).unsqueeze(2) * M_k_U +\
                        beta_k_3_pos.unsqueeze(0).repeat(weight.shape[0], 1).unsqueeze(2) * D_k_L + beta_k_3_neg.unsqueeze(0).repeat(weight.shape[0], 1).unsqueeze(2) * D_k_U
                    beta_k_6 = beta_k_0_pos * c_k_L.unsqueeze(0).repeat(weight.shape[0], 1) + beta_k_0_neg * c_k_U.unsqueeze(0).repeat(weight.shape[0], 1) +\
                        beta_k_1_pos.unsqueeze(0).repeat(weight.shape[0], 1) * m_k_L + beta_k_1_neg.unsqueeze(0).repeat(weight.shape[0], 1) * m_k_U +\
                        beta_k_3_pos.unsqueeze(0).repeat(weight.shape[0], 1) * d_k_L + beta_k_3_neg.unsqueeze(0).repeat(weight.shape[0], 1) * d_k_U +\
                        beta_k_4

                    # cache these computations because at the component level back-prop they are constant
                    self.component_non_backward_dependencies[n_layer] = {}
                    self.component_non_backward_dependencies[n_layer]["alpha_k_2"] = alpha_k_2
                    self.component_non_backward_dependencies[n_layer]["alpha_k_5"] = alpha_k_5
                    self.component_non_backward_dependencies[n_layer]["alpha_k_6"] = alpha_k_6

                    self.component_non_backward_dependencies[n_layer]["beta_k_2"] = beta_k_2
                    self.component_non_backward_dependencies[n_layer]["beta_k_5"] = beta_k_5
                    self.component_non_backward_dependencies[n_layer]["beta_k_6"] = beta_k_6

                # backward propagation through the partial derivative network
                if n_layer == n_layers - 1:
                    if is_final:
                        # penultimate layer of the full NN, take into account the weight of the final layer
                        last_W = layers[n_layers+1].weight
                        last_W_pos = last_W.clamp(min=0)
                        last_W_neg = last_W.clamp(max=0)

                        rho_0_U = last_W_pos @ alpha_k_2 + last_W_neg @ beta_k_2
                        rho_1_U = (last_W_pos @ alpha_k_5.reshape(alpha_k_5.shape[0], -1) + last_W_neg @ beta_k_5.reshape(beta_k_5.shape[0], -1)).reshape(last_W_pos.shape[0], *alpha_k_5.shape[1:])
                        rho_2_U = last_W_pos @ alpha_k_6 + last_W_neg @ beta_k_6

                        rho_0_L = last_W_pos @ beta_k_2 + last_W_neg @ alpha_k_2
                        rho_1_L = (last_W_pos @ beta_k_5.reshape(beta_k_5.shape[0], -1) + last_W_neg @ alpha_k_5.reshape(alpha_k_5.shape[0], -1)).reshape(last_W_pos.shape[0], *alpha_k_5.shape[1:])
                        rho_2_L = last_W_pos @ beta_k_6 + last_W_neg @ alpha_k_6
                    else:
                        # it's just a middle layer, we want to compute it's output directly
                        rho_0_U = alpha_k_2
                        rho_1_U = alpha_k_5
                        rho_2_U = alpha_k_6
                        rho_0_L = beta_k_2
                        rho_1_L = beta_k_5
                        rho_2_L = beta_k_6

                        # it's the first time we're exploring this layer, add intermediate coefficients, constants and bounds for further computations
                        # self.partial_z_k_z_k_1_coefficients_lbs.append(new_partial_z_k_z_k_1_coeffs_lbs)
                        # self.partial_z_k_z_k_1_constants_lbs.append(new_partial_z_k_z_k_1_consts_lbs)
                        # self.partial_z_k_z_k_1_coefficients_ubs.append(new_partial_z_k_z_k_1_coeffs_ubs)
                        # self.partial_z_k_z_k_1_constants_ubs.append(new_partial_z_k_z_k_1_consts_ubs)

                        # self.partial_z_k_z_k_1_ubs.append(first_matmul_ubs)
                        # self.partial_z_k_z_k_1_lbs.append(first_matmul_lbs)
                else:
                    new_rho_0_U = (rho_0_U.unsqueeze(2).clamp(min=0) * alpha_k_2.unsqueeze(0) + rho_0_U.unsqueeze(2).clamp(max=0) * beta_k_2.unsqueeze(0)).sum(dim=1)
                    new_rho_1_U = (rho_0_U.unsqueeze(2).unsqueeze(3).clamp(min=0) * alpha_k_5.unsqueeze(0) + rho_0_U.unsqueeze(2).unsqueeze(3).clamp(max=0) * beta_k_5.unsqueeze(0)).sum(dim=1) + (1/layer.weight.shape[1]) * rho_1_U.sum(dim=1).unsqueeze(1)
                    new_rho_2_U = (rho_0_U.unsqueeze(2).clamp(min=0) * alpha_k_6.unsqueeze(0) + rho_0_U.unsqueeze(2).clamp(max=0) * beta_k_6.unsqueeze(0)).sum(dim=1) + (1/layer.weight.shape[1]) * rho_2_U.sum(dim=1).unsqueeze(1)

                    new_rho_0_L = (rho_0_L.unsqueeze(2).clamp(min=0) * beta_k_2.unsqueeze(0) + rho_0_L.unsqueeze(2).clamp(max=0) * alpha_k_2.unsqueeze(0)).sum(dim=1)
                    new_rho_1_L = (rho_0_L.unsqueeze(2).unsqueeze(3).clamp(min=0) * beta_k_5.unsqueeze(0) + rho_0_L.unsqueeze(2).unsqueeze(3).clamp(max=0) * alpha_k_5.unsqueeze(0)).sum(dim=1) + (1/layer.weight.shape[1]) * rho_1_L.sum(dim=1).unsqueeze(1)
                    new_rho_2_L = (rho_0_L.unsqueeze(2).clamp(min=0) * beta_k_6.unsqueeze(0) + rho_0_L.unsqueeze(2).clamp(max=0) * alpha_k_6.unsqueeze(0)).sum(dim=1) + (1/layer.weight.shape[1]) * rho_2_L.sum(dim=1).unsqueeze(1)

                    rho_0_U = new_rho_0_U
                    rho_1_U = new_rho_1_U
                    rho_2_U = new_rho_2_U
                    rho_0_L = new_rho_0_L
                    rho_1_L = new_rho_1_L
                    rho_2_L = new_rho_2_L

        layer_crown_U_coefficients = rho_1_U.sum(dim=1)
        layer_crown_U_constants = rho_2_U.sum(dim=1)
        layer_crown_L_coefficients = rho_1_L.sum(dim=1)
        layer_crown_L_constants = rho_2_L.sum(dim=1)

        layer_output_upper_bounds = torch.sum(layer_crown_U_coefficients.clamp(min=0) * x_ub + layer_crown_U_coefficients.clamp(max=0) * x_lb, dim=1) + layer_crown_U_constants
        layer_output_lower_bounds = torch.sum(layer_crown_L_coefficients.clamp(min=0) * x_lb + layer_crown_L_coefficients.clamp(max=0) * x_ub, dim=1) + layer_crown_L_constants

        return layer_output_upper_bounds, layer_output_lower_bounds, (layer_crown_U_coefficients, layer_crown_U_constants, layer_crown_L_coefficients, layer_crown_L_constants)
    
    def compute_bounds(self, debug: bool = True, LP_bounds = None, backprop_mode: BackpropMode = BackpropMode.BLOCK_BACKPROP):
        self.x_lb = self.u_theta.x_lb
        self.x_ub = self.u_theta.x_ub

        # if we haven't computed the intermediate bounds on u_theta, do it now
        if not self.u_dxi_theta.computed_bounds:
            self.u_dxi_theta.compute_bounds(debug=debug)

        # proceed if the computation is successful and these variables are now populated
        assert self.u_theta.computed_bounds
        assert self.u_dxi_theta.computed_bounds

        u_theta = self.u_theta
        u_dxi_theta = self.u_dxi_theta

        rho_k = 0.5
        tau_k = 0.5
        eta_k, gamma_k = 0.5, 0.5
        zeta_k, delta_k = 0.5, 0.5

        # computing a bound on $\partial_t u_theta$ using the previously computed bounds
        u_dxixi_lower_bounds = self.lower_bounds
        u_dxixi_upper_bounds = self.upper_bounds
        self.first_sum_term_lbs = []
        self.first_sum_term_ubs = []
        self.second_sum_term_lbs = []
        self.second_sum_term_ubs = []

        # d_psi_0_dx_i is equal to 0 (second derivative of x with respect to x_i) and it'll remain the
        # same through the normalization layers
        zero_vec = [0 for _ in range(u_theta.fc_layers[0].weight.shape[1])]
        u_dxixi_lower_bounds.append(torch.tensor(zero_vec, dtype=torch.float))
        u_dxixi_upper_bounds.append(torch.tensor(zero_vec, dtype=torch.float))

        self.u_dxixi_crown_coefficients_lbs = [torch.zeros(u_dxixi_lower_bounds[-1].shape[0])]
        self.u_dxixi_crown_constants_lbs = [u_dxixi_lower_bounds[-1]]
        self.u_dxixi_crown_coefficients_ubs = [torch.zeros(u_dxixi_upper_bounds[-1].shape[0])]
        self.u_dxixi_crown_constants_ubs = [u_dxixi_upper_bounds[-1]]

        # debug computation with interval midpoint
        if debug:
            norm_point = self.u_theta.domain_bounds.mean(dim=0)
            norm_grad = copy.deepcopy(self.u_dxi_theta.lower_bounds[0]).unsqueeze(1)
            norm_d_xi_xi_d_z_k_1 = torch.tensor([[0, 0]], dtype=torch.float).T

            for norm_layer in self.u_theta.norm_layers:
                norm_point = norm_layer(norm_point)

        it_object = u_theta.fc_layers
        if debug:
            print(f"Propagating LP bounds through d^2u_theta/dx{self.component_idx}^2...")
            it_object = tqdm(u_theta.fc_layers)

        layers_activation_second_derivative_output_lines = []
        layers_activation_second_derivative_actual_bounds = []
        if backprop_mode is BackpropMode.COMPONENT_BACKPROP:
            self.component_non_backward_dependencies = {}

        for n_layer, layer in enumerate(it_object):
            if debug:
                norm_point = layer(norm_point)

            if not isinstance(layer, torch.nn.Linear):
                continue
        
            is_final = (n_layer == len(u_theta.fc_layers) - 1)
            u_dxixi_lbs = u_dxixi_lower_bounds[-1]
            u_dxixi_ubs = u_dxixi_upper_bounds[-1]
            u_dxixi_crown_coeffs_lbs = self.u_dxixi_crown_coefficients_lbs[-1]
            u_dxixi_crown_consts_lbs = self.u_dxixi_crown_constants_lbs[-1]
            u_dxixi_crown_coeffs_ubs = self.u_dxixi_crown_coefficients_ubs[-1]
            u_dxixi_crown_consts_ubs = self.u_dxixi_crown_constants_ubs[-1]

            weight = layer.weight
            W_pos = torch.clamp(weight, min=0)
            W_neg = torch.clamp(weight, max=0)

            if not is_final:
                # hybrid CROWN computation; fetch bounds, coefficients and constants from u_theta and u_dxi_theta
                # from u_theta: y^(k) lower and upper bounds
                y_k_lbs, y_k_ubs = u_theta.lower_bounds[n_layer+1], self.u_theta.upper_bounds[n_layer+1]
                A_k_U, a_k_U, A_k_L, a_k_L = u_theta.layer_CROWN_coefficients[n_layer // 2]

                # from u_dxi_theta: \partial_{x_i} z^{(k-1)} lower and upper bounds (layers in u_dxi_theta)
                partial_dxi_z_k_1_lbs, partial_dxi_z_k_1_ubs = u_dxi_theta.lower_bounds[n_layer // 2], u_dxi_theta.upper_bounds[n_layer // 2]
                C_k_L, c_k_L = u_dxi_theta.u_dxi_crown_coefficients_lbs[n_layer // 2], u_dxi_theta.u_dxi_crown_constants_lbs[n_layer // 2]
                C_k_U, c_k_U = u_dxi_theta.u_dxi_crown_coefficients_ubs[n_layer // 2], u_dxi_theta.u_dxi_crown_constants_ubs[n_layer // 2]

                # from u_dxi_theta: \partial_{z^{k-1}} z^k lower and upper bounds
                partial_dz_k_1_z_k_lbs, partial_dz_k_1_z_k_ubs = u_dxi_theta.partial_z_k_z_k_1_lbs[n_layer // 2], u_dxi_theta.partial_z_k_z_k_1_ubs[n_layer // 2]
                D_k_L, d_k_L = u_dxi_theta.partial_z_k_z_k_1_coefficients_lbs[n_layer // 2], u_dxi_theta.partial_z_k_z_k_1_constants_lbs[n_layer // 2]
                D_k_U, d_k_U = u_dxi_theta.partial_z_k_z_k_1_coefficients_ubs[n_layer // 2], u_dxi_theta.partial_z_k_z_k_1_constants_ubs[n_layer // 2]

                # use the bounds to relax sigma''(y^(k)) for all outputs
                layer_output_lines = torch.zeros(weight.shape[0], 4)
                layer_actual_bounds = torch.zeros(weight.shape[0], 2)
                for j in range(weight.shape[0]):
                    lb_lines, ub_lines = self.activation_derivative_derivative_relaxation.get_bounds(y_k_lbs[j], y_k_ubs[j])

                    # can only have a single bound, take a convex combination of them
                    # TODO: move convex combination code to a SINGLE_LINE relaxation inside the ActivationRelaxation classes
                    # m_combination_lb = np.mean([m for m, b in lb_lines])
                    # b_combination_lb = np.mean([b for m, b in lb_lines])

                    # m_combination_ub = np.mean([m for m, b in ub_lines])
                    # b_combination_ub = np.mean([b for m, b in ub_lines])

                    m_combination_lb, b_combination_lb = torch.Tensor(lb_lines).mean(dim=0)
                    m_combination_ub, b_combination_ub = torch.Tensor(ub_lines).mean(dim=0)

                    # CROWN line definition
                    if m_combination_lb != 0:
                        b_combination_lb /= m_combination_lb
                    
                    if m_combination_ub != 0:
                        b_combination_ub /= m_combination_ub

                    layer_output_lines[j, :] = torch.tensor([
                        m_combination_lb,
                        b_combination_lb,
                        m_combination_ub,
                        b_combination_ub
                    ])

                    layer_actual_bounds[j, :] = torch.tensor(
                        self.activation_derivative_derivative_relaxation.get_lb_ub_in_interval(y_k_lbs[j], y_k_ubs[j])
                    )
                
                layers_activation_second_derivative_output_lines.append(layer_output_lines)
                layers_activation_second_derivative_actual_bounds.append(layer_actual_bounds)

                if backprop_mode == BackpropMode.BLOCK_BACKPROP:
                    # define E, e, and H, h as per the paper notation
                    E_k_U = W_pos @ C_k_U + W_neg @ C_k_L
                    e_k_U = W_pos @ c_k_U + W_neg @ c_k_L
                    E_k_L = W_pos @ C_k_L + W_neg @ C_k_U
                    e_k_L = W_pos @ c_k_L + W_neg @ c_k_U

                    lambda_k_L, mu_k_L, lambda_k_U, mu_k_U = layer_output_lines.T
                    lambda_k_L, mu_k_L = lambda_k_L.unsqueeze(1), mu_k_L.unsqueeze(1)
                    lambda_k_U, mu_k_U = lambda_k_U.unsqueeze(1), mu_k_U.unsqueeze(1)

                    H_k_U = lambda_k_U.clamp(min=0) * A_k_U + lambda_k_U.clamp(max=0) * A_k_L
                    h_k_U = lambda_k_U.clamp(min=0) * a_k_U.unsqueeze(1) + lambda_k_U.clamp(max=0) * a_k_L.unsqueeze(1) + lambda_k_U * mu_k_U
                    H_k_L = lambda_k_L.clamp(min=0) * A_k_L + lambda_k_L.clamp(max=0) * A_k_U
                    h_k_L = lambda_k_L.clamp(min=0) * a_k_L.unsqueeze(1) + lambda_k_L.clamp(max=0) * a_k_U.unsqueeze(1) + lambda_k_L * mu_k_L

                    # compute \theta_j^{(k),U} and \theta_j^{(k),L} using E_k_U_j, e_k_U_j, E_k_L_j, e_k_L_j to use in following McCormick relaxation
                    theta_k_U = torch.sum(E_k_U.clamp(min=0) * self.x_ub + E_k_U.clamp(max=0) * self.x_lb, dim=1) + e_k_U
                    theta_k_L = torch.sum(E_k_L.clamp(min=0) * self.x_lb + E_k_L.clamp(max=0) * self.x_ub, dim=1) + e_k_L
                    theta_k_L, theta_k_U = theta_k_L.unsqueeze(1), theta_k_U.unsqueeze(1)

                    # define \nu's for McCormick relaxation of first, element-wise multiplication
                    # TODO: make rho_k_j and tau_k_j depend on actual bounds somehow
                    iota_k_L, iota_k_U = layer_actual_bounds.T
                    iota_k_L, iota_k_U = iota_k_L.unsqueeze(1), iota_k_U.unsqueeze(1)
                    nu_k_U_0 = rho_k * iota_k_U + (1 - rho_k) * iota_k_L
                    nu_k_U_1 = rho_k * theta_k_L + (1 - rho_k) * theta_k_U
                    nu_k_U_2 = -rho_k * iota_k_U * theta_k_L - (1 - rho_k) * iota_k_L * theta_k_U

                    nu_k_L_0 = tau_k * iota_k_L + (1 - tau_k) * iota_k_U
                    nu_k_L_1 = tau_k * theta_k_L + (1 - tau_k) * theta_k_U
                    nu_k_L_2 = -tau_k * iota_k_L * theta_k_L - (1 - tau_k) * iota_k_U * theta_k_U

                    # define \upsilon's for the linear layer that comes after that
                    upsilon_k_U_0 = nu_k_U_0 * W_pos + nu_k_L_0 * W_neg
                    upsilon_k_U_1 = nu_k_U_1 * W_pos + nu_k_L_1 * W_neg
                    upsilon_k_U_2 = nu_k_U_2 * W_pos + nu_k_L_2 * W_neg
                    upsilon_k_U_0_pos, upsilon_k_U_0_neg = upsilon_k_U_0.clamp(min=0), upsilon_k_U_0.clamp(max=0)
                    upsilon_k_U_1_pos, upsilon_k_U_1_neg = upsilon_k_U_1.clamp(min=0), upsilon_k_U_1.clamp(max=0)

                    upsilon_k_L_0 = nu_k_L_0 * W_pos + nu_k_U_0 * W_neg
                    upsilon_k_L_1 = nu_k_L_1 * W_pos + nu_k_U_1 * W_neg
                    upsilon_k_L_2 = nu_k_L_2 * W_pos + nu_k_U_2 * W_neg
                    upsilon_k_L_0_pos, upsilon_k_L_0_neg = upsilon_k_L_0.clamp(min=0), upsilon_k_L_0.clamp(max=0)
                    upsilon_k_L_1_pos, upsilon_k_L_1_neg = upsilon_k_L_1.clamp(min=0), upsilon_k_L_1.clamp(max=0)

                    # define M and m from the upsilon values
                    M_k_U = upsilon_k_U_0_pos.unsqueeze(2) * E_k_U.unsqueeze(1).repeat(1, weight.shape[1], 1) +\
                        upsilon_k_U_0_neg.unsqueeze(2) * E_k_L.unsqueeze(1).repeat(1, weight.shape[1], 1) +\
                        upsilon_k_U_1_pos.unsqueeze(2) * H_k_U.unsqueeze(1).repeat(1, weight.shape[1], 1) +\
                        upsilon_k_U_1_neg.unsqueeze(2) * H_k_L.unsqueeze(1).repeat(1, weight.shape[1], 1)
                    m_k_U = upsilon_k_U_0_pos * e_k_U.unsqueeze(1).repeat(1, weight.shape[1]) +\
                        upsilon_k_U_0_neg * e_k_L.unsqueeze(1).repeat(1, weight.shape[1]) +\
                        upsilon_k_U_1_pos * h_k_U.repeat(1, weight.shape[1]) +\
                        upsilon_k_U_1_neg * h_k_L.repeat(1, weight.shape[1]) +\
                        upsilon_k_U_2
                    
                    M_k_L = upsilon_k_L_0_pos.unsqueeze(2) * E_k_L.unsqueeze(1).repeat(1, weight.shape[1], 1) +\
                        upsilon_k_L_0_neg.unsqueeze(2) * E_k_U.unsqueeze(1).repeat(1, weight.shape[1], 1) +\
                        upsilon_k_L_1_pos.unsqueeze(2) * H_k_L.unsqueeze(1).repeat(1, weight.shape[1], 1) +\
                        upsilon_k_L_1_neg.unsqueeze(2) * H_k_U.unsqueeze(1).repeat(1, weight.shape[1], 1)
                    m_k_L = upsilon_k_L_0_pos * e_k_L.unsqueeze(1).repeat(1, weight.shape[1]) +\
                        upsilon_k_L_0_neg * e_k_U.unsqueeze(1).repeat(1, weight.shape[1]) +\
                        upsilon_k_L_1_pos * h_k_L.repeat(1, weight.shape[1]) +\
                        upsilon_k_L_1_neg * h_k_U.repeat(1, weight.shape[1]) +\
                        upsilon_k_L_2
                    
                    partial_dxiz_k_1_z_k_ubs = torch.sum(M_k_U * ((M_k_U >= 0) * self.x_ub + (M_k_U < 0) * self.x_lb), dim=2) + m_k_U
                    partial_dxiz_k_1_z_k_lbs = torch.sum(M_k_L * ((M_k_L >= 0) * self.x_lb + (M_k_L < 0) * self.x_ub), dim=2) + m_k_L

                    # define all alphas and betas starting with _0, _1, _2, _3 and _4
                    # TODO: eta_k_j_n and zeta_k_j_n depend on actual bounds somehow
                    alpha_k_0 = eta_k * partial_dxiz_k_1_z_k_ubs + (1 - eta_k) * partial_dxiz_k_1_z_k_lbs
                    alpha_k_1 = eta_k * partial_dxi_z_k_1_lbs + (1 - eta_k) * partial_dxi_z_k_1_ubs
                    alpha_k_2 = gamma_k * partial_dz_k_1_z_k_ubs + (1 - gamma_k) * partial_dz_k_1_z_k_lbs
                    alpha_k_3 = gamma_k * u_dxixi_lbs + (1 - gamma_k) * u_dxixi_ubs
                    alpha_k_4 = - eta_k * partial_dxiz_k_1_z_k_ubs * partial_dxi_z_k_1_lbs - (1 - eta_k) * partial_dxiz_k_1_z_k_lbs * partial_dxi_z_k_1_ubs + - gamma_k * partial_dz_k_1_z_k_ubs * u_dxixi_lbs - (1 - gamma_k) * partial_dz_k_1_z_k_lbs * u_dxixi_ubs

                    alpha_k_0_pos, alpha_k_0_neg = alpha_k_0.clamp(min=0), alpha_k_0.clamp(max=0)
                    alpha_k_1_pos, alpha_k_1_neg = alpha_k_1.clamp(min=0), alpha_k_1.clamp(max=0)
                    alpha_k_2_pos, alpha_k_2_neg = alpha_k_2.clamp(min=0), alpha_k_2.clamp(max=0)
                    alpha_k_3_pos, alpha_k_3_neg = alpha_k_3.clamp(min=0), alpha_k_3.clamp(max=0)

                    beta_k_0 = zeta_k * partial_dxiz_k_1_z_k_lbs + (1 - zeta_k) * partial_dxiz_k_1_z_k_ubs
                    beta_k_1 = zeta_k * partial_dxi_z_k_1_lbs + (1 - zeta_k) * partial_dxi_z_k_1_ubs
                    beta_k_2 = delta_k * partial_dz_k_1_z_k_lbs + (1 - delta_k) * partial_dz_k_1_z_k_ubs
                    beta_k_3 = delta_k * u_dxixi_lbs + (1 - delta_k) * u_dxixi_ubs
                    beta_k_4 = - zeta_k * partial_dxiz_k_1_z_k_lbs * partial_dxi_z_k_1_lbs - (1 - zeta_k) * partial_dxiz_k_1_z_k_ubs * partial_dxi_z_k_1_ubs +\
                        - delta_k * partial_dz_k_1_z_k_lbs * u_dxixi_lbs - (1 - delta_k) * partial_dz_k_1_z_k_ubs * u_dxixi_ubs

                    beta_k_0_pos, beta_k_0_neg = beta_k_0.clamp(min=0), beta_k_0.clamp(max=0)
                    beta_k_1_pos, beta_k_1_neg = beta_k_1.clamp(min=0), beta_k_1.clamp(max=0)
                    beta_k_2_pos, beta_k_2_neg = beta_k_2.clamp(min=0), beta_k_2.clamp(max=0)
                    beta_k_3_pos, beta_k_3_neg = beta_k_3.clamp(min=0), beta_k_3.clamp(max=0)

                    # define the alpha_5, alpha_6, beta_5, and beta_6 from Equation 19
                    alpha_k_5 = alpha_k_0_pos.unsqueeze(2) * C_k_U.unsqueeze(0).repeat(weight.shape[0], 1, 1) + alpha_k_0_neg.unsqueeze(2) * C_k_L.unsqueeze(0).repeat(weight.shape[0], 1, 1) +\
                        alpha_k_1_pos.unsqueeze(0).repeat(weight.shape[0], 1).unsqueeze(2) * M_k_U + alpha_k_1_neg.unsqueeze(0).repeat(weight.shape[0], 1).unsqueeze(2) * M_k_L +\
                        alpha_k_3_pos.unsqueeze(0).repeat(weight.shape[0], 1).unsqueeze(2) * D_k_U + alpha_k_3_neg.unsqueeze(0).repeat(weight.shape[0], 1).unsqueeze(2) * D_k_L
                    alpha_k_6 = alpha_k_0_pos * c_k_U.unsqueeze(0).repeat(weight.shape[0], 1) + alpha_k_0_neg * c_k_L.unsqueeze(0).repeat(weight.shape[0], 1) +\
                        alpha_k_1_pos.unsqueeze(0).repeat(weight.shape[0], 1) * m_k_U + alpha_k_1_neg.unsqueeze(0).repeat(weight.shape[0], 1) * m_k_L +\
                        alpha_k_3_pos.unsqueeze(0).repeat(weight.shape[0], 1) * d_k_U + alpha_k_3_neg.unsqueeze(0).repeat(weight.shape[0], 1) * d_k_L +\
                        alpha_k_4
                    
                    beta_k_5 = beta_k_0_pos.unsqueeze(2) * C_k_L.unsqueeze(0).repeat(weight.shape[0], 1, 1) + beta_k_0_neg.unsqueeze(2) * C_k_U.unsqueeze(0).repeat(weight.shape[0], 1, 1) +\
                        beta_k_1_pos.unsqueeze(0).repeat(weight.shape[0], 1).unsqueeze(2) * M_k_L + beta_k_1_neg.unsqueeze(0).repeat(weight.shape[0], 1).unsqueeze(2) * M_k_U +\
                        beta_k_3_pos.unsqueeze(0).repeat(weight.shape[0], 1).unsqueeze(2) * D_k_L + beta_k_3_neg.unsqueeze(0).repeat(weight.shape[0], 1).unsqueeze(2) * D_k_U
                    beta_k_6 = beta_k_0_pos * c_k_L.unsqueeze(0).repeat(weight.shape[0], 1) + beta_k_0_neg * c_k_U.unsqueeze(0).repeat(weight.shape[0], 1) +\
                        beta_k_1_pos.unsqueeze(0).repeat(weight.shape[0], 1) * m_k_L + beta_k_1_neg.unsqueeze(0).repeat(weight.shape[0], 1) * m_k_U +\
                        beta_k_3_pos.unsqueeze(0).repeat(weight.shape[0], 1) * d_k_L + beta_k_3_neg.unsqueeze(0).repeat(weight.shape[0], 1) * d_k_U +\
                        beta_k_4
                    
                    # replacing \partial_{x_ix_i} z_n^{(k-1)} in Eq. 19 using the linear bounds from the previous layer
                    # if isinstance(u_dxixi_crown_coeffs_lbs, list):
                    #     u_dxixi_crown_coeffs_lbs = torch.vstack(u_dxixi_crown_coeffs_lbs)
                    #     u_dxixi_crown_consts_lbs = torch.vstack(u_dxixi_crown_consts_lbs).flatten()

                    #     u_dxixi_crown_coeffs_ubs = torch.vstack(u_dxixi_crown_coeffs_ubs)
                    #     u_dxixi_crown_consts_ubs = torch.vstack(u_dxixi_crown_consts_ubs).flatten()

                    alpha_k_7 = alpha_k_2_pos.unsqueeze(2) * u_dxixi_crown_coeffs_ubs.unsqueeze(0).repeat(weight.shape[0], 1, 1) + alpha_k_2_neg.unsqueeze(2) * u_dxixi_crown_coeffs_lbs.unsqueeze(0).repeat(weight.shape[0], 1, 1) + alpha_k_5
                    alpha_k_8 = alpha_k_2_pos * u_dxixi_crown_consts_ubs.unsqueeze(0).repeat(weight.shape[0], 1) + alpha_k_2_neg * u_dxixi_crown_consts_lbs.unsqueeze(0).repeat(weight.shape[0], 1) + alpha_k_6
                    sum_alpha_k_7 = alpha_k_7.sum(dim=1)
                    sum_alpha_k_8 = alpha_k_8.sum(dim=1)
                    new_u_dxixi_crown_coeffs_ubs, new_u_dxixi_crown_consts_ubs = sum_alpha_k_7, sum_alpha_k_8

                    beta_k_7 = beta_k_2_pos.unsqueeze(2) * u_dxixi_crown_coeffs_lbs.unsqueeze(0).repeat(weight.shape[0], 1, 1) + beta_k_2_neg.unsqueeze(2) * u_dxixi_crown_coeffs_ubs.unsqueeze(0).repeat(weight.shape[0], 1, 1) + beta_k_5
                    beta_k_8 = beta_k_2_pos * u_dxixi_crown_consts_lbs.unsqueeze(0).repeat(weight.shape[0], 1) + beta_k_2_neg * u_dxixi_crown_consts_ubs.unsqueeze(0).repeat(weight.shape[0], 1) + beta_k_6
                    sum_beta_k_7 = beta_k_7.sum(dim=1)
                    sum_beta_k_8 = beta_k_8.sum(dim=1)
                    new_u_dxixi_crown_coeffs_lbs, new_u_dxixi_crown_consts_lbs = sum_beta_k_7, sum_beta_k_8

                    layer_output_upper_bounds = torch.sum(new_u_dxixi_crown_coeffs_ubs * ((new_u_dxixi_crown_coeffs_ubs >= 0) * self.x_ub + (new_u_dxixi_crown_coeffs_ubs < 0) * self.x_lb), dim=1) + new_u_dxixi_crown_consts_ubs
                    layer_output_lower_bounds = torch.sum(new_u_dxixi_crown_coeffs_lbs * ((new_u_dxixi_crown_coeffs_lbs >= 0) * self.x_lb + (new_u_dxixi_crown_coeffs_lbs < 0) * self.x_ub), dim=1) + new_u_dxixi_crown_consts_lbs

                    # for j in range(weight.shape[0]):
                        # define E, e, and H, h as per the paper notation
                        # E_k_U_j = sum([W_pos[j, n] * C_k_U[n] + W_neg[j, n] * C_k_L[n] for n in range(weight.shape[1])])
                        # e_k_U_j = sum([W_pos[j, n] * c_k_U[n] + W_neg[j, n] * c_k_L[n] for n in range(weight.shape[1])])
                        # E_k_L_j = sum([W_pos[j, n] * C_k_L[n] + W_neg[j, n] * C_k_U[n] for n in range(weight.shape[1])])
                        # e_k_L_j = sum([W_pos[j, n] * c_k_L[n] + W_neg[j, n] * c_k_U[n] for n in range(weight.shape[1])])

                        # try:
                        #     assert torch.linalg.norm(E_k_U[j] - E_k_U_j) <= 1e-5
                        #     assert torch.linalg.norm(e_k_U[j] - e_k_U_j) <= 1e-5
                        #     assert torch.linalg.norm(E_k_L[j] - E_k_L_j) <= 1e-5
                        #     assert torch.linalg.norm(e_k_L[j] - e_k_L_j) <= 1e-5
                        # except:
                        #     import pdb
                        #     pdb.set_trace()

                        # lambda_k_L_j, mu_k_L_j, lambda_k_U_j, mu_k_U_j = layer_output_lines[j]
                        # H_k_U_j = lambda_k_U_j * (float(lambda_k_U_j >= 0) * A_k_U[j] + float(lambda_k_U_j < 0) * A_k_L[j])
                        # h_k_U_j = lambda_k_U_j * (float(lambda_k_U_j >= 0) * a_k_U[j] + float(lambda_k_U_j < 0) * a_k_L[j]) + lambda_k_U_j * mu_k_U_j
                        # H_k_L_j = lambda_k_L_j * (float(lambda_k_L_j >= 0) * A_k_L[j] + float(lambda_k_L_j < 0) * A_k_U[j])
                        # h_k_L_j = lambda_k_L_j * (float(lambda_k_L_j >= 0) * a_k_L[j] + float(lambda_k_L_j < 0) * a_k_U[j]) + lambda_k_L_j * mu_k_L_j

                        # try:
                        #     assert torch.linalg.norm(H_k_U[j] - H_k_U_j) <= 1e-5
                        #     assert torch.linalg.norm(h_k_U[j] - h_k_U_j) <= 1e-5
                        #     assert torch.linalg.norm(H_k_L[j] - H_k_L_j) <= 1e-5
                        #     assert torch.linalg.norm(h_k_L[j] - h_k_L_j) <= 1e-5
                        # except:
                        #     import pdb
                        #     pdb.set_trace()

                        # # compute \theta_j^{(k),U} and \theta_j^{(k),L} using E_k_U_j, e_k_U_j, E_k_L_j, e_k_L_j to use in following McCormick relaxation
                        # theta_k_U_j = torch.sum(E_k_U_j * ((E_k_U_j >= 0) * self.x_ub + (E_k_U_j < 0) * self.x_lb)) + e_k_U_j
                        # theta_k_L_j = torch.sum(E_k_L_j * ((E_k_L_j >= 0) * self.x_lb + (E_k_L_j < 0) * self.x_ub)) + e_k_L_j

                        # try:
                        #     assert torch.linalg.norm(theta_k_U[j] - theta_k_U_j) <= 1e-5
                        #     assert torch.linalg.norm(theta_k_L[j] - theta_k_L_j) <= 1e-5
                        # except:
                        #     import pdb
                        #     pdb.set_trace()

                        # E_k_U_j, e_k_U_j, E_k_L_j, e_k_L_j = E_k_U[j], e_k_U[j], E_k_L[j], e_k_L[j]
                        # H_k_U_j, h_k_U_j, H_k_L_j, h_k_L_j = H_k_U[j], h_k_U[j], H_k_L[j], h_k_L[j]
                        # theta_k_U_j, theta_k_L_j = theta_k_U[j], theta_k_L[j]

                        # if debug:
                        #     # sigma''(y_k) bounds
                        #     sigma_prime_prime_y_k_j_U = torch.sum(H_k_U_j * ((H_k_U_j >= 0) * self.x_ub + (H_k_U_j < 0) * self.x_lb)) + h_k_U_j
                        #     sigma_prime_prime_y_k_j_L = torch.sum(H_k_L_j * ((H_k_L_j >= 0) * self.x_lb + (H_k_L_j < 0) * self.x_ub)) + h_k_L_j

                        #     assert (sigma_prime_prime_y_k_j_L <= sigma_prime_prime_y_k_j_U)

                        #     norm_point_sigma_prime_prime_y_k_j = self.activation_derivative_derivative_relaxation.evaluate(norm_point)[j]

                        #     assert (norm_point_sigma_prime_prime_y_k_j >= sigma_prime_prime_y_k_j_L - 1e-4)
                        #     assert (norm_point_sigma_prime_prime_y_k_j <= sigma_prime_prime_y_k_j_U + 1e-4)

                        #     # W \partial_{x_i} z
                        #     for n in range(weight.shape[1]):
                        #         norm_grad_ubs = torch.sum(C_k_U[n] * ((C_k_U[n] >= 0) * self.x_ub + (C_k_U[n] < 0) * self.x_lb)) + c_k_U[n]
                        #         norm_grad_lbs = torch.sum(C_k_L[n] * ((C_k_L[n] >= 0) * self.x_lb + (C_k_L[n] < 0) * self.x_ub)) + c_k_L[n]

                        #         assert (norm_grad[n] <= norm_grad_ubs + 1e-4)
                        #         assert (norm_grad[n] >= norm_grad_lbs - 1e-4)

                        #     assert (theta_k_L_j <= theta_k_U_j)

                        #     norm_point_W_partial_xi_z_k_1 = (weight @ norm_grad)[j]

                        #     assert (norm_point_W_partial_xi_z_k_1 >= theta_k_L_j - 1e-4)
                        #     assert (norm_point_W_partial_xi_z_k_1 <= theta_k_U_j + 1e-4)

                        # define \nu's for McCormick relaxation of first, element-wise multiplication
                        # rho_k_j, tau_k_j  = rho_k, tau_k
                        # iota_k_L_j, iota_k_U_j = layer_actual_bounds[j, :]
                        # nu_k_U_0_j = rho_k_j * iota_k_U_j + (1 - rho_k_j) * iota_k_L_j
                        # nu_k_U_1_j = rho_k_j * theta_k_L_j + (1 - rho_k_j) * theta_k_U_j
                        # nu_k_U_2_j = -rho_k_j * iota_k_U_j * theta_k_L_j - (1 - rho_k_j) * iota_k_L_j * theta_k_U_j

                        # nu_k_L_0_j = tau_k_j * iota_k_L_j + (1 - tau_k_j) * iota_k_U_j
                        # nu_k_L_1_j = tau_k_j * theta_k_L_j + (1 - tau_k_j) * theta_k_U_j
                        # nu_k_L_2_j = -tau_k_j * iota_k_L_j * theta_k_L_j - (1 - tau_k_j) * iota_k_U_j * theta_k_U_j

                        # assert torch.linalg.norm(nu_k_U_0[j] - nu_k_U_0_j) <= 1e-6
                        # assert torch.linalg.norm(nu_k_U_1[j] - nu_k_U_1_j) <= 1e-6
                        # assert torch.linalg.norm(nu_k_U_2[j] - nu_k_U_2_j) <= 1e-6
                        # assert torch.linalg.norm(nu_k_L_0[j] - nu_k_L_0_j) <= 1e-6
                        # assert torch.linalg.norm(nu_k_L_1[j] - nu_k_L_1_j) <= 1e-6
                        # assert torch.linalg.norm(nu_k_L_2[j] - nu_k_L_2_j) <= 1e-6

                        # nu_k_U_0_j, nu_k_U_1_j, nu_k_U_2_j = nu_k_U_0[j], nu_k_U_1[j], nu_k_U_2[j]
                        # nu_k_L_0_j, nu_k_L_1_j, nu_k_L_2_j = nu_k_L_0[j], nu_k_L_1[j], nu_k_L_2[j]

                        # if debug:
                        #     elem_wise_mul_j = (self.activation_derivative_derivative_relaxation.evaluate(norm_point) * (weight @ norm_grad).flatten())[j]
                            
                        #     elem_wise_coeff_U = nu_k_U_0_j.clamp(min=0) * E_k_U_j + nu_k_U_0_j.clamp(max=0) * E_k_L_j + nu_k_U_1_j.clamp(min=0) * H_k_U_j + nu_k_U_1_j.clamp(max=0) * H_k_L_j
                        #     elem_wise_const_U = nu_k_U_0_j.clamp(min=0) * e_k_U_j + nu_k_U_0_j.clamp(max=0) * e_k_L_j + nu_k_U_1_j.clamp(min=0) * h_k_U_j + nu_k_U_1_j.clamp(max=0) * h_k_L_j + nu_k_U_2_j
                        #     elem_wise_coeff_L = nu_k_L_0_j.clamp(min=0) * E_k_L_j + nu_k_L_0_j.clamp(max=0) * E_k_U_j + nu_k_L_1_j.clamp(min=0) * H_k_L_j + nu_k_L_1_j.clamp(max=0) * H_k_U_j
                        #     elem_wise_const_L = nu_k_L_0_j.clamp(min=0) * e_k_L_j + nu_k_L_0_j.clamp(max=0) * e_k_U_j + nu_k_L_1_j.clamp(min=0) * h_k_L_j + nu_k_L_1_j.clamp(max=0) * h_k_U_j + nu_k_L_2_j

                        #     elem_wise_j_U = torch.sum(elem_wise_coeff_U * ((elem_wise_coeff_U >= 0) * self.x_ub + (elem_wise_coeff_U < 0) * self.x_lb)) + elem_wise_const_U
                        #     elem_wise_j_L = torch.sum(elem_wise_coeff_L * ((elem_wise_coeff_L >= 0) * self.x_lb + (elem_wise_coeff_L < 0) * self.x_ub)) + elem_wise_const_L

                        #     try:
                        #         assert (elem_wise_j_U >= elem_wise_j_L)
                        #         assert (elem_wise_mul_j <= elem_wise_j_U + 1e-4)
                        #         assert (elem_wise_mul_j >= elem_wise_j_L - 1e-4)
                        #     except:
                        #         import pdb
                        #         pdb.set_trace()

                        # define \upsilon's for the linear layer that comes after that
                        # upsilon_k_U_0_j = nu_k_U_0_j * W_pos[j, :] + nu_k_L_0_j * W_neg[j, :]
                        # upsilon_k_U_1_j = nu_k_U_1_j * W_pos[j, :] + nu_k_L_1_j * W_neg[j, :]
                        # upsilon_k_U_2_j = nu_k_U_2_j * W_pos[j, :] + nu_k_L_2_j * W_neg[j, :]

                        # upsilon_k_L_0_j = nu_k_L_0_j * W_pos[j, :] + nu_k_U_0_j * W_neg[j, :]
                        # upsilon_k_L_1_j = nu_k_L_1_j * W_pos[j, :] + nu_k_U_1_j * W_neg[j, :]
                        # upsilon_k_L_2_j = nu_k_L_2_j * W_pos[j, :] + nu_k_U_2_j * W_neg[j, :]

                        # assert torch.linalg.norm(upsilon_k_U_0[j] - upsilon_k_U_0_j) <= 1e-6
                        # assert torch.linalg.norm(upsilon_k_U_1[j] - upsilon_k_U_1_j) <= 1e-6
                        # assert torch.linalg.norm(upsilon_k_U_2[j] - upsilon_k_U_2_j) <= 1e-6
                        # assert torch.linalg.norm(upsilon_k_L_0[j] - upsilon_k_L_0_j) <= 1e-6
                        # assert torch.linalg.norm(upsilon_k_L_1[j] - upsilon_k_L_1_j) <= 1e-6
                        # assert torch.linalg.norm(upsilon_k_L_2[j] - upsilon_k_L_2_j) <= 1e-6

                        # upsilon_k_U_0_j, upsilon_k_U_1_j, upsilon_k_U_2_j = upsilon_k_U_0[j], upsilon_k_U_1[j], upsilon_k_U_2[j]
                        # upsilon_k_L_0_j, upsilon_k_L_1_j, upsilon_k_L_2_j = upsilon_k_L_0[j], upsilon_k_L_1[j], upsilon_k_L_2[j]

                        # sum_alpha_k_7_j = 0
                        # sum_alpha_k_8_j = 0
                        # sum_beta_k_7_j = 0
                        # sum_beta_k_8_j = 0

                        # for n in range(weight.shape[1]):
                            # define M and m from the upsilon values
                            # M_k_U_j_n = upsilon_k_U_0_j[n].clamp(min=0) * E_k_U_j + upsilon_k_U_0_j[n].clamp(max=0) * E_k_L_j +\
                            #     upsilon_k_U_1_j[n].clamp(min=0) * H_k_U_j + upsilon_k_U_1_j[n].clamp(max=0) * H_k_L_j
                            # m_k_U_j_n = upsilon_k_U_0_j[n].clamp(min=0) * e_k_U_j + upsilon_k_U_0_j[n].clamp(max=0) * e_k_L_j +\
                            #     upsilon_k_U_1_j[n].clamp(min=0) * h_k_U_j + upsilon_k_U_1_j[n].clamp(max=0) * h_k_L_j + upsilon_k_U_2_j[n]
                            
                            # M_k_L_j_n = upsilon_k_L_0_j[n].clamp(min=0) * E_k_L_j + upsilon_k_L_0_j[n].clamp(max=0) * E_k_U_j +\
                            #     upsilon_k_L_1_j[n].clamp(min=0) * H_k_L_j + upsilon_k_L_1_j[n].clamp(max=0) * H_k_U_j
                            # m_k_L_j_n = upsilon_k_L_0_j[n].clamp(min=0) * e_k_L_j + upsilon_k_L_0_j[n].clamp(max=0) * e_k_U_j +\
                            #     upsilon_k_L_1_j[n].clamp(min=0) * h_k_L_j + upsilon_k_L_1_j[n].clamp(max=0) * h_k_U_j + upsilon_k_L_2_j[n]

                            # assert torch.linalg.norm(M_k_U[j,n] - M_k_U_j_n) <= 1e-6
                            # assert torch.linalg.norm(m_k_U[j,n] - m_k_U_j_n) <= 1e-6
                            # assert torch.linalg.norm(M_k_L[j,n] - M_k_L_j_n) <= 1e-6
                            # assert torch.linalg.norm(m_k_L[j,n] - m_k_L_j_n) <= 1e-6

                            # M_k_U_j_n, m_k_U_j_n, M_k_L_j_n, m_k_L_j_n = M_k_U[j, n], m_k_U[j, n], M_k_L[j, n], m_k_L[j, n]

                            # obtain the lower and upper bounds on \partial_{x_i z^{(k-1)}} z^k to be used in the McCormick relaxation afterwards
                            # partial_dxiz_k_1_z_k_ub_j_n = torch.sum(M_k_U_j_n * ((M_k_U_j_n >= 0) * self.x_ub + (M_k_U_j_n < 0) * self.x_lb)) + m_k_U_j_n
                            # partial_dxiz_k_1_z_k_lb_j_n = torch.sum(M_k_L_j_n * ((M_k_L_j_n >= 0) * self.x_lb + (M_k_L_j_n < 0) * self.x_ub)) + m_k_L_j_n

                            # assert torch.linalg.norm(partial_dxiz_k_1_z_k_ubs[j,n] - partial_dxiz_k_1_z_k_ub_j_n) <= 1e-6
                            # assert torch.linalg.norm(partial_dxiz_k_1_z_k_lbs[j,n] - partial_dxiz_k_1_z_k_lb_j_n) <= 1e-6

                            # import pdb
                            # pdb.set_trace()

                            # partial_dxiz_k_1_z_k_ubs[j, n] = partial_dxiz_k_1_z_k_ub_j_n
                            # partial_dxiz_k_1_z_k_lbs[j, n] = partial_dxiz_k_1_z_k_lb_j_n

                            # partial_dxiz_k_1_z_k_ub_j_n, partial_dxiz_k_1_z_k_lb_j_n = partial_dxiz_k_1_z_k_ubs[j,n], partial_dxiz_k_1_z_k_lbs[j,n]

                            # define all alphas and betas starting with _0, _1, _2, _3 and _4
                            # eta_k_j_n, gamma_k_j_n, zeta_k_j_n, delta_k_j_n = eta_k, gamma_k, zeta_k, delta_k
                            # alpha_k_0_j_n = eta_k_j_n * partial_dxiz_k_1_z_k_ub_j_n + (1 - eta_k_j_n) * partial_dxiz_k_1_z_k_lb_j_n
                            # alpha_k_1_j_n = eta_k_j_n * partial_dxi_z_k_1_lbs[n] + (1 - eta_k_j_n) * partial_dxi_z_k_1_ubs[n]
                            # alpha_k_2_j_n = gamma_k_j_n * partial_dz_k_1_z_k_ubs[j, n] + (1 - gamma_k_j_n) * partial_dz_k_1_z_k_lbs[j, n]
                            # alpha_k_3_j_n = gamma_k_j_n * u_dxixi_lbs[n] + (1 - gamma_k_j_n) * u_dxixi_ubs[n]
                            # alpha_k_4_j_n = - eta_k_j_n * partial_dxiz_k_1_z_k_ub_j_n * partial_dxi_z_k_1_lbs[n] - (1 - eta_k_j_n) * partial_dxiz_k_1_z_k_lb_j_n * partial_dxi_z_k_1_ubs[n] +\
                            #     - gamma_k_j_n * partial_dz_k_1_z_k_ubs[j, n] * u_dxixi_lbs[n] - (1 - gamma_k_j_n) * partial_dz_k_1_z_k_lbs[j, n] * u_dxixi_ubs[n]

                            # assert torch.linalg.norm(alpha_k_0[j, n] - alpha_k_0_j_n) <= 1e-6
                            # assert torch.linalg.norm(alpha_k_1[n] - alpha_k_1_j_n) <= 1e-6
                            # assert torch.linalg.norm(alpha_k_2[j, n] - alpha_k_2_j_n) <= 1e-6
                            # assert torch.linalg.norm(alpha_k_3[n] - alpha_k_3_j_n) <= 1e-6
                            # assert torch.linalg.norm(alpha_k_4[j, n] - alpha_k_4_j_n) <= 1e-6

                            # beta_k_0_j_n = zeta_k_j_n * partial_dxiz_k_1_z_k_lb_j_n + (1 - zeta_k_j_n) * partial_dxiz_k_1_z_k_ub_j_n
                            # beta_k_1_j_n = zeta_k_j_n * partial_dxi_z_k_1_lbs[n] + (1 - zeta_k_j_n) * partial_dxi_z_k_1_ubs[n]
                            # beta_k_2_j_n = delta_k_j_n * partial_dz_k_1_z_k_lbs[j, n] + (1 - delta_k_j_n) * partial_dz_k_1_z_k_ubs[j, n]
                            # beta_k_3_j_n = delta_k_j_n * u_dxixi_lbs[n] + (1 - delta_k_j_n) * u_dxixi_ubs[n]
                            # beta_k_4_j_n = - zeta_k_j_n * partial_dxiz_k_1_z_k_lb_j_n * partial_dxi_z_k_1_lbs[n] - (1 - zeta_k_j_n) * partial_dxiz_k_1_z_k_ub_j_n * partial_dxi_z_k_1_ubs[n] +\
                            #     - delta_k_j_n * partial_dz_k_1_z_k_lbs[j, n] * u_dxixi_lbs[n] - (1 - delta_k_j_n) * partial_dz_k_1_z_k_ubs[j, n] * u_dxixi_ubs[n]

                            # assert torch.linalg.norm(beta_k_0[j, n] - beta_k_0_j_n) <= 1e-6
                            # assert torch.linalg.norm(beta_k_1[n] - beta_k_1_j_n) <= 1e-6
                            # assert torch.linalg.norm(beta_k_2[j, n] - beta_k_2_j_n) <= 1e-6
                            # assert torch.linalg.norm(beta_k_3[n] - beta_k_3_j_n) <= 1e-6
                            # assert torch.linalg.norm(beta_k_4[j, n] - beta_k_4_j_n) <= 1e-6

                            # alpha_k_0_j_n, alpha_k_1_j_n, alpha_k_2_j_n, alpha_k_3_j_n, alpha_k_4_j_n = alpha_k_0[j, n], alpha_k_1[n], alpha_k_2[j, n], alpha_k_3[n], alpha_k_4[j, n]
                            # beta_k_0_j_n, beta_k_1_j_n, beta_k_2_j_n, beta_k_3_j_n, beta_k_4_j_n = beta_k_0[j, n], beta_k_1[n], beta_k_2[j, n], beta_k_3[n], beta_k_4[j, n]

                            # define the alpha_5, alpha_6, beta_5, and beta_6 from Equation 19
                            # alpha_k_5_j_n = alpha_k_0_j_n.clamp(min=0) * C_k_U[n] + alpha_k_0_j_n.clamp(max=0) * C_k_L[n] +\
                            #     alpha_k_1_j_n.clamp(min=0) * M_k_U_j_n + alpha_k_1_j_n.clamp(max=0) * M_k_L_j_n +\
                            #     alpha_k_3_j_n.clamp(min=0) * D_k_U[j][n] + alpha_k_3_j_n.clamp(max=0) * D_k_L[j][n]
                            # alpha_k_6_j_n = alpha_k_0_j_n.clamp(min=0) * c_k_U[n] + alpha_k_0_j_n.clamp(max=0) * c_k_L[n] +\
                            #     alpha_k_1_j_n.clamp(min=0) * m_k_U_j_n + alpha_k_1_j_n.clamp(max=0) * m_k_L_j_n +\
                            #     alpha_k_3_j_n.clamp(min=0) * d_k_U[j][n] + alpha_k_3_j_n.clamp(max=0) * d_k_L[j][n] +\
                            #     alpha_k_4_j_n

                            # assert torch.linalg.norm(alpha_k_5[j, n] - alpha_k_5_j_n) <= 1e-6
                            # assert torch.linalg.norm(alpha_k_6[j, n] - alpha_k_6_j_n) <= 1e-6

                            # import pdb
                            # pdb.set_trace()

                            # beta_k_5_j_n = beta_k_0_j_n.clamp(min=0) * C_k_L[n] + beta_k_0_j_n.clamp(max=0) * C_k_U[n] +\
                            #     beta_k_1_j_n.clamp(min=0) * M_k_L_j_n + beta_k_1_j_n.clamp(max=0) * M_k_U_j_n +\
                            #     beta_k_3_j_n.clamp(min=0) * D_k_L[j][n] + beta_k_3_j_n.clamp(max=0) * D_k_U[j][n]
                            # beta_k_6_j_n = beta_k_0_j_n.clamp(min=0) * c_k_L[n] + beta_k_0_j_n.clamp(max=0) * c_k_U[n] +\
                            #     beta_k_1_j_n.clamp(min=0) * m_k_L_j_n + beta_k_1_j_n.clamp(max=0) * m_k_U_j_n +\
                            #     beta_k_3_j_n.clamp(min=0) * d_k_L[j][n] + beta_k_3_j_n.clamp(max=0) * d_k_U[j][n] +\
                            #     beta_k_4_j_n

                            # assert torch.linalg.norm(beta_k_5[j, n] - beta_k_5_j_n) <= 1e-6
                            # assert torch.linalg.norm(beta_k_6[j, n] - beta_k_6_j_n) <= 1e-6

                            # alpha_k_5_j_n, alpha_k_6_j_n = alpha_k_5[j, n], alpha_k_6[j, n]
                            # beta_k_5_j_n, beta_k_6_j_n = beta_k_5[j, n], beta_k_6[j, n]

                            # replacing \partial_{x_ix_i} z_n^{(k-1)} in Eq. 19 using the linear bounds from the previous layer
                            # alpha_k_7_j_n = alpha_k_2_j_n.clamp(min=0) * u_dxixi_crown_coeffs_ubs[n] + alpha_k_2_j_n.clamp(max=0) * u_dxixi_crown_coeffs_lbs[n] + alpha_k_5_j_n
                            # alpha_k_8_j_n = alpha_k_2_j_n.clamp(min=0) * u_dxixi_crown_consts_ubs[n] + alpha_k_2_j_n.clamp(max=0) * u_dxixi_crown_consts_lbs[n] + alpha_k_6_j_n

                            # assert torch.linalg.norm(alpha_k_7[j, n] - alpha_k_7_j_n) <= 1e-6
                            # assert torch.linalg.norm(alpha_k_8[j, n] - alpha_k_8_j_n) <= 1e-6

                            # beta_k_7_j_n = beta_k_2_j_n.clamp(min=0) * u_dxixi_crown_coeffs_lbs[n] + beta_k_2_j_n.clamp(max=0) * u_dxixi_crown_coeffs_ubs[n] + beta_k_5_j_n
                            # beta_k_8_j_n = beta_k_2_j_n.clamp(min=0) * u_dxixi_crown_consts_lbs[n] + beta_k_2_j_n.clamp(max=0) * u_dxixi_crown_consts_ubs[n] + beta_k_6_j_n

                            # assert torch.linalg.norm(beta_k_7[j, n] - beta_k_7_j_n) <= 1e-6
                            # assert torch.linalg.norm(beta_k_8[j, n] - beta_k_8_j_n) <= 1e-6

                        #     alpha_k_7_j_n, alpha_k_8_j_n = alpha_k_7[j, n], alpha_k_8[j, n]
                        #     beta_k_7_j_n, beta_k_8_j_n = beta_k_7[j, n], beta_k_8[j, n]

                        #     sum_alpha_k_7_j += alpha_k_7_j_n
                        #     sum_alpha_k_8_j += alpha_k_8_j_n
                        #     sum_beta_k_7_j += beta_k_7_j_n
                        #     sum_beta_k_8_j += beta_k_8_j_n

                        # assert torch.linalg.norm(sum_alpha_k_7[j] - sum_alpha_k_7_j) <= 1e-5
                        # assert torch.linalg.norm(sum_alpha_k_8[j] - sum_alpha_k_8_j) <= 1e-5
                        
                        # sum_alpha_k_7_j, sum_alpha_k_8_j = sum_alpha_k_7[j], sum_alpha_k_8[j]
                        # sum_beta_k_7_j, sum_beta_k_8_j = sum_beta_k_7[j], sum_beta_k_8[j]

                        # partial_dxixi_z_k_j_U = torch.sum(sum_alpha_k_7_j * ((sum_alpha_k_7_j >= 0) * self.x_ub + (sum_alpha_k_7_j < 0) * self.x_lb)) + sum_alpha_k_8_j
                        # partial_dxixi_z_k_j_L = torch.sum(sum_beta_k_7_j * ((sum_beta_k_7_j >= 0) * self.x_lb + (sum_beta_k_7_j < 0) * self.x_ub)) + sum_beta_k_8_j

                        # assert torch.linalg.norm(partial_dxixi_z_k_U[j] - partial_dxixi_z_k_j_U) <= 1e-5
                        # assert torch.linalg.norm(partial_dxixi_z_k_L[j] - partial_dxixi_z_k_j_L) <= 1e-5

                        # import pdb
                        # pdb.set_trace()

                        # new_u_dxixi_lbs.append(partial_dxixi_z_k_j_L)
                        # new_u_dxixi_ubs.append(partial_dxixi_z_k_j_U)
                        # new_u_dxixi_crown_coeffs_lbs.append(sum_beta_k_7_j)
                        # new_u_dxixi_crown_consts_lbs.append(sum_beta_k_8_j)
                        # new_u_dxixi_crown_coeffs_ubs.append(sum_alpha_k_7_j)
                        # new_u_dxixi_crown_consts_ubs.append(sum_alpha_k_8_j)
                elif backprop_mode == BackpropMode.COMPONENT_BACKPROP:
                    layer_output_upper_bounds, layer_output_lower_bounds, CROWN_coefficients = self.backward_component_propagation(
                        self.u_theta.fc_layers[:n_layer+1],
                        layers_activation_second_derivative_output_lines,
                        layers_activation_second_derivative_actual_bounds,
                        self.upper_bounds,
                        self.lower_bounds,
                        self.x_lb,
                        self.x_ub,
                        is_final=False
                    )

                    new_u_dxixi_crown_coeffs_ubs, new_u_dxixi_crown_consts_ubs = CROWN_coefficients[0], CROWN_coefficients[1]
                    new_u_dxixi_crown_coeffs_lbs, new_u_dxixi_crown_consts_lbs = CROWN_coefficients[2], CROWN_coefficients[3]
                else:
                    raise NotImplementedError

                if debug:
                    try:
                        # 1. all lower bounds should be smaller than upper bounds
                        assert all(layer_output_lower_bounds <= layer_output_upper_bounds)

                        # 2. a random point in the interval needs to be inside the bounds
                        d_xi_z_k_1_d_z_k = torch.diag(self.activation_derivative_derivative_relaxation.evaluate(norm_point) * (weight @ norm_grad).flatten()) @ weight
                        d_xi_d_z_k = torch.diag(self.u_dxi_theta.activation_derivative_relaxation.evaluate(norm_point)) @ weight

                        d_xi_xi_d_z_k = d_xi_z_k_1_d_z_k @ norm_grad + d_xi_d_z_k @ norm_d_xi_xi_d_z_k_1

                        # all the partial derivative components must be fine
                        assert all(d_xi_xi_d_z_k.flatten() <= layer_output_upper_bounds + 1e-4)
                        assert all(d_xi_xi_d_z_k.flatten() >= layer_output_lower_bounds - 1e-4)
                        
                        # norm_point next computations
                        norm_grad = d_xi_d_z_k @ norm_grad
                        norm_d_xi_xi_d_z_k_1 = d_xi_xi_d_z_k

                        # 3. if LP bounds are passed, they must be tighter than the partial-CROWN ones
                        if LP_bounds is not None:
                            LP_lower_bounds_layer = LP_bounds[0][n_layer // 2 + 1]
                            LP_upper_bounds_layer = LP_bounds[1][n_layer // 2 + 1]

                            assert all(LP_lower_bounds_layer >= layer_output_lower_bounds - 1e-2)
                            assert all(LP_upper_bounds_layer <= layer_output_upper_bounds + 1e-2)
                    except:
                        import pdb
                        pdb.set_trace()
            else:
                # it's the last layer, there's no activations, just the linear part; exactly the same as in first partial derivative
                if backprop_mode == BackpropMode.BLOCK_BACKPROP:
                    # layer_output_upper_bounds = torch.zeros(weight.shape[0])
                    # layer_output_lower_bounds = torch.zeros(weight.shape[0])

                    # for j in range(weight.shape[0]):
                    #     # upper bound
                    #     output_j_U_coeffs = sum([
                    #         W_pos[j, i] * u_dxixi_crown_coeffs_ubs[i] + W_neg[j, i] * u_dxixi_crown_coeffs_lbs[i]
                    #         for i in range(weight.shape[1])
                    #     ])
                    #     output_j_U_const = sum([
                    #         W_pos[j, i] * u_dxixi_crown_consts_ubs[i] + W_neg[j, i] * u_dxixi_crown_consts_lbs[i]
                    #         for i in range(weight.shape[1])
                    #     ])

                    #     layer_output_upper_bounds[j] = torch.sum(output_j_U_coeffs * ((output_j_U_coeffs >= 0) * self.x_ub + (output_j_U_coeffs < 0) * self.x_lb)) + output_j_U_const

                    #     # lower bound
                    #     output_j_L_coeffs = sum([
                    #         W_pos[j, i] * u_dxixi_crown_coeffs_lbs[i] + W_neg[j, i] * u_dxixi_crown_coeffs_ubs[i]
                    #         for i in range(weight.shape[1])
                    #     ])
                    #     output_j_L_const = sum([
                    #         W_pos[j, i] * u_dxixi_crown_consts_lbs[i] + W_neg[j, i] * u_dxixi_crown_consts_ubs[i]
                    #         for i in range(weight.shape[1])
                    #     ])

                    #     layer_output_lower_bounds[j] = torch.sum(output_j_L_coeffs * ((output_j_L_coeffs >= 0) * self.x_lb + (output_j_L_coeffs < 0) * self.x_ub)) + output_j_L_const

                    new_u_dxixi_crown_coeffs_ubs = W_pos @ u_dxixi_crown_coeffs_ubs + W_neg @ u_dxixi_crown_coeffs_lbs
                    new_u_dxixi_crown_consts_ubs = W_pos @ u_dxixi_crown_consts_ubs + W_neg @ u_dxixi_crown_consts_lbs

                    layer_output_upper_bounds = torch.sum(new_u_dxixi_crown_coeffs_ubs.clamp(min=0) * self.x_ub + new_u_dxixi_crown_coeffs_ubs.clamp(max=0) * self.x_lb, dim=1) + new_u_dxixi_crown_consts_ubs

                    new_u_dxixi_crown_coeffs_lbs = W_pos @ u_dxixi_crown_coeffs_lbs + W_neg @ u_dxixi_crown_coeffs_ubs
                    new_u_dxixi_crown_consts_lbs = W_pos @ u_dxixi_crown_consts_lbs + W_neg @ u_dxixi_crown_consts_ubs

                    layer_output_lower_bounds = torch.sum(new_u_dxixi_crown_coeffs_lbs.clamp(min=0) * self.x_lb + new_u_dxixi_crown_coeffs_lbs.clamp(max=0) * self.x_ub, dim=1) + new_u_dxixi_crown_consts_lbs
                elif backprop_mode == BackpropMode.COMPONENT_BACKPROP:
                    layer_output_upper_bounds, layer_output_lower_bounds, CROWN_coefficients = self.backward_component_propagation(
                        self.u_theta.fc_layers[:n_layer+1],
                        layers_activation_second_derivative_output_lines,
                        layers_activation_second_derivative_actual_bounds,
                        self.upper_bounds,
                        self.lower_bounds,
                        self.x_lb,
                        self.x_ub,
                        is_final=True
                    )
                        
                    new_u_dxixi_crown_coeffs_ubs, new_u_dxixi_crown_consts_ubs = CROWN_coefficients[0], CROWN_coefficients[1]
                    new_u_dxixi_crown_coeffs_lbs, new_u_dxixi_crown_consts_lbs = CROWN_coefficients[2], CROWN_coefficients[3]
                else:
                    raise NotImplementedError
                
                if debug:
                    try:
                        # 1. all lower bounds should be smaller than upper bounds
                        assert all((layer_output_lower_bounds <= layer_output_upper_bounds).flatten())

                        # 2. a random point in the interval needs to be inside the bounds
                        norm_d_xi_xi_d_z_k_1 = weight @ norm_d_xi_xi_d_z_k_1
                        assert all(norm_d_xi_xi_d_z_k_1.flatten() <= layer_output_upper_bounds)
                        assert all(norm_d_xi_xi_d_z_k_1.flatten() >= layer_output_lower_bounds)

                        # 3. the CROWN bounds must be looser than the LP ones
                        if LP_bounds:
                            assert all(layer_output_upper_bounds >= LP_bounds[1][n_layer // 2 + 1] - 1e-3)
                            assert all(layer_output_lower_bounds <= LP_bounds[0][n_layer // 2 + 1] + 1e-2)
                    except:
                        print('--- exception ---')
                        import pdb
                        pdb.set_trace()

            u_dxixi_upper_bounds.append(layer_output_upper_bounds)
            u_dxixi_lower_bounds.append(layer_output_lower_bounds)

            self.u_dxixi_crown_coefficients_ubs.append(new_u_dxixi_crown_coeffs_ubs)
            self.u_dxixi_crown_constants_ubs.append(new_u_dxixi_crown_consts_ubs)
            self.u_dxixi_crown_coefficients_lbs.append(new_u_dxixi_crown_coeffs_lbs)
            self.u_dxixi_crown_constants_lbs.append(new_u_dxixi_crown_consts_lbs)

        self.computed_bounds = True


class CROWNBurgersVerifier():
    def __init__(
            self, model: List[torch.nn.Module],
            activation_relaxation: ActivationRelaxationType, 
            activation_derivative_relaxation: ActivationRelaxationType,
            activation_second_derivative_relaxation: ActivationRelaxationType
    ) -> None:
        self.u_theta = CROWNPINNSolution(
            model,
            activation_relaxation=activation_relaxation
        )

        self.u_dt_theta = CROWNPINNPartialDerivative(
            self.u_theta,
            component_idx=0,
            activation_derivative_relaxation=activation_derivative_relaxation
        )
        self.u_dx_theta = CROWNPINNPartialDerivative(
            self.u_theta,
            component_idx=1,
            activation_derivative_relaxation=activation_derivative_relaxation
        )
        self.u_dxdx_theta = CROWNPINNSecondPartialDerivative(
            self.u_dx_theta,
            component_idx=1,
            activation_derivative_derivative_relaxation=activation_second_derivative_relaxation
        )

        self.viscosity = (0.01/np.pi)

    def compute_residual_bound(
            self,
            domain_bounds: torch.tensor,
            debug: bool = True,
            derivatives_backprop_mode: BackpropMode = BackpropMode.COMPONENT_BACKPROP,
            second_derivatives_backprop_mode: BackpropMode = BackpropMode.COMPONENT_BACKPROP
    ):
        # compute all the intermediate bounds of the components
        u_theta = self.u_theta
        u_theta.domain_bounds = domain_bounds
        u_theta.compute_bounds(debug=debug)

        if debug:
            print("u_theta bounds:", (u_theta.lower_bounds[-1], u_theta.upper_bounds[-1]))

        u_dt_theta = self.u_dt_theta
        u_dt_theta.compute_bounds(debug=debug, backprop_mode=derivatives_backprop_mode)

        if debug:
            print("u_dt_theta bounds:", (u_dt_theta.lower_bounds[-1], u_dt_theta.upper_bounds[-1]))

        u_dx_theta = self.u_dx_theta
        u_dx_theta.compute_bounds(debug=debug, backprop_mode=derivatives_backprop_mode)

        if debug:
            print("u_dx_theta bounds:", (u_dx_theta.lower_bounds[-1], u_dx_theta.upper_bounds[-1]))

        u_dxdx_theta = self.u_dxdx_theta
        u_dxdx_theta.compute_bounds(debug=debug, backprop_mode=second_derivatives_backprop_mode)

        if debug:
            print("u_dxdx_theta bounds:", (u_dxdx_theta.lower_bounds[-1], u_dxdx_theta.upper_bounds[-1]))

        # residual is u_dt_theta + u_theta * u_dx_theta - self.viscosity * u_dxdx_theta

        u_theta_ubs, u_theta_lbs = u_theta.upper_bounds[-1], u_theta.lower_bounds[-1]
        u_theta_coeffs_ubs, u_theta_consts_ubs, u_theta_coeffs_lbs, u_theta_consts_lbs = u_theta.layer_CROWN_coefficients[-1]

        u_dt_theta_ubs, u_dt_theta_lbs = u_dt_theta.upper_bounds[-1], u_dt_theta.lower_bounds[-1]
        u_dt_theta_coeffs_ubs, u_dt_theta_consts_ubs = u_dt_theta.u_dxi_crown_coefficients_ubs[-1], u_dt_theta.u_dxi_crown_constants_ubs[-1]
        u_dt_theta_coeffs_lbs, u_dt_theta_consts_lbs = u_dt_theta.u_dxi_crown_coefficients_lbs[-1], u_dt_theta.u_dxi_crown_constants_lbs[-1]

        u_dx_theta_ubs, u_dx_theta_lbs = u_dx_theta.upper_bounds[-1], u_dx_theta.lower_bounds[-1]
        u_dx_theta_coeffs_ubs, u_dx_theta_consts_ubs = u_dx_theta.u_dxi_crown_coefficients_ubs[-1], u_dx_theta.u_dxi_crown_constants_ubs[-1]
        u_dx_theta_coeffs_lbs, u_dx_theta_consts_lbs = u_dx_theta.u_dxi_crown_coefficients_lbs[-1], u_dx_theta.u_dxi_crown_constants_lbs[-1]

        u_dxdx_theta_ubs, u_dxdx_theta_lbs = u_dxdx_theta.upper_bounds[-1], u_dxdx_theta.lower_bounds[-1]
        u_dxdx_theta_coeffs_ubs, u_dxdx_theta_consts_ubs = u_dxdx_theta.u_dxixi_crown_coefficients_ubs[-1], u_dxdx_theta.u_dxixi_crown_constants_ubs[-1]
        u_dxdx_theta_coeffs_lbs, u_dxdx_theta_consts_lbs = u_dxdx_theta.u_dxixi_crown_coefficients_lbs[-1], u_dxdx_theta.u_dxixi_crown_constants_lbs[-1]

        # McCormick relaxation of u_theta * u_dx_theta
        alpha_U, alpha_L = 0.5, 0.5

        beta_0_U = alpha_U * u_theta_ubs + (1 - alpha_U) * u_theta_lbs
        beta_1_U = alpha_U * u_dx_theta_lbs + (1 - alpha_U) * u_dx_theta_ubs
        beta_2_U = - alpha_U * u_theta_ubs * u_dx_theta_lbs - (1 - alpha_U) * u_theta_lbs * u_dx_theta_ubs

        beta_0_L = alpha_L * u_theta_lbs + (1 - alpha_L) * u_theta_ubs
        beta_1_L = alpha_L * u_dx_theta_lbs + (1 - alpha_L) * u_dx_theta_ubs
        beta_2_L = - alpha_L * u_theta_lbs * u_dx_theta_lbs - (1 - alpha_L) * u_theta_ubs * u_dx_theta_ubs

        mul_coefficient_U = beta_0_U.clamp(min=0) * u_dx_theta_coeffs_ubs + beta_0_U.clamp(max=0) * u_dx_theta_coeffs_lbs +\
            beta_1_U.clamp(min=0) * u_theta_coeffs_ubs + beta_1_U.clamp(max=0) * u_theta_coeffs_lbs
        mul_const_U = beta_0_U.clamp(min=0) * u_dx_theta_consts_ubs + beta_0_U.clamp(max=0) * u_dx_theta_consts_lbs +\
            beta_1_U.clamp(min=0) * u_theta_consts_ubs + beta_1_U.clamp(max=0) * u_theta_consts_lbs+\
            beta_2_U
        
        mul_coefficient_L = beta_0_L.clamp(min=0) * u_dx_theta_coeffs_lbs + beta_0_L.clamp(max=0) * u_dx_theta_coeffs_ubs +\
            beta_1_L.clamp(min=0) * u_theta_coeffs_lbs + beta_1_L.clamp(max=0) * u_theta_coeffs_ubs
        mul_const_L = beta_0_L.clamp(min=0) * u_dx_theta_consts_lbs + beta_0_L.clamp(max=0) * u_dx_theta_consts_ubs +\
            beta_1_L.clamp(min=0) * u_theta_consts_lbs + beta_1_L.clamp(max=0) * u_theta_consts_ubs+\
            beta_2_L

        # sum of the correct coefficients to obtain the final ones
        f_coefficients_U = u_dt_theta_coeffs_ubs + mul_coefficient_U - self.viscosity * u_dxdx_theta_coeffs_lbs
        f_constant_U = u_dt_theta_consts_ubs + mul_const_U - self.viscosity * u_dxdx_theta_consts_lbs

        f_coefficients_L = u_dt_theta_coeffs_lbs + mul_coefficient_L - self.viscosity * u_dxdx_theta_coeffs_ubs
        f_constant_L = u_dt_theta_consts_lbs + mul_const_L - self.viscosity * u_dxdx_theta_consts_ubs

        f_upper_bound = torch.sum(f_coefficients_U.clamp(min=0) * self.u_theta.x_ub + f_coefficients_U.clamp(max=0) * self.u_theta.x_lb, dim=1) + f_constant_U
        f_lower_bound = torch.sum(f_coefficients_L.clamp(min=0) * self.u_theta.x_lb + f_coefficients_L.clamp(max=0) * self.u_theta.x_ub, dim=1) + f_constant_L

        return f_upper_bound, f_lower_bound