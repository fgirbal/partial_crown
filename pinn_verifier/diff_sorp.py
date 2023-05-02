from typing import List, Optional, Tuple

import numpy as np
import torch

from pinn_verifier.activations.relu import ReLURelaxation, ReLUFirstDerivativeRelaxation
from pinn_verifier.crown import BackpropMode, CROWNPINNSolution, CROWNPINNPartialDerivative, CROWNPINNSecondPartialDerivative
from pinn_verifier.activations.activation_relaxations import ActivationRelaxationType
from pinn_verifier.activations.one_over_R_diff_sorp import OneOverRDiffSorpRelaxation


def get_single_line_mccormick_coefficients(x_L: torch.Tensor, x_U: torch.Tensor, y_L: torch.Tensor, y_U: torch.Tensor, alpha_upper_bounds: Optional[float] = 0.5, alpha_lower_bounds: Optional[float] = 0.5):
    coeffs_y_lbs = alpha_lower_bounds * x_L + (1 - alpha_lower_bounds) * x_U
    coeffs_x_lbs = alpha_lower_bounds * y_L + (1 - alpha_lower_bounds) * y_U
    coeffs_const_lbs = alpha_lower_bounds * (- x_L * y_L) + (1 - alpha_lower_bounds) * (- x_U * y_U)

    coeffs_y_ubs = alpha_upper_bounds * x_U + (1 - alpha_upper_bounds) * x_L
    coeffs_x_ubs = alpha_upper_bounds * y_L + (1 - alpha_upper_bounds) * y_U
    coeffs_const_ubs = alpha_upper_bounds * (- x_U * y_L) + (1 - alpha_upper_bounds) * (- x_L * y_U)

    return (coeffs_y_lbs, coeffs_x_lbs, coeffs_const_lbs), (coeffs_y_ubs, coeffs_x_ubs, coeffs_const_ubs)



class CROWNDiffusionSorpionVerifier():
    def __init__(
            self, model: List[torch.nn.Module],
            activation_relaxation: ActivationRelaxationType, 
            activation_derivative_relaxation: ActivationRelaxationType,
            activation_second_derivative_relaxation: ActivationRelaxationType,
            device: torch.device = torch.device('cpu'),
            D: float = 5e-4,
            por: float = 0.29,
            rho_s: float = 2880,
            k_f: float = 3.5e-4,
            n_f: float = 0.874
    ) -> None:
        self.u_theta = CROWNPINNSolution(
            model,
            activation_relaxation=activation_relaxation,
            device=device
        )

        self.u_dt_theta = CROWNPINNPartialDerivative(
            self.u_theta,
            component_idx=1,
            activation_derivative_relaxation=activation_derivative_relaxation
        )
        self.u_dx_theta = CROWNPINNPartialDerivative(
            self.u_theta,
            component_idx=0,
            activation_derivative_relaxation=activation_derivative_relaxation
        )
        self.u_dxdx_theta = CROWNPINNSecondPartialDerivative(
            self.u_dx_theta,
            component_idx=0,
            activation_derivative_derivative_relaxation=activation_second_derivative_relaxation
        )

        self.D = D
        self.por = por
        self.rho_s = rho_s
        self.k_f = k_f
        self.n_f = n_f

        self.relu = torch.nn.ReLU()

        self.one_over_R_relaxation = OneOverRDiffSorpRelaxation(
            ActivationRelaxationType.SINGLE_LINE,
            por=por, rho_s=rho_s, k_f=k_f, n_f=n_f
        )

    def compute_initial_conditions_bounds(
            self,
            domain_bounds: torch.tensor,
            debug: bool = True
    ):
        # THIS NETWORK IS INDEXED [x, t] INSTEAD OF THE NORMAL [t, x]
        # compute bounds of the network itself
        u_theta = self.u_theta
        u_theta.domain_bounds = domain_bounds
        u_theta.compute_bounds(debug=debug)

        # initial condition is ReLU(u_theta) = 0
        return self.relu(u_theta.upper_bounds[-1]), self.relu(u_theta.lower_bounds[-1])

    def compute_boundary_conditions_solution(
            self,
            domain_bounds: torch.tensor,
            debug: bool = True
    ):
        # THIS NETWORK IS INDEXED [x, t] INSTEAD OF THE NORMAL [t, x]
        # compute bounds of the network itself
        u_theta = self.u_theta
        u_theta.domain_bounds = domain_bounds
        u_theta.compute_bounds(debug=debug)

        return self.relu(u_theta.upper_bounds[-1]) - 1, self.relu(u_theta.lower_bounds[-1]) - 1

    def compute_boundary_conditions_partial(
            self,
            domain_bounds: torch.tensor,
            debug: bool = True,
            derivatives_backprop_mode: BackpropMode = BackpropMode.COMPONENT_BACKPROP,
    ):
        # THIS NETWORK IS INDEXED [x, t] INSTEAD OF THE NORMAL [t, x]
        # compute bounds of the network itself
        u_theta = self.u_theta
        u_theta.domain_bounds = domain_bounds
        u_theta.compute_bounds(debug=debug)

        u_dx_theta = self.u_dx_theta
        u_dx_theta.compute_bounds(debug=debug, backprop_mode=derivatives_backprop_mode)

        if debug:
            print("u_dx_theta bounds:", (u_dx_theta.lower_bounds[-1], u_dx_theta.upper_bounds[-1]))

        # get lower and upper bound coefficients for ReLU(u_theta) by doing linear relaxation of ReLU
        u_theta_lbs = u_theta.lower_bounds[-1]
        u_theta_ubs = u_theta.upper_bounds[-1]
        u_theta_coeffs_ubs, u_theta_consts_ubs, u_theta_coeffs_lbs, u_theta_consts_lbs = u_theta.layer_CROWN_coefficients[-1]

        relu_relaxation = ReLURelaxation(
            ActivationRelaxationType.SINGLE_LINE
        )
        relaxation_lines = torch.Tensor([relu_relaxation.get_bounds(lb, ub) for lb, ub in zip(u_theta_lbs, u_theta_ubs)]).reshape(-1, 4)
        alpha_L = relaxation_lines[:, 0]
        beta_L = relaxation_lines[:, 1]
        alpha_U = relaxation_lines[:, 2]
        beta_U = relaxation_lines[:, 3]

        relu_u_theta_coeffs_ubs = alpha_U.clamp(min=0).unsqueeze(1).unsqueeze(2) * u_theta_coeffs_ubs + alpha_U.clamp(max=0) .unsqueeze(1).unsqueeze(2)* u_theta_coeffs_lbs
        relu_u_theta_consts_ubs = alpha_U.clamp(min=0).unsqueeze(1) * u_theta_consts_ubs + alpha_U.clamp(max=0).unsqueeze(1) * u_theta_consts_lbs + beta_U.unsqueeze(1)

        relu_u_theta_coeffs_lbs = alpha_L.clamp(min=0).unsqueeze(1).unsqueeze(2) * u_theta_coeffs_lbs + alpha_L.clamp(max=0).unsqueeze(1).unsqueeze(2) * u_theta_coeffs_ubs
        relu_u_theta_consts_lbs = alpha_L.clamp(min=0).unsqueeze(1) * u_theta_consts_lbs + alpha_L.clamp(max=0).unsqueeze(1) * u_theta_consts_ubs + beta_L.unsqueeze(1)

        # get lower and upper bound coefficients for \partial_x ReLU(u_theta) = ReLU'(u_theta) * \partial_x u_theta
        # start by obtaining the coefficients for ReLU'(u_theta)
        relu_derivative_relaxation = ReLUFirstDerivativeRelaxation(
            ActivationRelaxationType.SINGLE_LINE
        )
        derivative_relaxation_lines = torch.Tensor([relu_derivative_relaxation.get_bounds(lb, ub) for lb, ub in zip(u_theta_lbs, u_theta_ubs)]).reshape(-1, 4)
        alpha_L = derivative_relaxation_lines[:, 0]
        beta_L = derivative_relaxation_lines[:, 1]
        alpha_U = derivative_relaxation_lines[:, 2]
        beta_U = derivative_relaxation_lines[:, 3]

        derivative_relu_u_theta_coeffs_ubs = alpha_U.clamp(min=0) * u_theta_coeffs_ubs + alpha_U.clamp(max=0) * u_theta_coeffs_lbs
        derivative_relu_u_theta_consts_ubs = alpha_U.clamp(min=0).unsqueeze(1) * u_theta_consts_ubs + alpha_U.clamp(max=0).unsqueeze(1) * u_theta_consts_lbs + beta_U.unsqueeze(1)

        derivative_relu_u_theta_coeffs_lbs = alpha_L.clamp(min=0) * u_theta_coeffs_lbs + alpha_L.clamp(max=0) * u_theta_coeffs_ubs
        derivative_relu_u_theta_consts_lbs = alpha_L.clamp(min=0).unsqueeze(1) * u_theta_consts_lbs + alpha_L.clamp(max=0).unsqueeze(1) * u_theta_consts_ubs + beta_L.unsqueeze(1)

        derivative_relu_u_theta_ubs = torch.sum(derivative_relu_u_theta_coeffs_ubs.clamp(min=0) * self.u_theta.x_ub.unsqueeze(1) + derivative_relu_u_theta_coeffs_ubs.clamp(max=0) * self.u_theta.x_lb.unsqueeze(1), dim=2) + derivative_relu_u_theta_consts_ubs
        derivative_relu_u_theta_lbs = torch.sum(derivative_relu_u_theta_coeffs_lbs.clamp(min=0) * self.u_theta.x_lb.unsqueeze(1) + derivative_relu_u_theta_coeffs_lbs.clamp(max=0) * self.u_theta.x_ub.unsqueeze(1), dim=2) + derivative_relu_u_theta_consts_lbs

        # McCormick relaxation of derivative_relu_u_theta * \partial_x u_theta
        u_dx_theta_lbs = u_dx_theta.lower_bounds[-1]
        u_dx_theta_ubs = u_dx_theta.upper_bounds[-1]
        u_dx_theta_coeffs_ubs, u_dx_theta_consts_ubs = u_dx_theta.u_dxi_crown_coefficients_ubs[-1], u_dx_theta.u_dxi_crown_constants_ubs[-1]
        u_dx_theta_coeffs_lbs, u_dx_theta_consts_lbs = u_dx_theta.u_dxi_crown_coefficients_lbs[-1], u_dx_theta.u_dxi_crown_constants_lbs[-1]

        alpha_U, alpha_L = 0.5, 0.5

        beta_0_U = alpha_U * derivative_relu_u_theta_ubs + (1 - alpha_U) * derivative_relu_u_theta_lbs
        beta_1_U = alpha_U * u_dx_theta_lbs + (1 - alpha_U) * u_dx_theta_ubs
        beta_2_U = - alpha_U * derivative_relu_u_theta_ubs * u_dx_theta_lbs - (1 - alpha_U) * derivative_relu_u_theta_lbs * u_dx_theta_ubs

        beta_0_L = alpha_L * derivative_relu_u_theta_lbs + (1 - alpha_L) * derivative_relu_u_theta_ubs
        beta_1_L = alpha_L * u_dx_theta_lbs + (1 - alpha_L) * u_dx_theta_ubs
        beta_2_L = - alpha_L * derivative_relu_u_theta_lbs * u_dx_theta_lbs - (1 - alpha_L) * derivative_relu_u_theta_ubs * u_dx_theta_ubs

        partial_relu_u_theta_coeffs_ubs = beta_0_U.unsqueeze(1).clamp(min=0) * u_dx_theta_coeffs_ubs + beta_0_U.unsqueeze(1).clamp(max=0) * u_dx_theta_coeffs_lbs +\
            beta_1_U.unsqueeze(1).clamp(min=0) * derivative_relu_u_theta_coeffs_ubs + beta_1_U.unsqueeze(1).clamp(max=0) * derivative_relu_u_theta_coeffs_lbs
        partial_relu_u_theta_consts_ubs = beta_0_U.clamp(min=0) * u_dx_theta_consts_ubs + beta_0_U.clamp(max=0) * u_dx_theta_consts_lbs +\
            beta_1_U.clamp(min=0) * derivative_relu_u_theta_consts_ubs + beta_1_U.clamp(max=0) * derivative_relu_u_theta_consts_lbs+\
            beta_2_U
        
        partial_relu_u_theta_coeffs_lbs = beta_0_L.unsqueeze(1).clamp(min=0) * u_dx_theta_coeffs_lbs + beta_0_L.unsqueeze(1).clamp(max=0) * u_dx_theta_coeffs_ubs +\
            beta_1_L.unsqueeze(1).clamp(min=0) * derivative_relu_u_theta_coeffs_lbs + beta_1_L.unsqueeze(1).clamp(max=0) * derivative_relu_u_theta_coeffs_ubs
        partial_relu_u_theta_consts_lbs = beta_0_L.clamp(min=0) * u_dx_theta_consts_lbs + beta_0_L.clamp(max=0) * u_dx_theta_consts_ubs +\
            beta_1_L.clamp(min=0) * derivative_relu_u_theta_consts_lbs + beta_1_L.clamp(max=0) * derivative_relu_u_theta_consts_ubs+\
            beta_2_L
        
        # sum the coefficients to obtain the boundary condition relu_u_theta - D * partial_relu_u_theta
        f_coefficients_U = relu_u_theta_coeffs_ubs - self.D * partial_relu_u_theta_coeffs_lbs
        f_constant_U = relu_u_theta_consts_ubs - self.D * partial_relu_u_theta_consts_lbs

        f_coefficients_L = relu_u_theta_coeffs_lbs - self.D * partial_relu_u_theta_coeffs_ubs
        f_constant_L = relu_u_theta_consts_lbs - self.D * partial_relu_u_theta_consts_ubs

        f_upper_bound = torch.sum(f_coefficients_U.clamp(min=0) * self.u_theta.x_ub.unsqueeze(1) + f_coefficients_U.clamp(max=0) * self.u_theta.x_lb.unsqueeze(1), dim=2) + f_constant_U
        f_lower_bound = torch.sum(f_coefficients_L.clamp(min=0) * self.u_theta.x_lb.unsqueeze(1) + f_coefficients_L.clamp(max=0) * self.u_theta.x_ub.unsqueeze(1), dim=2) + f_constant_L

        return f_upper_bound, f_lower_bound

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
    
        # residual is \partial_t ReLu(u) - D / retardation_factor * \partial_xx ReLu(u)
        u_theta_lbs = u_theta.lower_bounds[-1]
        u_theta_ubs = u_theta.upper_bounds[-1]
        u_theta_coeffs_ubs, u_theta_consts_ubs, u_theta_coeffs_lbs, u_theta_consts_lbs = u_theta.layer_CROWN_coefficients[-1]

        # get lower and upper bound coefficients for \partial_t ReLU(u_theta) = ReLU'(u_theta) * \partial_t u_theta
        # start by obtaining the coefficients for ReLU'(u_theta)
        relu_derivative_relaxation = ReLUFirstDerivativeRelaxation(
            ActivationRelaxationType.SINGLE_LINE
        )
        derivative_relaxation_lines = torch.Tensor([relu_derivative_relaxation.get_bounds(lb, ub) for lb, ub in zip(u_theta_lbs, u_theta_ubs)]).reshape(-1, 4)
        alpha_L = derivative_relaxation_lines[:, 0]
        beta_L = derivative_relaxation_lines[:, 1]
        alpha_U = derivative_relaxation_lines[:, 2]
        beta_U = derivative_relaxation_lines[:, 3]

        derivative_relu_u_theta_coeffs_ubs = alpha_U.clamp(min=0).unsqueeze(1).unsqueeze(2) * u_theta_coeffs_ubs + alpha_U.clamp(max=0).unsqueeze(1).unsqueeze(2) * u_theta_coeffs_lbs
        derivative_relu_u_theta_consts_ubs = alpha_U.clamp(min=0).unsqueeze(1) * u_theta_consts_ubs + alpha_U.clamp(max=0).unsqueeze(1) * u_theta_consts_lbs + beta_U.unsqueeze(1)

        derivative_relu_u_theta_coeffs_lbs = alpha_L.clamp(min=0).unsqueeze(1).unsqueeze(2) * u_theta_coeffs_lbs + alpha_L.clamp(max=0).unsqueeze(1).unsqueeze(2) * u_theta_coeffs_ubs
        derivative_relu_u_theta_consts_lbs = alpha_L.clamp(min=0).unsqueeze(1) * u_theta_consts_lbs + alpha_L.clamp(max=0).unsqueeze(1) * u_theta_consts_ubs + beta_L.unsqueeze(1)

        derivative_relu_u_theta_ubs = torch.sum(derivative_relu_u_theta_coeffs_ubs.clamp(min=0) * self.u_theta.x_ub.unsqueeze(1) + derivative_relu_u_theta_coeffs_ubs.clamp(max=0) * self.u_theta.x_lb.unsqueeze(1), dim=2) + derivative_relu_u_theta_consts_ubs
        derivative_relu_u_theta_lbs = torch.sum(derivative_relu_u_theta_coeffs_lbs.clamp(min=0) * self.u_theta.x_lb.unsqueeze(1) + derivative_relu_u_theta_coeffs_lbs.clamp(max=0) * self.u_theta.x_ub.unsqueeze(1), dim=2) + derivative_relu_u_theta_consts_lbs

        # McCormick relaxation of derivative_relu_u_theta * \partial_x u_theta
        u_dt_theta_lbs = u_dt_theta.lower_bounds[-1]
        u_dt_theta_ubs = u_dt_theta.upper_bounds[-1]
        u_dt_theta_coeffs_ubs, u_dt_theta_consts_ubs = u_dt_theta.u_dxi_crown_coefficients_ubs[-1], u_dt_theta.u_dxi_crown_constants_ubs[-1]
        u_dt_theta_coeffs_lbs, u_dt_theta_consts_lbs = u_dt_theta.u_dxi_crown_coefficients_lbs[-1], u_dt_theta.u_dxi_crown_constants_lbs[-1]

        alpha_U, alpha_L = 0.5, 0.5

        beta_0_U = alpha_U * derivative_relu_u_theta_ubs + (1 - alpha_U) * derivative_relu_u_theta_lbs
        beta_1_U = alpha_U * u_dt_theta_lbs + (1 - alpha_U) * u_dt_theta_ubs
        beta_2_U = - alpha_U * derivative_relu_u_theta_ubs * u_dt_theta_lbs - (1 - alpha_U) * derivative_relu_u_theta_lbs * u_dt_theta_ubs

        beta_0_L = alpha_L * derivative_relu_u_theta_lbs + (1 - alpha_L) * derivative_relu_u_theta_ubs
        beta_1_L = alpha_L * u_dt_theta_lbs + (1 - alpha_L) * u_dt_theta_ubs
        beta_2_L = - alpha_L * derivative_relu_u_theta_lbs * u_dt_theta_lbs - (1 - alpha_L) * derivative_relu_u_theta_ubs * u_dt_theta_ubs

        partial_t_relu_u_theta_coeffs_ubs = beta_0_U.unsqueeze(1).clamp(min=0) * u_dt_theta_coeffs_ubs + beta_0_U.unsqueeze(1).clamp(max=0) * u_dt_theta_coeffs_lbs +\
            beta_1_U.unsqueeze(1).clamp(min=0) * derivative_relu_u_theta_coeffs_ubs + beta_1_U.unsqueeze(1).clamp(max=0) * derivative_relu_u_theta_coeffs_lbs
        partial_t_relu_u_theta_consts_ubs = beta_0_U.clamp(min=0) * u_dt_theta_consts_ubs + beta_0_U.clamp(max=0) * u_dt_theta_consts_lbs +\
            beta_1_U.clamp(min=0) * derivative_relu_u_theta_consts_ubs + beta_1_U.clamp(max=0) * derivative_relu_u_theta_consts_lbs+\
            beta_2_U
        
        partial_t_relu_u_theta_coeffs_lbs = beta_0_L.unsqueeze(1).clamp(min=0) * u_dt_theta_coeffs_lbs + beta_0_L.unsqueeze(1).clamp(max=0) * u_dt_theta_coeffs_ubs +\
            beta_1_L.unsqueeze(1).clamp(min=0) * derivative_relu_u_theta_coeffs_lbs + beta_1_L.unsqueeze(1).clamp(max=0) * derivative_relu_u_theta_coeffs_ubs
        partial_t_relu_u_theta_consts_lbs = beta_0_L.clamp(min=0) * u_dt_theta_consts_lbs + beta_0_L.clamp(max=0) * u_dt_theta_consts_ubs +\
            beta_1_L.clamp(min=0) * derivative_relu_u_theta_consts_lbs + beta_1_L.clamp(max=0) * derivative_relu_u_theta_consts_ubs+\
            beta_2_L

        # get lower and upper bound coefficients for ReLU(u_theta) by doing linear relaxation of ReLU
        relu_relaxation = ReLURelaxation(
            ActivationRelaxationType.SINGLE_LINE
        )
        relaxation_lines = torch.Tensor([relu_relaxation.get_bounds(lb, ub) for lb, ub in zip(u_theta_lbs, u_theta_ubs)]).reshape(-1, 4)
        alpha_L = relaxation_lines[:, 0]
        beta_L = relaxation_lines[:, 1]
        alpha_U = relaxation_lines[:, 2]
        beta_U = relaxation_lines[:, 3]

        relu_u_theta_coeffs_ubs = alpha_U.clamp(min=0).unsqueeze(1).unsqueeze(2) * u_theta_coeffs_ubs + alpha_U.clamp(max=0).unsqueeze(1).unsqueeze(2) * u_theta_coeffs_lbs
        relu_u_theta_consts_ubs = alpha_U.clamp(min=0).unsqueeze(1) * u_theta_consts_ubs + alpha_U.clamp(max=0).unsqueeze(1) * u_theta_consts_lbs + beta_U.unsqueeze(1)

        relu_u_theta_coeffs_lbs = alpha_L.clamp(min=0).unsqueeze(1).unsqueeze(2) * u_theta_coeffs_lbs + alpha_L.clamp(max=0).unsqueeze(1).unsqueeze(2) * u_theta_coeffs_ubs
        relu_u_theta_consts_lbs = alpha_L.clamp(min=0).unsqueeze(1) * u_theta_consts_lbs + alpha_L.clamp(max=0).unsqueeze(1) * u_theta_consts_ubs + beta_L.unsqueeze(1)

        relu_u_theta_ubs = self.relu(
            torch.sum(relu_u_theta_coeffs_ubs.clamp(min=0) * self.u_theta.x_ub.unsqueeze(1) + relu_u_theta_coeffs_ubs.clamp(max=0) * self.u_theta.x_lb.unsqueeze(1), dim=2) + relu_u_theta_consts_ubs
        )
        relu_u_theta_lbs = self.relu(
            torch.sum(relu_u_theta_coeffs_lbs.clamp(min=0) * self.u_theta.x_lb.unsqueeze(1) + relu_u_theta_coeffs_lbs.clamp(max=0) * self.u_theta.x_ub.unsqueeze(1), dim=2) + relu_u_theta_consts_lbs
        )

        # get lower and upper bound coefficients for 1/ retardation_factor using relu_u_theta coeffs, consts and lbs/ubs
        one_over_R_lines = torch.Tensor([self.one_over_R_relaxation.get_bounds(lb, ub) for lb, ub in zip(relu_u_theta_lbs, relu_u_theta_ubs)]).reshape(-1, 4)
        alpha_L = one_over_R_lines[:, 0]
        beta_L = one_over_R_lines[:, 1]
        alpha_U = one_over_R_lines[:, 2]
        beta_U = one_over_R_lines[:, 3]

        one_over_R_coeffs_ubs = alpha_U.clamp(min=0).unsqueeze(1).unsqueeze(2) * relu_u_theta_coeffs_ubs + alpha_U.clamp(max=0).unsqueeze(1).unsqueeze(2) * relu_u_theta_coeffs_lbs
        one_over_R_consts_ubs = alpha_U.clamp(min=0).unsqueeze(1) * relu_u_theta_consts_ubs + alpha_U.clamp(max=0).unsqueeze(1) * relu_u_theta_consts_lbs + beta_U.unsqueeze(1)

        one_over_R_coeffs_lbs = alpha_L.clamp(min=0).unsqueeze(1).unsqueeze(2) * relu_u_theta_coeffs_lbs + alpha_L.clamp(max=0).unsqueeze(1).unsqueeze(2) * relu_u_theta_coeffs_ubs
        one_over_R_consts_lbs = alpha_L.clamp(min=0).unsqueeze(1) * relu_u_theta_consts_lbs + alpha_L.clamp(max=0).unsqueeze(1) * relu_u_theta_consts_ubs + beta_L.unsqueeze(1)

        one_over_R_ubs = torch.sum(one_over_R_coeffs_ubs.clamp(min=0) * self.u_theta.x_ub.unsqueeze(1) + one_over_R_coeffs_ubs.clamp(max=0) * self.u_theta.x_lb.unsqueeze(1), dim=2) + one_over_R_consts_ubs
        one_over_R_lbs = torch.sum(one_over_R_coeffs_lbs.clamp(min=0) * self.u_theta.x_lb.unsqueeze(1) + one_over_R_coeffs_lbs.clamp(max=0) * self.u_theta.x_ub.unsqueeze(1), dim=2) + one_over_R_consts_lbs

        # get lower and upper bound coefficients for:
        # \partial_xx ReLU(u_theta) =
        # \partial_x \partial_x ReLU(u_theta) = 
        # \partial_x (ReLU'(u_theta) * \partial_x u_theta) =
        # \partial_x (ReLU'(u_theta)) * \partial_x u_theta + ReLU'(u_theta) * \partial_xx u_theta =
        # ReLU'(u_theta) * \partial_xx u_theta (since ReLU''(u_theta) = 0)

        # McCormick relaxation of derivative_relu_u_theta * \partial_xx u_theta
        u_dxdx_theta_lbs = u_dxdx_theta.lower_bounds[-1]
        u_dxdx_theta_ubs = u_dxdx_theta.upper_bounds[-1]
        u_dxdx_theta_coeffs_ubs, u_dxdx_theta_consts_ubs = u_dxdx_theta.u_dxixi_crown_coefficients_ubs[-1], u_dxdx_theta.u_dxixi_crown_constants_ubs[-1]
        u_dxdx_theta_coeffs_lbs, u_dxdx_theta_consts_lbs = u_dxdx_theta.u_dxixi_crown_coefficients_lbs[-1], u_dxdx_theta.u_dxixi_crown_constants_lbs[-1]

        alpha_U, alpha_L = 0.5, 0.5

        beta_0_U = alpha_U * derivative_relu_u_theta_ubs + (1 - alpha_U) * derivative_relu_u_theta_lbs
        beta_1_U = alpha_U * u_dxdx_theta_lbs + (1 - alpha_U) * u_dxdx_theta_ubs
        beta_2_U = - alpha_U * derivative_relu_u_theta_ubs * u_dxdx_theta_lbs - (1 - alpha_U) * derivative_relu_u_theta_lbs * u_dxdx_theta_ubs

        beta_0_L = alpha_L * derivative_relu_u_theta_lbs + (1 - alpha_L) * derivative_relu_u_theta_ubs
        beta_1_L = alpha_L * u_dxdx_theta_lbs + (1 - alpha_L) * u_dxdx_theta_ubs
        beta_2_L = - alpha_L * derivative_relu_u_theta_lbs * u_dxdx_theta_lbs - (1 - alpha_L) * derivative_relu_u_theta_ubs * u_dxdx_theta_ubs

        partial_xx_relu_u_theta_coeffs_ubs = beta_0_U.unsqueeze(1).clamp(min=0) * u_dxdx_theta_coeffs_ubs + beta_0_U.unsqueeze(1).clamp(max=0) * u_dxdx_theta_coeffs_lbs +\
            beta_1_U.unsqueeze(1).clamp(min=0) * derivative_relu_u_theta_coeffs_ubs + beta_1_U.unsqueeze(1).clamp(max=0) * derivative_relu_u_theta_coeffs_lbs
        partial_xx_relu_u_theta_consts_ubs = beta_0_U.clamp(min=0) * u_dxdx_theta_consts_ubs + beta_0_U.clamp(max=0) * u_dxdx_theta_consts_lbs +\
            beta_1_U.clamp(min=0) * derivative_relu_u_theta_consts_ubs + beta_1_U.clamp(max=0) * derivative_relu_u_theta_consts_lbs+\
            beta_2_U
        
        partial_xx_relu_u_theta_coeffs_lbs = beta_0_L.unsqueeze(1).clamp(min=0) * u_dxdx_theta_coeffs_lbs + beta_0_L.unsqueeze(1).clamp(max=0) * u_dxdx_theta_coeffs_ubs +\
            beta_1_L.unsqueeze(1).clamp(min=0) * derivative_relu_u_theta_coeffs_lbs + beta_1_L.unsqueeze(1).clamp(max=0) * derivative_relu_u_theta_coeffs_ubs
        partial_xx_relu_u_theta_consts_lbs = beta_0_L.clamp(min=0) * u_dxdx_theta_consts_lbs + beta_0_L.clamp(max=0) * u_dxdx_theta_consts_ubs +\
            beta_1_L.clamp(min=0) * derivative_relu_u_theta_consts_lbs + beta_1_L.clamp(max=0) * derivative_relu_u_theta_consts_ubs+\
            beta_2_L
        
        partial_xx_relu_u_theta_ubs = torch.sum(partial_xx_relu_u_theta_coeffs_ubs.clamp(min=0) * self.u_theta.x_ub.unsqueeze(1) + partial_xx_relu_u_theta_coeffs_ubs.clamp(max=0) * self.u_theta.x_lb.unsqueeze(1), dim=2) + partial_xx_relu_u_theta_consts_ubs
        partial_xx_relu_u_theta_lbs = torch.sum(partial_xx_relu_u_theta_coeffs_lbs.clamp(min=0) * self.u_theta.x_lb.unsqueeze(1) + partial_xx_relu_u_theta_coeffs_lbs.clamp(max=0) * self.u_theta.x_ub.unsqueeze(1), dim=2) + partial_xx_relu_u_theta_consts_lbs

        # McCormick relaxation of one_over_R * partial_xx_relu_u_theta
        alpha_U, alpha_L = 0.5, 0.5

        beta_0_U = alpha_U * partial_xx_relu_u_theta_ubs + (1 - alpha_U) * partial_xx_relu_u_theta_lbs
        beta_1_U = alpha_U * one_over_R_lbs + (1 - alpha_U) * one_over_R_ubs
        beta_2_U = - alpha_U * partial_xx_relu_u_theta_ubs * one_over_R_lbs - (1 - alpha_U) * partial_xx_relu_u_theta_lbs * one_over_R_ubs

        beta_0_L = alpha_L * partial_xx_relu_u_theta_lbs + (1 - alpha_L) * partial_xx_relu_u_theta_ubs
        beta_1_L = alpha_L * one_over_R_lbs + (1 - alpha_L) * one_over_R_ubs
        beta_2_L = - alpha_L * partial_xx_relu_u_theta_lbs * one_over_R_lbs - (1 - alpha_L) * partial_xx_relu_u_theta_ubs * one_over_R_ubs

        mul_one_over_partial_xx_coeffs_ubs = beta_0_U.unsqueeze(1).clamp(min=0) * one_over_R_coeffs_ubs + beta_0_U.unsqueeze(1).clamp(max=0) * one_over_R_coeffs_lbs +\
            beta_1_U.unsqueeze(1).clamp(min=0) * partial_xx_relu_u_theta_coeffs_ubs + beta_1_U.unsqueeze(1).clamp(max=0) * partial_xx_relu_u_theta_coeffs_lbs
        mul_one_over_partial_xx_consts_ubs = beta_0_U.clamp(min=0) * one_over_R_consts_ubs + beta_0_U.clamp(max=0) * one_over_R_consts_lbs +\
            beta_1_U.clamp(min=0) * partial_xx_relu_u_theta_consts_ubs + beta_1_U.clamp(max=0) * partial_xx_relu_u_theta_consts_lbs+\
            beta_2_U
        
        mul_one_over_partial_xx_coeffs_lbs = beta_0_L.unsqueeze(1).clamp(min=0) * one_over_R_coeffs_lbs + beta_0_L.unsqueeze(1).clamp(max=0) * one_over_R_coeffs_ubs +\
            beta_1_L.unsqueeze(1).clamp(min=0) * partial_xx_relu_u_theta_coeffs_lbs + beta_1_L.unsqueeze(1).clamp(max=0) * partial_xx_relu_u_theta_coeffs_ubs
        mul_one_over_partial_xx_consts_lbs = beta_0_L.clamp(min=0) * one_over_R_consts_lbs + beta_0_L.clamp(max=0) * one_over_R_consts_ubs +\
            beta_1_L.clamp(min=0) * partial_xx_relu_u_theta_consts_lbs + beta_1_L.clamp(max=0) * partial_xx_relu_u_theta_consts_ubs+\
            beta_2_L

        # sum of the correct coefficients/constants to obtain the final ones:
        # partial_t_relu_u_theta - D * mul_one_over_partial_xx

        f_coefficients_U = partial_t_relu_u_theta_coeffs_ubs - self.D * mul_one_over_partial_xx_coeffs_lbs
        f_constant_U = partial_t_relu_u_theta_consts_ubs - self.D * mul_one_over_partial_xx_consts_lbs

        f_coefficients_L = partial_t_relu_u_theta_coeffs_lbs - self.D * mul_one_over_partial_xx_coeffs_ubs
        f_constant_L = partial_t_relu_u_theta_consts_lbs - self.D * mul_one_over_partial_xx_consts_ubs

        f_upper_bound = torch.sum(f_coefficients_U.clamp(min=0) * self.u_theta.x_ub.unsqueeze(1) + f_coefficients_U.clamp(max=0) * self.u_theta.x_lb.unsqueeze(1), dim=2) + f_constant_U
        f_lower_bound = torch.sum(f_coefficients_L.clamp(min=0) * self.u_theta.x_lb.unsqueeze(1) + f_coefficients_L.clamp(max=0) * self.u_theta.x_ub.unsqueeze(1), dim=2) + f_constant_L

        return f_upper_bound, f_lower_bound
