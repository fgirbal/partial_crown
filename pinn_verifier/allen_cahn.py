from typing import List, Optional

import numpy as np
import torch

from pinn_verifier.crown import BackpropMode, CROWNPINNSolution, CROWNPINNPartialDerivative, CROWNPINNSecondPartialDerivative
from pinn_verifier.activations.activation_relaxations import ActivationRelaxationType
from pinn_verifier.activations.cos import CosRelaxation


def get_single_line_mccormick_coefficients(x_L: torch.Tensor, x_U: torch.Tensor, y_L: torch.Tensor, y_U: torch.Tensor, alpha_upper_bounds: Optional[float] = 0.5, alpha_lower_bounds: Optional[float] = 0.5):
    coeffs_y_lbs = alpha_lower_bounds * x_L + (1 - alpha_lower_bounds) * x_U
    coeffs_x_lbs = alpha_lower_bounds * y_L + (1 - alpha_lower_bounds) * y_U
    coeffs_const_lbs = alpha_lower_bounds * (- x_L * y_L) + (1 - alpha_lower_bounds) * (- x_U * y_U)

    coeffs_y_ubs = alpha_upper_bounds * x_U + (1 - alpha_upper_bounds) * x_L
    coeffs_x_ubs = alpha_upper_bounds * y_L + (1 - alpha_upper_bounds) * y_U
    coeffs_const_ubs = alpha_upper_bounds * (- x_U * y_L) + (1 - alpha_upper_bounds) * (- x_L * y_U)

    return (coeffs_y_lbs, coeffs_x_lbs, coeffs_const_lbs), (coeffs_y_ubs, coeffs_x_ubs, coeffs_const_ubs)



class CROWNAllenCahnVerifier():
    def __init__(
            self, model: List[torch.nn.Module],
            activation_relaxation: ActivationRelaxationType, 
            activation_derivative_relaxation: ActivationRelaxationType,
            activation_second_derivative_relaxation: ActivationRelaxationType,
            device: torch.device = torch.device('cpu')
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

        self.rho = 5
        self.nu = 0.0001

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

        # initial condition is u_theta - x**2 * cos(pi*x)

        # linear coefficients for cos(pi*x) don't depend on t (index 1), only on x (index 0)
        cos_relaxation = CosRelaxation(
            ActivationRelaxationType.SINGLE_LINE
        )
        relaxation_lines = torch.Tensor([cos_relaxation.get_bounds(subdomain[0, 0], subdomain[1, 0]) for subdomain in domain_bounds]).reshape(-1, 4)
        cos_pi_x_coeffs_lbs = torch.stack([relaxation_lines[:, 0], torch.zeros(relaxation_lines.shape[0])]).T.unsqueeze(1)
        cos_pi_x_consts_lbs = relaxation_lines[:, 1].unsqueeze(1)
        cos_pi_x_coeffs_ubs = torch.stack([relaxation_lines[:, 2], torch.zeros(relaxation_lines.shape[0])]).T.unsqueeze(1)
        cos_pi_x_consts_ubs = relaxation_lines[:, 3].unsqueeze(1)

        cos_pi_x_U = torch.sum(cos_pi_x_coeffs_ubs.clamp(min=0) * self.u_theta.x_ub.unsqueeze(1) + cos_pi_x_coeffs_ubs.clamp(max=0) * self.u_theta.x_lb.unsqueeze(1), dim=2) + cos_pi_x_consts_ubs
        cos_pi_x_L = torch.sum(cos_pi_x_coeffs_lbs.clamp(min=0) * self.u_theta.x_lb.unsqueeze(1) + cos_pi_x_coeffs_lbs.clamp(max=0) * self.u_theta.x_ub.unsqueeze(1), dim=2) + cos_pi_x_consts_lbs

        # McCormick relaxation of x^2
        mccormick_coeffs = torch.Tensor([
            get_single_line_mccormick_coefficients(
                x_L=subdomain[0, 0], 
                x_U=subdomain[1, 0],
                y_L=subdomain[0, 0],
                y_U=subdomain[1, 0]
            )
            for subdomain in domain_bounds
        ])

        mccormick_coeffs_U = mccormick_coeffs[:, 1]
        mccormick_coeffs_L = mccormick_coeffs[:, 0]

        x_squared_coefficient_U = torch.stack([mccormick_coeffs_U[:, 0] + mccormick_coeffs_U[:, 1], torch.zeros(relaxation_lines.shape[0])]).T.unsqueeze(1)
        x_squared_const_U = mccormick_coeffs_U[:, 2].unsqueeze(1)
        
        x_squared_coefficient_L = torch.stack([mccormick_coeffs_L[:, 0] + mccormick_coeffs_L[:, 1], torch.zeros(relaxation_lines.shape[0])]).T.unsqueeze(1)
        x_squared_const_L = mccormick_coeffs_L[:, 2].unsqueeze(1)

        x_squared_U = torch.sum(x_squared_coefficient_U.clamp(min=0) * self.u_theta.x_ub.unsqueeze(1) + x_squared_coefficient_U.clamp(max=0) * self.u_theta.x_lb.unsqueeze(1), dim=2) + x_squared_const_U
        x_squared_L = torch.sum(x_squared_coefficient_L.clamp(min=0) * self.u_theta.x_lb.unsqueeze(1) + x_squared_coefficient_L.clamp(max=0) * self.u_theta.x_ub.unsqueeze(1), dim=2) + x_squared_const_L

        # McCormick relaxation of x_squared * cos_pi_x
        alpha_U, alpha_L = 0.5, 0.5

        beta_0_U = alpha_U * x_squared_U + (1 - alpha_U) * x_squared_L
        beta_1_U = alpha_U * cos_pi_x_L + (1 - alpha_U) * cos_pi_x_U
        beta_2_U = - alpha_U * x_squared_U * cos_pi_x_L - (1 - alpha_U) * x_squared_L * cos_pi_x_U

        beta_0_L = alpha_L * x_squared_L + (1 - alpha_L) * x_squared_U
        beta_1_L = alpha_L * cos_pi_x_L + (1 - alpha_L) * cos_pi_x_U
        beta_2_L = - alpha_L * x_squared_L * cos_pi_x_L - (1 - alpha_L) * x_squared_U * cos_pi_x_U

        mul_coefficient_U = beta_0_U.unsqueeze(1).clamp(min=0) * cos_pi_x_coeffs_ubs + beta_0_U.unsqueeze(1).clamp(max=0) * cos_pi_x_coeffs_lbs +\
            beta_1_U.unsqueeze(1).clamp(min=0) * x_squared_coefficient_U + beta_1_U.unsqueeze(1).clamp(max=0) * x_squared_coefficient_L
        
        mul_const_U = beta_0_U.clamp(min=0) * cos_pi_x_consts_ubs + beta_0_U.clamp(max=0) * cos_pi_x_consts_lbs +\
            beta_1_U.clamp(min=0) * x_squared_const_U + beta_1_U.clamp(max=0) * x_squared_const_L+\
            beta_2_U
        
        mul_coefficient_L = beta_0_L.unsqueeze(1).clamp(min=0) * cos_pi_x_coeffs_lbs + beta_0_L.unsqueeze(1).clamp(max=0) * cos_pi_x_coeffs_ubs +\
            beta_1_L.unsqueeze(1).clamp(min=0) * x_squared_coefficient_L + beta_1_L.unsqueeze(1).clamp(max=0) * x_squared_coefficient_U
        
        mul_const_L = beta_0_L.clamp(min=0) * cos_pi_x_consts_lbs + beta_0_L.clamp(max=0) * cos_pi_x_consts_ubs +\
            beta_1_L.clamp(min=0) * x_squared_const_L + beta_1_L.clamp(max=0) * x_squared_const_U+\
            beta_2_L

        u_theta_coeffs_ubs, u_theta_consts_ubs, u_theta_coeffs_lbs, u_theta_consts_lbs = u_theta.layer_CROWN_coefficients[-1]

        # it's u_theta - x_squared * cos_pi_x, so switch the bounds of the function
        sum_coefficients_U = u_theta_coeffs_ubs - mul_coefficient_L
        sum_constant_U = u_theta_consts_ubs - mul_const_L

        sum_coefficients_L = u_theta_coeffs_lbs - mul_coefficient_U
        sum_constant_L = u_theta_consts_lbs - mul_const_U

        sum_upper_bound = torch.sum(sum_coefficients_U.clamp(min=0) * self.u_theta.x_ub.unsqueeze(1) + sum_coefficients_U.clamp(max=0) * self.u_theta.x_lb.unsqueeze(1), dim=2) + sum_constant_U
        sum_lower_bound = torch.sum(sum_coefficients_L.clamp(min=0) * self.u_theta.x_lb.unsqueeze(1) + sum_coefficients_L.clamp(max=0) * self.u_theta.x_ub.unsqueeze(1), dim=2) + sum_constant_L

        return sum_upper_bound, sum_lower_bound

    def compute_boundary_conditions(
            self,
            domain_bounds: torch.tensor,
            debug: bool = True
    ):
        # mirror the extended bounds
        n_domains = domain_bounds.shape[0]
        extended_domain_bounds = domain_bounds.repeat(2, 1, 1)
        extended_domain_bounds[n_domains:, :, 0] = -extended_domain_bounds[n_domains:, :, 0]

        # compute bounds of the network itself
        u_theta = self.u_theta
        u_theta.domain_bounds = extended_domain_bounds
        u_theta.compute_bounds(debug=debug)

        difference_upper_bound = u_theta.upper_bounds[-1][n_domains:, 0] - u_theta.lower_bounds[-1][:n_domains, 0]
        difference_lower_bound = u_theta.lower_bounds[-1][n_domains:, 0] - u_theta.upper_bounds[-1][:n_domains, 0]

        return difference_upper_bound, difference_lower_bound


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
    
        # residual is u_dt_theta + self.rho * u_theta**3 - self.rho * u_theta - self.nu * u_dxdx_theta
        u_dt_theta_coeffs_ubs, u_dt_theta_consts_ubs = u_dt_theta.u_dxi_crown_coefficients_ubs[-1], u_dt_theta.u_dxi_crown_constants_ubs[-1]
        u_dt_theta_coeffs_lbs, u_dt_theta_consts_lbs = u_dt_theta.u_dxi_crown_coefficients_lbs[-1], u_dt_theta.u_dxi_crown_constants_lbs[-1]

        u_dxdx_theta_coeffs_ubs, u_dxdx_theta_consts_ubs = u_dxdx_theta.u_dxixi_crown_coefficients_ubs[-1], u_dxdx_theta.u_dxixi_crown_constants_ubs[-1]
        u_dxdx_theta_coeffs_lbs, u_dxdx_theta_consts_lbs = u_dxdx_theta.u_dxixi_crown_coefficients_lbs[-1], u_dxdx_theta.u_dxixi_crown_constants_lbs[-1]

        # first McCormick relax u_theta_squared = u_theta * u_theta
        u_theta_lbs = u_theta.lower_bounds[-1]
        u_theta_ubs = u_theta.upper_bounds[-1]
        u_theta_coeffs_ubs, u_theta_consts_ubs, u_theta_coeffs_lbs, u_theta_consts_lbs = u_theta.layer_CROWN_coefficients[-1]

        alpha_U, alpha_L = 0.5, 0.5

        beta_0_U = alpha_U * u_theta_ubs + (1 - alpha_U) * u_theta_lbs
        beta_1_U = alpha_U * u_theta_lbs + (1 - alpha_U) * u_theta_ubs
        beta_2_U = - alpha_U * u_theta_ubs * u_theta_lbs - (1 - alpha_U) * u_theta_lbs * u_theta_ubs

        beta_0_L = alpha_L * u_theta_lbs + (1 - alpha_L) * u_theta_ubs
        beta_1_L = alpha_L * u_theta_lbs + (1 - alpha_L) * u_theta_ubs
        beta_2_L = - alpha_L * u_theta_lbs * u_theta_lbs - (1 - alpha_L) * u_theta_ubs * u_theta_ubs

        u_theta_squared_coeffs_ubs = beta_0_U.unsqueeze(1).clamp(min=0) * u_theta_coeffs_ubs + beta_0_U.unsqueeze(1).clamp(max=0) * u_theta_coeffs_lbs +\
            beta_1_U.unsqueeze(1).clamp(min=0) * u_theta_coeffs_ubs + beta_1_U.unsqueeze(1).clamp(max=0) * u_theta_coeffs_lbs
        u_theta_squared_consts_ubs = beta_0_U.clamp(min=0) * u_theta_consts_ubs + beta_0_U.clamp(max=0) * u_theta_consts_lbs +\
            beta_1_U.clamp(min=0) * u_theta_consts_ubs + beta_1_U.clamp(max=0) * u_theta_consts_lbs+\
            beta_2_U
        
        u_theta_squared_coeffs_lbs = beta_0_L.unsqueeze(1).clamp(min=0) * u_theta_coeffs_lbs + beta_0_L.unsqueeze(1).clamp(max=0) * u_theta_coeffs_ubs +\
            beta_1_L.unsqueeze(1).clamp(min=0) * u_theta_coeffs_lbs + beta_1_L.unsqueeze(1).clamp(max=0) * u_theta_coeffs_ubs
        u_theta_squared_consts_lbs = beta_0_L.clamp(min=0) * u_theta_consts_lbs + beta_0_L.clamp(max=0) * u_theta_consts_ubs +\
            beta_1_L.clamp(min=0) * u_theta_consts_lbs + beta_1_L.clamp(max=0) * u_theta_consts_ubs+\
            beta_2_L

        u_theta_squared_ubs = torch.sum(u_theta_squared_coeffs_ubs.clamp(min=0) * self.u_theta.x_ub.unsqueeze(1) + u_theta_squared_coeffs_ubs.clamp(max=0) * self.u_theta.x_lb.unsqueeze(1), dim=2) + u_theta_squared_consts_ubs
        u_theta_squared_lbs = torch.sum(u_theta_squared_coeffs_lbs.clamp(min=0) * self.u_theta.x_lb.unsqueeze(1) + u_theta_squared_coeffs_lbs.clamp(max=0) * self.u_theta.x_ub.unsqueeze(1), dim=2) + u_theta_squared_consts_lbs

        # now relax u_theta_cubed = u_theta_squared * u_theta
        alpha_U, alpha_L = 0.5, 0.5

        beta_0_U = alpha_U * u_theta_ubs + (1 - alpha_U) * u_theta_lbs
        beta_1_U = alpha_U * u_theta_squared_lbs + (1 - alpha_U) * u_theta_squared_ubs
        beta_2_U = - alpha_U * u_theta_ubs * u_theta_squared_lbs - (1 - alpha_U) * u_theta_lbs * u_theta_squared_ubs

        beta_0_L = alpha_L * u_theta_lbs + (1 - alpha_L) * u_theta_ubs
        beta_1_L = alpha_L * u_theta_squared_lbs + (1 - alpha_L) * u_theta_squared_ubs
        beta_2_L = - alpha_L * u_theta_lbs * u_theta_squared_lbs - (1 - alpha_L) * u_theta_ubs * u_theta_squared_ubs

        u_theta_cubed_coeffs_ubs = beta_0_U.unsqueeze(1).clamp(min=0) * u_theta_squared_coeffs_ubs + beta_0_U.unsqueeze(1).clamp(max=0) * u_theta_squared_coeffs_lbs +\
            beta_1_U.unsqueeze(1).clamp(min=0) * u_theta_coeffs_ubs + beta_1_U.unsqueeze(1).clamp(max=0) * u_theta_coeffs_lbs
        u_theta_cubed_consts_ubs = beta_0_U.clamp(min=0) * u_theta_squared_consts_ubs + beta_0_U.clamp(max=0) * u_theta_squared_consts_lbs +\
            beta_1_U.clamp(min=0) * u_theta_consts_ubs + beta_1_U.clamp(max=0) * u_theta_consts_lbs+\
            beta_2_U
        
        u_theta_cubed_coeffs_lbs = beta_0_L.unsqueeze(1).clamp(min=0) * u_theta_squared_coeffs_lbs + beta_0_L.unsqueeze(1).clamp(max=0) * u_theta_squared_coeffs_ubs +\
            beta_1_L.unsqueeze(1).clamp(min=0) * u_theta_coeffs_lbs + beta_1_L.unsqueeze(1).clamp(max=0) * u_theta_coeffs_ubs
        u_theta_cubed_consts_lbs = beta_0_L.clamp(min=0) * u_theta_squared_consts_lbs + beta_0_L.clamp(max=0) * u_theta_squared_consts_ubs +\
            beta_1_L.clamp(min=0) * u_theta_consts_lbs + beta_1_L.clamp(max=0) * u_theta_consts_ubs+\
            beta_2_L

        # sum of the correct coefficients/constants to obtain the final ones:
        # u_dt_theta + self.rho * u_theta_cubed - self.rho * u_theta - self.nu * u_dxdx_theta

        f_coefficients_U = u_dt_theta_coeffs_ubs + self.rho * u_theta_cubed_coeffs_ubs - self.rho * u_theta_coeffs_lbs - self.nu * u_dxdx_theta_coeffs_lbs
        f_constant_U = u_dt_theta_consts_ubs + self.rho * u_theta_cubed_consts_ubs - self.rho * u_theta_consts_lbs - self.nu * u_dxdx_theta_consts_lbs

        f_coefficients_L = u_dt_theta_coeffs_lbs + self.rho * u_theta_cubed_coeffs_lbs - self.rho * u_theta_coeffs_ubs - self.nu * u_dxdx_theta_coeffs_ubs
        f_constant_L = u_dt_theta_consts_lbs + self.rho * u_theta_cubed_consts_lbs - self.rho * u_theta_consts_ubs - self.nu * u_dxdx_theta_consts_ubs

        f_upper_bound = torch.sum(f_coefficients_U.clamp(min=0) * self.u_theta.x_ub.unsqueeze(1) + f_coefficients_U.clamp(max=0) * self.u_theta.x_lb.unsqueeze(1), dim=2) + f_constant_U
        f_lower_bound = torch.sum(f_coefficients_L.clamp(min=0) * self.u_theta.x_lb.unsqueeze(1) + f_coefficients_L.clamp(max=0) * self.u_theta.x_ub.unsqueeze(1), dim=2) + f_constant_L

        return f_upper_bound, f_lower_bound
