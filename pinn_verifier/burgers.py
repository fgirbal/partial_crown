from typing import List

import numpy as np
import torch

from pinn_verifier.crown import BackpropMode, CROWNPINNSolution, CROWNPINNPartialDerivative, CROWNPINNSecondPartialDerivative
from pinn_verifier.activations.activation_relaxations import ActivationRelaxationType
from pinn_verifier.activations.sin import SinRelaxation


class CROWNBurgersVerifier():
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

    def compute_sin_initial_conditions_bounds(
            self,
            domain_bounds: torch.tensor,
            debug: bool = True
    ):
        # compute bounds of the network itself
        u_theta = self.u_theta
        u_theta.domain_bounds = domain_bounds
        u_theta.compute_bounds(debug=debug)

        # initial condition is u_theta + sin(pi*x)

        # linear coefficients for sin(pi*x) don't depend on t, only on x
        sin_relaxation = SinRelaxation(
            ActivationRelaxationType.SINGLE_LINE
        )
        relaxation_lines = torch.Tensor([sin_relaxation.get_bounds(subdomain[0, 1], subdomain[1, 1]) for subdomain in domain_bounds]).reshape(-1, 4)
        sin_pi_x_coeffs_lbs = torch.stack([torch.zeros(relaxation_lines.shape[0]), relaxation_lines[:, 0]]).T.unsqueeze(1)
        sin_pi_x_consts_lbs = relaxation_lines[:, 1].unsqueeze(1)
        sin_pi_x_coeffs_ubs = torch.stack([torch.zeros(relaxation_lines.shape[0]), relaxation_lines[:, 2]]).T.unsqueeze(1)
        sin_pi_x_consts_ubs = relaxation_lines[:, 3].unsqueeze(1)

        u_theta_coeffs_ubs, u_theta_consts_ubs, u_theta_coeffs_lbs, u_theta_consts_lbs = u_theta.layer_CROWN_coefficients[-1]

        sum_coefficients_U = u_theta_coeffs_ubs + sin_pi_x_coeffs_ubs
        sum_constant_U = u_theta_consts_ubs + sin_pi_x_consts_ubs

        sum_coefficients_L = u_theta_coeffs_lbs + sin_pi_x_coeffs_lbs
        sum_constant_L = u_theta_consts_lbs + sin_pi_x_consts_lbs

        sum_upper_bound = torch.sum(sum_coefficients_U.clamp(min=0) * self.u_theta.x_ub.unsqueeze(1) + sum_coefficients_U.clamp(max=0) * self.u_theta.x_lb.unsqueeze(1), dim=2) + sum_constant_U
        sum_lower_bound = torch.sum(sum_coefficients_L.clamp(min=0) * self.u_theta.x_lb.unsqueeze(1) + sum_coefficients_L.clamp(max=0) * self.u_theta.x_ub.unsqueeze(1), dim=2) + sum_constant_L

        return sum_upper_bound, sum_lower_bound

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

        # u_dt_theta_ubs, u_dt_theta_lbs = u_dt_theta.upper_bounds[-1], u_dt_theta.lower_bounds[-1]
        u_dt_theta_coeffs_ubs, u_dt_theta_consts_ubs = u_dt_theta.u_dxi_crown_coefficients_ubs[-1], u_dt_theta.u_dxi_crown_constants_ubs[-1]
        u_dt_theta_coeffs_lbs, u_dt_theta_consts_lbs = u_dt_theta.u_dxi_crown_coefficients_lbs[-1], u_dt_theta.u_dxi_crown_constants_lbs[-1]

        u_dx_theta_ubs, u_dx_theta_lbs = u_dx_theta.upper_bounds[-1], u_dx_theta.lower_bounds[-1]
        u_dx_theta_coeffs_ubs, u_dx_theta_consts_ubs = u_dx_theta.u_dxi_crown_coefficients_ubs[-1], u_dx_theta.u_dxi_crown_constants_ubs[-1]
        u_dx_theta_coeffs_lbs, u_dx_theta_consts_lbs = u_dx_theta.u_dxi_crown_coefficients_lbs[-1], u_dx_theta.u_dxi_crown_constants_lbs[-1]

        # u_dxdx_theta_ubs, u_dxdx_theta_lbs = u_dxdx_theta.upper_bounds[-1], u_dxdx_theta.lower_bounds[-1]
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

        mul_coefficient_U = beta_0_U.unsqueeze(1).clamp(min=0) * u_dx_theta_coeffs_ubs + beta_0_U.unsqueeze(1).clamp(max=0) * u_dx_theta_coeffs_lbs +\
            beta_1_U.unsqueeze(1).clamp(min=0) * u_theta_coeffs_ubs + beta_1_U.unsqueeze(1).clamp(max=0) * u_theta_coeffs_lbs
        mul_const_U = beta_0_U.clamp(min=0) * u_dx_theta_consts_ubs + beta_0_U.clamp(max=0) * u_dx_theta_consts_lbs +\
            beta_1_U.clamp(min=0) * u_theta_consts_ubs + beta_1_U.clamp(max=0) * u_theta_consts_lbs+\
            beta_2_U
        
        mul_coefficient_L = beta_0_L.unsqueeze(1).clamp(min=0) * u_dx_theta_coeffs_lbs + beta_0_L.unsqueeze(1).clamp(max=0) * u_dx_theta_coeffs_ubs +\
            beta_1_L.unsqueeze(1).clamp(min=0) * u_theta_coeffs_lbs + beta_1_L.unsqueeze(1).clamp(max=0) * u_theta_coeffs_ubs
        mul_const_L = beta_0_L.clamp(min=0) * u_dx_theta_consts_lbs + beta_0_L.clamp(max=0) * u_dx_theta_consts_ubs +\
            beta_1_L.clamp(min=0) * u_theta_consts_lbs + beta_1_L.clamp(max=0) * u_theta_consts_ubs+\
            beta_2_L

        # sum of the correct coefficients to obtain the final ones
        f_coefficients_U = u_dt_theta_coeffs_ubs + mul_coefficient_U - self.viscosity * u_dxdx_theta_coeffs_lbs
        f_constant_U = u_dt_theta_consts_ubs + mul_const_U - self.viscosity * u_dxdx_theta_consts_lbs

        f_coefficients_L = u_dt_theta_coeffs_lbs + mul_coefficient_L - self.viscosity * u_dxdx_theta_coeffs_ubs
        f_constant_L = u_dt_theta_consts_lbs + mul_const_L - self.viscosity * u_dxdx_theta_consts_ubs

        f_upper_bound = torch.sum(f_coefficients_U.clamp(min=0) * self.u_theta.x_ub.unsqueeze(1) + f_coefficients_U.clamp(max=0) * self.u_theta.x_lb.unsqueeze(1), dim=2) + f_constant_U
        f_lower_bound = torch.sum(f_coefficients_L.clamp(min=0) * self.u_theta.x_lb.unsqueeze(1) + f_coefficients_L.clamp(max=0) * self.u_theta.x_ub.unsqueeze(1), dim=2) + f_constant_L

        return f_upper_bound, f_lower_bound
