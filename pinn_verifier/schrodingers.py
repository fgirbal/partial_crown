from typing import List

import numpy as np
import torch
from tools.custom_torch_modules import Add, Mul

from pinn_verifier.crown import BackpropMode, CROWNPINNSolution, CROWNPINNPartialDerivative, CROWNPINNSecondPartialDerivative, get_single_line_mccormick_coefficients
from pinn_verifier.activations.activation_relaxations import ActivationRelaxationType
from pinn_verifier.activations.sech import SechRelaxation


class CROWNSchrodingersVerifier():
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

    def compute_sech_initial_conditions_bounds(
            self,
            domain_bounds: torch.tensor,
            debug: bool = True
    ):
        # compute bounds of the network itself
        u_theta = self.u_theta
        u_theta.domain_bounds = domain_bounds
        u_theta.compute_bounds(debug=debug)

        h_0_relaxation = SechRelaxation(
            ActivationRelaxationType.SINGLE_LINE
        )
        relaxation_lines = torch.Tensor([h_0_relaxation.get_bounds(subdomain[0, 1], subdomain[1, 1]) for subdomain in domain_bounds]).reshape(-1, 4)
        sech_x_coeffs_lbs = torch.stack([torch.zeros(relaxation_lines.shape[0]), relaxation_lines[:, 0]]).T
        sech_x_consts_lbs = relaxation_lines[:, 1]
        sech_x_coeffs_ubs = torch.stack([torch.zeros(relaxation_lines.shape[0]), relaxation_lines[:, 2]]).T
        sech_x_consts_ubs = relaxation_lines[:, 3]

        sech_x_upper_bound = torch.sum(sech_x_coeffs_ubs.clamp(min=0) * domain_bounds[:, 1] + sech_x_coeffs_ubs.clamp(max=0) * domain_bounds[:, 0], dim=1) + sech_x_consts_ubs
        sech_x_lower_bound = torch.sum(sech_x_coeffs_lbs.clamp(min=0) * domain_bounds[:, 0] + sech_x_coeffs_lbs.clamp(max=0) * domain_bounds[:, 1], dim=1) + sech_x_consts_lbs

        # apply normalization layers on the bounds
        import copy
        current_domain_bounds = copy.deepcopy(domain_bounds)
        for layer in self.u_theta.norm_layers:
            if type(layer) == Add:
                if layer.const.shape[0] > 1:
                    sech_x_consts_lbs -= sech_x_coeffs_lbs[:, 1] * layer.const[1]
                    sech_x_consts_ubs -= sech_x_coeffs_ubs[:, 1] * layer.const[1]
                else:
                    sech_x_consts_lbs -= sech_x_coeffs_lbs[:, 1] * layer.const[0]
                    sech_x_consts_ubs -= sech_x_coeffs_ubs[:, 1] * layer.const[0]
            elif type(layer) == Mul:
                if layer.const.shape[0] > 1:
                    mul_const = layer.const[1]
                else:
                    mul_const = layer.const[0]

                sech_x_coeffs_lbs /= mul_const
                sech_x_coeffs_ubs /= mul_const

                if mul_const < 0:
                    sech_x_coeffs_lbs, sech_x_coeffs_ubs = sech_x_coeffs_ubs, sech_x_coeffs_lbs
                    sech_x_consts_lbs, sech_x_consts_ubs = sech_x_consts_ubs, sech_x_consts_lbs
            
            current_domain_bounds = layer(current_domain_bounds)
            
            if type(layer) == Mul:
                # a multiplication can change the direction of the bounds, sort them accordingly
                current_domain_bounds = current_domain_bounds.sort(dim=1).values

            sech_x_upper_bound_ = torch.sum(sech_x_coeffs_ubs.clamp(min=0) * current_domain_bounds[:, 1] + sech_x_coeffs_ubs.clamp(max=0) * current_domain_bounds[:, 0], dim=1) + sech_x_consts_ubs
            sech_x_lower_bound_ = torch.sum(sech_x_coeffs_lbs.clamp(min=0) * current_domain_bounds[:, 0] + sech_x_coeffs_lbs.clamp(max=0) * current_domain_bounds[:, 1], dim=1) + sech_x_consts_lbs

            try:
                assert torch.norm(sech_x_upper_bound_ - sech_x_upper_bound) <= 1e-5
                assert torch.norm(sech_x_lower_bound_ - sech_x_lower_bound) <= 1e-5
            except:
                print('relaxation error...')
                import pdb
                pdb.set_trace()

        # sech coefficients are now with respect to the normalized domain bounds, can join them to compute final bound
        u_theta_coeffs_ubs, u_theta_consts_ubs, u_theta_coeffs_lbs, u_theta_consts_lbs = u_theta.layer_CROWN_coefficients[-1]

        h_theta_coeffs_ubs = u_theta_coeffs_ubs[:, 0]
        h_theta_consts_ubs = u_theta_consts_ubs[:, 0]
        h_theta_coeffs_lbs = u_theta_coeffs_lbs[:, 0]
        h_theta_consts_lbs = u_theta_consts_lbs[:, 0]

        h_error_coeffs_ubs = h_theta_coeffs_ubs - sech_x_coeffs_lbs
        h_error_consts_ubs = h_theta_consts_ubs - sech_x_consts_lbs
        h_error_coeffs_lbs = h_theta_coeffs_lbs - sech_x_coeffs_ubs
        h_error_consts_lbs = h_theta_consts_lbs - sech_x_consts_ubs

        h_error_upper_bound = torch.sum(h_error_coeffs_ubs.clamp(min=0) * self.u_theta.x_ub + h_error_coeffs_ubs.clamp(max=0) * self.u_theta.x_lb, dim=1) + h_error_consts_ubs
        h_error_lower_bound = torch.sum(h_error_coeffs_lbs.clamp(min=0) * self.u_theta.x_lb + h_error_coeffs_lbs.clamp(max=0) * self.u_theta.x_ub, dim=1) + h_error_consts_lbs
        h_error_upper_bound, h_error_lower_bound = h_error_upper_bound.unsqueeze(1), h_error_lower_bound.unsqueeze(1)
        h_error_consts_ubs, h_error_consts_lbs = h_error_consts_ubs.unsqueeze(1), h_error_consts_lbs.unsqueeze(1)

        v_error_coeffs_ubs = u_theta_coeffs_ubs[:, 1]
        v_error_consts_ubs = u_theta_consts_ubs[:, 1].unsqueeze(1)
        v_error_coeffs_lbs = u_theta_coeffs_lbs[:, 1]
        v_error_consts_lbs = u_theta_consts_lbs[:, 1].unsqueeze(1)

        v_error_upper_bound = u_theta.upper_bounds[-1][:, 1].unsqueeze(1)
        v_error_lower_bound = u_theta.lower_bounds[-1][:, 1].unsqueeze(1)

        # perform the mccormick relaxation of h_error * h_error and v_error * v_error
        alpha_h_error = 0.5
        h_error_mccormick_0_ubs = h_error_lower_bound + h_error_upper_bound
        h_error_mccormick_1_ubs = - h_error_lower_bound * h_error_upper_bound

        h_error_mccormick_0_lbs = 2 * (alpha_h_error * h_error_lower_bound + (1 - alpha_h_error) * h_error_upper_bound)
        h_error_mccormick_1_lbs = - alpha_h_error * (h_error_lower_bound)**2 - (1 - alpha_h_error) * (h_error_upper_bound)**2

        h_error_squared_coeffs_ubs = h_error_mccormick_0_ubs.clamp(min=0) * h_error_coeffs_ubs + h_error_mccormick_0_ubs.clamp(max=0) * h_error_coeffs_lbs
        h_error_squared_consts_ubs = h_error_mccormick_0_ubs.clamp(min=0) * h_error_consts_ubs + h_error_mccormick_0_ubs.clamp(max=0) * h_error_consts_lbs + h_error_mccormick_1_ubs

        h_error_squared_coeffs_lbs = h_error_mccormick_0_lbs.clamp(min=0) * h_error_coeffs_lbs + h_error_mccormick_0_lbs.clamp(max=0) * h_error_coeffs_ubs
        h_error_squared_consts_lbs = h_error_mccormick_0_lbs.clamp(min=0) * h_error_consts_lbs + h_error_mccormick_0_lbs.clamp(max=0) * h_error_consts_ubs + h_error_mccormick_1_lbs

        h_error_squared_upper_bound = torch.sum(h_error_squared_coeffs_ubs.clamp(min=0) * self.u_theta.x_ub + h_error_squared_coeffs_ubs.clamp(max=0) * self.u_theta.x_lb, dim=1) + h_error_squared_consts_ubs.flatten()
        h_error_squared_lower_bound = torch.clamp(
            torch.sum(h_error_squared_coeffs_lbs.clamp(min=0) * self.u_theta.x_lb + h_error_squared_coeffs_lbs.clamp(max=0) * self.u_theta.x_ub, dim=1) + h_error_squared_consts_lbs.flatten(),
            min=0.0
        )

        alpha_v_error = 0.5
        v_error_mccormick_0_ubs = v_error_lower_bound + v_error_upper_bound
        v_error_mccormick_1_ubs = - v_error_lower_bound * v_error_upper_bound

        v_error_mccormick_0_lbs = 2 * (alpha_v_error * v_error_lower_bound + (1 - alpha_v_error) * v_error_upper_bound)
        v_error_mccormick_1_lbs = - alpha_v_error * (v_error_lower_bound)**2 - (1 - alpha_v_error) * (v_error_upper_bound)**2

        v_error_squared_coeffs_ubs = v_error_mccormick_0_ubs.clamp(min=0) * v_error_coeffs_ubs + v_error_mccormick_0_ubs.clamp(max=0) * v_error_coeffs_lbs
        v_error_squared_consts_ubs = v_error_mccormick_0_ubs.clamp(min=0) * v_error_consts_ubs + v_error_mccormick_0_ubs.clamp(max=0) * v_error_consts_lbs + v_error_mccormick_1_ubs

        v_error_squared_coeffs_lbs = v_error_mccormick_0_lbs.clamp(min=0) * v_error_coeffs_lbs + v_error_mccormick_0_lbs.clamp(max=0) * v_error_coeffs_ubs
        v_error_squared_consts_lbs = v_error_mccormick_0_lbs.clamp(min=0) * v_error_consts_lbs + v_error_mccormick_0_lbs.clamp(max=0) * v_error_consts_ubs + v_error_mccormick_1_lbs

        v_error_squared_upper_bound = torch.sum(v_error_squared_coeffs_ubs.clamp(min=0) * self.u_theta.x_ub + v_error_squared_coeffs_ubs.clamp(max=0) * self.u_theta.x_lb, dim=1) + v_error_squared_consts_ubs.flatten()
        v_error_squared_lower_bound = torch.clamp(
            torch.sum(v_error_squared_coeffs_lbs.clamp(min=0) * self.u_theta.x_lb + v_error_squared_coeffs_lbs.clamp(max=0) * self.u_theta.x_ub, dim=1) + v_error_squared_consts_lbs.flatten(),
            min=0.0
        )

        total_error_upper_bound = h_error_squared_upper_bound + v_error_squared_upper_bound
        total_error_lower_bound = h_error_squared_lower_bound + v_error_squared_lower_bound

        self.initial_conditions_intermediate_bounds = {
            'h_error_lower_bound': h_error_lower_bound,
            'h_error_upper_bound': h_error_upper_bound,
            'v_error_lower_bound': v_error_lower_bound,
            'v_error_upper_bound': v_error_upper_bound,
        }

        return total_error_upper_bound, total_error_lower_bound
    
    def compute_solution_boundary_conditions(
            self,
            domain_bounds: torch.tensor,
            debug: bool = True
    ):
        # mirror the extended bounds
        n_domains = domain_bounds.shape[0]
        extended_domain_bounds = domain_bounds.repeat(2, 1, 1)
        extended_domain_bounds[n_domains:, :, 1] = -extended_domain_bounds[n_domains:, :, 1]

        # compute bounds of the network itself
        u_theta = self.u_theta
        u_theta.domain_bounds = extended_domain_bounds
        u_theta.compute_bounds(debug=debug)

        # get coefficients of the lower and upper boundaries of the domain
        u_theta_coeffs_ubs, u_theta_consts_ubs, u_theta_coeffs_lbs, u_theta_consts_lbs = u_theta.layer_CROWN_coefficients[-1]

        h_lower_lower_bound = u_theta.lower_bounds[-1][:n_domains, 0]
        h_lower_upper_bound = u_theta.upper_bounds[-1][:n_domains, 0]
        h_lower_coeffs_ubs = u_theta_coeffs_ubs[:n_domains, 0]
        h_lower_consts_ubs = u_theta_consts_ubs[:n_domains, 0]
        h_lower_coeffs_lbs = u_theta_coeffs_lbs[:n_domains, 0]
        h_lower_consts_lbs = u_theta_consts_lbs[:n_domains, 0]

        h_upper_lower_bound = u_theta.lower_bounds[-1][n_domains:, 0]
        h_upper_upper_bound = u_theta.upper_bounds[-1][n_domains:, 0]
        h_upper_coeffs_ubs = u_theta_coeffs_ubs[n_domains:, 0]
        h_upper_consts_ubs = u_theta_consts_ubs[n_domains:, 0]
        h_upper_coeffs_lbs = u_theta_coeffs_lbs[n_domains:, 0]
        h_upper_consts_lbs = u_theta_consts_lbs[n_domains:, 0]

        h_error_upper_bound = u_theta.upper_bounds[-1][n_domains:, 0] - u_theta.lower_bounds[-1][:n_domains, 0]
        h_error_lower_bound = u_theta.lower_bounds[-1][n_domains:, 0] - u_theta.upper_bounds[-1][:n_domains, 0]

        # return h_error_upper_bound, h_error_lower_bound

        v_lower_lower_bound = u_theta.lower_bounds[-1][:n_domains, 1]
        v_lower_upper_bound = u_theta.upper_bounds[-1][:n_domains, 1]
        v_lower_coeffs_ubs = u_theta_coeffs_ubs[:n_domains, 1]
        v_lower_consts_ubs = u_theta_consts_ubs[:n_domains, 1]
        v_lower_coeffs_lbs = u_theta_coeffs_lbs[:n_domains, 1]
        v_lower_consts_lbs = u_theta_consts_lbs[:n_domains, 1]

        v_upper_lower_bound = u_theta.lower_bounds[-1][n_domains:, 1]
        v_upper_upper_bound = u_theta.upper_bounds[-1][n_domains:, 1]
        v_upper_coeffs_ubs = u_theta_coeffs_ubs[n_domains:, 1]
        v_upper_consts_ubs = u_theta_consts_ubs[n_domains:, 1]
        v_upper_coeffs_lbs = u_theta_coeffs_lbs[n_domains:, 1]
        v_upper_consts_lbs = u_theta_consts_lbs[n_domains:, 1]

        v_error_upper_bound = u_theta.upper_bounds[-1][n_domains:, 1] - u_theta.lower_bounds[-1][:n_domains, 1]
        v_error_lower_bound = u_theta.lower_bounds[-1][n_domains:, 1] - u_theta.upper_bounds[-1][:n_domains, 1]

        # return v_error_upper_bound, v_error_lower_bound

        # perform the mccormick relaxation of h_upper * h_upper, -2 * h_upper * h_lower, h_lower * h_lower
        alpha_h = 0.5

        # h_upper^2 relaxation
        h_upper_squared_mccormick_0_ubs = h_upper_lower_bound + h_upper_upper_bound
        h_upper_squared_mccormick_1_ubs = - h_upper_lower_bound * h_upper_upper_bound

        h_upper_squared_mccormick_0_lbs = 2 * (alpha_h * h_upper_lower_bound + (1 - alpha_h) * h_upper_upper_bound)
        h_upper_squared_mccormick_1_lbs = - alpha_h * (h_upper_lower_bound)**2 - (1 - alpha_h) * (h_upper_upper_bound)**2

        # h_lower^2 relaxation
        h_lower_squared_mccormick_0_ubs = h_lower_lower_bound + h_lower_upper_bound
        h_lower_squared_mccormick_1_ubs = - h_lower_lower_bound * h_lower_upper_bound

        h_lower_squared_mccormick_0_lbs = 2 * (alpha_h * h_lower_lower_bound + (1 - alpha_h) * h_lower_upper_bound)
        h_lower_squared_mccormick_1_lbs = - alpha_h * (h_lower_lower_bound)**2 - (1 - alpha_h) * (h_lower_upper_bound)**2

        # -2 * h_upper * h_lower relaxation
        h_upper_lower_mccormick_0_ubs = (alpha_h * h_upper_upper_bound + (1 - alpha_h) * h_upper_lower_bound)
        h_upper_lower_mccormick_1_ubs = (alpha_h * h_lower_lower_bound + (1 - alpha_h) * h_lower_upper_bound)
        h_upper_lower_mccormick_2_ubs = (alpha_h * (- h_upper_upper_bound * h_lower_lower_bound) + (1 - alpha_h) * (- h_upper_lower_bound * h_lower_upper_bound))

        h_upper_lower_mccormick_0_lbs = (alpha_h * h_upper_lower_bound + (1 - alpha_h) * h_upper_upper_bound)
        h_upper_lower_mccormick_1_lbs = (alpha_h * h_lower_lower_bound + (1 - alpha_h) * h_lower_upper_bound)
        h_upper_lower_mccormick_2_lbs = (alpha_h * (- h_upper_lower_bound * h_lower_lower_bound) + (1 - alpha_h) * (- h_upper_upper_bound * h_lower_upper_bound))

        h_error_h_upper_coeff_ubs = h_upper_squared_mccormick_0_ubs - 2 * h_upper_lower_mccormick_1_lbs
        h_error_h_lower_coeff_ubs = h_lower_squared_mccormick_0_ubs - 2 * h_upper_lower_mccormick_0_lbs
        h_error_const_ubs = h_upper_squared_mccormick_1_ubs + h_lower_squared_mccormick_1_ubs - 2 * h_upper_lower_mccormick_2_lbs

        h_error_h_upper_coeff_lbs = h_upper_squared_mccormick_0_lbs - 2 * h_upper_lower_mccormick_1_ubs
        h_error_h_lower_coeff_lbs = h_lower_squared_mccormick_0_lbs - 2 * h_upper_lower_mccormick_0_ubs
        h_error_const_lbs = h_upper_squared_mccormick_1_lbs + h_lower_squared_mccormick_1_lbs - 2 * h_upper_lower_mccormick_2_ubs

        h_error_h_upper_final_coeffs_ubs = h_error_h_upper_coeff_ubs.clamp(min=0).unsqueeze(1) * h_upper_coeffs_ubs + h_error_h_upper_coeff_ubs.clamp(max=0).unsqueeze(1) * h_upper_coeffs_lbs
        h_error_h_upper_final_consts_ubs = h_error_h_upper_coeff_ubs.clamp(min=0) * h_upper_consts_ubs + h_error_h_upper_coeff_ubs.clamp(max=0) * h_upper_consts_lbs
        h_error_h_lower_final_coeffs_ubs = h_error_h_lower_coeff_ubs.clamp(min=0).unsqueeze(1) * h_lower_coeffs_ubs + h_error_h_lower_coeff_ubs.clamp(max=0).unsqueeze(1) * h_lower_coeffs_lbs
        h_error_h_lower_final_consts_ubs = h_error_h_lower_coeff_ubs.clamp(min=0) * h_lower_consts_ubs + h_error_h_lower_coeff_ubs.clamp(max=0) * h_lower_consts_lbs
        h_error_final_consts_ubs = h_error_h_upper_final_consts_ubs + h_error_h_lower_final_consts_ubs + h_error_const_ubs

        h_error_h_upper_final_coeffs_lbs = h_error_h_upper_coeff_lbs.clamp(min=0).unsqueeze(1) * h_upper_coeffs_lbs + h_error_h_upper_coeff_lbs.clamp(max=0).unsqueeze(1) * h_upper_coeffs_ubs
        h_error_h_upper_final_consts_lbs = h_error_h_upper_coeff_lbs.clamp(min=0) * h_upper_consts_lbs + h_error_h_upper_coeff_lbs.clamp(max=0) * h_upper_consts_ubs
        h_error_h_lower_final_coeffs_lbs = h_error_h_lower_coeff_lbs.clamp(min=0).unsqueeze(1) * h_lower_coeffs_lbs + h_error_h_lower_coeff_lbs.clamp(max=0).unsqueeze(1) * h_lower_coeffs_ubs
        h_error_h_lower_final_consts_lbs = h_error_h_lower_coeff_lbs.clamp(min=0) * h_lower_consts_lbs + h_error_h_lower_coeff_lbs.clamp(max=0) * h_lower_consts_ubs
        h_error_final_consts_lbs = h_error_h_upper_final_consts_lbs + h_error_h_lower_final_consts_lbs + h_error_const_lbs

        h_error_squared_upper_bound = torch.sum(h_error_h_upper_final_coeffs_ubs.clamp(min=0) * u_theta.x_ub[n_domains:] + h_error_h_upper_final_coeffs_ubs.clamp(max=0) * u_theta.x_lb[n_domains:], dim=1) +\
            torch.sum(h_error_h_lower_final_coeffs_ubs.clamp(min=0) * u_theta.x_ub[:n_domains] + h_error_h_lower_final_coeffs_ubs.clamp(max=0) * u_theta.x_lb[:n_domains], dim=1) +\
            h_error_final_consts_ubs.flatten()
        h_error_squared_lower_bound = torch.clamp(
            torch.sum(h_error_h_upper_final_coeffs_lbs.clamp(min=0) * u_theta.x_lb[n_domains:] + h_error_h_upper_final_coeffs_lbs.clamp(max=0) * u_theta.x_ub[n_domains:], dim=1) +\
            torch.sum(h_error_h_lower_final_coeffs_lbs.clamp(min=0) * u_theta.x_lb[:n_domains] + h_error_h_lower_final_coeffs_lbs.clamp(max=0) * u_theta.x_ub[:n_domains], dim=1) +\
            h_error_final_consts_lbs.flatten(),
            min=0
        )

        # perform the mccormick relaxation of v_upper * v_upper, -2 * v_upper * v_lower, v_lower * v_lower
        alpha_v = 0.5

        # h_upper^2 relaxation
        v_upper_squared_mccormick_0_ubs = v_upper_lower_bound + v_upper_upper_bound
        v_upper_squared_mccormick_1_ubs = - v_upper_lower_bound * v_upper_upper_bound

        v_upper_squared_mccormick_0_lbs = 2 * (alpha_v * v_upper_lower_bound + (1 - alpha_v) * v_upper_upper_bound)
        v_upper_squared_mccormick_1_lbs = - alpha_v * (v_upper_lower_bound)**2 - (1 - alpha_v) * (v_upper_upper_bound)**2

        # v_lower^2 relaxation
        v_lower_squared_mccormick_0_ubs = v_lower_lower_bound + v_lower_upper_bound
        v_lower_squared_mccormick_1_ubs = - v_lower_lower_bound * v_lower_upper_bound

        v_lower_squared_mccormick_0_lbs = 2 * (alpha_v * v_lower_lower_bound + (1 - alpha_v) * v_lower_upper_bound)
        v_lower_squared_mccormick_1_lbs = - alpha_v * (v_lower_lower_bound)**2 - (1 - alpha_v) * (v_lower_upper_bound)**2

        # -2 * v_upper * v_lower relaxation
        v_upper_lower_mccormick_0_ubs = (alpha_v * v_upper_upper_bound + (1 - alpha_v) * v_upper_lower_bound)
        v_upper_lower_mccormick_1_ubs = (alpha_v * v_lower_lower_bound + (1 - alpha_v) * v_lower_upper_bound)
        v_upper_lower_mccormick_2_ubs = (alpha_v * (- v_upper_upper_bound * v_lower_lower_bound) + (1 - alpha_v) * (- v_upper_lower_bound * v_lower_upper_bound))

        v_upper_lower_mccormick_0_lbs = (alpha_v * v_upper_lower_bound + (1 - alpha_v) * v_upper_upper_bound)
        v_upper_lower_mccormick_1_lbs = (alpha_v * v_lower_lower_bound + (1 - alpha_v) * v_lower_upper_bound)
        v_upper_lower_mccormick_2_lbs = (alpha_v * (- v_upper_lower_bound * v_lower_lower_bound) + (1 - alpha_v) * (- v_upper_upper_bound * v_lower_upper_bound))

        v_error_v_upper_coeff_ubs = v_upper_squared_mccormick_0_ubs - 2 * v_upper_lower_mccormick_1_lbs
        v_error_v_lower_coeff_ubs = v_lower_squared_mccormick_0_ubs - 2 * v_upper_lower_mccormick_0_lbs
        v_error_const_ubs = v_upper_squared_mccormick_1_ubs + v_lower_squared_mccormick_1_ubs - 2 * v_upper_lower_mccormick_2_lbs

        v_error_v_upper_coeff_lbs = v_upper_squared_mccormick_0_lbs - 2 * v_upper_lower_mccormick_1_ubs
        v_error_v_lower_coeff_lbs = v_lower_squared_mccormick_0_lbs - 2 * v_upper_lower_mccormick_0_ubs
        v_error_const_lbs = v_upper_squared_mccormick_1_lbs + v_lower_squared_mccormick_1_lbs - 2 * v_upper_lower_mccormick_2_ubs

        v_error_v_upper_final_coeffs_ubs = v_error_v_upper_coeff_ubs.clamp(min=0).unsqueeze(1) * v_upper_coeffs_ubs + v_error_v_upper_coeff_ubs.clamp(max=0).unsqueeze(1) * v_upper_coeffs_lbs
        v_error_v_upper_final_consts_ubs = v_error_v_upper_coeff_ubs.clamp(min=0) * v_upper_consts_ubs + v_error_v_upper_coeff_ubs.clamp(max=0) * v_upper_consts_lbs
        v_error_v_lower_final_coeffs_ubs = v_error_v_lower_coeff_ubs.clamp(min=0).unsqueeze(1) * v_lower_coeffs_ubs + v_error_v_lower_coeff_ubs.clamp(max=0).unsqueeze(1) * v_lower_coeffs_lbs
        v_error_v_lower_final_consts_ubs = v_error_v_lower_coeff_ubs.clamp(min=0) * v_lower_consts_ubs + v_error_v_lower_coeff_ubs.clamp(max=0) * v_lower_consts_lbs
        v_error_final_consts_ubs = v_error_v_upper_final_consts_ubs + v_error_v_lower_final_consts_ubs + v_error_const_ubs

        v_error_v_upper_final_coeffs_lbs = v_error_v_upper_coeff_lbs.clamp(min=0).unsqueeze(1) * v_upper_coeffs_lbs + v_error_v_upper_coeff_lbs.clamp(max=0).unsqueeze(1) * v_upper_coeffs_ubs
        v_error_v_upper_final_consts_lbs = v_error_v_upper_coeff_lbs.clamp(min=0) * v_upper_consts_lbs + v_error_v_upper_coeff_lbs.clamp(max=0) * v_upper_consts_ubs
        v_error_v_lower_final_coeffs_lbs = v_error_v_lower_coeff_lbs.clamp(min=0).unsqueeze(1) * v_lower_coeffs_lbs + v_error_v_lower_coeff_lbs.clamp(max=0).unsqueeze(1) * v_lower_coeffs_ubs
        v_error_v_lower_final_consts_lbs = v_error_v_lower_coeff_lbs.clamp(min=0) * v_lower_consts_lbs + v_error_v_lower_coeff_lbs.clamp(max=0) * v_lower_consts_ubs
        v_error_final_consts_lbs = v_error_v_upper_final_consts_lbs + v_error_v_lower_final_consts_lbs + v_error_const_lbs

        v_error_squared_upper_bound = torch.sum(v_error_v_upper_final_coeffs_ubs.clamp(min=0) * u_theta.x_ub[n_domains:] + v_error_v_upper_final_coeffs_ubs.clamp(max=0) * u_theta.x_lb[n_domains:], dim=1) +\
            torch.sum(v_error_v_lower_final_coeffs_ubs.clamp(min=0) * u_theta.x_ub[:n_domains] + v_error_v_lower_final_coeffs_ubs.clamp(max=0) * u_theta.x_lb[:n_domains], dim=1) +\
            v_error_final_consts_ubs.flatten()
        v_error_squared_lower_bound = torch.clamp(
            torch.sum(v_error_v_upper_final_coeffs_lbs.clamp(min=0) * u_theta.x_lb[n_domains:] + v_error_v_upper_final_coeffs_lbs.clamp(max=0) * u_theta.x_ub[n_domains:], dim=1) +\
            torch.sum(v_error_v_lower_final_coeffs_lbs.clamp(min=0) * u_theta.x_lb[:n_domains] + v_error_v_lower_final_coeffs_lbs.clamp(max=0) * u_theta.x_ub[:n_domains], dim=1) +\
            v_error_final_consts_lbs.flatten(),
            min=0
        )

        self.initial_conditions_intermediate_bounds = {
            'h_error_lower_bound': h_error_lower_bound,
            'h_error_upper_bound': h_error_upper_bound,
            'v_error_lower_bound': v_error_lower_bound,
            'v_error_upper_bound': v_error_upper_bound,
        }

        if any(h_error_squared_upper_bound < h_error_squared_lower_bound) or any(v_error_squared_upper_bound < v_error_squared_lower_bound):
            import pdb
            pdb.set_trace()

        return h_error_squared_upper_bound + v_error_squared_upper_bound, h_error_squared_lower_bound + v_error_squared_lower_bound

    def compute_partial_x_boundary_conditions(
            self,
            domain_bounds: torch.tensor,
            debug: bool = True
    ):
        # mirror the extended bounds
        n_domains = domain_bounds.shape[0]
        extended_domain_bounds = domain_bounds.repeat(2, 1, 1)
        extended_domain_bounds[n_domains:, :, 1] = -extended_domain_bounds[n_domains:, :, 1]

        # compute bounds of the network itself
        u_theta = self.u_theta
        u_theta.domain_bounds = extended_domain_bounds
        u_theta.compute_bounds(debug=debug)

        u_dx_theta = self.u_dx_theta
        u_dx_theta.compute_bounds(debug=debug, backprop_mode=BackpropMode.COMPONENT_BACKPROP)

        # get coefficients of the lower and upper boundaries of the domain
        # u_theta_coeffs_ubs, u_theta_consts_ubs, u_theta_coeffs_lbs, u_theta_consts_lbs = u_theta.layer_CROWN_coefficients[-1]

        u_dx_theta_coeffs_ubs, u_dx_theta_consts_ubs = u_dx_theta.u_dxi_crown_coefficients_ubs[-1], u_dx_theta.u_dxi_crown_constants_ubs[-1]
        u_dx_theta_coeffs_lbs, u_dx_theta_consts_lbs = u_dx_theta.u_dxi_crown_coefficients_lbs[-1], u_dx_theta.u_dxi_crown_constants_lbs[-1]

        h_lower_lower_bound = u_dx_theta.lower_bounds[-1][:n_domains, 0]
        h_lower_upper_bound = u_dx_theta.upper_bounds[-1][:n_domains, 0]
        h_lower_coeffs_ubs = u_dx_theta_coeffs_ubs[:n_domains, 0]
        h_lower_consts_ubs = u_dx_theta_consts_ubs[:n_domains, 0]
        h_lower_coeffs_lbs = u_dx_theta_coeffs_lbs[:n_domains, 0]
        h_lower_consts_lbs = u_dx_theta_consts_lbs[:n_domains, 0]

        h_upper_lower_bound = u_dx_theta.lower_bounds[-1][n_domains:, 0]
        h_upper_upper_bound = u_dx_theta.upper_bounds[-1][n_domains:, 0]
        h_upper_coeffs_ubs = u_dx_theta_coeffs_ubs[n_domains:, 0]
        h_upper_consts_ubs = u_dx_theta_consts_ubs[n_domains:, 0]
        h_upper_coeffs_lbs = u_dx_theta_coeffs_lbs[n_domains:, 0]
        h_upper_consts_lbs = u_dx_theta_consts_lbs[n_domains:, 0]

        h_error_upper_bound = u_dx_theta.upper_bounds[-1][n_domains:, 0] - u_dx_theta.lower_bounds[-1][:n_domains, 0]
        h_error_lower_bound = u_dx_theta.lower_bounds[-1][n_domains:, 0] - u_dx_theta.upper_bounds[-1][:n_domains, 0]

        # return h_error_upper_bound, h_error_lower_bound

        v_lower_lower_bound = u_dx_theta.lower_bounds[-1][:n_domains, 1]
        v_lower_upper_bound = u_dx_theta.upper_bounds[-1][:n_domains, 1]
        v_lower_coeffs_ubs = u_dx_theta_coeffs_ubs[:n_domains, 1]
        v_lower_consts_ubs = u_dx_theta_consts_ubs[:n_domains, 1]
        v_lower_coeffs_lbs = u_dx_theta_coeffs_lbs[:n_domains, 1]
        v_lower_consts_lbs = u_dx_theta_consts_lbs[:n_domains, 1]

        v_upper_lower_bound = u_dx_theta.lower_bounds[-1][n_domains:, 1]
        v_upper_upper_bound = u_dx_theta.upper_bounds[-1][n_domains:, 1]
        v_upper_coeffs_ubs = u_dx_theta_coeffs_ubs[n_domains:, 1]
        v_upper_consts_ubs = u_dx_theta_consts_ubs[n_domains:, 1]
        v_upper_coeffs_lbs = u_dx_theta_coeffs_lbs[n_domains:, 1]
        v_upper_consts_lbs = u_dx_theta_consts_lbs[n_domains:, 1]

        v_error_upper_bound = u_dx_theta.upper_bounds[-1][n_domains:, 1] - u_dx_theta.lower_bounds[-1][:n_domains, 1]
        v_error_lower_bound = u_dx_theta.lower_bounds[-1][n_domains:, 1] - u_dx_theta.upper_bounds[-1][:n_domains, 1]

        # return v_error_upper_bound, v_error_lower_bound

        # perform the mccormick relaxation of h_upper * h_upper, -2 * h_upper * h_lower, h_lower * h_lower
        alpha_h = 0.5

        # h_upper^2 relaxation
        h_upper_squared_mccormick_0_ubs = h_upper_lower_bound + h_upper_upper_bound
        h_upper_squared_mccormick_1_ubs = - h_upper_lower_bound * h_upper_upper_bound

        h_upper_squared_mccormick_0_lbs = 2 * (alpha_h * h_upper_lower_bound + (1 - alpha_h) * h_upper_upper_bound)
        h_upper_squared_mccormick_1_lbs = - alpha_h * (h_upper_lower_bound)**2 - (1 - alpha_h) * (h_upper_upper_bound)**2

        # h_lower^2 relaxation
        h_lower_squared_mccormick_0_ubs = h_lower_lower_bound + h_lower_upper_bound
        h_lower_squared_mccormick_1_ubs = - h_lower_lower_bound * h_lower_upper_bound

        h_lower_squared_mccormick_0_lbs = 2 * (alpha_h * h_lower_lower_bound + (1 - alpha_h) * h_lower_upper_bound)
        h_lower_squared_mccormick_1_lbs = - alpha_h * (h_lower_lower_bound)**2 - (1 - alpha_h) * (h_lower_upper_bound)**2

        # -2 * h_upper * h_lower relaxation
        h_upper_lower_mccormick_0_ubs = (alpha_h * h_upper_upper_bound + (1 - alpha_h) * h_upper_lower_bound)
        h_upper_lower_mccormick_1_ubs = (alpha_h * h_lower_lower_bound + (1 - alpha_h) * h_lower_upper_bound)
        h_upper_lower_mccormick_2_ubs = (alpha_h * (- h_upper_upper_bound * h_lower_lower_bound) + (1 - alpha_h) * (- h_upper_lower_bound * h_lower_upper_bound))

        h_upper_lower_mccormick_0_lbs = (alpha_h * h_upper_lower_bound + (1 - alpha_h) * h_upper_upper_bound)
        h_upper_lower_mccormick_1_lbs = (alpha_h * h_lower_lower_bound + (1 - alpha_h) * h_lower_upper_bound)
        h_upper_lower_mccormick_2_lbs = (alpha_h * (- h_upper_lower_bound * h_lower_lower_bound) + (1 - alpha_h) * (- h_upper_upper_bound * h_lower_upper_bound))

        h_error_h_upper_coeff_ubs = h_upper_squared_mccormick_0_ubs - 2 * h_upper_lower_mccormick_1_lbs
        h_error_h_lower_coeff_ubs = h_lower_squared_mccormick_0_ubs - 2 * h_upper_lower_mccormick_0_lbs
        h_error_const_ubs = h_upper_squared_mccormick_1_ubs + h_lower_squared_mccormick_1_ubs - 2 * h_upper_lower_mccormick_2_lbs

        h_error_h_upper_coeff_lbs = h_upper_squared_mccormick_0_lbs - 2 * h_upper_lower_mccormick_1_ubs
        h_error_h_lower_coeff_lbs = h_lower_squared_mccormick_0_lbs - 2 * h_upper_lower_mccormick_0_ubs
        h_error_const_lbs = h_upper_squared_mccormick_1_lbs + h_lower_squared_mccormick_1_lbs - 2 * h_upper_lower_mccormick_2_ubs

        h_error_h_upper_final_coeffs_ubs = h_error_h_upper_coeff_ubs.clamp(min=0).unsqueeze(1) * h_upper_coeffs_ubs + h_error_h_upper_coeff_ubs.clamp(max=0).unsqueeze(1) * h_upper_coeffs_lbs
        h_error_h_upper_final_consts_ubs = h_error_h_upper_coeff_ubs.clamp(min=0) * h_upper_consts_ubs + h_error_h_upper_coeff_ubs.clamp(max=0) * h_upper_consts_lbs
        h_error_h_lower_final_coeffs_ubs = h_error_h_lower_coeff_ubs.clamp(min=0).unsqueeze(1) * h_lower_coeffs_ubs + h_error_h_lower_coeff_ubs.clamp(max=0).unsqueeze(1) * h_lower_coeffs_lbs
        h_error_h_lower_final_consts_ubs = h_error_h_lower_coeff_ubs.clamp(min=0) * h_lower_consts_ubs + h_error_h_lower_coeff_ubs.clamp(max=0) * h_lower_consts_lbs
        h_error_final_consts_ubs = h_error_h_upper_final_consts_ubs + h_error_h_lower_final_consts_ubs + h_error_const_ubs

        h_error_h_upper_final_coeffs_lbs = h_error_h_upper_coeff_lbs.clamp(min=0).unsqueeze(1) * h_upper_coeffs_lbs + h_error_h_upper_coeff_lbs.clamp(max=0).unsqueeze(1) * h_upper_coeffs_ubs
        h_error_h_upper_final_consts_lbs = h_error_h_upper_coeff_lbs.clamp(min=0) * h_upper_consts_lbs + h_error_h_upper_coeff_lbs.clamp(max=0) * h_upper_consts_ubs
        h_error_h_lower_final_coeffs_lbs = h_error_h_lower_coeff_lbs.clamp(min=0).unsqueeze(1) * h_lower_coeffs_lbs + h_error_h_lower_coeff_lbs.clamp(max=0).unsqueeze(1) * h_lower_coeffs_ubs
        h_error_h_lower_final_consts_lbs = h_error_h_lower_coeff_lbs.clamp(min=0) * h_lower_consts_lbs + h_error_h_lower_coeff_lbs.clamp(max=0) * h_lower_consts_ubs
        h_error_final_consts_lbs = h_error_h_upper_final_consts_lbs + h_error_h_lower_final_consts_lbs + h_error_const_lbs

        h_error_squared_upper_bound = torch.sum(h_error_h_upper_final_coeffs_ubs.clamp(min=0) * u_theta.x_ub[n_domains:] + h_error_h_upper_final_coeffs_ubs.clamp(max=0) * u_theta.x_lb[n_domains:], dim=1) +\
            torch.sum(h_error_h_lower_final_coeffs_ubs.clamp(min=0) * u_theta.x_ub[:n_domains] + h_error_h_lower_final_coeffs_ubs.clamp(max=0) * u_theta.x_lb[:n_domains], dim=1) +\
            h_error_final_consts_ubs.flatten()
        h_error_squared_lower_bound = torch.clamp(
            torch.sum(h_error_h_upper_final_coeffs_lbs.clamp(min=0) * u_theta.x_lb[n_domains:] + h_error_h_upper_final_coeffs_lbs.clamp(max=0) * u_theta.x_ub[n_domains:], dim=1) +\
            torch.sum(h_error_h_lower_final_coeffs_lbs.clamp(min=0) * u_theta.x_lb[:n_domains] + h_error_h_lower_final_coeffs_lbs.clamp(max=0) * u_theta.x_ub[:n_domains], dim=1) +\
            h_error_final_consts_lbs.flatten(),
            min=0
        )

        # perform the mccormick relaxation of v_upper * v_upper, -2 * v_upper * v_lower, v_lower * v_lower
        alpha_v = 0.5

        # h_upper^2 relaxation
        v_upper_squared_mccormick_0_ubs = v_upper_lower_bound + v_upper_upper_bound
        v_upper_squared_mccormick_1_ubs = - v_upper_lower_bound * v_upper_upper_bound

        v_upper_squared_mccormick_0_lbs = 2 * (alpha_v * v_upper_lower_bound + (1 - alpha_v) * v_upper_upper_bound)
        v_upper_squared_mccormick_1_lbs = - alpha_v * (v_upper_lower_bound)**2 - (1 - alpha_v) * (v_upper_upper_bound)**2

        # v_lower^2 relaxation
        v_lower_squared_mccormick_0_ubs = v_lower_lower_bound + v_lower_upper_bound
        v_lower_squared_mccormick_1_ubs = - v_lower_lower_bound * v_lower_upper_bound

        v_lower_squared_mccormick_0_lbs = 2 * (alpha_v * v_lower_lower_bound + (1 - alpha_v) * v_lower_upper_bound)
        v_lower_squared_mccormick_1_lbs = - alpha_v * (v_lower_lower_bound)**2 - (1 - alpha_v) * (v_lower_upper_bound)**2

        # -2 * v_upper * v_lower relaxation
        v_upper_lower_mccormick_0_ubs = (alpha_v * v_upper_upper_bound + (1 - alpha_v) * v_upper_lower_bound)
        v_upper_lower_mccormick_1_ubs = (alpha_v * v_lower_lower_bound + (1 - alpha_v) * v_lower_upper_bound)
        v_upper_lower_mccormick_2_ubs = (alpha_v * (- v_upper_upper_bound * v_lower_lower_bound) + (1 - alpha_v) * (- v_upper_lower_bound * v_lower_upper_bound))

        v_upper_lower_mccormick_0_lbs = (alpha_v * v_upper_lower_bound + (1 - alpha_v) * v_upper_upper_bound)
        v_upper_lower_mccormick_1_lbs = (alpha_v * v_lower_lower_bound + (1 - alpha_v) * v_lower_upper_bound)
        v_upper_lower_mccormick_2_lbs = (alpha_v * (- v_upper_lower_bound * v_lower_lower_bound) + (1 - alpha_v) * (- v_upper_upper_bound * v_lower_upper_bound))

        v_error_v_upper_coeff_ubs = v_upper_squared_mccormick_0_ubs - 2 * v_upper_lower_mccormick_1_lbs
        v_error_v_lower_coeff_ubs = v_lower_squared_mccormick_0_ubs - 2 * v_upper_lower_mccormick_0_lbs
        v_error_const_ubs = v_upper_squared_mccormick_1_ubs + v_lower_squared_mccormick_1_ubs - 2 * v_upper_lower_mccormick_2_lbs

        v_error_v_upper_coeff_lbs = v_upper_squared_mccormick_0_lbs - 2 * v_upper_lower_mccormick_1_ubs
        v_error_v_lower_coeff_lbs = v_lower_squared_mccormick_0_lbs - 2 * v_upper_lower_mccormick_0_ubs
        v_error_const_lbs = v_upper_squared_mccormick_1_lbs + v_lower_squared_mccormick_1_lbs - 2 * v_upper_lower_mccormick_2_ubs

        v_error_v_upper_final_coeffs_ubs = v_error_v_upper_coeff_ubs.clamp(min=0).unsqueeze(1) * v_upper_coeffs_ubs + v_error_v_upper_coeff_ubs.clamp(max=0).unsqueeze(1) * v_upper_coeffs_lbs
        v_error_v_upper_final_consts_ubs = v_error_v_upper_coeff_ubs.clamp(min=0) * v_upper_consts_ubs + v_error_v_upper_coeff_ubs.clamp(max=0) * v_upper_consts_lbs
        v_error_v_lower_final_coeffs_ubs = v_error_v_lower_coeff_ubs.clamp(min=0).unsqueeze(1) * v_lower_coeffs_ubs + v_error_v_lower_coeff_ubs.clamp(max=0).unsqueeze(1) * v_lower_coeffs_lbs
        v_error_v_lower_final_consts_ubs = v_error_v_lower_coeff_ubs.clamp(min=0) * v_lower_consts_ubs + v_error_v_lower_coeff_ubs.clamp(max=0) * v_lower_consts_lbs
        v_error_final_consts_ubs = v_error_v_upper_final_consts_ubs + v_error_v_lower_final_consts_ubs + v_error_const_ubs

        v_error_v_upper_final_coeffs_lbs = v_error_v_upper_coeff_lbs.clamp(min=0).unsqueeze(1) * v_upper_coeffs_lbs + v_error_v_upper_coeff_lbs.clamp(max=0).unsqueeze(1) * v_upper_coeffs_ubs
        v_error_v_upper_final_consts_lbs = v_error_v_upper_coeff_lbs.clamp(min=0) * v_upper_consts_lbs + v_error_v_upper_coeff_lbs.clamp(max=0) * v_upper_consts_ubs
        v_error_v_lower_final_coeffs_lbs = v_error_v_lower_coeff_lbs.clamp(min=0).unsqueeze(1) * v_lower_coeffs_lbs + v_error_v_lower_coeff_lbs.clamp(max=0).unsqueeze(1) * v_lower_coeffs_ubs
        v_error_v_lower_final_consts_lbs = v_error_v_lower_coeff_lbs.clamp(min=0) * v_lower_consts_lbs + v_error_v_lower_coeff_lbs.clamp(max=0) * v_lower_consts_ubs
        v_error_final_consts_lbs = v_error_v_upper_final_consts_lbs + v_error_v_lower_final_consts_lbs + v_error_const_lbs

        v_error_squared_upper_bound = torch.sum(v_error_v_upper_final_coeffs_ubs.clamp(min=0) * u_theta.x_ub[n_domains:] + v_error_v_upper_final_coeffs_ubs.clamp(max=0) * u_theta.x_lb[n_domains:], dim=1) +\
            torch.sum(v_error_v_lower_final_coeffs_ubs.clamp(min=0) * u_theta.x_ub[:n_domains] + v_error_v_lower_final_coeffs_ubs.clamp(max=0) * u_theta.x_lb[:n_domains], dim=1) +\
            v_error_final_consts_ubs.flatten()
        v_error_squared_lower_bound = torch.clamp(
            torch.sum(v_error_v_upper_final_coeffs_lbs.clamp(min=0) * u_theta.x_lb[n_domains:] + v_error_v_upper_final_coeffs_lbs.clamp(max=0) * u_theta.x_ub[n_domains:], dim=1) +\
            torch.sum(v_error_v_lower_final_coeffs_lbs.clamp(min=0) * u_theta.x_lb[:n_domains] + v_error_v_lower_final_coeffs_lbs.clamp(max=0) * u_theta.x_ub[:n_domains], dim=1) +\
            v_error_final_consts_lbs.flatten(),
            min=0
        )

        self.initial_conditions_intermediate_bounds = {
            'h_error_lower_bound': h_error_lower_bound,
            'h_error_upper_bound': h_error_upper_bound,
            'v_error_lower_bound': v_error_lower_bound,
            'v_error_upper_bound': v_error_upper_bound,
        }

        if any(h_error_squared_upper_bound < h_error_squared_lower_bound) or any(v_error_squared_upper_bound < v_error_squared_lower_bound):
            import pdb
            pdb.set_trace()

        # return h_error_squared_upper_bound, h_error_squared_lower_bound
        return h_error_squared_upper_bound + v_error_squared_upper_bound, h_error_squared_lower_bound + v_error_squared_lower_bound

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

        # both parts of the residual require bounding norm of u_theta^2 = (h_theta^2 + v_theta^2); do this using McCormick relaxation
        u_theta_coeffs_ubs, u_theta_consts_ubs, u_theta_coeffs_lbs, u_theta_consts_lbs = u_theta.layer_CROWN_coefficients[-1]

        h_theta_coeffs_ubs = u_theta_coeffs_ubs[:, 0]
        h_theta_consts_ubs = u_theta_consts_ubs[:, 0].unsqueeze(1)
        h_theta_coeffs_lbs = u_theta_coeffs_lbs[:, 0]
        h_theta_consts_lbs = u_theta_consts_lbs[:, 0].unsqueeze(1)
        h_theta_upper_bound = u_theta.upper_bounds[-1][:, 0].unsqueeze(1)
        h_theta_lower_bound = u_theta.lower_bounds[-1][:, 0].unsqueeze(1)

        v_theta_coeffs_ubs = u_theta_coeffs_ubs[:, 1]
        v_theta_consts_ubs = u_theta_consts_ubs[:, 1].unsqueeze(1)
        v_theta_coeffs_lbs = u_theta_coeffs_lbs[:, 1]
        v_theta_consts_lbs = u_theta_consts_lbs[:, 1].unsqueeze(1)
        v_theta_upper_bound = u_theta.upper_bounds[-1][:, 1].unsqueeze(1)
        v_theta_lower_bound = u_theta.lower_bounds[-1][:, 1].unsqueeze(1)

        alpha_h_error = 0.5
        h_theta_mccormick_0_ubs = h_theta_lower_bound + h_theta_upper_bound
        h_theta_mccormick_1_ubs = - h_theta_lower_bound * h_theta_upper_bound

        h_theta_mccormick_0_lbs = 2 * (alpha_h_error * h_theta_lower_bound + (1 - alpha_h_error) * h_theta_upper_bound)
        h_theta_mccormick_1_lbs = - alpha_h_error * (h_theta_lower_bound)**2 - (1 - alpha_h_error) * (h_theta_upper_bound)**2

        h_theta_squared_coeffs_ubs = h_theta_mccormick_0_ubs.clamp(min=0) * h_theta_coeffs_ubs + h_theta_mccormick_0_ubs.clamp(max=0) * h_theta_coeffs_lbs
        h_theta_squared_consts_ubs = h_theta_mccormick_0_ubs.clamp(min=0) * h_theta_consts_ubs + h_theta_mccormick_0_ubs.clamp(max=0) * h_theta_consts_lbs + h_theta_mccormick_1_ubs

        h_theta_squared_coeffs_lbs = h_theta_mccormick_0_lbs.clamp(min=0) * h_theta_coeffs_lbs + h_theta_mccormick_0_lbs.clamp(max=0) * h_theta_coeffs_ubs
        h_theta_squared_consts_lbs = h_theta_mccormick_0_lbs.clamp(min=0) * h_theta_consts_lbs + h_theta_mccormick_0_lbs.clamp(max=0) * h_theta_consts_ubs + h_theta_mccormick_1_lbs

        # h_theta_squared_upper_bound = torch.sum(h_theta_squared_coeffs_ubs.clamp(min=0) * self.u_theta.x_ub + h_theta_squared_coeffs_ubs.clamp(max=0) * self.u_theta.x_lb, dim=1) + h_theta_squared_consts_ubs.flatten()
        # h_theta_squared_lower_bound = torch.clamp(
        #     torch.sum(h_theta_squared_coeffs_lbs.clamp(min=0) * self.u_theta.x_lb + h_theta_squared_coeffs_lbs.clamp(max=0) * self.u_theta.x_ub, dim=1) + h_theta_squared_consts_lbs.flatten(),
        #     min=0.0
        # )

        alpha_v_error = 0.5
        v_theta_mccormick_0_ubs = v_theta_lower_bound + v_theta_upper_bound
        v_theta_mccormick_1_ubs = - v_theta_lower_bound * v_theta_upper_bound

        v_theta_mccormick_0_lbs = 2 * (alpha_v_error * v_theta_lower_bound + (1 - alpha_v_error) * v_theta_upper_bound)
        v_theta_mccormick_1_lbs = - alpha_v_error * (v_theta_lower_bound)**2 - (1 - alpha_v_error) * (v_theta_upper_bound)**2

        v_theta_squared_coeffs_ubs = v_theta_mccormick_0_ubs.clamp(min=0) * v_theta_coeffs_ubs + v_theta_mccormick_0_ubs.clamp(max=0) * v_theta_coeffs_lbs
        v_theta_squared_consts_ubs = v_theta_mccormick_0_ubs.clamp(min=0) * v_theta_consts_ubs + v_theta_mccormick_0_ubs.clamp(max=0) * v_theta_consts_lbs + v_theta_mccormick_1_ubs

        v_theta_squared_coeffs_lbs = v_theta_mccormick_0_lbs.clamp(min=0) * v_theta_coeffs_lbs + v_theta_mccormick_0_lbs.clamp(max=0) * v_theta_coeffs_ubs
        v_theta_squared_consts_lbs = v_theta_mccormick_0_lbs.clamp(min=0) * v_theta_consts_lbs + v_theta_mccormick_0_lbs.clamp(max=0) * v_theta_consts_ubs + v_theta_mccormick_1_lbs

        # v_theta_squared_upper_bound = torch.sum(v_theta_squared_coeffs_ubs.clamp(min=0) * self.u_theta.x_ub + v_theta_squared_coeffs_ubs.clamp(max=0) * self.u_theta.x_lb, dim=1) + v_theta_squared_consts_ubs.flatten()
        # v_theta_squared_lower_bound = torch.clamp(
        #     torch.sum(v_theta_squared_coeffs_lbs.clamp(min=0) * self.u_theta.x_lb + v_theta_squared_coeffs_lbs.clamp(max=0) * self.u_theta.x_ub, dim=1) + v_theta_squared_consts_lbs.flatten(),
        #     min=0.0
        # )

        u_theta_norm_coeffs_ubs = h_theta_squared_coeffs_ubs + v_theta_squared_coeffs_ubs
        u_theta_norm_consts_ubs = h_theta_squared_consts_ubs + v_theta_squared_consts_ubs
        u_theta_norm_coeffs_lbs = h_theta_squared_coeffs_lbs + v_theta_squared_coeffs_lbs
        u_theta_norm_consts_lbs = h_theta_squared_consts_lbs + v_theta_squared_consts_lbs

        u_theta_norm_upper_bound = torch.sum(u_theta_norm_coeffs_ubs.clamp(min=0) * self.u_theta.x_ub + u_theta_norm_coeffs_ubs.clamp(max=0) * self.u_theta.x_lb, dim=1) + u_theta_norm_consts_ubs.flatten()
        u_theta_norm_lower_bound = torch.clamp(
            torch.sum(u_theta_norm_coeffs_lbs.clamp(min=0) * self.u_theta.x_lb + u_theta_norm_coeffs_lbs.clamp(max=0) * self.u_theta.x_ub, dim=1) + u_theta_norm_consts_lbs.flatten(),
            min=0.0
        )
        u_theta_norm_upper_bound, u_theta_norm_lower_bound = u_theta_norm_upper_bound.unsqueeze(1), u_theta_norm_lower_bound.unsqueeze(1)

        # compute residual relaxations
        u_dt_theta_coeffs_ubs, u_dt_theta_consts_ubs = u_dt_theta.u_dxi_crown_coefficients_ubs[-1], u_dt_theta.u_dxi_crown_constants_ubs[-1]
        u_dt_theta_coeffs_lbs, u_dt_theta_consts_lbs = u_dt_theta.u_dxi_crown_coefficients_lbs[-1], u_dt_theta.u_dxi_crown_constants_lbs[-1]

        h_dt_theta_coeffs_ubs = u_dt_theta_coeffs_ubs[:, 0]
        h_dt_theta_consts_ubs = u_dt_theta_consts_ubs[:, 0].unsqueeze(1)
        h_dt_theta_coeffs_lbs = u_dt_theta_coeffs_lbs[:, 0]
        h_dt_theta_consts_lbs = u_dt_theta_consts_lbs[:, 0].unsqueeze(1)
        h_dt_theta_upper_bound = u_dt_theta.upper_bounds[-1][:, 0].unsqueeze(1)
        h_dt_theta_lower_bound = u_dt_theta.lower_bounds[-1][:, 0].unsqueeze(1)

        v_dt_theta_coeffs_ubs = u_dt_theta_coeffs_ubs[:, 1]
        v_dt_theta_consts_ubs = u_dt_theta_consts_ubs[:, 1].unsqueeze(1)
        v_dt_theta_coeffs_lbs = u_dt_theta_coeffs_lbs[:, 1]
        v_dt_theta_consts_lbs = u_dt_theta_consts_lbs[:, 1].unsqueeze(1)
        v_dt_theta_upper_bound = u_dt_theta.upper_bounds[-1][:, 1].unsqueeze(1)
        v_dt_theta_lower_bound = u_dt_theta.lower_bounds[-1][:, 1].unsqueeze(1)

        u_dxdx_theta_coeffs_ubs, u_dxdx_theta_consts_ubs = u_dxdx_theta.u_dxixi_crown_coefficients_ubs[-1], u_dxdx_theta.u_dxixi_crown_constants_ubs[-1]
        u_dxdx_theta_coeffs_lbs, u_dxdx_theta_consts_lbs = u_dxdx_theta.u_dxixi_crown_coefficients_lbs[-1], u_dxdx_theta.u_dxixi_crown_constants_lbs[-1]

        h_dxdx_theta_coeffs_ubs = u_dxdx_theta_coeffs_ubs[:, 0]
        h_dxdx_theta_consts_ubs = u_dxdx_theta_consts_ubs[:, 0].unsqueeze(1)
        h_dxdx_theta_coeffs_lbs = u_dxdx_theta_coeffs_lbs[:, 0]
        h_dxdx_theta_consts_lbs = u_dxdx_theta_consts_lbs[:, 0].unsqueeze(1)
        h_dxdx_theta_upper_bound = u_dxdx_theta.upper_bounds[-1][:, 0].unsqueeze(1)
        h_dxdx_theta_lower_bound = u_dxdx_theta.lower_bounds[-1][:, 0].unsqueeze(1)

        v_dxdx_theta_coeffs_ubs = u_dxdx_theta_coeffs_ubs[:, 1]
        v_dxdx_theta_consts_ubs = u_dxdx_theta_consts_ubs[:, 1].unsqueeze(1)
        v_dxdx_theta_coeffs_lbs = u_dxdx_theta_coeffs_lbs[:, 1]
        v_dxdx_theta_consts_lbs = u_dxdx_theta_consts_lbs[:, 1].unsqueeze(1)
        v_dxdx_theta_upper_bound = u_dxdx_theta.upper_bounds[-1][:, 1].unsqueeze(1)
        v_dxdx_theta_lower_bound = u_dxdx_theta.lower_bounds[-1][:, 1].unsqueeze(1)

        # real part of the residual is v_dt_theta - 0.5 * h_dxdx_theta - u_theta_norm * h_theta
        # start by doing a McCormick relaxation of u_theta_norm * h_theta, with x = u_theta_norm and y = h_theta
        alpha_u_theta_h = 0.5
        u_theta_norm_h_theta_lbs, u_theta_norm_h_theta_ubs = get_single_line_mccormick_coefficients(
            x_L=u_theta_norm_lower_bound,
            x_U=u_theta_norm_upper_bound,
            y_L=h_theta_lower_bound,
            y_U=h_theta_upper_bound,
            alpha_lower_bounds=alpha_u_theta_h,
            alpha_upper_bounds=alpha_u_theta_h
        )
        u_theta_norm_h_theta_0_ubs, u_theta_norm_h_theta_1_ubs, u_theta_norm_h_theta_2_ubs = u_theta_norm_h_theta_ubs
        u_theta_norm_h_theta_0_lbs, u_theta_norm_h_theta_1_lbs, u_theta_norm_h_theta_2_lbs = u_theta_norm_h_theta_lbs

        u_theta_norm_h_theta_coeffs_ubs = u_theta_norm_h_theta_0_ubs.clamp(min=0) * h_theta_coeffs_ubs + u_theta_norm_h_theta_0_ubs.clamp(max=0) * h_theta_coeffs_lbs +\
            u_theta_norm_h_theta_1_ubs.clamp(min=0) * u_theta_norm_coeffs_ubs + u_theta_norm_h_theta_1_ubs.clamp(max=0) * u_theta_norm_coeffs_lbs
        u_theta_norm_h_theta_consts_ubs = u_theta_norm_h_theta_0_ubs.clamp(min=0) * h_theta_consts_ubs + u_theta_norm_h_theta_0_ubs.clamp(max=0) * h_theta_consts_lbs +\
            u_theta_norm_h_theta_1_ubs.clamp(min=0) * u_theta_norm_consts_ubs + u_theta_norm_h_theta_1_ubs.clamp(max=0) * u_theta_norm_consts_lbs +\
            u_theta_norm_h_theta_2_ubs
        
        u_theta_norm_h_theta_coeffs_lbs = u_theta_norm_h_theta_0_lbs.clamp(min=0) * h_theta_coeffs_lbs + u_theta_norm_h_theta_0_lbs.clamp(max=0) * h_theta_coeffs_ubs +\
            u_theta_norm_h_theta_1_lbs.clamp(min=0) * u_theta_norm_coeffs_lbs + u_theta_norm_h_theta_1_lbs.clamp(max=0) * u_theta_norm_coeffs_ubs
        u_theta_norm_h_theta_consts_lbs = u_theta_norm_h_theta_0_lbs.clamp(min=0) * h_theta_consts_lbs + u_theta_norm_h_theta_0_lbs.clamp(max=0) * h_theta_consts_ubs +\
            u_theta_norm_h_theta_1_lbs.clamp(min=0) * u_theta_norm_consts_lbs + u_theta_norm_h_theta_1_lbs.clamp(max=0) * u_theta_norm_consts_ubs +\
            u_theta_norm_h_theta_2_lbs
        
        real_residual_coeffs_ubs = v_dt_theta_coeffs_ubs + (-0.5) * h_dxdx_theta_coeffs_lbs - u_theta_norm_h_theta_coeffs_lbs
        real_residual_consts_ubs = v_dt_theta_consts_ubs + (-0.5) * h_dxdx_theta_consts_lbs - u_theta_norm_h_theta_consts_lbs

        real_residual_coeffs_lbs = v_dt_theta_coeffs_lbs + (-0.5) * h_dxdx_theta_coeffs_ubs - u_theta_norm_h_theta_coeffs_ubs
        real_residual_consts_lbs = v_dt_theta_consts_lbs + (-0.5) * h_dxdx_theta_consts_ubs - u_theta_norm_h_theta_consts_ubs

        real_residual_upper_bound = torch.sum(real_residual_coeffs_ubs.clamp(min=0) * self.u_theta.x_ub + real_residual_coeffs_ubs.clamp(max=0) * self.u_theta.x_lb, dim=1) + real_residual_consts_ubs.flatten()
        real_residual_lower_bound = torch.sum(real_residual_coeffs_lbs.clamp(min=0) * self.u_theta.x_lb + real_residual_coeffs_lbs.clamp(max=0) * self.u_theta.x_ub, dim=1) + real_residual_consts_lbs.flatten()
        real_residual_upper_bound, real_residual_lower_bound = real_residual_upper_bound.unsqueeze(1), real_residual_lower_bound.unsqueeze(1)

        # imaginary part of the residual is h_dt_theta + 0.5 * v_dxdx_theta + u_theta_norm * v_theta
        alpha_u_theta_v = 0.5
        u_theta_norm_v_theta_lbs, u_theta_norm_v_theta_ubs = get_single_line_mccormick_coefficients(
            x_L=u_theta_norm_lower_bound,
            x_U=u_theta_norm_upper_bound,
            y_L=v_theta_lower_bound,
            y_U=v_theta_upper_bound,
            alpha_lower_bounds=alpha_u_theta_v,
            alpha_upper_bounds=alpha_u_theta_v
        )
        u_theta_norm_v_theta_0_ubs, u_theta_norm_v_theta_1_ubs, u_theta_norm_v_theta_2_ubs = u_theta_norm_v_theta_ubs
        u_theta_norm_v_theta_0_lbs, u_theta_norm_v_theta_1_lbs, u_theta_norm_v_theta_2_lbs = u_theta_norm_v_theta_lbs

        u_theta_norm_v_theta_coeffs_ubs = u_theta_norm_v_theta_0_ubs.clamp(min=0) * v_theta_coeffs_ubs + u_theta_norm_v_theta_0_ubs.clamp(max=0) * v_theta_coeffs_lbs +\
            u_theta_norm_v_theta_1_ubs.clamp(min=0) * u_theta_norm_coeffs_ubs + u_theta_norm_v_theta_1_ubs.clamp(max=0) * u_theta_norm_coeffs_lbs
        u_theta_norm_v_theta_consts_ubs = u_theta_norm_v_theta_0_ubs.clamp(min=0) * v_theta_consts_ubs + u_theta_norm_v_theta_0_ubs.clamp(max=0) * v_theta_consts_lbs +\
            u_theta_norm_v_theta_1_ubs.clamp(min=0) * u_theta_norm_consts_ubs + u_theta_norm_v_theta_1_ubs.clamp(max=0) * u_theta_norm_consts_lbs +\
            u_theta_norm_v_theta_2_ubs
        
        u_theta_norm_v_theta_coeffs_lbs = u_theta_norm_v_theta_0_lbs.clamp(min=0) * v_theta_coeffs_lbs + u_theta_norm_v_theta_0_lbs.clamp(max=0) * v_theta_coeffs_ubs +\
            u_theta_norm_v_theta_1_lbs.clamp(min=0) * u_theta_norm_coeffs_lbs + u_theta_norm_v_theta_1_lbs.clamp(max=0) * u_theta_norm_coeffs_ubs
        u_theta_norm_v_theta_consts_lbs = u_theta_norm_v_theta_0_lbs.clamp(min=0) * v_theta_consts_lbs + u_theta_norm_v_theta_0_lbs.clamp(max=0) * v_theta_consts_ubs +\
            u_theta_norm_v_theta_1_lbs.clamp(min=0) * u_theta_norm_consts_lbs + u_theta_norm_v_theta_1_lbs.clamp(max=0) * u_theta_norm_consts_ubs +\
            u_theta_norm_v_theta_2_lbs
        
        imag_residual_coeffs_ubs = h_dt_theta_coeffs_ubs + 0.5 * v_dxdx_theta_coeffs_ubs + u_theta_norm_v_theta_coeffs_ubs
        imag_residual_consts_ubs = h_dt_theta_consts_ubs + 0.5 * v_dxdx_theta_consts_ubs + u_theta_norm_v_theta_consts_ubs

        imag_residual_coeffs_lbs = h_dt_theta_coeffs_lbs + 0.5 * v_dxdx_theta_coeffs_lbs + u_theta_norm_v_theta_coeffs_lbs
        imag_residual_consts_lbs = h_dt_theta_consts_lbs + 0.5 * v_dxdx_theta_consts_lbs + u_theta_norm_v_theta_consts_lbs

        imag_residual_upper_bound = torch.sum(imag_residual_coeffs_ubs.clamp(min=0) * self.u_theta.x_ub + imag_residual_coeffs_ubs.clamp(max=0) * self.u_theta.x_lb, dim=1) + imag_residual_consts_ubs.flatten()
        imag_residual_lower_bound = torch.sum(imag_residual_coeffs_lbs.clamp(min=0) * self.u_theta.x_lb + imag_residual_coeffs_lbs.clamp(max=0) * self.u_theta.x_ub, dim=1) + imag_residual_consts_lbs.flatten()
        imag_residual_upper_bound, imag_residual_lower_bound = imag_residual_upper_bound.unsqueeze(1), imag_residual_lower_bound.unsqueeze(1)

        # residual norm squared is given by real_residual**2 + imag_residual**2
        alpha_real_residual_error = 0.5
        real_residual_squared_mccormick_0_ubs = real_residual_lower_bound + real_residual_upper_bound
        real_residual_squared_mccormick_1_ubs = - real_residual_lower_bound * real_residual_upper_bound

        real_residual_squared_mccormick_0_lbs = 2 * (alpha_real_residual_error * real_residual_lower_bound + (1 - alpha_real_residual_error) * real_residual_upper_bound)
        real_residual_squared_mccormick_1_lbs = - alpha_real_residual_error * (real_residual_lower_bound)**2 - (1 - alpha_real_residual_error) * (real_residual_upper_bound)**2

        real_residual_squared_coeffs_ubs = real_residual_squared_mccormick_0_ubs.clamp(min=0) * real_residual_coeffs_ubs + real_residual_squared_mccormick_0_ubs.clamp(max=0) * real_residual_coeffs_lbs
        real_residual_squared_consts_ubs = real_residual_squared_mccormick_0_ubs.clamp(min=0) * real_residual_consts_ubs + real_residual_squared_mccormick_0_ubs.clamp(max=0) * real_residual_consts_lbs + real_residual_squared_mccormick_1_ubs

        real_residual_squared_coeffs_lbs = real_residual_squared_mccormick_0_lbs.clamp(min=0) * real_residual_coeffs_lbs + real_residual_squared_mccormick_0_lbs.clamp(max=0) * real_residual_coeffs_ubs
        real_residual_squared_consts_lbs = real_residual_squared_mccormick_0_lbs.clamp(min=0) * real_residual_consts_lbs + real_residual_squared_mccormick_0_lbs.clamp(max=0) * real_residual_consts_ubs + real_residual_squared_mccormick_1_lbs

        real_residual_squared_upper_bound = torch.sum(real_residual_squared_coeffs_ubs.clamp(min=0) * self.u_theta.x_ub + real_residual_squared_coeffs_ubs.clamp(max=0) * self.u_theta.x_lb, dim=1) + real_residual_squared_consts_ubs.flatten()
        real_residual_squared_lower_bound = torch.clamp(
            torch.sum(real_residual_squared_coeffs_lbs.clamp(min=0) * self.u_theta.x_lb + real_residual_squared_coeffs_lbs.clamp(max=0) * self.u_theta.x_ub, dim=1) + real_residual_squared_consts_lbs.flatten(),
            min=0.0
        )

        alpha_imag_residual_error = 0.5
        imag_residual_squared_mccormick_0_ubs = imag_residual_lower_bound + imag_residual_upper_bound
        imag_residual_squared_mccormick_1_ubs = - imag_residual_lower_bound * imag_residual_upper_bound

        imag_residual_squared_mccormick_0_lbs = 2 * (alpha_imag_residual_error * imag_residual_lower_bound + (1 - alpha_imag_residual_error) * imag_residual_upper_bound)
        imag_residual_squared_mccormick_1_lbs = - alpha_imag_residual_error * (imag_residual_lower_bound)**2 - (1 - alpha_imag_residual_error) * (imag_residual_upper_bound)**2

        imag_residual_squared_coeffs_ubs = imag_residual_squared_mccormick_0_ubs.clamp(min=0) * imag_residual_coeffs_ubs + imag_residual_squared_mccormick_0_ubs.clamp(max=0) * imag_residual_coeffs_lbs
        imag_residual_squared_consts_ubs = imag_residual_squared_mccormick_0_ubs.clamp(min=0) * imag_residual_consts_ubs + imag_residual_squared_mccormick_0_ubs.clamp(max=0) * imag_residual_consts_lbs + imag_residual_squared_mccormick_1_ubs

        imag_residual_squared_coeffs_lbs = imag_residual_squared_mccormick_0_lbs.clamp(min=0) * imag_residual_coeffs_lbs + imag_residual_squared_mccormick_0_lbs.clamp(max=0) * imag_residual_coeffs_ubs
        imag_residual_squared_consts_lbs = imag_residual_squared_mccormick_0_lbs.clamp(min=0) * imag_residual_consts_lbs + imag_residual_squared_mccormick_0_lbs.clamp(max=0) * imag_residual_consts_ubs + imag_residual_squared_mccormick_1_lbs

        imag_residual_squared_upper_bound = torch.sum(imag_residual_squared_coeffs_ubs.clamp(min=0) * self.u_theta.x_ub + imag_residual_squared_coeffs_ubs.clamp(max=0) * self.u_theta.x_lb, dim=1) + imag_residual_squared_consts_ubs.flatten()
        imag_residual_squared_lower_bound = torch.clamp(
            torch.sum(imag_residual_squared_coeffs_lbs.clamp(min=0) * self.u_theta.x_lb + imag_residual_squared_coeffs_lbs.clamp(max=0) * self.u_theta.x_ub, dim=1) + imag_residual_squared_consts_lbs.flatten(),
            min=0.0
        )

        self.residual_intermediate_bounds = {
            'h_theta_lower_bound': h_theta_lower_bound,
            'h_theta_upper_bound': h_theta_upper_bound,
            'v_theta_lower_bound': v_theta_lower_bound,
            'v_theta_upper_bound': v_theta_upper_bound,
            'u_theta_norm_lower_bound': u_theta_norm_lower_bound,
            'u_theta_norm_upper_bound': u_theta_norm_upper_bound,
            'h_dt_theta_lower_bound': h_dt_theta_lower_bound,
            'h_dt_theta_upper_bound': h_dt_theta_upper_bound,
            'v_dt_theta_lower_bound': v_dt_theta_lower_bound,
            'v_dt_theta_upper_bound': v_dt_theta_upper_bound,
            'h_dxdx_theta_lower_bound': h_dxdx_theta_lower_bound,
            'h_dxdx_theta_upper_bound': h_dxdx_theta_upper_bound,
            'v_dxdx_theta_lower_bound': v_dxdx_theta_lower_bound,
            'v_dxdx_theta_upper_bound': v_dxdx_theta_upper_bound,
            'real_residual_lower_bound': real_residual_lower_bound,
            'real_residual_upper_bound': real_residual_upper_bound,
            'imag_residual_lower_bound': imag_residual_lower_bound,
            'imag_residual_upper_bound': imag_residual_upper_bound
        }

        return real_residual_squared_upper_bound + imag_residual_squared_upper_bound, real_residual_squared_lower_bound + imag_residual_squared_lower_bound
