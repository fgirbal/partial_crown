"""
Library of generally useful functions.
"""
import copy
from typing import Tuple, List

import torch

import tools.bab_tools.vnnlib_utils as vnnlib_utils


def load_compliant_model(model_path: str) -> Tuple[torch.nn.Module, List[torch.nn.Module]]:
    assert model_path.endswith(".onnx")

    model, in_shape, out_shape, dtype, model_correctness = vnnlib_utils.onnx_to_pytorch(model_path)
    model.eval()

    if not model_correctness:
        raise ValueError(
            "model has not been loaded successfully; some operations are not compatible, please check manually"
        )

    # should fail if some maxpools are actually in the network
    layers = vnnlib_utils.remove_maxpools(copy.deepcopy(list(model.children())), [], dtype=dtype)

    return model, layers
