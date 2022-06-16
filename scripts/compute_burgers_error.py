import scipy.io
import numpy as np

import torch

from pinn_verifier.utils import load_compliant_model

model, layers = load_compliant_model("../../trained/tanh_adam.onnx")

data = scipy.io.loadmat('../../PINNs/appendix/Data/burgers_shock.mat')
    
t = data['t'].flatten()
x = data['x'].flatten()
usol = data['usol'].flatten()

T, X = torch.meshgrid(torch.tensor(t, dtype=torch.float), torch.tensor(x, dtype=torch.float), indexing="ij")
grid_points = torch.dstack([T, X]).reshape(-1, 2)

outputs = model(grid_points)

import pdb
pdb.set_trace()