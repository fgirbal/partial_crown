import argparse

from pyDOE import lhs
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn.functional as F


class SchrodingerModel(torch.nn.Module):
    def __init__(self, lb: torch.tensor, ub: torch.tensor, num_hidden_layers: int, num_neurons_per_layer: int, activation: torch.nn.Module = torch.nn.Softplus()) -> None:
        super().__init__()
        self.lb = lb
        self.ub = ub

        self.layers = torch.nn.ModuleDict()
        self.activation = activation

        self.layers['first_layer'] = torch.nn.Linear(2, num_neurons_per_layer)
        for i in range(num_hidden_layers):
            self.layers[f'hidden_{i}'] = torch.nn.Linear(num_neurons_per_layer, num_neurons_per_layer)
        self.layers['last_fc'] = torch.nn.Linear(num_neurons_per_layer, 2)
        self.num_hidden_layers = num_hidden_layers

    def forward(self, x):
        # Scale the input to [lb, ub]
        x = 2.0 * (x - self.lb)/(self.ub - self.lb) - 1.0
        x = self.activation(self.layers['first_layer'](x))

        # Add hidden layers
        for layer_i in range(self.num_hidden_layers):
            layer = self.layers[f'hidden_{layer_i}']
            x = self.activation(layer(x))
        
        x = self.layers['last_fc'](x)
        return x

def model_get_residual(model, X_r):
    t, x = X_r[:, 0:1], X_r[:, 1:2]
    t.requires_grad_()
    x.requires_grad_()

    h = model(torch.hstack([t, x]))
    u, v = h[:, 0:1], h[:, 1:2]

    u_x = torch.autograd.grad(u.sum(), x, create_graph=True, retain_graph=True, allow_unused=True)[0]
    v_x = torch.autograd.grad(v.sum(), x, create_graph=True, retain_graph=True, allow_unused=True)[0]

    u_t = torch.autograd.grad(u.sum(), t, create_graph=True, retain_graph=True, allow_unused=True)[0]
    v_t = torch.autograd.grad(v.sum(), t, create_graph=True, retain_graph=True, allow_unused=True)[0]

    u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True, retain_graph=True, allow_unused=True)[0]
    v_xx = torch.autograd.grad(v_x.sum(), x, create_graph=True, retain_graph=True, allow_unused=True)[0]

    f_u = u_t + 0.5*v_xx + (u**2 + v**2)*v
    f_v = v_t - 0.5*u_xx - (u**2 + v**2)*u

    return f_u, f_v

def compute_loss(model, X_r, X_data, u_data, X_b):
    # Compute phi^r
    f_u, f_v = model_get_residual(model, X_r)
    loss = torch.mean(f_u**2 + f_v**2)

    # Add phi^0 terms to the loss
    for i in range(len(X_data)):
        initial_data_outputs = model(X_data[i])
        initial_u, initial_v = initial_data_outputs[:, 0:1], initial_data_outputs[:, 1:2]

        loss += torch.mean((u_data[i][0] - initial_u)**2)
        loss += torch.mean((u_data[i][1] - initial_v)**2)

    # Add phi^b terms to the loss
    t_b = X_b[0][:, 0:1]
    t_b.requires_grad_()

    x_lb_b = X_b[0][:, 1:2]
    x_lb_b.requires_grad_()

    x_ub_b = X_b[1][:, 1:2]
    x_ub_b.requires_grad_()

    h_lb = model(torch.hstack([t_b, x_lb_b]))
    u_lb, v_lb = h_lb[:, 0:1], h_lb[:, 1:2]
    h_ub = model(torch.hstack([t_b, x_ub_b]))
    u_ub, v_ub = h_ub[:, 0:1], h_ub[:, 1:2]

    u_x_lb = torch.autograd.grad(u_lb.sum(), x_lb_b, create_graph=True, retain_graph=True, allow_unused=True)[0]
    v_x_lb = torch.autograd.grad(v_lb.sum(), x_lb_b, create_graph=True, retain_graph=True, allow_unused=True)[0]

    u_x_ub = torch.autograd.grad(u_ub.sum(), x_ub_b, create_graph=True, retain_graph=True, allow_unused=True)[0]
    v_x_ub = torch.autograd.grad(v_ub.sum(), x_ub_b, create_graph=True, retain_graph=True, allow_unused=True)[0]

    loss += ((h_lb - h_ub)**2).mean()
    loss += ((u_x_lb - u_x_ub)**2).mean()
    loss += ((v_x_lb - v_x_ub)**2).mean()

    return loss

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-filename', required=True, type=str, help='where to write the onnx model')
    parser.add_argument(
        '--activation',
        type=str,
        choices=["tanh", "softplus", "relu"],
        default="tanh",
        help='type of activation to use for the FC network'
    )
    parser.add_argument(
        '--optimizer',
        type=str,
        choices=["adam", "lbfgs", "adam+lbfgs"],
        default="lbfgs",
        help='optimizer to use in training'
    )
    args = parser.parse_args()

    def fun_u_0(x):
        return 2 * (1 / torch.cosh(x))

    residual_fn = lambda model, X_r: model_get_residual(model, X_r)

    # Set number of data points
    N_0 = 50
    N_b = 50
    N_r = 20000

    # Set boundary
    tmin = 0.0
    tmax = np.pi/2
    xmin = -5.0
    xmax = 5.0

    # Lower bounds
    lb = torch.Tensor([tmin, xmin])
    # Upper bounds
    ub = torch.Tensor([tmax, xmax])

    # Set random seed for reproducible results
    torch.random.manual_seed(0)

    # Draw uniform sample points for initial boundary data
    t_0 = torch.ones((N_0,1))*lb[0]
    # x_0 = torch.zeros((N_0,1)).uniform_(lb[1], ub[1])
    x_0 = torch.linspace(lb[1], ub[1], N_0).reshape(-1, 1)
    X_0 = torch.concat([t_0, x_0], axis=1)

    # Evaluate intitial condition at x_0
    h_0_u = fun_u_0(x_0)

    # Boundary data
    t_b = torch.zeros((N_b,1)).uniform_(lb[0], ub[0])
    lb_x_b = torch.ones((N_b,1)) * lb[1]
    ub_x_b = torch.ones((N_b,1)) * ub[1]
    lb_b = torch.concat([t_b, lb_x_b], axis=1)
    ub_b = torch.concat([t_b, ub_x_b], axis=1)

    X_r = lb + (ub - lb)*lhs(2, N_r)
    X_r = X_r.to(torch.float32)
    X_r = torch.concat([X_r, X_0], axis=0)

    # Collect boundary and inital data in lists
    X_data = [X_0]
    u_data = [(h_0_u, torch.zeros(N_0, 1))]

    if args.activation == "softplus":
        activation_fn = torch.nn.Softplus()
    elif args.activation == "tanh":
        activation_fn = torch.nn.Tanh()
    elif args.activation == "relu":
        print("Warning - ReLU training does not usually work for PINNs!")
        activation_fn = torch.nn.ReLU()

    model = SchrodingerModel(
        lb,
        ub,
        num_hidden_layers=5,
        num_neurons_per_layer=100,
        activation=activation_fn
    )

    def init_weights(m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_normal_(m.weight)

    model.apply(init_weights)

    if args.optimizer == "adam":
        N = 15000
        adam_optim = torch.optim.Adam(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(adam_optim, [2500, 12000], gamma=0.1)
        lbfgs_optim = None
    elif args.optimizer == "lbfgs":
        N = 0
        adam_optim = None
        lbfgs_optim = torch.optim.LBFGS(
            model.parameters(),
            lr=1.0,
            max_iter=50000, 
            max_eval=50000, 
            history_size=50,
            tolerance_grad=1e-5, 
            tolerance_change=1.0 * np.finfo(float).eps,
            line_search_fn="strong_wolfe"       # can be "strong_wolfe"
        )
    elif args.optimizer == "adam+lbfgs":
        N = 1000
        adam_optim = torch.optim.Adam(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(adam_optim, [250, 750], gamma=0.1)
        lbfgs_optim = torch.optim.LBFGS(
            model.parameters(),
            lr=1.0,
            max_iter=50000, 
            max_eval=50000, 
            history_size=50,
            tolerance_grad=1e-5, 
            tolerance_change=1.0 * np.finfo(float).eps,
            line_search_fn="strong_wolfe"       # can be "strong_wolfe"
        )

    # Number of training epochs
    hist = []

    import time
    start_time = time.time()

    if adam_optim:
        print(f"Optimizing using Adam for {N} epochs...")
        for epoch in range(N+1):
            adam_optim.zero_grad()
            loss = compute_loss(model, X_r, X_data, u_data, [lb_b, ub_b])
            loss.backward()
            
            adam_optim.step()
            
            # Append current loss to hist
            hist.append(loss.detach().numpy())
            
            # Output current loss after 50 iterates
            if epoch % 50 == 0:
                if scheduler:
                    print(f'Adam, iteration {epoch}: loss = {loss}, lr = {scheduler.get_last_lr()[0]}')
                else:
                    print(f'Adam, iteration {epoch}: loss = {loss}')

            if scheduler:
                scheduler.step()
    
    if lbfgs_optim:
        print("Optimizing using LBFGS...")
        global iter
        iter = 0

        def loss_function_closure():
            lbfgs_optim.zero_grad()
            loss = compute_loss(model, X_r, X_data, u_data, [lb_b, ub_b])
            loss.backward()
            global iter

            if iter % 50 == 0:
                print(f'LBGFS, iteration {iter}: loss = {loss.item()}')
            
            iter += 1

            return loss

        loss = lbfgs_optim.step(loss_function_closure)

    print(f"Training time: {time.time() - start_time} seconds")

    model.eval()

    x = torch.ones([1, 2])
    torch.onnx.export(
        model,               # model being run
        x,                         # model input (or a tuple for multiple inputs)
        args.output_filename,   # where to save the model (can be a file or file-like object)
        export_params=True,        # store the trained parameter weights inside the model file
        opset_version=10,          # the ONNX version to export the model to
        do_constant_folding=True,  # whether to execute constant folding for optimization
        input_names = ['input'],   # the model's input names
        output_names = ['output'], # the model's output names
        dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                    'output' : {0 : 'batch_size'}}
    )

    import pdb
    pdb.set_trace()
