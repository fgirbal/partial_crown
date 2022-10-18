import argparse

from pyDOE import lhs
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn.functional as F


class Model(torch.nn.Module):
    def __init__(self, lb: torch.tensor, ub: torch.tensor, num_hidden_layers: int, num_neurons_per_layer: int, activation: torch.nn.Module = torch.nn.Softplus()) -> None:
        super().__init__()
        self.lb = lb
        self.ub = ub

        self.layers = torch.nn.ModuleDict()
        self.activation = activation

        self.layers['first_layer'] = torch.nn.Linear(2, num_neurons_per_layer)
        for i in range(num_hidden_layers):
            self.layers[f'hidden_{i}'] = torch.nn.Linear(num_neurons_per_layer, num_neurons_per_layer)
        self.layers['last_fc'] = torch.nn.Linear(num_neurons_per_layer, 1)
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

def model_get_residual(model, X_r, viscosity):
    t, x = X_r[:, 0:1], X_r[:, 1:2]
    t.requires_grad_()
    x.requires_grad_()

    u = model(torch.hstack([t, x]))

    u_x = torch.autograd.grad(u.sum(), x, create_graph=True, retain_graph=True, allow_unused=True)[0]
    u_t = torch.autograd.grad(u.sum(), t, create_graph=True, retain_graph=True, allow_unused=True)[0]

    u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True, retain_graph=True, allow_unused=True)[0]

    return u_t + u * u_x - viscosity * u_xx

def compute_loss(model, X_r, X_data, u_data, residual_fn):
    # Compute phi^r
    r = residual_fn(model, X_r)
    phi_r = torch.mean(r**2)
    
    # Initialize loss
    loss = phi_r

    # Add phi^0 and phi^b to the loss
    for i in range(len(X_data)):
        loss += torch.mean((u_data[i] - model(X_data[i]))**2)
 
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
        choices=["adam", "lbfgs"],
        default="lbfgs",
        help='optimizer to use in training'
    )
    args = parser.parse_args()

    # Set constants
    viscosity = .01/np.pi

    # Define initial condition
    def fun_u_0(x):
        return - torch.sin(np.pi * x)

    # Define boundary condition
    def fun_u_b(t, x):
        n = x.shape[0]
        return torch.zeros((n,1))

    residual_fn = lambda model, X_r: model_get_residual(model, X_r, viscosity=viscosity)

    # Set number of data points
    N_0 = 50
    N_b = 50
    N_r = 10000

    # Set boundary
    tmin = 0.
    tmax = 1.
    xmin = -1.
    xmax = 1.

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
    u_0 = fun_u_0(x_0)

    # Boundary data
    t_b = torch.zeros((N_b,1)).uniform_(lb[0], ub[0])
    x_b = lb[1] + (ub[1] - lb[1]) * torch.bernoulli(0.5*torch.ones([N_b, 1]))
    X_b = torch.concat([t_b, x_b], axis=1)

    # Evaluate boundary condition at (t_b,x_b)
    u_b = fun_u_b(t_b, x_b)

    # Draw uniformly sampled collocation points
    # t_r = torch.zeros((N_r,1)).uniform_(lb[0], ub[0])
    # x_r = torch.zeros((N_r,1)).uniform_(lb[1], ub[1])
    # X_r = torch.concat([t_r, x_r], axis=1)

    X_r = lb + (ub - lb)*lhs(2, N_r)
    X_r = X_r.to(torch.float32)
    X_r = torch.concat([X_r, X_0, X_b], axis=0)

    # Collect boundary and inital data in lists
    X_data = [X_0, X_b]
    u_data = [u_0, u_b]

    def plot_collocation_points():
        fig = plt.figure(figsize=(9,6))
        plt.scatter(t_0, x_0, c=u_0, marker='X', vmin=-1, vmax=1)
        plt.scatter(t_b, x_b, c=u_b, marker='X', vmin=-1, vmax=1)
        plt.scatter(t_r, x_r, c='r', marker='.', alpha=0.1)
        plt.xlabel('$t$')
        plt.ylabel('$x$')

        plt.title('Positions of collocation points and boundary data')
        plt.show()

    # plot_collocation_points()

    if args.activation == "softplus":
        activation_fn = torch.nn.Softplus()
    elif args.activation == "tanh":
        activation_fn = torch.nn.Tanh()
    elif args.activation == "relu":
        print("Warning - ReLU training does not usually work for PINNs!")
        activation_fn = torch.nn.ReLU()

    model = Model(
        lb,
        ub,
        num_hidden_layers=8,
        num_neurons_per_layer=20,
        activation=activation_fn
    )

    def init_weights(m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_normal_(m.weight)

    model.apply(init_weights)

    if args.optimizer == "adam":
        optim = torch.optim.Adam(model.parameters(), lr=0.01)
        # optim = torch.optim.SGD(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, [2000, 10000], gamma=0.1)
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, [1000], gamma=0.1)
    elif args.optimizer == "lbfgs":
        optim = torch.optim.LBFGS(
            model.parameters(),
            lr=1.0,
            max_iter=50000, 
            max_eval=50000, 
            history_size=50,
            tolerance_grad=1e-5, 
            tolerance_change=1.0 * np.finfo(float).eps,
            line_search_fn="strong_wolfe"       # can be "strong_wolfe"
        )
        scheduler = None

    # Number of training epochs
    N = 12500
    hist = []

    if args.optimizer == "adam":
        for epoch in range(N+1):
            optim.zero_grad()
            loss = compute_loss(model, X_r, X_data, u_data, residual_fn)
            loss.backward()
            
            optim.step()
            
            # Append current loss to hist
            hist.append(loss.detach().numpy())
            
            # Output current loss after 50 iterates
            if epoch % 50 == 0:
                if scheduler:
                    print(f'Iteration {epoch}: loss = {loss}, lr = {scheduler.get_last_lr()[0]}')
                else:
                    print(f'Iteration {epoch}: loss = {loss}')

            if scheduler:
                scheduler.step()
    else:
        global iter
        iter = 0

        def loss_function_closure():
            optim.zero_grad()
            loss = compute_loss(model, X_r, X_data, u_data, residual_fn)
            loss.backward()
            global iter

            if iter % 50 == 0:
                print(f'Iteration {iter}: loss = {loss.item()}')
            
            iter += 1

            return loss

        loss = optim.step(loss_function_closure)

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
