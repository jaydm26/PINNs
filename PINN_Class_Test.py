import numpy as np
import torch
import torch.nn as nn

np.random.seed(2608)
torch.manual_seed(2608)

layers = [2, 20, 20, 20, 20, 20, 20, 20, 20, 1]
model = nn.Sequential()
for l in range(0, len(layers) - 1):
    model.add_module("layer_"+str(l), nn.Linear(layers[l],layers[l+1], bias=True))
    model.add_module("tanh_"+str(l), nn.Tanh())

def initialize_NN(m):
    """
    Initialize the neural network with the required layers, the weights and the biases. The input "layers" in an array that contains the number of nodes (neurons) in each layer.
    """

    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
        # print(m.weight)
model.apply(initialize_NN)

def net_u(model, x, t):
    """
    Forward pass through the network to obtain the U field.
    """

    u = model(torch.cat((x,t)))
    return u

def net_f(model, x, t):
    u = net_u(model, x, t)
    u_x = torch.autograd.grad(u, x, create_graph = True)
    u_xx = torch.autograd.grad(u_x, x, create_graph = True)
    u_t = torch.autograd.grad(u,t, create_graph = True)

    f = u_t[0] + u * u_x[0] - 0.01 * u_xx[0]

    return f

def calc_loss(u_pred, u_tf, f_pred):
    losses = torch.mean(torch.mul(u_pred - u_tf, u_pred - u_tf)) + torch.mean(torch.mul(f_pred, f_pred))
    return losses

optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)

x = torch.tensor([0.1],requires_grad=True)
t = torch.tensor([0.],requires_grad=True)
u_true = torch.tensor([1.],requires_grad=True)

for epoch in range(0,100):
    u_pred = net_u(model,x,t)
    f_pred = net_f(model,x,t)

    loss = calc_loss(u_pred,u_true,f_pred)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(u_pred, f_pred)
