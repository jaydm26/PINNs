"""
@author: Jay Mehta
"""


import sys
# Include the path that contains a number of files that have txt files containing solutions to the Burger's problem.
sys.path.insert(0,'../../Utilities/')

import os
os.getcwd()


# Import required modules
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.interpolate import griddata
from pyDOE import lhs
from mpl_toolkits.mplot3d import Axes3D
import time
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable


np.random.seed(1234)
torch.manual_seed(1234)


class PhysicsInformedNN:
    # Initialize the class
    """
    This class defined the Physics Informed Neural Network. The class is first initialized by the __init__ function. Additional functions related to the class are also defined subsequently.
    """

    def __init__(self, X_u, u, X_f, layers, lb, ub, nu, epochs):

        # Defining the lower and upper bound of the domain.
        self.lb = lb
        self.ub = ub
        self.epochs = epochs

        # X_u = 2.0 * (X_u - self.lb)/(self.ub - self.lb) - 1.0
        # X_f = 2.0 * (X_f - self.lb)/(self.ub - self.lb) - 1.0

        #$ Define the initial conditions for X and t
        self.x_u = torch.tensor(X_u[:,0:1]).float()
        self.x_u.requires_grad = True
        self.t_u = torch.tensor(X_u[:,1:2]).float()
        self.t_u.requires_grad = True

        #$ Define the final conditions for X and t
        self.x_f = torch.tensor(X_f[:,0:1]).float()
        self.x_f.requires_grad = True
        self.t_f = torch.tensor(X_f[:,1:2]).float()
        self.t_f.requires_grad = True

        # Declaring the field for the variable to be solved for
        self.u = torch.tensor(u).float()
        self.u_true = torch.tensor(u).float()

        # Declaring the number of layers in the Neural Network
        self.layers = layers
        # Defininf the diffusion constant in the problem (?)
        self.nu = torch.tensor(nu)

        # Create the structure of the neural network here, or build a function below to build the architecture and send the model here.

        self.model = self.neural_net(layers)

        # Define the initialize_NN function to obtain the initial weights and biases for the network.
        self.model.apply(self.initialize_NN)

        # Select the optimization method for the network. Currently, it is just a placeholder.

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr = 0.01)

        self.losses = []
        # train(model,epochs,self.x_u_tf,self.t_u_tf,self.x_f_tf,self.t_f_tf,self.u_tf)

    def neural_net(self, layers):
        """
        A function to build the neural network of the required size using the weights and biases provided. Instead of doing this, can we use a simple constructor method and initalize them post the construction? That would be sensible and faster.
        """
        model = nn.Sequential()
        for l in range(0, len(layers) - 1):
            model.add_module("layer_"+str(l), nn.Linear(layers[l],layers[l+1], bias=True))
            if l != len(layers) - 2:
                model.add_module("tanh_"+str(l), nn.Tanh())

        return model


    def initialize_NN(self, m):
        """
        Initialize the neural network with the required layers, the weights and the biases. The input "layers" in an array that contains the number of nodes (neurons) in each layer.
        """

        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
            # print(m.weight)


    def net_u(self, x, t):
        """
        Forward pass through the network to obtain the U field.
        """

        u = self.model(torch.cat((x,t),1))
        return u


    def net_f(self, x, t):
        u = self.net_u(x, t)
        u_x = torch.autograd.grad(u, x, grad_outputs = torch.ones([len(x),1], dtype = torch.float), create_graph = True)
        u_xx = torch.autograd.grad(u_x, x, grad_outputs = torch.ones([len(x),1], dtype = torch.float), create_graph = True)
        u_t = torch.autograd.grad(u,t,grad_outputs = torch.ones([len(t),1], dtype = torch.float), create_graph = True)

        f = u_t[0] - self.nu * u_xx[0]

        return f

    def calc_loss(self, u_pred, u_true, f_pred):
        u_error = u_pred - u_true
        loss_u = torch.mean(torch.mul(u_error, u_error))
        loss_f = torch.mean(torch.mul(f_pred, f_pred))
        losses = loss_u + loss_f
        print('Loss: %.4f, U_loss: %.4f, F_loss: %.4f' %(losses, loss_u, loss_f))
        return losses


    def set_optimizer(self,optimizer):
        self.optimizer = optimizer


    def train(self):

        for epoch in range(0,self.epochs):
            # Now, one can perform a forward pass through the network to predict the value of u and f for various locations of x and at various times t. The function to call here is net_u and net_f.

            # Here it is crucial to remember to provide x and t as columns and not as rows. Concatenation in the prediction step will fail otherwise.

            u_pred = self.net_u(self.x_u, self.t_u)
            f_pred = self.net_f(self.x_f, self.t_f)

            # Now, we can define the loss of the network. The loss here is broken into two components: one is the loss due to miscalculating the predicted value of u, the other is for not satisfying the physical governing equation in f which must be equal to 0 at all times and all locations (strong form).

            loss = self.calc_loss(u_pred, self.u_true, f_pred)
            self.losses.append(loss.detach().numpy())

            # Clear out the previous gradients
            self.optimizer.zero_grad()

            # Calculate the gradients using the backward() method.

            loss.backward() # Here, a tensor may need to be passed so that the gradients can be calculated.

            # Optimize the parameters through the optimization step and the learning rate.

            self.optimizer.step()

            # Repeat the prediction, calculation of losses, and optimization a number of times to optimize the network.

    def closure(self):
        self.optimizer.zero_grad()
        u_pred = self.net_u(self.x_u, self.t_u)
        f_pred = self.net_f(self.x_f, self.t_f)
        loss = self.calc_loss(u_pred, self.u_true, f_pred)
        loss.backward()
        return loss

if __name__ == "__main__":

    noise = 0.0
    n_epochs = 100

    N_u = 100
    N_f = 10000

    # Layer Map

    layers = [2, 20, 20, 20, 20, 20, 20, 20, 20, 1]

    data = scipy.io.loadmat('Fluid Flows/uSol.mat')
    Exact = np.fliplr(data['u_true_store']).T
    dt = data['dt']
    dx = data['dx']
    nu = data['nu']

    t = np.array([x*dt for x in range(0,Exact.shape[1])]).flatten()[:,None]
    x = np.array([x*dx for x in range(0,Exact.shape[0])]).flatten()[:,None]
    t.shape
    T, X = np.meshgrid(t, x)
    X_star = np.hstack((X.flatten()[:,None],T.flatten()[:,None]))
    u_star = Exact.flatten()[:,None]


    # Doman bounds
    lb = X_star.min(0)
    ub = X_star.max(0)

    Exact[-1:,:].T
    xx1 = np.hstack((X[0:1,:].T, T[0:1,:].T)) # Boundary condition x = 0, for all time t
    uu1 = Exact[0:1,:].T
    xx2 = np.hstack((X[:,0:1], T[:,0:1])) # Initial condition for all values of x
    uu2 = Exact[:,0:1]
    xx3 = np.hstack((X[-1:,:].T, T[-1:,:].T))
    uu3 = Exact[-1:,:].T

    X_u_train = np.vstack([xx1, xx2, xx3])
    X_f_train = lb + (ub-lb)*lhs(2, N_f)
    X_f_train = np.vstack((X_f_train, X_u_train))
    u_train = np.vstack([uu1, uu2, uu3])

    idx = np.random.choice(X_u_train.shape[0], N_u, replace=False)
    X_u_train = X_u_train[idx, :]
    u_train = u_train[idx,:]


    for i in range(0,u_train.shape[0]):
        print(X_u_train[i,:],u_train[i])

    pinn = PhysicsInformedNN(X_u_train,u_train,X_f_train,layers,lb,ub,nu,n_epochs)

pinn.model

pinn.set_optimizer(torch.optim.Adam(pinn.model.parameters(),lr = 1e-4))

for _ in range(0,5):
    pinn.train()

plt.plot(np.linspace(0,len(pinn.losses),num=len(pinn.losses)),pinn.losses)

u_pred = pinn.model(torch.tensor(X_star).float())


u_pred2 = pinn.net_u(torch.tensor(X_star[:,0:1]).float(), torch.tensor(X_star[:,1:2]).float())

u_pred == u_pred2 # Sanity Check

np.linalg.norm(u_star - u_pred.detach().numpy(),2)/np.linalg.norm(u_star,2)

U_pred = griddata(X_star, u_pred.detach().numpy().flatten(), (X, T), method='cubic')
np.mean(np.abs(Exact-U_pred))/np.mean(np.abs(Exact))
pinn.losses[-1]

# %%
"""
Contour plot of the Exact Solution
"""

plt.contourf(T,X,Exact)
# %%

# %%
"""
Contour plot of the Predicted Solution
"""
plt.contourf(T,X,u_pred.detach().numpy().reshape([257,3427]))
# %%
