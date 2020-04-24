"""
@author: Jay Mehta

Based on the work of Maziar Raissi
"""


import sys
# Include the path that contains a number of files that have txt files containing solutions to the Burger's problem.
sys.path.insert(0,'../../Utilities/')


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

    def __init__(self, X, u, layers, lb, ub,  epochs):

        # Defining the lower and upper bound of the domain.
        self.lb = lb
        self.ub = ub
        self.epochs = epochs

        # X_u = 2.0 * (X_u - self.lb)/(self.ub - self.lb) - 1.0
        # X_f = 2.0 * (X_f - self.lb)/(self.ub - self.lb) - 1.0

        #$ Define the initial conditions for X and t
        self.x = torch.tensor(X[:,0:1]).float()
        self.x.requires_grad = True
        self.t = torch.tensor(X[:,1:2]).float()
        self.t.requires_grad = True

        # Declaring the field for the variable to be solved for
        self.u = torch.tensor(u).float()
        self.u_true = torch.tensor(u).float()

        # Declaring the number of layers in the Neural Network
        self.layers = layers
        # Defining the diffusion constant in the problem
        self.lambda_1 = torch.tensor([0.0], requires_grad = True)
        self.lambda_2 = torch.tensor([-6.0], requires_grad = True)

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

        f = u_t[0] + self.lambda_1 * u * u_x[0] - torch.exp(self.lambda_2) * u_xx[0]

        return f

    def calc_loss(self, u_pred, u_true, f_pred):
        u_error = u_pred - u_true
        loss_u = torch.mean(torch.mul(u_error, u_error))
        loss_f = torch.mean(torch.mul(f_pred, f_pred))
        losses = loss_u + loss_f
        print('Loss: %.4f, U_loss: %.4f, F_loss: %.4f' %(losses, loss_u, loss_f))
        print('Lambda 1: %.4f, Lambda 2: %.4f \n' %(self.lambda_1, self.lambda_2))
        return losses


    def set_optimizer(self,optimizer):
        self.optimizer = optimizer


    def train(self):

        for epoch in range(0,self.epochs):
            # Now, one can perform a forward pass through the network to predict the value of u and f for various locations of x and at various times t. The function to call here is net_u and net_f.

            # Here it is crucial to remember to provide x and t as columns and not as rows. Concatenation in the prediction step will fail otherwise.

            u_pred = self.net_u(self.x, self.t)
            f_pred = self.net_f(self.x, self.t)

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

    # nu = 0.01/np.pi
    noise = 0.0
    n_epochs = 100

    N_u = 2000
    # N_f = 10000

    # Layer Map

    layers = [2, 20, 20, 20, 20, 20, 20, 20, 20, 1]

    data = scipy.io.loadmat('appendix/Data/burgers_shock.mat')

    t = data['t'].flatten()[:,None]
    x = data['x'].flatten()[:,None]
    Exact = np.real(data['usol']).T

    X, T = np.meshgrid(x,t)
    X_star = np.hstack((X.flatten()[:,None],T.flatten()[:,None]))
    u_star = Exact.flatten()[:,None]

    # Doman bounds
    lb = X_star.min(0)
    ub = X_star.max(0)

    idx = np.random.choice(X_star.shape[0], N_u, replace=False)
    X_u_train = X_star[idx, :]
    u_train = u_star[idx,:]

    pinn = PhysicsInformedNN(X_u_train,u_train,layers,lb,ub,n_epochs)

pinn.model

# Run this model with SGD for 2K-3K epochs. Then switch to Adam. Use default Learning Rates. You should observe a good convergence by the time you 25K epochs.
pinn.set_optimizer(torch.optim.Adam([{'params': pinn.model.parameters()}, {'params': pinn.lambda_1}, {'params': pinn.lambda_2}],lr = 1e-3))

for _ in range(0,40):
    pinn.train()
plt.plot(np.linspace(0,len(pinn.losses),num=len(pinn.losses)),pinn.losses)

u_pred = pinn.model(torch.tensor(X_star).float())

u_pred2 = pinn.net_u(torch.tensor(X_star[:,0:1]).float(), torch.tensor(X_star[:,1:2]).float())

u_pred == u_pred2 # Sanity Check

np.linalg.norm(u_star - pinn.model(torch.tensor(X_star).float()).detach().numpy(),2)/np.linalg.norm(u_star,2)

U_pred = griddata(X_star, pinn.model(torch.tensor(X_star).float()).detach().numpy().flatten(), (X, T), method='cubic')
np.mean(np.abs(Exact-U_pred))/np.mean(np.abs(Exact))
pinn.losses[-1], pinn.lambda_1, torch.exp(pinn.lambda_2)

# %%
"""
Contour plot of the Exact Solution
"""
plt.contourf(X,T,Exact)
# %%

# %%
"""
Contour plot of the Predicted Solution
"""
plt.contourf(X,T,u_pred.detach().numpy().reshape([100,256]))
# %%


fig,ax = plt.subplots()

ax.axis('off')

gs0 = gridspec.GridSpec(1, 2)
gs0.update(top=1-0.06, bottom=1-1/3, left=0.15, right=0.85, wspace=0)
ax = plt.subplot(gs0[:, :])
h = ax.imshow(U_pred.T, interpolation='nearest', cmap='rainbow',
              extent=[t.min(), t.max(), x.min(), x.max()],
              origin='lower', aspect='auto')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(h, cax=cax)
ax.plot(X_u_train[:,1], X_u_train[:,0], 'kx', label = 'Data (%d points)' % (u_train.shape[0]), markersize = 4, clip_on = False)
line = np.linspace(x.min(), x.max(), 2)[:,None]
ax.plot(t[25]*np.ones((2,1)), line, 'w-', linewidth = 1)
ax.plot(t[50]*np.ones((2,1)), line, 'w-', linewidth = 1)
ax.plot(t[75]*np.ones((2,1)), line, 'w-', linewidth = 1)
ax.set_xlabel('$t$')
ax.set_ylabel('$x$')
ax.legend(frameon=False, loc = 'best')
ax.set_title('$u(t,x)$', fontsize = 10)

gs1 = gridspec.GridSpec(1, 3)
gs1.update(top=1-1/3, bottom=0, left=0.1, right=0.9, wspace=0.5)

ax = plt.subplot(gs1[0, 0])
ax.plot(x,Exact[25,:], 'b-', linewidth = 2, label = 'Exact')
ax.plot(x,U_pred[25,:], 'r--', linewidth = 2, label = 'Prediction')
ax.set_xlabel('$x$')
ax.set_ylabel('$u(t,x)$')
ax.set_title('$t = 0.25$', fontsize = 10)
ax.axis('square')
ax.set_xlim([-1.1,1.1])
ax.set_ylim([-1.1,1.1])

ax = plt.subplot(gs1[0, 1])
ax.plot(x,Exact[50,:], 'b-', linewidth = 2, label = 'Exact')
ax.plot(x,U_pred[50,:], 'r--', linewidth = 2, label = 'Prediction')
ax.set_xlabel('$x$')
ax.set_ylabel('$u(t,x)$')
ax.axis('square')
ax.set_xlim([-1.1,1.1])
ax.set_ylim([-1.1,1.1])
ax.set_title('$t = 0.50$', fontsize = 10)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.35), ncol=5, frameon=False)

ax = plt.subplot(gs1[0, 2])
ax.plot(x,Exact[75,:], 'b-', linewidth = 2, label = 'Exact')
ax.plot(x,U_pred[75,:], 'r--', linewidth = 2, label = 'Prediction')
ax.set_xlabel('$x$')
ax.set_ylabel('$u(t,x)$')
ax.axis('square')
ax.set_xlim([-1.1,1.1])
ax.set_ylim([-1.1,1.1])
ax.set_title('$t = 0.75$', fontsize = 10)
