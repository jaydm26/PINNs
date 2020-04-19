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

nu = 0.01/np.pi
noise = 0.0

N_u = 100
N_f = 10000

# Layer Map

layers = [2, 20, 20, 20, 20, 20, 20, 20, 20, 1]

data = scipy.io.loadmat('../../appendix/Data/burgers_shock.mat')

t = data['t'].flatten()[:,None]
x = data['x'].flatten()[:,None]
Exact = np.real(data['usol']).T

X, T = np.meshgrid(x,t)
X_star = np.hstack((X.flatten()[:,None],T.flatten()[:,None]))
u_star = Exact.flatten()[:,None]

# Doman bounds
lb = X_star.min(0)
ub = X_star.max(0)

xx1 = np.hstack((X[0:1,:].T, T[0:1,:].T))
uu1 = Exact[0:1,:].T
xx2 = np.hstack((X[:,0:1], T[:,0:1]))
uu2 = Exact[:,0:1]
xx3 = np.hstack((X[:,-1:], T[:,-1:]))
uu3 = Exact[:,-1:]

X_u_train = np.vstack([xx1, xx2, xx3])
X_f_train = lb + (ub-lb)*lhs(2, N_f)
X_f_train = np.vstack((X_f_train, X_u_train))
u_train = np.vstack([uu1, uu2, uu3])

idx = np.random.choice(X_u_train.shape[0], N_u, replace=False)
X_u_train = X_u_train[idx, :]
u_train = u_train[idx,:]

# model = PhysicsInformedNN(X_u_train,u_train,X_f_train,layers,lb,ub,nu,5)
X_u_train = torch.from_numpy(X_u_train)
X_u_train.requires_grad = True
u_train = torch.from_numpy(u_train)
u_train.requires_grad = True

x_u = X_u_train[:,0:1]

t_u = X_u_train[:,1:2]
model = nn.Sequential()
for l in range(0, len(layers) - 1):
    model.add_module("layer_"+str(l), nn.Linear(layers[l],layers[l+1], bias=True))
    model.add_module("tanh_"+str(l), nn.Tanh())

optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)

losses = []
for epoch in range(0,1e5):

    u_pred = model(torch.cat((x_u,t_u),1).float())
    u_x = torch.autograd.grad(u_pred,x_u,grad_outputs = torch.ones([len(x_u),1],dtype = torch.float),create_graph=True)
    u_xx = torch.autograd.grad(u_x,x_u,grad_outputs = torch.ones([len(x_u),1],dtype = torch.float),create_graph=True)
    u_t = torch.autograd.grad(u_pred,t_u,grad_outputs = torch.ones([len(t_u),1],dtype = torch.float),create_graph=True)
    f = u_t[0] + u_pred * u_x[0] - nu * u_xx[0]

    loss = torch.mean(torch.mul(u_pred - u_train, u_pred - u_train)) + torch.mean(torch.mul(f,f))
    losses.append(loss.detach().numpy())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
