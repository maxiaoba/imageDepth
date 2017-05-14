import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data import sampler

import torchvision.datasets as dset
import torchvision.transforms as T
import numpy as np

import timeit


dtype = torch.FloatTensor
# dtype = torch.cuda.FloatTensor # Uncomment this to run on GPU


class Flatten(nn.Module):
    def forward(self, x):
        N, C, H, W = x.size() # read in N, C, H, W
        return x.view(N, -1)  # \"flatten\" the C * H * W values into a single vector per image
    

class Unflatten(nn.Module):
    def __init__(self,w,h):
        super(Unflatten,self).__init__()
        self.w=w
        self.h=h
    def forward(self, x):
        N, S = x.size() # read in N, C, H, W
        return x.view(N, self.w, self.h)  # \"flatten\" the C * H * W values into a single vector per image


import h5py
import scipy.misc
data = h5py.File("../Data/nyu_depth_v2_labeled.mat")
depths = data['depths'][0:10];
depths=depths[:,::8,::8]
depths_pytorch = torch.from_numpy(depths);
print(depths_pytorch.size())
images = data['images'][0:10];
images=images[:,:,::2,::2]
images_pytorch = torch.from_numpy(images);
print(images_pytorch.size())
images_pytorch = images_pytorch.int()
depths_pytorch = depths_pytorch.float()

course_model=nn.Sequential(
        nn.Conv2d(3,96,11,stride=4,padding=5),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(96,256,5,padding=2),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(256,384,3,padding=1),
        nn.ReLU(),
        nn.Conv2d(384,384,3,padding=1),
        nn.ReLU(),
        nn.Conv2d(384,256,3,padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        Flatten(),
        nn.Linear(17920,4096),
        nn.ReLU(),
        nn.Linear(4096,4800),
        Unflatten(80,60)
    )
course_model.type(dtype)

lr=1e-3
adam_optim=optim.Adam(course_model.parameters(),lr=lr)

images_var=Variable(images_pytorch.type(dtype),requires_grad=False)
depths_var=Variable(depths_pytorch.type(dtype),requires_grad=False)
def RMSE_log(pred,y):
    pred=pred-pred.min()+1
    N,W,H=pred.size()
    log_pred=pred.log()
    log_y=y.log()
    a=(log_y-log_pred).sum()
    b=(log_pred-log_y)+a
    return b.pow(2).sum()

RMSE_linear=nn.MSELoss().type(dtype)

def train(model, loss_fn, optimizer, num_epochs = 1):
    for epoch in range(num_epochs):
        model.train() # set the model to training mode, only effect batchnorm and dropout
        print(images_pytorch.size())
        pred=model(images_var)
        loss = loss_fn(pred, depths_var)
        print('t = %d, loss = %.4f' % (1, loss.data[0]))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

train(course_model,RMSE_linear,adam_optim,2)