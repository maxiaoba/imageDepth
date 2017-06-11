import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data import sampler
from torch.utils.data import TensorDataset
from torch.utils.data import Dataset

import torchvision.datasets as dset
import torchvision.transforms as T
import numpy as np

import timeit
import h5py

import matplotlib.pyplot as plt

import os
class DepthDataset(Dataset):
    def __init__(self, data_dir):
        self.data_files=[]
        folders=os.listdir(data_dir+'/RGB')
        for folder in folders:
            subfolders=os.listdir(data_dir+'/RGB/'+folder)
            for subfolder in subfolders:
                if subfolder.startswith('.'):
                    continue
                files=os.listdir(data_dir+'/RGB/'+folder+'/'+subfolder)
                for file in files:
                    if file.endswith('.mat'):
                        self.data_files.append(folder+'/'+ subfolder+'/'+file)
            self.data_dir=data_dir
#         sort(self.data_files)      

    def __getitem__(self, index):
        name=self.data_files[index]
        img=torch.from_numpy(h5py.File(self.data_dir+'/RGB/'+name,'r')['rgbOut'].value).float()
        depth=torch.from_numpy(h5py.File(self.data_dir+'/DEP/'+name,'r')['depthOut'].value).float()
        return img,depth

    def __len__(self):
        return len(self.data_files)

dataset=DepthDataset('../../Data_liv')
class ChunkSampler(sampler.Sampler):
    """Samples elements sequentially from some offset. 
    Arguments:
        num_samples: # of desired datapoints
        start: offset where we should start selecting from
    """
    def __init__(self, num_samples, start = 0):
        self.num_samples = num_samples
        self.start = start

    def __iter__(self):
        return iter((torch.randperm(self.num_samples)+self.start).long())

    def __len__(self):
        return self.num_samples

N=dataset.__len__()
NUM_TRAIN = int(N*0.9)
NUM_VAL = N-NUM_TRAIN
print("NUM_TRAIN:",NUM_TRAIN,",NUM_VAL:",NUM_VAL)
batch_size=6
loader_train = DataLoader(dataset, batch_size=batch_size, sampler=ChunkSampler(NUM_TRAIN, 0),num_workers=8)
loader_val = DataLoader(dataset, batch_size=batch_size, sampler=ChunkSampler(NUM_VAL, NUM_TRAIN),num_workers=8)

print(len(loader_train))

import torchvision.models as models
dtype = torch.cuda.FloatTensor
model = models.resnet50(pretrained=True)
mod = list(model.children())
mod.pop()
mod.pop()
resnet50 = torch.nn.Sequential(*mod)
del model
class cropMore(nn.Module):
    def forward(self,x):
        N,C,H,W = x.size()
        return x[:,:,0:H-6,0:W-9].squeeze()

class unpooling(nn.Module):
    def __init__(self,H,W):
        super(unpooling, self).__init__()
        self.indices = torch.zeros(H,W)
        for i in range(H):
            for j in range(W):
                self.indices[i,j] = i*2*2*W+j*2
        self.unpool = nn.MaxUnpool2d(2)
    def forward(self, x):
        N,C,H,W = x.size()
        indices = self.indices.expand(N,C,H,W)
        y = self.unpool(x, Variable(indices.type(torch.cuda.LongTensor),requires_grad=False))
        return y

class upsampling(nn.Module):
    def __init__(self,C,H,W):
        super(upsampling, self).__init__()
        self.unpooling = unpooling(H,W)
        self.conv5 = nn.Conv2d(C,int(C/2),5,stride = 1,padding=2)
        self.ReLU = nn.ReLU()
        self.conv3 = nn.Conv2d(int(C/2),int(C/2),3,stride = 1,padding=1)
        self.conv5_2 = nn.Conv2d(C,int(C/2),5,stride = 1,padding=2)
    def forward(self, x):
        y0 = self.unpooling(x)
        y1 = self.conv5(y0)
        y1 = self.ReLU(y1)
        y1 = self.conv3(y1)
        y2 = self.conv5_2(y0)
        y3 = y1+y2
        y = self.ReLU(y3)
        return y

upsample = torch.nn.Sequential(
    nn.Conv2d(2048,1024,1,stride = 1,padding=0),
    nn.BatchNorm2d(1024), #1024*10*8
    upsampling(1024,10,8),
    upsampling(512,20,16),
    upsampling(256,40,32),
    #upsampling(128,80,64),
    nn.Conv2d(128,1,3,stride = 1,padding=1),
    nn.ReLU(),
    cropMore()
    )

resnet50.type(dtype)
upsample.type(dtype)

upsample.load_state_dict(torch.load('alldata_dict_ep8'))

#images_var=Variable(images_pytorch.type(dtype),requires_grad=False)
#depths_var=Variable(depths_pytorch.type(dtype),requires_grad=False)

def loss_log(pred,y):
    ep = 1e-6
    N,W,H = pred.size()
    pred = pred.contiguous().view(N,-1)
    y = y.view(N,-1)
    y = y+ep
    d = pred - y.log()
    d[y <= 0] = 0
    n = W*H
    loss = (d.pow(2).sum(1) / n - 0.5 / n/n * d.sum(1).pow(2)).sum()
    loss /= N
    return loss
    
for param in resnet50.parameters():
    param.requires_grad = False

lr=1e-6
reg=1e-4
adam_optim=optim.Adam(upsample.parameters(),lr=lr,weight_decay=reg)

import pickle
print_every=50

def train(model, loss_fn, optimizer, num_epochs = 1, plot_every = 2):
    train_losses = []
    val_losses = []
    for tepoch in range(num_epochs):
        epoch=tepoch+9
        model.train() # set the model to training mode, only effect batchnorm and dropout
        avg_train_loss=0
        num_batches=0
        for t,(x,y) in enumerate(loader_train):
            x_var=Variable(x.type(dtype),requires_grad=False)
            y_var=Variable(y.type(dtype),requires_grad=False)
            x1 = resnet50(x_var)
            pred = model(x1)
            loss = loss_fn(pred, y_var)
            #losses.append(loss.data.cpu().numpy())
            
            if (t+1) % print_every==0:
                print('t = %d, loss = %.4f' % (t+1, loss.data[0]))
            avg_train_loss+=loss.data[0]
            num_batches+=1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            del x,y
        avg_train_loss/=num_batches
        train_losses.append(avg_train_loss)
        
        num_batches=0
        avg_val_loss=0
        for t,(x,y) in enumerate(loader_val):
            x_var=Variable(x.type(dtype),requires_grad=False)
            y_var=Variable(y.type(dtype),requires_grad=False)
            x1 = resnet50(x_var)
            pred = model(x1)
            loss=loss_fn(pred,y_var)
            avg_val_loss+=loss.data[0]
            num_batches+=1
            del x,y
        avg_val_loss/=num_batches
        val_losses.append(avg_val_loss)
        print("epoch:",epoch,"average training loss: %.2f"%avg_train_loss,"validation loss: %.2f" %avg_val_loss)
        if(epoch % plot_every == 0):
            with open('losses_ep'+ str(epoch)+ '.pickle', 'wb') as f:  # Python 3: open(..., 'wb')
                pickle.dump([train_losses,val_losses], f)
            torch.save(model.state_dict(), 'alldata_dict_ep'+str(epoch))
train(upsample,loss_log,adam_optim,num_epochs = 40)
