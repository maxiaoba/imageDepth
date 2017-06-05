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



# load data ##############################################################################################

import os
import scipy
class DepthDataset(Dataset):
    def __init__(self, data_dir):
        sub_folders=os.listdir(data_dir+'/RGB')
        self.data_files=[]
        for folder in sub_folders:
            files=os.listdir(data_dir+'/RGB/'+folder)
            for file in files:
                self.data_files.append(folder+'/'+file)
        self.data_dir=data_dir
#         sort(self.data_files)      

    def __getitem__(self, index):
        name=self.data_files[index]
        img=torch.from_numpy(h5py.File(self.data_dir+'/RGB/'+name,'r')['rgbOut'].value).float()
        depth=torch.from_numpy(h5py.File(self.data_dir+'/DEP/'+name,'r')['depthOut'].value).float()
        return img,depth

    def __len__(self):
        print('len',len(self.data_files))
        return len(self.data_files)


dataset=DepthDataset('../Data')
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
        return iter(range(self.start, self.start + self.num_samples))

    def __len__(self):
        return self.num_samples

N=dataset.__len__()
NUM_TRAIN = int(N*0.9)
NUM_VAL = N-NUM_TRAIN
print("NUM_TRAIN:",NUM_TRAIN,",NUM_VAL:",NUM_VAL)
batch_size=32
loader_train = DataLoader(dataset, batch_size=batch_size, sampler=ChunkSampler(NUM_TRAIN, 0),num_workers=8)
loader_val = DataLoader(dataset, batch_size=batch_size, sampler=ChunkSampler(NUM_VAL, NUM_TRAIN),num_workers=8)



# define network ################################################################################################3
dtype = torch.cuda.FloatTensor # Uncomment this to run on GPU


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

class crop(nn.Module):
    def forward(self,x):
        N,C,H,W = x.size()
        return x[:,:,0:H-1,0:W-1].squeeze()
class printLayer(nn.Module):
    def forward(self,x):
        #print('layer size:',x.size())
        print('max element:', x.max())
        print('min element:', x.min())
        return x


dtype = torch.cuda.FloatTensor
#del coarse_model
coarse_model=nn.Sequential(
        nn.Conv2d(3,60,3,stride=1, padding=1),
        printLayer(),
        nn.BatchNorm2d(60),
        nn.ReLU(),
        nn.Conv2d(60,100,3,stride=1, padding=1),
        printLayer(),
        nn.BatchNorm2d(100),
        nn.ReLU(),
        nn.Conv2d(100,100,3,stride=1, padding=1),
        printLayer(),
        nn.BatchNorm2d(100),
        nn.ReLU(),
        nn.Conv2d(100,200,3,stride = 2,padding=0),
        printLayer(),
        nn.BatchNorm2d(200),
        nn.ReLU(),
        nn.Conv2d(200,220,3,stride = 1,padding=1),
        printLayer(),
        nn.BatchNorm2d(220),
        nn.ReLU(),
        nn.Conv2d(220,300,3,stride=2,padding=0),
        printLayer(),
        nn.BatchNorm2d(300),
        nn.ReLU(),
        nn.Conv2d(300,350,3,stride=1,padding=1),
        printLayer(),
        nn.BatchNorm2d(350),
        nn.ReLU(),
        #nn.ConvTranspose2d(360, 100, 3, stride = 4, padding=2,output_padding=1),
        #nn.BatchNorm2d(100),
        #nn.ReLU(),
        nn.Conv2d(350,1,1,stride=1,padding=0),
        crop()
    )
coarse_model.type(dtype)



print_every=400
lr=1e-3
reg=1e-4
adam_optim=optim.Adam(coarse_model.parameters(),lr=lr,weight_decay=reg)

#images_var=Variable(images_pytorch.type(dtype),requires_grad=False)
#depths_var=Variable(depths_pytorch.type(dtype),requires_grad=False)

def loss_log(pred,y):
    ep = 1e-6
    N,W,H = pred.size()
    pred = pred.contiguous().view(N,-1)
    y = y.view(N,-1)
    y = y+ep
    d = pred - y.log()
    d[y == 0] = 0
    n = W*H
    loss = (d.pow(2).sum(1) / n - 0.5 / n/n * d.sum(1).pow(2)).sum()
    loss /= N
    return loss
    


def imshow_noax(img, normalize=True):
    """ Tiny helper to show images as uint8 and remove axis labels """

    if normalize:
        img_max, img_min = np.max(img), np.min(img)
        img = 255.0 * (img - img_min) / (img_max - img_min)

    plt.imshow(img.astype('uint8'))
    plt.gca().axis('off')



def train(model, loss_fn, optimizer, num_epochs = 1, plot_every = 10):
    losses = []
    for epoch in range(num_epochs):
        model.train() # set the model to training mode, only effect batchnorm and dropout
        avg_train_loss=0
        num_batches=0
        for t,(x,y) in enumerate(loader_train):
            x_var=Variable(x.type(dtype),requires_grad=False)
            y_var=Variable(y.type(dtype),requires_grad=False)
            print(x_var.size())
            pred=model(x_var)
            loss = loss_fn(pred, y_var)
            losses.append(loss.data.cpu().numpy())
            
            if (t+1) % print_every==0:
                print('t = %d, loss = %.4f' % (t+1, loss.data[0]))
            avg_train_loss+=loss.data[0]
            num_batches+=1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        avg_train_loss/=num_batches
        num_batches=0
        avg_val_loss=0
        for t,(x,y) in enumerate(loader_val):
            x_var=Variable(x.type(dtype),requires_grad=False)
            y_var=Variable(y.type(dtype),requires_grad=False)
            pred=coarse_model(x_var)
            loss=loss_fn(pred,y_var)
            avg_val_loss+=loss.data[0]
            num_batches+=1
        avg_val_loss/=num_batches
        print("epoch:",epoch,"average training loss: %.2f"%avg_train_loss,"validation loss: %.2f" %avg_val_loss)
        if(epoch % plot_every == 0):
            plt.plot(losses)
            

train(coarse_model,loss_log,adam_optim,num_epochs = 20)


