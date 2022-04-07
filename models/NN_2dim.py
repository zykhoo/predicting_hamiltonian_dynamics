# this neural network takes in q and p and learns the derivative dq/dt, dp/dt

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data
import numpy as np
import os
import time

from sklearn.model_selection import train_test_split

import torch.optim as optim

from tqdm import tqdm

def classicIntNN(z,h,net):
	## classical symplectic Euler scheme
		dim = int(len(z)/2)
		q=z[:dim]
		p=z[dim:]		
		fstage = lambda stg: h * torch.squeeze(net(torch.tensor([q+stg,p]).float().transpose(1,0)),0).detach().numpy().transpose()[:dim]

		stageold=np.zeros(dim) 
		stage = fstage(stageold) +0.
		Iter = 0

		while (np.amax(abs(stage - stageold)*25) > 1e-10 and Iter<100):
			stageold = stage+0.
			stage = fstage(stage)+0.
			Iter = Iter+1
		q = q+stage
		p = p +h*torch.squeeze(net(torch.tensor([q,p]).float().transpose(1,0)),0).detach().numpy().transpose()[dim:]
		return np.block([q,p])

def classicTrajectoryNN(z,h,net,N=1):
	## trajectory computed with classicInt
  z = z.reshape(1,-1)[0]
  trj = np.zeros((len(z),N+2))
  trj[:,0] = z.copy()
  if N == 1:
    return z.reshape(-1,1), classicIntNN(trj[:,0],h,net).reshape(-1,1)
  else:
    for j in tqdm(range(0,N+1)):
      trj[:,j+1] = classicIntNN(trj[:,j].copy(),h,net)
  return trj[:, :-1], trj[:, 1:]

def naiveIntNN(z,h,net):
		dim = int(len(z)/2)
		q=z[:dim]
		p=z[dim:]		
		dH = torch.squeeze(net(torch.tensor(z).float()),0).detach().numpy().transpose()
		q = q +h*dH[:dim]
		p = p +h*dH[dim:]
		return np.block([q,p])

def naiveTrajectoryNN(z,h,net,N=1):
	## trajectory computed with classicInt
  z = z.reshape(1,-1)[0]
  trj = np.zeros((len(z),N+2))
  trj[:,0] = z.copy()
  if N == 1:
    return z.reshape(-1,1), naiveIntNN(trj[:,0],h,net).reshape(-1,1)
  else:
    for j in tqdm(range(0,N+1)):
      trj[:,j+1] = naiveIntNN(trj[:,j].copy(),h,net)
  return trj[:, :-1], trj[:, 1:]


# define model
def softplus(x):
    return torch.log(torch.exp(x)+1)

class SoftPlus(nn.Module):
    def __init__(self):
        '''
        Init method.
        '''
        super().__init__() # init the base class

    def forward(self, x):
        return softplus(x)

class Net(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(Net , self).__init__()
        self.hidden_layer_1 = nn.Linear( input_size, hidden_size, bias=True)
        self.hidden_layer_2 = nn.Linear( hidden_size, hidden_size, bias=True)
        self.output_layer = nn.Linear( hidden_size, output_size , bias=True)
        
    def forward(self, x):
        x = softplus(self.hidden_layer_1(x)) # F.relu(self.hidden_layer_1(x)) # 
        x = softplus(self.hidden_layer_2(x)) # F.relu(self.hidden_layer_2(x)) # 
        x = self.output_layer(x)

        return x

## Run


def data_preprocessing(start, delta, device):
    mat = np.hstack((start.transpose(), delta.transpose()))
    mat = torch.tensor(mat)
    mat = mat.to(device)
    wholemat,evalmat=train_test_split(mat, train_size=0.8, shuffle = True, random_state=1)
    return wholemat, evalmat





class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            上次验证集损失值改善后等待几个epoch
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            如果是True，为每个验证集损失值改善打印一条信息
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            监测数量的最小变化，以符合改进的要求
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if abs(self.counter-self.patience)<5:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''
        Saves model when validation loss decrease.
        验证损失减少时保存模型。
        '''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        # torch.save(model.state_dict(), 'checkpoint.pt')     # 这里会存储迄今最优模型的参数
        torch.save(model, 'checkpoint.pt')                 # 这里会存储迄今最优的模型
        self.val_loss_min = val_loss
	
def train(net, wholemat, evalmat, optimizer, batchsize=10, iter=1600, ):
    for epoch in range(iter):  # loop over the dataset multiple times

        # for count in range(0,len(wholemat),batchsize):
        # get the inputs; data is a list of [inputs, labels]
        input = wholemat[:,0:2].float()

        # zero the parameter gradients
        early_stopping = EarlyStopping(patience=100, verbose=False,delta=0.00001)
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(input)
        loss = torch.mean((outputs - wholemat[:,2:4])**2)
        loss.backward()
        optimizer.step()

        # print statistics
        # running_loss += loss.item()
        if epoch % 200 == 0:    # print every 2000 mini-batches
            # print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            print("training loss", loss)
            net.eval()
            val_loss = torch.mean((net(evalmat[:,0:2].float()) - evalmat[:,2:4])**2)
            print("validation loss", val_loss)
		
        early_stopping(val_loss,net)
        if early_stopping.early_stop:
            print('Early Stopping')
            break
            
    net=torch.load('checkpoint.pt')

    print('Finished Training')
    return net

def classicqbarpbar(z,h,net):
		dim = int(len(z)/2)		
		qbar = torch.squeeze(net(torch.tensor(z).float()),0).detach().numpy().transpose()[:dim]
		pbar = torch.squeeze(net(torch.tensor(z).float()),0).detach().numpy().transpose()[dim:]
		return np.block([qbar,pbar])

def classicTrajectorybar(z,h,net,N=1):
	## trajectory computed with classicInt
  z = z.reshape(1,-1)[0]
  trj = np.zeros((len(z),N+2))
  trj[:,0] = z.copy()
  if N == 1:
    return z.reshape(-1,1), classicqbarpbar(trj[:,0],h,net).reshape(-1,1)
  else:
    for j in tqdm(range(0,N+1)):
      trj[:,j+1] = classicqbarpbar(trj[:,j].copy(),h,net)
  return trj[:, :-1], trj[:, 1:]