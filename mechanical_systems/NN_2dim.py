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




def train(net, wholemat, evalmat, optimizer, batchsize=10, iter=1600, ):
    for epoch in range(iter):  # loop over the dataset multiple times

        # for count in range(0,len(wholemat),batchsize):
        # get the inputs; data is a list of [inputs, labels]
        input = wholemat[:,0:2].float()

        # zero the parameter gradients
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
            print("validation loss", torch.mean((net(evalmat[:,0:2].float()) - evalmat[:,2:4])**2))

    print('Finished Training')
    return net
