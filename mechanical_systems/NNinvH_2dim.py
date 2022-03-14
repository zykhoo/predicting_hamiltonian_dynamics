import numpy as np 
from numpy import sin, cos
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data

from sklearn.model_selection import train_test_split

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

# calculate loss, c1,c2,c3,c4 = 1,10,1,1 default
def lossfuc(mat,x,y,model,device,c1=1,c2=1,c3=1,x0=0.,H0=0.,verbose=False):
    # function to calculate loss
    # mat: [batch,trajectory_len,7] matrix for one trajectory [q,p,qprime,pprime]
    # x: first 2 columns in mat which is q and p, input of net
    # y: output of net, estimation of Hamiltonian
    # c1~c4: hyperparameter for loss function

    # def Hamiltonian(q,p):
    #     value = (p**2 / 2) + (1 - torch.cos(q)) #*m*g*h
    #     return value
    # [q,p,qbar,pbar,qprime,pprime,qbarprime,pbarprime,h]
    
    # y0_1=y[:,0,0:1]
    # f3_1=(y0_1-mat[:,0,-1:])**2
    y0_2=model(torch.tensor([[x0,x0]]).to(device))
    f3_2=(y0_2-torch.tensor([[H0]]).to(device))**2
    f3=f3_2 # f3_1 # +f3_2
    # print(f3)
    dH=torch.autograd.grad(y, x, grad_outputs=y.data.new(y.shape).fill_(1),create_graph=True)[0]
    #     print(dH)
    dHdq=dH[:,:,0]
    dHdp=dH[:,:,1]
    deltaq=(mat[:,:,2]-mat[:,:,0])
    deltap=(mat[:,:,3]-mat[:,:,1])
    
    h=mat[:,:,-1]
    f1=torch.mean((dHdp*h-deltaq)**2,dim=1)

    f2=torch.mean((dHdq*h+deltap)**2,dim=1)

    # f4=torch.mean((dHdq*q_prime+dHdp*pprime)**2,dim=1)
    # f4=torch.mean((dHdq*deltaq/h+dHdp*deltap/h)**2,dim=1)

    loss=torch.mean(c1*f1+c2*f2+c3*f3)
    meanf1,meanf2,meanf3=torch.mean(c1*f1),torch.mean(c2*f2),torch.mean(c3*f3)
    if verbose:
      print(x)
      print(loss,meanf1,meanf2,meanf3)
    return loss,meanf1,meanf2,meanf3

"""## train"""

# evaluate loss of dataset use c1,c2,c3,c4=1,10,1,1
def get_loss(model,mat,bs,device,trainset=False,verbose=False):
    # this function is used to calculate average loss of a whole dataset
    # rootpath: path of set to be calculated loss
    # model: model
    # trainset: is training set or not


    avg_loss=0
    avg_f1=0
    avg_f2=0
    avg_f3=0

    for count in range(0,len(mat),bs):
      curmat=mat[count:count+bs]
      # curHbar=Hbar[count:count+bs]
      x=Variable((curmat[:,:,[2,1]]).float(),requires_grad=True)
      # x=Variable((curmat[:,:,[2,1]]).float(),requires_grad=True)
      y=model(x)
      x=x.to(device)
      loss,f1,f2,f3=lossfuc(curmat,x,y,model,device)
      avg_loss+=loss.detach().cpu().item()
      avg_f1+=f1.detach().cpu().item()
      avg_f2+=f2.detach().cpu().item()
      avg_f3+=f3.detach().cpu().item()
      # avg_f4+=f4.detach().cpu().item()
    num_batches=len(mat)//bs
    avg_loss/=num_batches
    avg_f1/=num_batches
    avg_f2/=num_batches
    avg_f3/=num_batches
    # avg_f4/=num_batches
    if verbose:
        print(' loss=',avg_loss,' f1=',avg_f1,' f2=',avg_f2,' f3=',avg_f3) # ,' f4=',avg_f4)
    return avg_loss

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

def train(net,wholemat,evalmat,device,lr,bs,num_epoch,patience,c1,c2,c3,verbose=True):
    # function of training process
    # net: the model
    # bs: batch size 
    # num_epoch: max of epoch to run
    # initial_conditions: number of trajectory in train set
    # patience: EarlyStopping parameter
    # c1~c4: hyperparameter for loss function


    avg_lossli,avg_f1li,avg_f2li,avg_f3li=[],[],[],[]
    avg_vallosses=[]
    
    start = time.time()
     # initial learning rate

    early_stopping = EarlyStopping(patience=patience, verbose=True,delta=0.00001) # delta
    optimizer=torch.optim.Adam( net.parameters() , lr=lr )
    for epoch in range(num_epoch):

        running_loss=0

        running_f1=0
        running_f2=0
        running_f3=0
        num_batches=0
        
        # train
        shuffled_indices=torch.randperm(len(wholemat))
        net.train()
        for count in range(0,len(wholemat),bs):
            optimizer.zero_grad()

            indices=shuffled_indices[count:count+bs]
            mat=wholemat[indices]
            # Hbar=trainHbar[indices]

            x=Variable(torch.tensor(mat[:,:,[2,1]]).float(),requires_grad=True)
            # x=Variable(torch.tensor(mat[:,:,[2,1]]).float(),requires_grad=True)
            y=net(x)

            loss,f1,f2,f3=lossfuc(mat,x,y,net,device,c1,c2,c3)
            if torch.isnan(loss):
              return net,avg_vallosses,avg_lossli,avg_f1li,avg_f2li,avg_f3li,count
            loss.backward()

            optimizer.step()

            # compute some stats
            running_loss += loss.detach().item()
            running_f1 += f1.detach().item()
            running_f2 += f2.detach().item()
            running_f3 += f3.detach().item()

            num_batches+=1
            torch.cuda.empty_cache()



        avg_loss = running_loss/num_batches
        avg_f1 = running_f1/num_batches
        avg_f2 = running_f2/num_batches
        avg_f3 = running_f3/num_batches
        elapsed_time = time.time() - start
        
        avg_lossli.append(avg_loss)
        avg_f1li.append(avg_f1)
        avg_f2li.append(avg_f2)
        avg_f3li.append(avg_f3)
        
        
        # evaluate
        net.eval()
        avg_val_loss=get_loss(net,evalmat,bs,device)
        avg_vallosses.append(avg_val_loss)
        
        if verbose and epoch % 10 == 0 : 
            print(' ')
            print('epoch=',epoch, ' time=', elapsed_time,
                  ' loss=', avg_loss ,' val_loss=',avg_val_loss,' f1=', avg_f1 ,' f2=', avg_f2 ,
                  ' f3=', avg_f3 )
        
        
        
        early_stopping(avg_val_loss,net)
        if early_stopping.early_stop:
            print('Early Stopping')
            break
            
    net=torch.load('checkpoint.pt')
    return net,avg_vallosses,avg_lossli,avg_f1li,avg_f2li,avg_f3li




def data_preprocessing(start,delta,epsilon,device):
    wholemat,evalmat=train_test_split(np.expand_dims(np.hstack((start.transpose(), start.transpose()+delta.transpose()*epsilon, 
                                                                np.asarray([[epsilon]*64]).transpose())),1), train_size=0.8, random_state=2,shuffle=True)	

    wholemat=torch.tensor(wholemat).to(device)
    evalmat=torch.tensor(evalmat).to(device)
    
    return wholemat, evalmat
