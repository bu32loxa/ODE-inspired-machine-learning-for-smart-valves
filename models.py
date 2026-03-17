import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
import time

import os
from copy import deepcopy


class RecursiveRegression(nn.Module):
    """
    | class for a recurrent layer that depends on a number of inputs and a window 
    | of previous values
    """
    
    def __init__(self, h_window=1, x_window=2, h_lag=0, x_lag=0, channels=1, no_bias=False,
                 h_weight_init=None, x_weight_init=None):
        super().__init__()
        
        self.channels = channels
        self.no_bias = no_bias
        
        self.h_window = h_window
        self.x_window = x_window
        
        self.h_lag = h_lag
        self.x_lag = x_lag
        
        self.h_weights = nn.Parameter(torch.zeros((h_window, channels)))
        with torch.no_grad():
            if h_weight_init is not None:
                self.h_weights += h_weight_init
            else:
                self.h_weights += torch.normal(0,.05, (h_window, channels))
        
        self.x_weights = nn.Parameter(torch.zeros((x_window, channels)))
        with torch.no_grad():
            if x_weight_init is not None:
                self.x_weights += x_weight_init
            else:
                self.x_weights += torch.normal(0,.05, (x_window, channels))
        
        self.biases = nn.Parameter(torch.zeros(1,channels))
        
        if self.no_bias:
            self.biases.requires_grad = False
        
    def forward(self,h,x=None):
        
        start_ix = max(self.h_window+self.h_lag-1, self.x_window,self.x_lag-1)
        output = torch.zeros(h.shape[:-1]+(self.channels,), device=h.device)
        x_window_ = torch.zeros((h.shape[0],self.x_window), device=h.device)
        h_window_ = h[:,:self.h_window].view((-1,self.h_window))
        biases = self.biases.repeat(h.shape[0],1)
        
        for i in range(start_ix,h.shape[1]):
            new_output = x_window_ @ self.x_weights + h_window_ @ self.h_weights + biases
            output[:,i] = new_output 

            if i+1 < h.shape[1]:
                h_window_ = torch.cat([h_window_,h[:,i+1-self.h_lag]], dim=1)[:,-self.h_window:]
                if x is None:
                    x_window_ = torch.cat([x_window_,new_output], dim=1)[:,-self.x_window:]
                else:
                    x_window_ = torch.cat([x_window_,x[:,i+1-self.x_lag]], dim=1)[:,-self.x_window:]
        return output
    

class CurrentFlowPrediction(nn.Module):
    """
    | temporal processing unit designed to make diaphragm-specific predictions of 
    | current flow
    """
    
    def __init__(self, channels=3):
        super().__init__()
        self.channels = channels
        self.recursive_layer = RecursiveRegression(1,2,1,1,
                                                   channels=channels-1, 
                                                   no_bias=True,
                                                   x_weight_init=torch.tensor([[-1],[2]],dtype=torch.float32).repeat(1,channels-1))
        self.eval_layers = nn.ModuleList([
                                RecursiveRegression(1,2,1,0,
                                                    channels=1,
                                                    h_weight_init=torch.ones((1,1)))
                                for _ in range(channels)])
        
    def forward(self,h):
        x = self.recursive_layer(h)
        # x = x - torch.cat([torch.zeros(x.shape[0],1,x.shape[2]),x[:,:-1,:]], dim=1) # calculate 1-step difference
        h_ = torch.cat([self.eval_layers[i](h,x[:,:,i].unsqueeze(-1)) for i in range(x.shape[2])], dim=2)
        return torch.cat([h,h_], dim=-1).transpose(1,2)
    
    
class ValvePositionPrediction(nn.Module):
    """
    | temporal processing unit designed to calculate valve position in two different
    | ways, of which one is dependent on the diaphragm type
    """
    
    def __init__(self, channels=3):
        super().__init__()
        self.channels = channels
        self.recursive_layer = RecursiveRegression(2,1,0,1,
                                                   channels=1,
                                                   h_weight_init=torch.tensor(np.array([[1.],[-1.]]),dtype=torch.float32),
                                                   x_weight_init=torch.ones((1,1)))
        self.eval_layer = RecursiveRegression(1,2,1,1,
                                              channels=channels-1, no_bias=True)
        
    def forward(self, h):
        x = self.recursive_layer(h)
        x_ = self.eval_layer(h,x)
        return torch.cat([x,x_], dim=-1).transpose(1,2)
    

class Upsampling(nn.Module):
    """
    | a layer that concatenates the input multiple times along the last axis
    """
    
    def __init__(self,channels=3):
        super().__init__()
        
        self.channels = channels
        
    def forward(self,h):
        output = torch.cat([h for _ in range(self.channels)], dim=-1)
        return output.transpose(1,2)
    
    
class RecurrentPrediction(nn.Module):
    """
    | a wrapper for a RNN-layer, with the option to extend the input by calculating
    | numerical differentials
    """
    
    def __init__(self,channels=3, num_diff=True):
        super().__init__()
        
        self.channels = channels
        self.num_diff = num_diff
        inputs = 2 if self.num_diff else 1
        self.model = nn.RNN(inputs, channels, batch_first=True)
        
    def forward(self, h):
        if self.num_diff:
            h_lag = torch.cat([torch.zeros(h.shape[0],1,1),h[:,:-1,:]], dim=1)
            h = torch.cat([h,h-h_lag], dim=2)
        output = self.model(h)[0]
        return output.transpose(1,2)
    
    
class LSTMPrediction(nn.Module):
    """
    | same as 'RecurrentPrediction', with an LSTM layer instead of RNN
    """
    
    def __init__(self, channels=3, num_diff=True):
        super().__init__()
        
        self.channels = channels
        self.num_diff = num_diff
        inputs = 2 if self.num_diff else 1
        self.model = torch.nn.LSTM(inputs, channels, batch_first=True)
        
    def forward(self, h):
        if self.num_diff:
            h_lag = torch.cat([torch.zeros(h.shape[0],1,1),h[:,:-1,:]], dim=1)
            h = torch.cat([h,h-h_lag], dim=2)
        output = self.model(h)[0]
        return output.transpose(1,2)
    
    
class PDEInspiredModel(nn.Module):
    """
    | model inspired by a discretized version of the PDEs describing the valve motion and current
    """
    
    def __init__(self, classes=3, ts_model=CurrentFlowPrediction, model_channels=3):
        super().__init__()
        self.model_channels = model_channels
        
        self.pool1 = nn.AvgPool1d(4,4) # initial downsampling layer
        
        self.ts_layer = ts_model(channels=self.model_channels) # time-attentive layer
        
        # embedding/classification layers
        self.pool2 = nn.AvgPool1d(4,4)
        self.conv = nn.Conv1d(self.model_channels,self.model_channels,4,1)
        self.pool3 = nn.MaxPool1d(5,5)
        self.classifier = nn.Linear(4*self.model_channels,classes)
        
        
    def forward(self,x):
        
        if len(x.shape)==2:
            x = x.unsqueeze(2)
        elif len(x.shape)==1:
            x = x.unsqueeze(0).unsqueeze(2)
        
        batch_size = x.shape[0]
        
        x = self.pool1(x.transpose(1,2)).transpose(1,2)
        
        x = self.ts_layer(x)
        
        x = self.pool2(x)
        x = self.conv(x)
        x = self.pool3(x)
        
        x = torch.tanh(x)
        x = self.classifier(x.view(batch_size,1,-1)).view(batch_size,-1)
        return F.softmax(x,dim=-1)
    
    
    def fit(self, X, Y, x_valid=None, y_valid=None,epochs=10000, batch_size=500, alpha=.1, cat_weights=None, save_best='./pde_model.pt'):
        
        history = [] # for returning training progress
        device = next(self.parameters()).device
        
        if x_valid is not None:
            x_valid, y_valid = torch.tensor(x_valid,dtype=torch.float32,device=device), torch.tensor(y_valid,dtype=torch.float32,device=device)
        
        if cat_weights is None:
            cat_weights = torch.ones(Y.shape[1:])
        
        # calculate network parameters
        n_params = 0
        for tens in list(self.parameters()):
            tens_params = 1
            for dim in tens.shape:
                tens_params *= dim
            n_params += tens_params
        print('parameter count:', n_params)
        
        
        optim = torch.optim.Adam(self.parameters(), lr=.01, eps=1e-6)
        
        # categorical cross entropy loss
        loss_fn = lambda x,y: ((-x.log()) * y * cat_weights).sum(dim=1).mean()
        
        # 2nd loss function for keeping input-gradients small
        def input_grad(f,x): 
            grads = torch.cat([
                        torch.autograd.grad(f[i],x, retain_graph=True,grad_outputs=torch.ones_like(f[i]))[0].unsqueeze(-1)
                        for i in range(f.shape[-1])
                    ], dim=-1)
            sq_grads = torch.bmm(grads.view(-1,1,grads.shape[-1]),grads.view(-1,grads.shape[-1],1)).view(grads.shape[:-1])
            return sq_grads.sum(dim=-1).mean()
        
        
        n_batches = len(X)//batch_size
        max_acc = 0.
        
        for epoch in range(epochs):  
            p = np.random.permutation(len(X))
            X, Y = X[p], Y[p]    

            epoch_start = time.time()
            epoch_avg_loss = 0.
            epoch_avg_grad = 0.
            
            for batch in range(n_batches):
                
                batch_X = torch.tensor(X[batch_size*batch:batch_size*(batch+1)],dtype=torch.float32, requires_grad=True, device=device)
                batch_Y = torch.tensor(Y[batch_size*batch:batch_size*(batch+1)],dtype=torch.float32, device=device)
                
                optim.zero_grad()
                pred = self(batch_X)
                l1 = 1e12 * input_grad(pred,batch_X)
                l2 = loss_fn(pred, batch_Y)
                loss = alpha * l1 + (1-alpha) * l2
                epoch_avg_grad += l1.item()/n_batches
                epoch_avg_loss += l2.item()/n_batches
                loss.backward()
                optim.step()
            
            if np.isnan(epoch_avg_loss):
                print('loss is nan! Loading best performing checkpoint')
                self.load_state_dict(torch.load(('./pde_model.pt')))
            
            if x_valid is not None:
                val_loss = loss_fn(self(x_valid),y_valid).item()
                acc_count = torch.sum(torch.argmax(self(x_valid),dim=1) == torch.argmax(y_valid,dim=1))
                val_acc = acc_count/len(x_valid)
                
                if np.isnan(val_loss):
                    print('val. loss is nan! Loading best performing checkpoint')
                    self.load_state_dict(torch.load(('./pde_model.pt')))
               
                if val_acc > max_acc:
                    print('\nsaving model\n')
                    torch.save(deepcopy(self.state_dict()), save_best)
                    max_acc = val_acc
                epoch_time = int(1000*(time.time()-epoch_start))
                history.append([epoch_avg_loss, val_loss, val_acc, epoch_time])
                print(epoch_avg_grad)
                print(f'epoch {epoch}, loss={epoch_avg_loss}, val. loss={val_loss}, val. acc.={val_acc}, time: {epoch_time}ms')
            else:
                epoch_time = int(1000*(time.time()-epoch_start))
                history.append([epoch_avg_loss, epoch_time])
                print(epoch_avg_grad)
                print(f'epoch {epoch}, loss={epoch_avg_loss}, time: {epoch_time}ms')
                    
        return history
