# Author QFIUNE
# coding=utf-8
# @Time: 2022/6/9 9:49
# @File: lrpDes.py
# @Software: PyCharm
# @contact: 1760812842@qq.com
import os
from copy import deepcopy

import torch
import torch as th
from torch import nn
from dgl.nn.pytorch import TAGConv
import numpy as np



class RelevancePropagationTAGCN(nn.Module):
    def __init__(self, layer: TAGConv):
        super().__init__()

        self.layer = layer

        # From source code: dgl/nn/pytorch/conv/tagconv.py
        self.W = self.layer.lin.weight         
        self.b = self.layer.lin.bias           
        self.K = self.layer._k                 
        # print(self.W.shape, self.b.shape, self.K)

    @torch.no_grad()
    def forward(self, a: torch.tensor, r: torch.tensor,  graph) -> torch.tensor:
        """
        a: represents the input for each layer
        r: represents the relevance of the latter layer
        """
        rho = lambda Wk: [W.abs() for W in Wk]
        num_nodes = len(a)   # a's shape is torch.Size([31, 50]
        dW = self.W.shape[1]//(self.K+1)
        Wk = [self.W[:, k*dW:(k+1)*dW].t() for k in range(self.K+1)]  

        Dm = th.diag(th.pow(graph.in_degrees().float().clamp(min=1), -0.5))
        A = Dm.matmul(graph.adj().to_dense()).matmul(Dm)  
        Ak = [th.matrix_power(A, k) for k in range(self.K+1)]

        rhoWk = rho(Wk)
        dimY, dimJ = rhoWk[0].shape
        U = th.zeros(num_nodes, dimJ, num_nodes, dimY)     # U's shape is torch.Size([31, 50, 31, 32])

        for i in range(num_nodes):
            for j in range(dimJ):
                U[i, j] = sum(Ak[k][i, :].unsqueeze(-1).matmul(rhoWk[k][:, j].unsqueeze(0)) for k in range(self.K + 1))

        r = th.einsum("ijxy,ij->xy", [U, r / U.sum(dim=(2, 3))])

        return r


    def Validate(self, graph):
        Dm = th.diag(th.pow(graph.in_degrees().float().clamp(min=1), -0.5))
        A = Dm.matmul(graph.adj().to_dense()).matmul(Dm)  
        
        Ak = [th.matrix_power(A, k) for k in range(self.maxK+1)]
   
        hs = []
        h = graph.ndata[self.net.features_str].float()
        hs.append(h)
        for layer_param in self.layer_params:
            
            h = sum(Ak[k].matmul(h).matmul(layer_param[2][k]) for k in range(layer_param[1]+1))+layer_param[0]
         
            h = h.clamp(min=0)
            hs.append(h)
        return hs, Ak


