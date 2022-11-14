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


from dgl.nn.pytorch import GraphConv, TAGConv, ChebConv, GATConv
import numpy as np

top_k_percent = 0.04  # Proportion of relevance scores that are allowed to pass.


class linear_ww_lrp(nn.Module):
    """
    Layer-wise relevance propagation for linear transformation.
    Optionally modifies layer weights according to propagation rule.

    LRR according to Eq(12) in https://arxiv.org/pdf/1512.02479v1.pdf

    Attributes:
        layer: linear transformation layer.
        eps: a value added to the denominator for numerical stability.
    """

    def __init__(self, layer: torch.nn.Linear) -> None:
        super().__init__()

        self.layer = layer
        self.layer.weight = torch.nn.Parameter(self.layer.weight.clamp(min=0.0))
        self.layer.bias = torch.nn.Parameter(torch.zeros_like(self.layer.bias))
        # print('Linear weight is ', self.layer.weight.shape)

    @torch.no_grad()
    def forward(self, a: torch.tensor, r: torch.tensor) -> torch.tensor:
        """
        LRR according to Eq(12) in https://arxiv.org/pdf/1512.02479v1.pdf

        暂时通不过
        """
        # Z = th.square(th.unsqueeze(self.layer.weight, 0))
        # print('Z shape: ', Z.shape)
        # Zs = th.unsqueeze(th.sum(Z, 1), 1)
        # print('Zs shape: ', Zs.shape)
        # r = th.sum((Z/Zs) * th.unsqueeze(r, 1), 2)
        # print('r shpe: ', r)

        Z = th.square(self.layer.weight)
        Zs = th.sum(Z)
        d = th.div(Z, Zs)
        # d = Z / Zs

        Dr = th.matmul(torch.unsqueeze(r, 0),d)
        print(d.shape, r.shape)
        # Dr = r*d

        Rs = th.sum(Dr, 0)
        return Rs


class RelevancePropagationLinear(nn.Module):
    """Layer-wise relevance propagation for linear transformation.
    Optionally modifies layer weights according to propagation rule. Here z^+-rule
    Attributes:
        layer: linear transformation layer.
        eps: a value added to the denominator for numerical stability.
    """

    def __init__(self, layer: torch.nn.Linear, mode: str = "z_plus", eps: float = 1.0e-05) -> None:
        super().__init__()

        self.layer = layer

        if mode == "z_plus":
            self.layer.weight = torch.nn.Parameter(self.layer.weight.clamp(min=0.0))
            self.layer.bias = torch.nn.Parameter(torch.zeros_like(self.layer.bias))
            # print('Linear weight is ', self.layer.weight.shape)

        self.eps = eps

    @torch.no_grad()
    def forward(self, a: torch.tensor, r: torch.tensor) -> torch.tensor:
        z = self.layer.forward(a) + self.eps
        s = r / z
        c = torch.matmul(s, self.layer.weight)
        r = (a * c).data
        return r


class RelevancePropagationReLU(nn.Module):
    """Layer-wise relevance propagation for ReLU activation.
    Passes the relevance scores without modification. Might be of use later.
    """

    def __init__(self, layer: torch.nn.ReLU) -> None:
        super().__init__()

    @torch.no_grad()
    def forward(self, a: torch.tensor, r: torch.tensor) -> torch.tensor:
        return r


class RelevancePropagationELU(nn.Module):
    """Layer-wise relevance propagation for ReLU activation.
    Passes the relevance scores without modification. Might be of use later.
    """

    def __init__(self, layer: torch.nn.ELU()) -> None:
        super().__init__()

    @torch.no_grad()
    def forward(self, a: torch.tensor, r: torch.tensor) -> torch.tensor:
        return r


class RelevancePropagationDropout(nn.Module):
    """Layer-wise relevance propagation for dropout layer.
    Passes the relevance scores without modification. Might be of use later.
    """

    def __init__(self, layer: torch.nn.Dropout) -> None:
        super().__init__()

    @torch.no_grad()
    def forward(self, a: torch.tensor, r: torch.tensor) -> torch.tensor:
        return r


class RelevancePropagationTAGCN(nn.Module):
    def __init__(self, layer: TAGConv):
        super().__init__()

        self.layer = layer

        # From source code: dgl/nn/pytorch/conv/tagconv.py
        self.W = self.layer.lin.weight         # 获取该层的权重
        self.b = self.layer.lin.bias           # 获取该层的偏置
        self.K = self.layer._k                 # 获取该层的K阶
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
        Wk = [self.W[:, k*dW:(k+1)*dW].t() for k in range(self.K+1)]  # 把每一个阶的信息抽取出来

        Dm = th.diag(th.pow(graph.in_degrees().float().clamp(min=1), -0.5))
        A = Dm.matmul(graph.adj().to_dense()).matmul(Dm)  # 邻接矩阵标准化
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
        A = Dm.matmul(graph.adj().to_dense()).matmul(Dm)  # 邻接矩阵标准化
        # 分别计算A的k次方
        Ak = [th.matrix_power(A, k) for k in range(self.maxK+1)]
        # 分别计算 k 个多项式卷积核提取图节点的邻域信息, 计算 k 阶多项式, 将K个多项式卷积核提取的 feature_map 拼接,并以此将结果存储到 hs 中
        hs = []
        h = graph.ndata[self.net.features_str].float()
        hs.append(h)
        for layer_param in self.layer_params:
            # k个卷积核在图结构数据上提取特征并和bias进行线性组·合
            h = sum(Ak[k].matmul(h).matmul(layer_param[2][k]) for k in range(layer_param[1]+1))+layer_param[0]
            # 将输入h中每个元素的范围限制到区间 [min,max], 返回结果到一个新张量
            h = h.clamp(min=0)
            hs.append(h)
        return hs, Ak


class RelevancePropagationChebNet(nn.Module):
    def __init__(self, layer: ChebConv):
        super().__init__()

        self.layer = layer

        # From source code: dgl/nn/pytorch/conv/tagconv.py
        self.W = self.layer.linear.weight         # 获取该层的权重
        self.b = self.layer.linear.bias           # 获取该层的偏置
        self.K = self.layer._k                 # 获取该层的K阶
        # print(self.W.shape, self.b.shape, self.K)

    @torch.no_grad()
    def forward(self, a: torch.tensor, r: torch.tensor, graph) -> torch.tensor:
        """
        a: represents the input for each layer
        r: represents the relevance of the latter layer
        """
        rho = lambda Wk: [W.abs() for W in Wk]
        # rho = lambda Wk: [W for W in Wk]
        num_nodes = len(a)

        dW = self.W.shape[1] // (self.K)
        Wk = [self.W[:, k * dW:(k + 1) * dW].t() for k in range(self.K + 1)]

        L = RelevancePropagationChebNet.get_laplacian(graph)  # [N, N]
        Ak = self.cheb_polynomial(L)  # [K, N, N]

        rhoWk = rho(Wk)
        dimY, dimJ = rhoWk[0].shape
        U = th.zeros(num_nodes, dimJ, num_nodes, dimY)  # U's shape is torch.Size([30, 50, 30, 32])

        for i in range(num_nodes):
            for j in range(dimJ):
                U[i, j] = sum(Ak[k][i, :].unsqueeze(-1).matmul(rhoWk[k][:, j].unsqueeze(0)) for k in range(self.K))

        r = th.einsum("ijxy,ij->xy", [U, r / U.sum(dim=(2, 3))])

        return r

    # 将Z一次性计算出来，存到列表中备用
    def cheb_polynomial(self, laplacian):
        """
        Compute the Chebyshev Polynomial, according to the graph laplacian.

        :param laplacian: the graph laplacian, [N, N].
        :return: the multi order Chebyshev laplacian, [K, N, N].
        """
        N = laplacian.size(0)  # [N, N]
        multi_order_laplacian = torch.zeros([self.K, N, N], device=laplacian.device, dtype=torch.float)  # [K, N, N]
        multi_order_laplacian[0] = torch.eye(N, device=laplacian.device, dtype=torch.float)

        if self.K == 1:
            return multi_order_laplacian
        else:
            multi_order_laplacian[1] = laplacian
            if self.K == 2:
                return multi_order_laplacian
            else:
                for k in range(2, self.K):
                    multi_order_laplacian[k] = 2 * torch.mm(laplacian, multi_order_laplacian[k - 1]) - \
                                               multi_order_laplacian[k - 2]
        return multi_order_laplacian

    @staticmethod
    def get_laplacian(graph):
        """
        return the laplacian of the graph.

        :param graph: the graph structure without self loop, [N, N].
        :return: graph laplacian.
        """

        Dm = th.diag(th.pow(graph.in_degrees().float().clamp(min=1), -0.5))
        D = Dm.matmul(graph.adj().to_dense()).matmul(Dm)
        L = torch.eye(D.size(0)) - D
        return L

