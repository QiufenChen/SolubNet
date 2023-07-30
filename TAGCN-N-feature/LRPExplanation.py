# Author QFIUNE
# coding=utf-8
# @Time: 2022/6/17 11:20
# @File: LRPExplanation.py
# @Software: PyCharm
# @contact: 1760812842@qq.com
import os
from copy import deepcopy

import dgl
import torch
import torch as th
import mtMolDes
from torch import nn
from lrp import RelevancePropagationLinear, RelevancePropagationReLU, RelevancePropagationELU,\
    RelevancePropagationDropout, RelevancePropagationTAGCN, RelevancePropagationChebNet, linear_ww_lrp
from dgl.nn.pytorch import GraphConv, TAGConv, ChebConv, GATConv


def layers_lookup() -> dict:
    """
    Lookup table to map network layer to associated lrpDes operation.
    Returns:
        Dictionary holding class mappings.
    """
    lookup_table = {
        # # torch.nn.modules.linear.Linear: RelevancePropagationLinear,
        # torch.nn.modules.linear.Linear: linear_ww_lrp,
        # torch.nn.modules.activation.ELU: RelevancePropagationELU,
        # torch.nn.modules.dropout.Dropout: RelevancePropagationDropout,
        dgl.nn.pytorch.conv.tagconv.TAGConv: RelevancePropagationTAGCN,
        # dgl.nn.pytorch.conv.chebconv.ChebConv: RelevancePropagationChebNet
    }
    return lookup_table


class LRPModel(nn.Module):
    """Class wraps PyTorch model to perform layer-wise relevance propagation."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model
        self.model.eval()  # self.model.train() activates dropout etc.!

        # Parse solubNet
        self.layers = self._get_layer_operations(model)

        # Create LRP network
        self.lrp_layers = self._create_lrp_model()

    def _get_layer_operations(self, model) -> torch.nn.ModuleList:
        """
        Get all network operations and store them in a list.
        Returns:
            Layers of original model stored in module list.
        """
        layers = torch.nn.ModuleList()

        # Parse solubNet
        for layer in model.gcn_layers:
            layers.append(layer)

        return layers

    def _create_lrp_model(self) -> torch.nn.ModuleList:
        """Method builds the model for layer-wise relevance propagation.
        Returns:
            LRP-model as module list.
        """
        # Clone layers from original model. This is necessary as we might modify the weights.
        layers = deepcopy(self.layers)
        lookup_table = layers_lookup()

        # Run backwards through layers
        for i, layer in enumerate(layers[::-1]):
            try:
                layers[i] = lookup_table[layer.__class__](layer=layer)
            except KeyError:
                message = f"Layer-wise relevance propagation not implemented for " \
                          f"{layer.__class__.__name__} layer."
                raise NotImplementedError(message)
        return layers

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        Forward method that first performs standard inference followed by layer-wise relevance propagation.
        Args:
            x: Input tensor representing a molecule.
        Returns:
            Tensor holding relevance scores.
        """
        activations = list()
        all_relevance = []

        # Run inference and collect activations.
        graph = x
        x = x.ndata['h'].float()

        elu = nn.ELU()
        relu = nn.ReLU()
        drop = nn.Dropout(0.5)

        with torch.no_grad():
            activations.append(torch.ones_like(x))

            # x = relu(self.layers[0].forward(graph, x))
            x = self.layers[0].forward(graph, x)
            activations.append(x)

            # x = relu(self.layers[1].forward(graph, x))
            x = self.layers[1].forward(graph, x)
            activations.append(x)

            x = self.layers[2].forward(graph, x)
            activations.append(x)


        # Reverse order of activations to run backwards through model
        activations = activations[::-1]
        activations = [a.data.requires_grad_(True) for a in activations]
        #
        # for item in activations:
        #     print(item.shape)
        #     print('-'*50)

        # Initial relevance scores are the network's output activations
        relevance = activations.pop(0)
        # print('Last layer: ', relevance)
        # relevance = torch.tensor([[-4.3845],
        #         [-5.6434],
        #         [1.6733],
        #         [-7.3403],
        #         [-5.0330],
        #         [-5.3756],
        #         [-5.1321],
        #         [-5.0837],
        #         [-5.2046],
        #         [-4.7496],
        #         [-5.7200],
        #         [-4.9097],
        #         [-5.2097],
        #         [-5.8794],
        #         [-5.3714],
        #         [-5.3850],
        #         [-5.5021],
        #         [-5.0978],
        #         [-6.6148],
        #         [-5.6805],
        #         [-4.4220],
        #         [-4.5804],
        #         [-5.5642],
        #         [-4.5659],
        #         [-5.1246],
        #         [-4.7308],
        #         [6.6576]])
        all_relevance.append(relevance)
        # print('The relevance of last layer: ', relevance.shape)

        # Perform relevance propagation
        for i, layer in enumerate(self.lrp_layers):
            relevance = layer.forward(activations.pop(0), relevance, graph)
            all_relevance.append(relevance)

        # print(all_relevance[-1])
        return all_relevance


