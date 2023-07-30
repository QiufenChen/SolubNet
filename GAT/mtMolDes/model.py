import torch as th
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv, TAGConv, ChebConv, GATConv

class GCNNet(nn.Module):
    """The block unit to percept the topology (bonding) of a molecule."""

    def __init__(self, num_features, num_labels, features_str):
        """Initialize the class.

        Args:
            num_features (int): The dimension of features for all atoms.
            num_labels (int):   The dimension of labels for all atoms.
            feature_str (str):  The string to access the atomic features.
        """
        super(GCNNet, self).__init__()
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        #  Begin of network structure definition.
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        self.gcn_layers = nn.ModuleList()
        num_hiddens = 32

        self.gcn_layers.append(GATConv(num_features, num_hiddens, num_heads=3))
        self.gcn_layers.append(GATConv(num_hiddens, num_hiddens, num_heads=3))
        self.gcn_layers.append(GATConv(num_hiddens, num_labels, num_heads=3))

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        #  End of network structure definition.
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        self.features_str = features_str

    def forward(self, graph):
        """Forward function.

        The input and output is a :math:`N_\text{atom}\times N_\text{feature}` and :math:`N_\text{atom}\times N_\text{label}` matrix, respectively.

        Args:
            graph (dgl graph): The graphs input to the layer.        

        Returns:
           The output vectors. 
        """
        h = graph.ndata[self.features_str]
        relu = nn.ReLU()
        h = relu(self.gcn_layers[0](graph, h))
        h = relu(self.gcn_layers[1](graph, h))
        h = self.gcn_layers[2](graph, h)
        
        return h

    
    def save(self, fn):
        """Save the model for later use.
        
        Args:
            fn (str): The model will be saved as fn.
        """
        pass
        # th.save(self.gcn_layers, fn)

    def load(self, fn):
        """Load the model from a file.

        Both the structure and parameters will be loaded.
        
        Args:
            fn (str): The file to load into the model.
        """
        saved_net = th.load(fn, map_location=th.device('cpu'))
        self.gcn_layers = saved_net

    def to(self, device):
        """Store the model to device (CPU or GPU)

        Args:
            device (str): The device (CPU or GPU) to store the model.
        """        
        self.gcn_layers.to(device)

    def export(self):
        """Export the parameters to reproduce the results.

        Returns:
            [[b, K, Wk]]: The bias, number of hops, and weights for all hops. 
        """
        params = []
        for layer in self.gcn_layers:
            W = layer.lin.weight
            b = layer.lin.bias
            K = layer._k
            dW = W.shape[1]//(K+1)
            Wk = [W[:, k*dW:(k+1)*dW].t() for k in range(K+1)]
            params.append([b, K, Wk])
        # print(len(params))
        return params
