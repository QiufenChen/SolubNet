import torch as th
import torch.nn as nn

class AtomRedBlock(nn.Module):
    """The block unit to reduce all atomic features to a single low-dimension vector."""

    def __init__(self, num_features, num_labels):
        """Initialize the class.

        Args:
            num_features (int): The dimension of input features.
            num_labels (int):   The dimension of output labels.
        """
        super(AtomRedBlock, self).__init__()
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        #  Begin of network structure definition.
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #        
        self.fc_layers = nn.ModuleList()
        num_hiddens = 32
        self.fc_layers.append(nn.Dropout(0.5))
        self.fc_layers.append(nn.Linear(num_features, num_hiddens))
        self.fc_layers.append(nn.Dropout(0.5))
        self.fc_layers.append(nn.Linear(num_hiddens, num_hiddens*2))
        self.fc_layers.append(nn.Dropout(0.5))
        self.fc_layers.append(nn.Linear(num_hiddens*2, num_labels))
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        #  End of network structure definition.
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    def forward(self, x):
        """Forward function.

        The input and output is a :math:`N_\text{atom}\times N_\text{feature}` and :math:`1 times N_\text{label}` matrix, respectively.

        Args:
            x (torch tensor): The input features.

        Returns:
           The reduced vector.
        """
        h = x  # num_atoms x num_features
        
        leakyrelu = nn.LeakyReLU(negative_slope=0.1, inplace=False)
        elu = nn.ELU()
        # softplus = nn.Softplus(beta=1, threshold=20)
        relu = nn.ReLU()

        h = self.fc_layers[0](h)
        h = elu(self.fc_layers[1](h))
        h = self.fc_layers[2](h)
        h = elu(self.fc_layers[3](h))
        h = self.fc_layers[4](h)
        h = self.fc_layers[5](h)
        # h = h.sum(0)  # 1 x num_output
        # h = th.min(h, dim=0)[0]

        # h = th.max(h, dim=0)[0]
        # print('this is last h', h.shape, h)
        h = th.mean(h, dim=0)[0]
      
        return h

    def save(self, fn):
        """Save the model for later use.
        
        Args:
            fn (str): The model will be saved as fn.
        """
        th.save(self.fc_layers, fn)

    def load(self, fn):
        """Load the model from a file.

        Both the structure and parameters will be loaded.
        
        Args:
            fn (str): The file to load into the model.
        """
        saved_net = th.load(fn, map_location = th.device('cpu'))
        self.fc_layers = saved_net

    def to(self, device):
        """Store the model to device (CPU or GPU)

        Args:
            device (str): The device (CPU or GPU) to store the model.
        """        
        self.fc_layers.to(device)