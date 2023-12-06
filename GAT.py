import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GATv2Conv, GATConv
import torch


class GAT(torch.nn.Module):
    """
    Configurations to test:
        - change the hidden dimensions; 128 -> 16
        - change the in_heads 4 -> 2
        - try with and without dropout, also changing the degree of dropout
        - different activations functions
        - try dropedge
        - different GAT layers
    """
    def __init__(self, n_in=768, hid=64, in_head=2, out_head=1, n_classes=3, dropout=0., activation=None, v2=False):
        super(GAT, self).__init__()        
        
        self.linear = Linear(n_in, hid)
        if v2:
            self.conv1 = GATv2Conv(hid, hid//2, heads=in_head, dropout=dropout)
            self.conv2 = GATv2Conv((hid//2) * in_head, n_classes, concat=False,
                             heads=out_head, dropout=dropout)
        else:
            self.conv1 = GATConv(hid, hid//2, heads=in_head, dropout=dropout)
            self.conv2 = GATConv((hid//2) * in_head, n_classes, concat=False,
                                heads=out_head, dropout=dropout)
        self.activation = activation

    def forward(self, x, edge_index, batch=None):
        """Need the batch variable for it to work as predefined GAT model."""
        
        # reduce dimensionality of node features
        x = self.linear(x)

        if self.activation is not None:
            x = self.activation(x)

        # first GAT layer
        x = self.conv1(x, edge_index)

        if self.activation is not None:
            x = self.activation(x)

        # second GAT layer
        x = self.conv2(x, edge_index)

        # DO THIS HERE?
        # get predictions
        # x = x.sigmoid().round()
        # final_one = (x==1).nonzero(as_tuple=True)[0][-1]
        # x[:final_one+1] = 1.
        
        return x