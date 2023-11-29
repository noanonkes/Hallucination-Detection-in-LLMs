import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
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
    def __init__(self, n_in=768, hid=8, in_head=2, out_head=1, n_classes=3):
        super(GAT, self).__init__()        
        
        self.conv1 = GATv2Conv(n_in, hid, heads=in_head, dropout=0.6)
        self.conv2 = GATv2Conv(hid*in_head, n_classes, concat=False,
                             heads=out_head, dropout=0.6)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
                
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        
        return x