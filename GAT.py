import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
import torch
class GAT(torch.nn.Module):
    def __init__(self, n_in=768, hid=768//2, in_head=8, out_head=1, n_classes=3):
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