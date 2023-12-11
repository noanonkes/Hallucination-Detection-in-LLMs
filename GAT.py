from torch.nn import Linear
from torch_geometric.nn import GATConv
import torch

class GAT(torch.nn.Module):
    """
    Implementation of a Graph Attention Network (GAT) model.

    Args:
        embedder (torch.nn.ModuleList): List of torch.nn.Module instances constituting
            the embedding layers.
        n_in (int): Dimensionality of input features (default: 768).
        hid (int): Dimensionality of hidden layer (default: 32).
        in_head (int): Number of attention heads in the first GAT layer (default: 2).
        out_head (int): Number of attention heads in the second GAT layer (default: 1).
        n_classes (int): Number of output classes (default: 3).
        dropout (float): Dropout probability (default: 0.0).

    Raises:
        ValueError: If the input dimension of the embedder does not match `n_in`.

    Attributes:
        embedder (torch.nn.ModuleList): List of torch.nn.Module instances for embedding.
        linear (torch.nn.Linear): Linear layer for dimensionality reduction.
        conv1 (torch_geometric.nn.GATConv): First Graph Attention layer.
        conv2 (torch_geometric.nn.GATConv): Second Graph Attention layer.

    """

    def __init__(self, embedder, n_in=768, hid=32, in_head=2, out_head=1, n_classes=3, dropout=0.):
        super(GAT, self).__init__()

        if embedder[0].in_features != n_in:
            raise ValueError("The embedder does not have the correct input dimension.")

        self.embedder = embedder
        self.linear = Linear(embedder[-1].out_features, hid)
        self.conv1 = GATConv(hid, hid//2, heads=in_head, dropout=dropout)
        self.conv2 = GATConv((hid//2) * in_head, n_classes, concat=False,
                            heads=out_head, dropout=dropout)

    def forward(self, x, edge_index):
        """
        Forward pass of the GAT model.

        Args:
            x (Tensor): Node features.
            edge_index (LongTensor): Graph edge indices.

        Returns:
            Tensor: Output tensor after passing through the GAT layers.
        """
        # reduce dimensionality of node features
        x = self.embedder(x)

        # linear layer
        x = self.linear(x)

        # first GAT layer
        x = self.conv1(x, edge_index)

        # second GAT layer
        x = self.conv2(x, edge_index)

        return x
