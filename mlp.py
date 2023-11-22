import torch.nn as nn

class MisinformationMLP(nn.Module):
    def __init__(self, s_in=768, s_out=3):
        """
        Multi-layer Perceptron (MLP) for misinformation detection.

        Args:
        - s_in (int): Input size, typically the embedding size of BERT (default: 768).
        - s_out (int): Output size, number of classes for classification (default: 3).
        """
        super(MisinformationMLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(s_in, 2 * s_in),
            nn.ReLU(),
            nn.Linear(2 * s_in, s_in),
            nn.ReLU(),
            nn.Linear(s_in, s_out),
            nn.Sigmoid()
        )
        self.loss = nn.CrossEntropyLoss()

    def forward(self, X):
        """
        Forward pass of the MLP model.

        Args:
        - X (torch.Tensor): Input tensor of sentence query combination.

        Returns:
        - Out (torch.Tensor): Output tensor after passing through the MLP.
        """
        Out = self.model(X)
        return Out
