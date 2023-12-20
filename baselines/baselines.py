from torch import nn
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class MisinformationCrossEncoder(nn.Module):
    """
    The Cross-Encoder jointly encodes a pair of query and document into a transformer's [CLS] vector,
    which is then fed into a linear layer for calculating the relevance scores (logits).
    Attributes
    ----------
    model: result of transformers.AutoModelForSequenceClassification.from_pretrained()
        The model is composed of a linear classifier on top of the transformer's backbone.
    loss: torch.nn.CrossEntropyLoss
        Cross entropy loss for training
    """

    def __init__(self, model_name, num_labels) -> None:
        """
        Constructing Cross Encoder
        Parameters
        ----------
        model_name: str
            name of the pretrained model which is used as an argument for
            the method AutoModelForSequenceClassification.from_pretrained
            (See: https://huggingface.co/transformers/v3.0.2/model_doc/auto.html#automodelforsequenceclassification)
        """
        super(MisinformationCrossEncoder, self).__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, batch):
        pairs = [(query, answer) for query, answer in zip(*batch)]
        
        inputs = self.tokenizer(pairs, return_tensors='pt', padding=True, truncation=True)
        inputs = {key: val.to(self.device) for key, val in inputs.items()}  # Move inputs to GPU
        
        outputs = self.model(**inputs)
        logits = outputs.logits
        
        return logits

    @classmethod
    def from_pretrained(cls, model_name_or_dir):
        """
        Load model checkpoint for a path or directory
        Parameters
        ----------
        model_name_or_dir: str
            a HuggingFace's model or path to a local checkpoint
        """
        return cls(model_name_or_dir)


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
            nn.Linear(s_in, s_in // 2),
            nn.ReLU(),
            nn.Linear(s_in // 2, s_in // 3),
            nn.ReLU(),
            nn.Linear(s_in // 3, s_out),
        )

    def forward(self, X):
        """
        Forward pass of the MLP model.

        Args:
        - X (torch.Tensor): Input tensor of sentence query combination.

        Returns:
        - Out (torch.Tensor): Output tensor after passing through the MLP.
        """
        return self.model(X)


class MisinformationPCA(nn.Module):
    def __init__(self, s_in=11, s_out=3):
        """
        Multi-layer Perceptron (MLP) for misinformation detection.

        Args:
        - s_in (int): Input size, typically the embedding size of BERT (default: 768).
        - s_out (int): Output size, number of classes for classification (default: 3).
        """
        super(MisinformationPCA, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(s_in, s_out),
        )

    def forward(self, X):
        """
        Forward pass of the MLP model.

        Args:
        - X (torch.Tensor): Input tensor of sentence query combination.

        Returns:
        - Out (torch.Tensor): Output tensor after passing through the MLP.
        """
        return self.model(X)