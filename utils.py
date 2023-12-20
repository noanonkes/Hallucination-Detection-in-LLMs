import torch
import numpy as np

from sklearn.decomposition import PCA
from torchmetrics.regression import MeanSquaredError
from torcheval.metrics import MultilabelAUPRC, MultilabelAccuracy, MulticlassPrecision, MulticlassRecall, BinaryRecall


class EvalManager:
    def __init__(self, device):
        self.auprc = MultilabelAUPRC(num_labels=4).to(device)
        self.accuracy = MultilabelAccuracy().to(device)
        self.precision = MulticlassPrecision(num_classes=4, average="macro").to(device)
        self.recall = MulticlassRecall(num_classes=4, average="macro").to(device)
        self.b_recall = BinaryRecall()
        

    def reset_all(self):
        """
        Resets all metrics to their initial state.
        """
        self.auprc.reset()
        self.accuracy.reset()
        self.precision.reset()
        self.recall.reset()
        self.b_recall.reset()

    def update_all(self, output, target):
        """
        Updates and computes metrics based on output and target tensors.

        Args:
            output (tensor): Predicted output tensor.
            target (tensor): Ground truth target tensor.
        """
        self.auprc.update(output, target)
        self.accuracy.update(output, target)
        self.precision.update(output, target)
        self.recall.update(output, target)
        self.b_recall.update(output, target)
        
    def compute_all(self):
        """
        Returns a dictionary containing all metrics.

        Returns:
            dict: A dictionary containing metric names as keys and corresponding metric instances.
        """
        return {
            'AUPRC': self.auprc.compute().item(),
            'Accuracy': self.accuracy.compute().item(),
            'Precision': self.precision.compute().item(),
            'Recall': self.recall.compute().item(),
            'Binary': self.b_recall.compute().item()
        }

def compute_embeddings(pairs, tokenizer, pretrained, device):
    """
    Calculate embeddings for the input pairs.

    Args:
        pairs: Input pairs.
        tokenizer: Tokenizer associated with the pre-trained model.
        pretrained: Pre-trained model for embedding.
        device (torch.device): Device to run the model (CPU or GPU).
        reduce (bool): If True, reduce the input embeddings using PCA.

    Returns:
        embeddings: Embeddings for the input pairs.
    """
    # Concatenate pairs and add special tokens, then encode
    combined_inputs = ["[CLS] " + query + " [SEP] " + answer for query, answer in zip(*pairs)]
    encoded_inputs = tokenizer(combined_inputs, return_tensors='pt', padding=True, truncation=True).to(device)

    # Forward pass through the model to get embeddings
    with torch.no_grad():
        outputs = pretrained(**encoded_inputs)

    # Extract embeddings for [CLS] tokens (the first token)
    embeddings = outputs.last_hidden_state[:, 0, :]

    return embeddings

def reduce_dimensionality(embeddings, n_components):
    # Apply PCA to reduce dimensionality
    pca = PCA(n_components)
    reduced_embeddings = pca.fit_transform(embeddings.cpu().numpy())
    reduced_embeddings = torch.tensor(reduced_embeddings)
    
    return reduced_embeddings

def train(model, dataloader, criterion, optimizer, device, reduce=0):
    model.train()
    train_loss = 0.0

    for inputs, targets in dataloader:
        inputs = reduce_dimensionality(inputs, reduce) if reduce != 0 else inputs

        inputs = inputs.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        # Track total loss
        train_loss += loss.item()

    # Get the average loss per sample
    train_loss /= len(dataloader)
    return train_loss

def validate(model, dataloader, criterion, metrics, device, reduce=0):
    model.eval()
    val_loss = 0.0
    metrics.reset_all()

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = reduce_dimensionality(inputs, reduce) if reduce != 0 else inputs
            print(inputs)
            
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Track total loss
            val_loss += loss.item()

            # Update metrics
            metrics.update_all(outputs.sigmoid(), targets)

    # Get the average loss per sample
    val_loss /= len(dataloader)
    return val_loss, metrics.compute_all()

def set_seed(seed):
    """
    Function for setting the seed for reproducibility.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
