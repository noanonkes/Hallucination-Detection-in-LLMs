import torch
import torch.nn.functional as F
import numpy as np


def set_seed(seed):
    """
    Function for setting the seed for reproducibility.

    Args:
    - seed (int): Seed value for random number generators.
    """

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_embeddings(model, tokenizer, dataloader, device):
    """
    Given a model and tokenizer, maps all the sentences
    in the dataset to the given model embedding.

    Args:
    - model (torch.nn.Module): Model for generating embeddings.
    - tokenizer (object): Tokenizer object.
    - dataloader (torch.utils.data.DataLoader): DataLoader object.
    - device (torch.device): Device where the model and data reside.

    Returns:
    - torch.Tensor: Concatenated embeddings of the sentences.
    """

    embeddings = []
    for i, (_, inputs, _) in enumerate(dataloader):
        # manually add the CLS token
        inputs = ["[CLS] " + s for mult in inputs for s in mult]
        encoded_inputs = tokenizer(inputs, return_tensors='pt', padding=True).to(device)

        # forward pass through the model to get embeddings
        with torch.no_grad():
            output = model(**encoded_inputs)

        # extract the CLS embedding
        cls_embedding = output.last_hidden_state[:, 0, :]
        embeddings.append(cls_embedding)

    return torch.concat(embeddings)


def get_labels(dataloader):
    """
    Collects all the labels of the dataloader.

    Args:
    - dataloader (torch.utils.data.DataLoader): DataLoader object.

    Returns:
    - torch.Tensor: Concatenated labels.
    """

    labels = []
    for i, (_, _, targets) in enumerate(dataloader):
        labels.append(targets[0])
    
    return torch.concat(labels)


def get_distances(node_features):
    """
    Computes cosine similarity between node features to derive distances.

    Args:
    - node_features (torch.Tensor): Node features tensor.

    Returns:
    - torch.Tensor: Matrix of distances between node features.
    """

    distances = torch.zeros((node_features.shape[0], node_features.shape[0]), dtype=node_features.dtype, device=node_features.device)
    for i, node in enumerate(node_features):
        distances[i] = F.cosine_similarity(node, node_features, -1)    
    return distances


def get_edge_index(node_features, distances=None, threshold=.85):
    """
    Generates edge indices based on cosine similarity and a given threshold.

    Args:
    - node_features (torch.Tensor): Node features tensor.
    - distances (torch.Tensor, optional): Precomputed distances matrix (default: None).
    - threshold (float): Similarity threshold for creating edges (default: 0.85).

    Returns:
    - torch.Tensor: Matrix of distances between node features.
    - torch.Tensor: Edge indices satisfying the similarity threshold.
    """

    if distances is None:
        distances = get_distances(node_features)

    # do not want to connect node to itself, inplace operation
    distances.fill_diagonal_(-1.)

    edge_index = torch.nonzero(distances >= threshold)

    return distances, edge_index


def rewrite_labels(labels):
    """
    Rewrite the labels from vectors to integers based on specific conditions.

    Args:
    - labels (torch.Tensor): Labels tensor.

    Returns:
    - torch.Tensor: Transformed labels tensor adhering to defined conditions.
    """

    new_labels = torch.empty_like(labels[:, 0])
    for i, label in enumerate(labels):
        round_out = label.sigmoid().round()
        if round_out[-1] == 1.:
            new_labels[i] = 3
        elif round_out[-2] == 1.:
            new_labels[i] = 2
        elif round_out[-3] == 1.:
            new_labels[i] = 1
        elif round_out.sum() == 0.:
            new_labels[i] = 0
    return new_labels

def rewrite_labels_binary(labels):
    "Assumes only label 0 to be false and all other positive labels to be true"
    return torch.where(labels == 0, 0, 1)

def confusion_matrix(conf_metric, inputs, targets):
    """
    Computes confusion matrix based on predicted and target labels.

    Args:
    - conf_metric: Confusion matrix metric object.
    - inputs (torch.Tensor): Predicted labels.
    - targets (torch.Tensor): Target labels.

    Returns:
    - torch.Tensor: Computed confusion matrix value.
    """

    conf_metric.reset()
    conf_metric.update(inputs, targets)
    return conf_metric.compute()


def accuracy(acc_metric, inputs, targets):
    """
    Computes accuracy metric based on predicted and target labels.

    Args:
    - acc_metric: Accuracy metric object.
    - inputs (torch.Tensor): Predicted labels.
    - targets (torch.Tensor): Target labels.

    Returns:
    - float: Computed accuracy value.
    """

    acc_metric.reset()
    acc_metric.update(inputs, targets)
    return acc_metric.compute().item()


def macro_recall(recall_metric, inputs, targets):
    """
    Computes macro recall metric based on predicted and target labels.

    Args:
    - recall_metric: Macro recall metric object.
    - inputs (torch.Tensor): Predicted labels.
    - targets (torch.Tensor): Target labels.

    Returns:
    - float: Computed macro recall value.
    """

    recall_metric.reset()
    recall_metric.update(inputs, targets)
    return recall_metric.compute().item()


def binary_accuracy(acc_metric, inputs, targets):
    """
    Computes binary accuracy metric based on predicted and target labels.

    Args:
    - recall_metric: Binary accuracy metric object.
    - inputs (torch.Tensor): Predicted labels.
    - targets (torch.Tensor): Target labels.

    Returns:
    - float: Computed binary accuracy value.
    """

    acc_metric.reset()
    acc_metric.update(inputs, targets)
    return acc_metric.compute().item()


def binary_recall(recall_metric, inputs, targets):
    """
    Computes binary recall metric based on predicted and target labels.

    Args:
    - recall_metric: Binary recall metric object.
    - inputs (torch.Tensor): Predicted labels.
    - targets (torch.Tensor): Target labels.

    Returns:
    - float: Computed binary recall value.
    """

    recall_metric.reset()
    recall_metric.update(inputs, targets)
    return recall_metric.compute().item()


def get_optimizer(optimizer, model, lr):
    """
    Returns the specified optimizer based on the given parameters.

    Args:
    - optimizer (str): Name of the optimizer.
    - model (torch.nn.Module): Model to optimize.
    - lr (float): Learning rate.

    Returns:
    - torch.optim.Optimizer: Optimizer object.
    """

    if optimizer == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    elif optimizer == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    else:
        raise ValueError("Unknown optimizer")
    return optimizer


def train_loop(data, model, loss_func, optimizer):
    """
    Train a GAT model for a single epoch.

    Args:
    - data: Graph data.
    - model (torch.nn.Module): Model to be trained.
    - loss_func: Loss function.
    - optimizer: Optimizer for training.

    Returns:
    - float: Training loss value.
    - torch.Tensor: Model output.
    """

    model.train()

    optimizer.zero_grad()

    out = model(data.x, data.edge_index)

    loss = loss_func(out[data.train_idx], data.y[data.train_idx].float())
    loss.backward()

    optimizer.step()
  
    return loss.item(), out