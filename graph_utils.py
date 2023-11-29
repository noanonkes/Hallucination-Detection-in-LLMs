import torch
from tqdm import tqdm
import torch.nn.functional as F

def get_embeddings(model, tokenizer, dataloader, device):
    """
    Expects batch size of dataloader to be 1.

    Given a model and tokenizer, maps all the sentences
    in the dataset to the givne model embedding.
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
    Expects batch size of dataloader to be 1.

    Collects all the labels of the dataloader.
    """
    labels = []
    for i, (_, _, targets) in enumerate(dataloader):
        labels.append(targets[0])
    
    return torch.concat(labels)


def get_distances(node_features):
    distances = torch.zeros((node_features.shape[0], node_features.shape[0]), dtype=node_features.dtype, device=node_features.device)
    for i, node in enumerate(node_features):
        distances[i] = F.cosine_similarity(node, node_features, -1)    
    return distances


def get_edge_index(node_features, distances=None, threshold=.85):
    """
    If the cosine similarity between two node features is greater than 
    the threshold, they will be connected via an edge.
    """
    if distances is None:
        distances = get_distances(node_features)

    # do not want to connect node to itself, inplace operation
    distances.fill_diagonal_(-1.)

    edge_index = torch.nonzero(distances >= threshold)

    return distances, edge_index


def train_loop(data, model, loss_func, optimizer):
    """Train a GAT model."""

    model.train()

    optimizer.zero_grad()

    out = model(data)

    loss = loss_func(out[data.train_idx], data.y[data.train_idx].float())
    loss.backward()

    optimizer.step()
  
    return loss.item()


def val_loop(data, model, loss_func, metric, acc, mse):
    """Validate a GAT model."""

    model.eval()

    metric.reset()
    acc.reset()
    mse.reset()
    
    with torch.no_grad():
        out = model(data)

        loss = loss_func(out[data.val_idx], data.y[data.val_idx].float())

        # do not need these to be on GPU, save some space for graph :)
        out_val, y_val = out[data.val_idx].detach().cpu(), data.y[data.val_idx].float().detach().cpu()

        metric.update(out_val.sigmoid(), y_val)
        acc.update(out_val.sigmoid(), y_val)
        mse.update(out_val.sigmoid(), y_val)

    return loss.item(), metric.compute(), acc.compute(), mse.compute()