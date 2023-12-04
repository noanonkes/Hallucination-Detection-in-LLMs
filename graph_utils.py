import torch
import torch.nn.functional as F
from torch_geometric.nn import GAT

from GAT import GAT as ManualGAT


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

def get_model(in_channels, out_channels, hidden_channels=32, activation=F.leaky_relu,
              v2 = False, in_head=2, out_head=1, dropout=0., manual=True, num_layers=2):
    if manual:
        return ManualGAT(n_in=in_channels, hid=hidden_channels,
                     in_head=in_head, out_head=out_head, 
                     n_classes=out_channels, dropout=dropout, 
                     activation=activation, v2=v2)
    else:
        return GAT(in_channels, hidden_channels, num_layers, out_channels, v2=v2, act=activation)


def frequency(graph, model_output):
    freq_matrix = torch.zeros((4, 4), device=graph.y.device)
    # print("Validation output\n", out_val, "\nTargets\n", y_val)
    for i, output in enumerate(model_output[graph.train_idx]):
        round_out = output.sigmoid().round()
        true_class = torch.sum(graph.y[graph.train_idx][i])
        if all(round_out == torch.tensor([1.,1.,1.], device=graph.y.device)):
            freq_matrix[true_class, 3] += 1
        elif all(round_out == torch.tensor([1.,1.,0.], device=graph.y.device)):
            freq_matrix[true_class, 2] += 1
        elif all(round_out == torch.tensor([1.,0.,0.], device=graph.y.device)):
            freq_matrix[true_class, 1] += 1
        elif all(round_out == torch.tensor([0.,0.,0.], device=graph.y.device)):
            freq_matrix[true_class, 0] += 1

    return freq_matrix.long()

def get_optimizer(optimizer, model, lr):
    if optimizer == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    elif optimizer == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    else:
        # uhh, that should be impossible
        ...
    return optimizer


def train_loop(data, model, loss_func, optimizer, batch_indices=None):
    """Train a GAT model for a single epoch."""

    model.train()

    optimizer.zero_grad()

    out = model(data.x, data.edge_index, batch=batch_indices)

    loss = loss_func(out[data.train_idx], data.y[data.train_idx].float())
    loss.backward()

    optimizer.step()
  
    return loss.item(), out


def val_loop(data, out, loss_func, metric, acc, mse, recall):
    """
    Validate a GAT model for a single epoch.
    Do not need model here, because in the train loop it already
    gave output for all data, so use that instead of re-calculating.
    """

    metric.reset()
    acc.reset()
    mse.reset()
    
    with torch.no_grad():

        loss = loss_func(out[data.val_idx], data.y[data.val_idx].float())

        # do not need these to be on GPU, save some space for graph :)
        out_val, y_val = out[data.val_idx].detach().cpu(), data.y[data.val_idx].float().detach().cpu()

        metric.update(out_val.sigmoid(), y_val)
        acc.update(out_val.sigmoid(), y_val)
        mse.update(out_val.sigmoid(), y_val)

    return loss.item(), metric.compute(), acc.compute(), mse.compute(), recall(out_val.sigmoid(), y_val)