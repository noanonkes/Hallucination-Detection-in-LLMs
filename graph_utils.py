import torch
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

def get_edge_index(node_features, distances=None, threshold=.9):
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
