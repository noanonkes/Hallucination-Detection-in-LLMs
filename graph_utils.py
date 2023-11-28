import torch
import numpy as np
import sklearn.metrics

def get_embeddings(model, tokenizer, dataloader, device):
    """
    Expects batch size of dataloader to be 1.

    Given a model and tokenizer, maps all the sentences
    in the dataset to the givne model embedding.
    """
    embeddings = []
    for _, inputs, _ in dataloader:
        inputs = ["[CLS] " + s for s in inputs]
        # manually add the CLS token
        encoded_inputs = tokenizer(inputs, return_tensors='pt').to(device)

        # forward pass through the model to get embeddings
        with torch.no_grad():
            output = model(**encoded_inputs)

        # extract the CLS embedding
        cls_embedding = output.last_hidden_state[:, 0, :]
        embeddings.append(cls_embedding)
    return torch.stack(embeddings)

def get_labels(dataloader):
    """
    Expects batch size of dataloader to be 1.

    Collects all the labels of the dataloader.
    """
    labels = []
    for _, _, targets in dataloader:
        labels.append(targets)
    return torch.stack(labels)

def get_edge_index(node_features, threshold=.7):
    """
    If the cosine similarity between two node features is greater than 
    the threshold, they will be connected via an edge.
    """
    node_features = node_features.cpu().detach().numpy()

    # calculate the cosine similarity between each node pair
    distances = sklearn.metrics.pairwise.cosine_similarity(node_features)

    # do not want to connect node to itself, inplace operation
    np.fill_diagonal(distances, -1.)

    edge_index = []
    for node_i, distances_i in enumerate(distances):
        for node_j, distance_ij in enumerate(distances_i):
            if distance_ij >= threshold:
                edge_index.append(torch.tensor([node_i, node_j]))

    return torch.stack(edge_index)
