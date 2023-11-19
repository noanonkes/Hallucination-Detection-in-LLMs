import torch
import numpy as np
import sklearn.metrics

def get_embeddings(model, tokenizer, dataloader, device):
    embeddings = []
    for input, _ in dataloader:
        encoded_input = tokenizer(input, return_tensors='pt').to(device) # padding=True, truncation=True ???

        # Forward pass through the model to get embeddings
        with torch.no_grad():
            output = model(**encoded_input)

        # Extract the embeddings
        cls_embedding = output.last_hidden_state[:, 0, :]
        embeddings.append(cls_embedding)
    
    return torch.tensor(embeddings, dtype=torch.float)

def get_labels(dataloader):
    labels = []
    for _, label in dataloader:
        labels.append(label)
    return torch.tensor(labels, dtype=torch.long)


def get_edge_index(node_features, top_k=5):
    """For now, each node also connects to itself."""

    distances = sklearn.metrics.pairwise.cosine_similarity(node_features)
    indices = np.argsort(distances, axis=1)[:, -top_k:]
    edge_index = []
    for node, idxs in enumerate(indices):
        for idx in idxs:
            edge_index.append([node, idx])
    return torch.tensor(edge_index, dtype=torch.long)
