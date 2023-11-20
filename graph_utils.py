import torch
import numpy as np
import sklearn.metrics

def get_embeddings(model, tokenizer, dataloader, device):
    """Expects batch size of dataloader to be 1!"""
    embeddings = []
    for input, _ in dataloader:
        encoded_input = tokenizer(input, return_tensors='pt').to(device) # padding=True, truncation=True ???

        # Forward pass through the model to get embeddings
        with torch.no_grad():
            output = model(**encoded_input)

        # Extract the embeddings
        cls_embedding = output.last_hidden_state[:, 0, :]
        embeddings.append(cls_embedding[0])
    return torch.stack(embeddings)

def get_labels(dataloader):
    """Expects batch size of dataloader to be 1!"""
    labels = []
    for _, label in dataloader:
        labels.append(label[0])
    return torch.stack(labels)

def get_edge_index(node_features, top_k=5):
    """For now, each node also connects to itself."""
    node_features = node_features.cpu().detach().numpy()
    distances = sklearn.metrics.pairwise.cosine_similarity(node_features)
    indices = np.argsort(distances, axis=1)[:, -top_k:]
    edge_index = []
    for node, idxs in enumerate(indices):
        for idx in idxs:
            edge_index.append(torch.tensor([node, idx]))
    return torch.stack(edge_index)
