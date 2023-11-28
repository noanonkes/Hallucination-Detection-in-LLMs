import torch
from torch_geometric.utils import degree
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np

if torch.cuda.is_available():
    distances = torch.load("data/distances.pt")
    dataset = torch.load("data/graph.pt")
else:
    distances = torch.load("data/distances.pt", map_location=torch.device('cpu'))
    dataset = torch.load("data/graph.pt", map_location=torch.device('cpu'))


for threshold in np.arange(0.7, 1, 0.01):
    edge_index = torch.nonzero(distances >= threshold)
    dataset.edge_index = edge_index.t().contiguous()

    # Get the list of degrees for each node
    degrees = degree(dataset.edge_index[0]).numpy()
    print(degrees)

    # Count the number of nodes for each degree
    numbers = Counter(degrees)

    # Bar plot
    fig, ax = plt.subplots(figsize=(18, 6))
    ax.set_title(f"Threshold = {threshold}")
    ax.set_xlabel('Node degree')
    ax.set_ylabel('Number of nodes')
    plt.bar(numbers.keys(),
            numbers.values(),
            color='#0A047A')
    plt.savefig(f"images/barplot_{threshold}.png")
