import torch
from torch_geometric.utils import degree
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np

LIMIT = True

if torch.cuda.is_available():
    distances = torch.load("data/distances.pt")
    dataset = torch.load("data/graph.pt")
else:
    distances = torch.load("data/distances.pt", map_location=torch.device('cpu'))
    dataset = torch.load("data/graph.pt", map_location=torch.device('cpu'))

# don't want node to connect to itself
distances.fill_diagonal_(-1.)

for threshold in np.arange(0.7, 1., 0.01):
    t = round(threshold, 2)
    edge_index = torch.nonzero(distances >= t)

    dataset.edge_index = edge_index.t().contiguous()
    print(f"Threshold {t}: Total number of edges = {dataset.edge_index.shape[1]}")

    # Get the list of degrees for each node
    degrees = degree(dataset.edge_index[0]).numpy()
    
    isolated_labels = torch.sum(dataset.y[degrees==0], axis = 1)
    class_3 = int(torch.sum(isolated_labels == 3))
    class_2 = int(torch.sum(isolated_labels == 2))
    class_1 = int(torch.sum(isolated_labels == 1))
    class_0 = int(torch.sum(isolated_labels == 0))

    print("Isolated nodes:\t", class_3 + class_2 + class_1 + class_0)
    print("Per class:\t", "3:", class_3, "\t", "2:", class_2, "\t", "1:", class_1, "\t", "0:", class_0, "\n")

    # Count the number of nodes for each degree
    numbers = Counter(degrees)

    # Bar plot
    fig, ax = plt.subplots(figsize=(18, 6))
    ax.set_title(f"Threshold = {t}")
    ax.set_xlabel('Node degree')
    ax.set_ylabel('Number of nodes')
    plt.bar(numbers.keys(),
            numbers.values(),
            color='#0A047A')
    if LIMIT:
        ax.set_xlim(0, 23000)
        plt.savefig(f"images/limit_barplot_{t}.png")
    else:
        plt.savefig(f"images/barplot_{t}.png")
