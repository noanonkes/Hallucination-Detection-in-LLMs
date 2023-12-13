import torch, argparse
from torch_geometric.utils import remove_isolated_nodes, to_undirected
from os.path import join as path_join
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import umap

import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # command line args for specifying the situation
    parser.add_argument("--use-cuda", action="store_true", default=False,
                        help="Use GPU acceleration if available")
    parser.add_argument("--path", type=str, default="data/",
                        help="Path to the data folder")
    parser.add_argument("--output_dir", type=str, default="images/",
                        help="Path to save plot to")

    args = parser.parse_args()

    print("STARTING...  setup:")
    print(args)
    print("-" * 120)
    print("\n" * 2)

    # some paramaters
    if args.use_cuda:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    else:
        device = torch.device("cpu")

    # load graph
    graph = torch.load(path_join(args.path, "graph.pt"), map_location=device)
    graph.to(device)

    # # removing isolated nodes
    # isolated = (remove_isolated_nodes(graph["edge_index"])[2] == False).sum(dim=0).item()
    # print(f"Number of isolated nodes = {isolated}\n")

    freq = torch.zeros((4, 4))
    all_labels = graph.y.sum(-1)
    for i, node_i in enumerate(graph.edge_index[0]):
        node_j = graph.edge_index[1, i]
        label_i, label_j = all_labels[node_i], all_labels[node_j]
        freq[label_i, label_j] += 1

    print(freq.long())
    
    reducer = umap.UMAP()
    embeddings = reducer.fit_transform(graph.x.detach().cpu().numpy())

    classes = ["False", "TwoC", "TwC", "T"]
    values = torch.sum(graph.y, dim=-1).detach().cpu().numpy()
    colors = ListedColormap(['tab:red', 'tab:blue', 'tab:orange', 'tab:green'])

    plt.figure(figsize=(18,10))
    scatter = plt.scatter(
    embeddings[:, 0],
    embeddings[:, 1],
    c = values,
    cmap=colors,
    s=5,
    alpha=0.7)

    plt.legend(handles=scatter.legend_elements()[0], labels=classes)

    plt.title('UMAP projection of the dataset', fontsize=24)

    plt.savefig(path_join(args.output_dir, "plot.png"))

