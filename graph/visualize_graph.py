import torch, argparse
from torch_geometric.utils import remove_isolated_nodes
from os.path import join as path_join
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import umap


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # command line args for specifying the situation
    parser.add_argument("--use-cuda", action="store_true", default=False,
                        help="Use GPU acceleration if available")
    parser.add_argument("--path", type=str, default="../data/",
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

    # removing isolated nodes
    isolated = (remove_isolated_nodes(graph["edge_index"])[2] == False).sum(dim=0).item()
    print(f"Number of isolated nodes = {isolated}\n")

    embedder_file = f"embedder_act_ReLU_opt_AdamW_lr_0.0001_bs_256_t_0.07_998.pt"
    embedder = torch.nn.Sequential(*[torch.nn.Linear(768, 768), torch.nn.ReLU(), torch.nn.Linear(768, 128)])
    embedder.load_state_dict(torch.load(path_join("../weights", embedder_file), map_location=device)["state_dict"])
    embedder.to(device)
    embeddings = embedder(graph.x)
    
    reducer = umap.UMAP()
    embedded_cl = reducer.fit_transform(embeddings.detach().cpu().numpy())

    reducer = umap.UMAP()
    embedded_graph = reducer.fit_transform(graph.x.detach().cpu().numpy())

    classes = ["False", "TwoC", "TwC", "T"]
    values = torch.sum(graph.y, dim=-1).detach().cpu().numpy()
    colors = ListedColormap(['tab:red', 'tab:blue', 'tab:orange', 'tab:green'])

    fig, (ax2, ax1) = plt.subplots(1, 2, figsize=(8,4), sharex=True, sharey=True)

    ax1.scatter(
    embedded_cl[:, 0],
    embedded_cl[:, 1],
    c = values,
    cmap=colors,
    s=5,
    alpha=0.7)

    ax1.set_xlabel("d1", fontsize=10)
    ax1.set_ylabel("d2", fontsize=10)

    ax1.set_title('Contrastive learning', fontsize=14)

    ax2.scatter(
    embedded_graph[:, 0],
    embedded_graph[:, 1],
    c = values,
    cmap=colors,
    s=5,
    alpha=0.7)

    ax2.set_xlabel("d1", fontsize=10)
    ax2.set_ylabel("d2", fontsize=10)

    ax2.set_title('Graph node features', fontsize=14)

    plt.savefig(path_join(args.output_dir, "plot_full.png"))