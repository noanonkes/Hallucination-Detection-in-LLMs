import torch, argparse
from torcheval.metrics import MulticlassConfusionMatrix, MulticlassAccuracy, MulticlassRecall, BinaryRecall
from torch_geometric.utils import remove_isolated_nodes

from GAT import GAT
import graph_utils

from os.path import join as path_join
torch.set_printoptions(profile="full")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # command line args for specifying the situation
    parser.add_argument("--use-cuda", action="store_true", default=False,
                        help="Use GPU acceleration if available")
    parser.add_argument("--path", type=str, default="data/",
                        help="Path to the data folder")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of cores to use when loading the data")
    parser.add_argument("--load-model", type=str, default="weights/950_GAT_483.pt",
                        help="GAT model weights to use.")
    parser.add_argument("--mode", type=str, default="val",
                        choices=["train", "val", "holdout", "test"],
                        help="Mode for evaluation")
    args = parser.parse_args()

    # for reproducibility
    graph_utils.set_seed(42)
    
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

    if args.mode == "train":
        idx = graph.train_idx
    elif args.mode == "val":
        idx = graph.val_idx
    elif args.mode == "holdout":
        idx = graph.holdout_idx
    elif args.mode == "test":
        idx = graph.test_idx
    else:
        raise ValueError("Invalid mode")
    
    # removing isolated nodes
    isolated = (remove_isolated_nodes(graph["edge_index"])[2] == False).sum(dim=0).item()
    print(f"Number of isolated nodes = {isolated}\n")

    # define model
    in_channels = graph.num_features
    out_channels = graph.num_classes
    hidden_channels = 32
    in_head = 2
    out_head = 1
    dropout = 0.

    embedder = torch.nn.Sequential(*[torch.nn.Linear(768, 768), torch.nn.ReLU(), torch.nn.Linear(768, 128)])
    gat = GAT(embedder, n_in=in_channels, hid=hidden_channels,
                     in_head=in_head, out_head=out_head, 
                     n_classes=out_channels, dropout=dropout)
    gat.load_state_dict(torch.load(args.load_model, map_location=device)["state_dict"])
    gat.to(device)

    # cross entropy loss -- w/ logits
    loss_func = torch.nn.BCEWithLogitsLoss()

    # evaluation metrics
    acc = MulticlassAccuracy(num_classes=4)
    conf = MulticlassConfusionMatrix(num_classes=4)
    macro_recall = MulticlassRecall(num_classes=4, average="macro")
    binary_recall = BinaryRecall()

    gat.eval()
    with torch.no_grad():
        model_output = gat(graph.x, graph.edge_index)


    loss = loss_func(model_output[idx], graph.y[idx].float()).item()

    # Rewrite the labels from vectors to integers
    y_pred, y = graph_utils.rewrite_labels(model_output[idx].sigmoid().round()).long(), torch.sum(graph.y[idx], dim=-1).long()

    # Valuation confusion matrices
    conf = graph_utils.confusion_matrix(conf, y_pred, y)

    # Valuation accuracy
    acc = graph_utils.accuracy(acc, y_pred, y)

    # Valuation macro recall
    macro_recall = graph_utils.macro_recall(macro_recall, y_pred, y)

    # Train and valuation binary accuracy
    binary_mask = torch.logical_or((y == 0), (y == 3))
    y_binary = graph_utils.rewrite_labels_binary(y[binary_mask])
    y_pred_binary = graph_utils.rewrite_labels_binary(y_pred[binary_mask])
    binary_recall = graph_utils.binary_recall(binary_recall, y_pred_binary, y_binary)


    # Print valuation loss
    print(f"Loss: {loss}")
    # Print train and valuation confusion matrices
    print(f"Confusion matrix:\n\t{conf.long()}")
    # Print valuation accuracy
    print(f"Accuracy: {acc}")
    # Print valuation macro recall
    print(f"Macro recall: {macro_recall}")
    # Print valuation binary accuracy
    print(f"Binary recall: {binary_recall}")

 