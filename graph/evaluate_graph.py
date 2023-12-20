import torch, argparse
from torcheval.metrics import MulticlassAUPRC, MulticlassConfusionMatrix, MulticlassAccuracy, MulticlassPrecision, MulticlassRecall, BinaryRecall
from torch_geometric.utils import remove_isolated_nodes

from GAT import GAT
import utils_graph

from os.path import join as path_join
torch.set_printoptions(profile="full")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # command line args for specifying the situation
    parser.add_argument("--use-cuda", action="store_true", default=False,
                        help="Use GPU acceleration if available")
    parser.add_argument("--path", type=str, default="../data/",
                        help="Path to the data folder")
    parser.add_argument("--load-model", type=str, default="../weights/998_GAT_431.pt",
                        help="GAT model weights to use.")
    parser.add_argument("--mode", type=str, default="val",
                        choices=["train", "val", "test"],
                        help="Mode for evaluation")
    args = parser.parse_args()

    # for reproducibility
    utils_graph.set_seed(42)
    
    print("STARTING...  setup:")
    print(args)
    print("-" * 120)
    print("\n" * 2)

    # Some paramaters
    if args.use_cuda:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    else:
        device = torch.device("cpu")

    # Load graph
    graph = torch.load(path_join(args.path, "graph.pt"), map_location=device)
    graph.to(device)

    # Get the indices for slice to be evaluated
    if args.mode == "train":
        idx = graph.train_idx
    elif args.mode == "val":
        idx = graph.val_idx
    elif args.mode == "test":
        idx = graph.test_idx
    else:
        raise ValueError("Invalid mode")
    
    # Removing isolated nodes
    isolated = (remove_isolated_nodes(graph["edge_index"])[2] == False).sum(dim=0).item()
    print(f"Number of isolated nodes = {isolated}\n")

    # Define model
    in_channels = graph.num_features
    out_channels = graph.y.shape[1] # number of columns
    hidden_channels = 32
    in_head = 2
    dropout = 0.

    embedder = torch.nn.Sequential(*[torch.nn.Linear(in_channels, in_channels), torch.nn.ReLU(), torch.nn.Linear(in_channels, 128)])
    gat = GAT(embedder, n_in=in_channels, hid=hidden_channels,
                     in_head=in_head, 
                     n_classes=out_channels, dropout=dropout)
    gat.load_state_dict(torch.load(args.load_model, map_location=device)["state_dict"])
    gat.to(device)

    # Cross entropy loss -- w/ logits
    loss_func = torch.nn.BCEWithLogitsLoss()

    # Evaluation metrics
    acc = MulticlassAccuracy(num_classes=4)
    conf = MulticlassConfusionMatrix(num_classes=4)
    macro_recall = MulticlassRecall(num_classes=4, average="macro")
    macro_precision = MulticlassPrecision(num_classes=4, average="macro")
    binary_recall = BinaryRecall()
    macro_AUPRC = MulticlassAUPRC(num_classes=4, average="macro")

    gat.eval()
    with torch.no_grad():
        model_output = gat(graph.x, graph.edge_index)

    loss = loss_func(model_output[idx], graph.y[idx].float()).item()

    # Rewrite the labels from vectors to integers
    y_pred, y = utils_graph.rewrite_labels(model_output[idx].sigmoid().round()).long(), torch.sum(graph.y[idx], dim=-1).long()

    # Valuation confusion matrices
    conf_mat = utils_graph.confusion_matrix(conf, y_pred, y)

    # Valuation accuracy
    accuracy = utils_graph.accuracy(acc, y_pred, y)

    # Valuation macro recall
    m_recall = utils_graph.macro_recall(macro_recall, y_pred, y)

    # Valuation macro precision
    m_precision = utils_graph.macro_recall(macro_precision, y_pred, y)

    # Valuation macro area under the precision-recall curve
    m_AUPRC = utils_graph.macro_AUPRC(macro_AUPRC, y_pred, y)

    # One frame agreement
    ofa = utils_graph.k_frame_agreement(y_pred, y, k=1)

    # Train and valuation binary accuracy
    binary_mask = torch.logical_or((y == 0), (y == 3))
    y_binary = utils_graph.rewrite_labels_binary(y[binary_mask])
    y_pred_binary = utils_graph.rewrite_labels_binary(y_pred[binary_mask])
    b_recall = utils_graph.binary_recall(binary_recall, y_pred_binary, y_binary)

    # Print valuation loss
    print(f"Loss: {loss}")
    # Print train and valuation confusion matrices
    print(f"Confusion matrix:\n\t{conf_mat.long()}")
    # Print valuation accuracy
    print(f"Accuracy: {accuracy.item()}")
    # Print valuation macro recall
    print(f"Macro recall: {m_recall.item()}")
    # Print valuation macro precision
    print(f"Macro precision: {m_precision.item()}")
    # Print valuation binary accuracy
    print(f"Binary recall: {b_recall.item()}")
    # Print valuation one frame agreement
    print(f"One frame agreement: {ofa}")
    # Print valuation macro AUPRC
    print(f"Macro AUPRC: {m_AUPRC.item()}")
 