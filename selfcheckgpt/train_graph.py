import torch, argparse
from torcheval.metrics import MulticlassConfusionMatrix, MulticlassRecall, MulticlassAUPRC
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
    parser.add_argument("--output_dir", type=str, default="../weights/",
                        help="Path to save model weights to")
    parser.add_argument("--epochs", type=int, default=500,
                        help="Number of epochs to train the model")
    parser.add_argument("--optimizer", type=str, default="Adam",
                        choices=["SGD", "Adam"],
                        help="Which optimizer to use for training")
    parser.add_argument("--learning-rate", type=float, default=1e-3,
                        help="Learning rate for the optimizer")
    parser.add_argument("--save-model", action="store_true", default=False,
                        help="Whether to save best model weights")
    args = parser.parse_args()

    # for reproducibility
    utils_graph.set_seed(42)
    
    print("\n" * 2)
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
    graph = torch.load(path_join(args.path, "768_SCGPT_graph.pt"), map_location=device)
    graph.to(device)

    # removing isolated nodes
    isolated = (remove_isolated_nodes(graph["edge_index"])[2] == False).sum(dim=0).item()
    print(f"Number of isolated nodes = {isolated}\n")

    ######## DATA LEAKAGE PREVENTION WITH SPECIFIC EDGE ATTRIBUTES ########
    train_idx = graph.train_idx # [t, ]
    val_idx = graph.val_idx # [v, ]
    edge_index = graph.edge_index.T # [N, 2]; (i, j) node pairs as rows

    # load the distances
    distances = torch.load(path_join(args.path, "768_SCGPT_distances.pt"), map_location=device)
    # get the distances corresponding to the nodes that have edges
    edge_attr = distances[edge_index[:, 0], edge_index[:, 1]] # [N, ]

    # these are all the edges between only train nodes
    train_mask = (torch.isin(edge_index[:, 0], train_idx)) & (torch.isin(edge_index[:, 1], train_idx))
    # these are all the edges only between train and/or validation
    val_mask = ((torch.isin(edge_index[:, 0], train_idx)) | (torch.isin(edge_index[:, 0], val_idx))) \
                    & ((torch.isin(edge_index[:, 1], train_idx)) | (torch.isin(edge_index[:, 1], val_idx)))

    # make all non train attributes zero
    train_edge_attr = edge_attr.detach().clone()
    train_edge_attr[~train_mask] = 0.    

    # make all non train and validation attributes zero
    val_edge_attr = edge_attr.detach().clone()
    val_edge_attr[~val_mask] = 0.
    ######## DATA LEAKAGE PREVENTION WITH SPECIFIC EDGE ATTRIBUTES ########

    # define model
    in_channels = graph.num_features
    out_channels = graph.y.shape[1] # number of columns
    hidden_channels = 32
    in_head = 2
    dropout = 0.2

    embedder_file = "SCGPT_embedder_act_ReLU_opt_AdamW_lr_0.0001_bs_256_t_0.07.pt"
    embedder = torch.nn.Sequential(*[torch.nn.Linear(in_channels, in_channels), torch.nn.ReLU(), torch.nn.Linear(in_channels, 128)])
    embedder.load_state_dict(torch.load(path_join(args.output_dir, embedder_file), map_location=device)["state_dict"])
    gat = GAT(embedder, n_in=in_channels, hid=hidden_channels,
                     in_head=in_head, 
                     n_classes=out_channels, dropout=dropout)
    gat.to(device)

    # cross entropy loss -- w/ logits
    loss_func = torch.nn.BCEWithLogitsLoss()

    # evaluation metrics
    macro_recall = MulticlassRecall(num_classes=3, average="macro")
    macro_AUPRC = MulticlassAUPRC(num_classes=2, average="none")
    confm = MulticlassConfusionMatrix(num_classes=3)

    optimizer = utils_graph.get_optimizer(args.optimizer, gat, args.learning_rate)

    best_recall = 0.
    for i in range(args.epochs):

        # Train epoch and train loss; edge attributes between non train nodes are zero
        train_loss, model_output = utils_graph.train_loop(graph, gat, loss_func, optimizer, train_edge_attr)

        # Rewrite the train labels from vectors to integers
        y_pred_train, y_train = utils_graph.rewrite_labels(model_output[graph.train_idx].sigmoid().round()).long(), torch.sum(graph.y[graph.train_idx], dim=-1).long()

        # Validation epoch and valuation loss; edge attributes between non train/val nodes are zero
        val_loss, model_output = utils_graph.val_loop(graph, gat, loss_func, val_edge_attr)
        
        # Rewrite the labels from vectors to integers
        y_pred_val, y_val = utils_graph.rewrite_labels(model_output[graph.val_idx].sigmoid().round()).long(), torch.sum(graph.y[graph.val_idx], dim=-1).long()

        # Train and valuation macro recall
        try:
            train_macro_recall = utils_graph.macro_recall(macro_recall, y_pred_train, y_train)
            val_macro_recall = utils_graph.macro_recall(macro_recall, y_pred_val, y_val)
        except:
            train_macro_recall = torch.tensor(0)
            val_macro_recall = torch.tensor(0)
            print("Could not calculate recall. See error log.")

        val_conf = utils_graph.confusion_matrix(confm, y_pred_val, y_val)
        
        y_pred_val, y_val = utils_graph.rewrite_labels_binary(y_pred_val), utils_graph.rewrite_labels_binary(y_val) 
        
        # Valuation macro area under the precision-recall curve
        m_AUPRC = utils_graph.macro_AUPRC(macro_AUPRC, y_pred_val, y_val, num_classes=2)

        # Print train and valuation loss
        print(f"Epoch: {i}\n\ttrain loss: {train_loss}\n\tval loss: {val_loss}")
        # Print train and valuation macro recall
        print(f"\ttrain macro recall: {train_macro_recall.item()}\n\tval macro recall: {val_macro_recall.item()}\n\tval AUPRC: {m_AUPRC}\n{val_conf}")

        if val_macro_recall.item() > best_recall:
            best_recall = val_macro_recall.item() 
            best_i = i        
            save = {
                "state_dict": gat.state_dict(),
                }
    if args.save_model:
        torch.save(save, path_join(args.output_dir, f"SCGPT_GAT_{best_i}.pt"))