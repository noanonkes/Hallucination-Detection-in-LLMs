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
    parser.add_argument("--output_dir", type=str, default="weights/",
                        help="Path to save model weights to")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of cores to use when loading the data")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of epochs to train the model")
    parser.add_argument("--optimizer", type=str, default="Adam",
                        choices=["SGD", "Adam"],
                        help="Which optimizer to use for training")
    parser.add_argument("--learning-rate", type=float, default=1e-3,
                        help="Learning rate for the optimizer")
    parser.add_argument("--pt-epoch", type=int, default=950,
                        help="Which epoch to use for the embedder weights")
    parser.add_argument("--load-model", type=str, default=None,
                        help="GAT model weights to use.")
    parser.add_argument("--save-model", action="store_true", default=False,
                        help="Whether to save all model weights")
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

    embedder_file = f"embedder_act_ReLU_opt_AdamW_lr_0.0001_bs_256_t_0.07_{args.pt_epoch}.pt"
    embedder = torch.nn.Sequential(*[torch.nn.Linear(768, 768), torch.nn.ReLU(), torch.nn.Linear(768, 128)])
    embedder.load_state_dict(torch.load(path_join(args.output_dir, embedder_file), map_location=device)["state_dict"])
    gat = GAT(embedder, n_in=in_channels, hid=hidden_channels,
                     in_head=in_head, out_head=out_head, 
                     n_classes=out_channels, dropout=dropout)
    if args.load_model is not None:
        gat.load_state_dict(torch.load(path_join(args.output_dir, args.load_model), map_location=device)["state_dict"])
    gat.to(device)

    # cross entropy loss -- w/ logits
    loss_func = torch.nn.BCEWithLogitsLoss()

    # evaluation metrics
    acc = MulticlassAccuracy(num_classes=4)
    conf = MulticlassConfusionMatrix(num_classes=4)
    macro_recall = MulticlassRecall(num_classes=4, average="macro")
    binary_recall = BinaryRecall()

    optimizer = graph_utils.get_optimizer(args.optimizer, gat, args.learning_rate)

    best_recall = 0.
    for i in range(args.epochs):

        # Train epoch and valuation loss
        train_loss, model_output = graph_utils.train_loop(graph, gat, loss_func, optimizer)
        val_loss = loss_func(model_output[graph.val_idx], graph.y[graph.val_idx].float()).item()

        # Rewrite the labels from vectors to integers
        y_pred_train, y_train = graph_utils.rewrite_labels(model_output[graph.train_idx].sigmoid().round()).long(), torch.sum(graph.y[graph.train_idx], dim=-1).long()
        y_pred_val, y_val = graph_utils.rewrite_labels(model_output[graph.val_idx].sigmoid().round()).long(), torch.sum(graph.y[graph.val_idx], dim=-1).long()

        # Train and valuation confusion matrices
        train_conf = graph_utils.confusion_matrix(conf, y_pred_train, y_train)
        val_conf = graph_utils.confusion_matrix(conf, y_pred_val, y_val)

        # Train and valuation accuracy
        train_acc = graph_utils.accuracy(acc, y_pred_train, y_train)
        val_acc = graph_utils.accuracy(acc, y_pred_val, y_val)

        # Train and valuation macro recall
        train_macro_recall = graph_utils.macro_recall(macro_recall, y_pred_train, y_train)
        val_macro_recall = graph_utils.macro_recall(macro_recall, y_pred_val, y_val)

        # Train and valuation binary accuracy
        binary_mask = torch.logical_or((y_train == 0), (y_train == 3))
        y_binary_train = graph_utils.rewrite_labels_binary(y_train[binary_mask])
        y_binary_pred_train = graph_utils.rewrite_labels_binary(y_pred_train[binary_mask])
        train_binary_recall = graph_utils.binary_recall(binary_recall, y_binary_pred_train, y_binary_train)
        
        binary_mask = torch.logical_or((y_val == 0), (y_val == 3))
        y_binary_val = graph_utils.rewrite_labels_binary(y_val[binary_mask])
        y_binary_pred_val = graph_utils.rewrite_labels_binary(y_pred_val[binary_mask])
        val_binary_recall = graph_utils.binary_recall(binary_recall, y_binary_pred_val, y_binary_val)

        # Print train and valuation loss
        print(f"Epoch: {i}\n\ttrain loss: {train_loss}\n\tval loss: {val_loss}")
        # Print train and valuation confusion matrices
        print(f"\ttrain confusion matrix:\n\t{train_conf.long()}\n\tval confusion matrix:\n\t{val_conf.long()}")
        # Print train and valuation accuracy
        print(f"\ttrain accuracy: {train_acc}\n\tval accuracy: {val_acc}")
        # Print train and valuation macro recall
        print(f"\ttrain macro recall: {train_macro_recall}\n\tval macro recall: {val_macro_recall}")
        # Print train and valuation binary accuracy
        print(f"\ttrain binary recall: {train_binary_recall}\n\tval binary recall: {val_binary_recall}")

        if val_macro_recall > best_recall and args.save_model:
            best_recall = macro_recall            
            save = {
                "state_dict": gat.state_dict(),
                }
            torch.save(save, path_join(args.output_dir, f"{args.pt_epoch}_GAT_{i}.pt"))