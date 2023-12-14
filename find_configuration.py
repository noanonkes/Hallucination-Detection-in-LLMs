import torch, argparse
from torch_geometric.utils import remove_isolated_nodes
from torcheval.metrics import MulticlassConfusionMatrix, MulticlassAccuracy, MulticlassRecall, MulticlassPrecision, BinaryRecall

import graph_utils

from os.path import join as path_join
from itertools import product
import GAT

from torch.utils.tensorboard import SummaryWriter


def write_to_summary(writer, config, i,
                    train_loss, val_loss,
                    val_acc,
                    val_macro_recall,
                    val_macro_precision,
                    val_binary_recall):
    
    writer.add_scalar(f"Loss/train/{config}", train_loss, i)
    writer.add_scalar(f"Loss/val/{config}", val_loss, i)
    writer.add_scalar(f"Accuracy/{config}", val_acc, i)
    writer.add_scalar(f"Macro_recall/{config}", val_macro_recall, i)
    writer.add_scalar(f"Macro_precision/{config}", val_macro_precision, i)
    writer.add_scalar(f"Binary_recall/{config}", val_binary_recall, i)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # command line args for specifying the situation
    parser.add_argument("--use-cuda", action="store_true", default=False,
                        help="Use GPU acceleration if available")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of epochs to train the model")
    parser.add_argument("--output_dir", type=str, default="weights/",
                        help="Path to save model weights to")
    parser.add_argument("--path", type=str, default="data/",
                        help="Path to the data folder")
    parser.add_argument("--pt-epoch", type=int, default=998,
                        help="Which epoch to use for the embedder weights")
    args = parser.parse_args()

    nhids = [1, 2]
    dropouts = [0., 0.2, 0.4]
    optimizers = ["SGD", "Adam"]
    learning_rates = [1e-2, 1e-3, 1e-4]
    combinations = product(nhids, dropouts,
                           optimizers, learning_rates)

    # for reproducibility
    graph_utils.set_seed(42)
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # load the graph
    graph = torch.load(path_join(args.path, "graph.pt"), map_location=device)
    graph.to(device)

    in_channels = graph.num_features
    out_channels = graph.y.shape[1] # number of columns

    # removing isolated nodes
    isolated = (remove_isolated_nodes(graph['edge_index'])[2] == False).sum(dim=0).item()
    print(f'Number of isolated nodes = {isolated}\n')

    # cross entropy loss -- w/ logits
    loss_func = torch.nn.BCEWithLogitsLoss()

    # validation metrics
    acc = MulticlassAccuracy(num_classes=4)
    conf = MulticlassConfusionMatrix(num_classes=4)
    macro_recall = MulticlassRecall(num_classes=4, average="macro")
    macro_precision = MulticlassPrecision(num_classes=4, average="macro")
    binary_recall = BinaryRecall()

    # for tensorboard support
    writer = SummaryWriter()

    for nhid, dropout, optimizer, lr in combinations:
        # for every new configuration, empty cache
        torch.cuda.empty_cache()

        config = f"nhid_{nhid}_dropout_{dropout}_optimizer_{optimizer}_lr_{lr}"
        print(f"Configuration:\n\t{config}")

        # not sure if this needs to be done every configuration
        embedder_file = f"embedder_act_ReLU_opt_AdamW_lr_0.0001_bs_256_t_0.07_{args.pt_epoch}.pt"
        embedder = torch.nn.Sequential(*[torch.nn.Linear(in_channels, in_channels), torch.nn.ReLU(), torch.nn.Linear(in_channels, 128)])
        embedder.load_state_dict(torch.load(path_join(args.output_dir, embedder_file), map_location=device)["state_dict"])

        if nhid == 1:
            gat = GAT.GAT(embedder, in_channels, dropout=dropout)
        else:
            gat = GAT.GAT2(embedder, in_channels, dropout=dropout)
        gat.to(device)

        optimizer = graph_utils.get_optimizer(optimizer, gat, lr)

        best_recall = 0.
        error = False
        for i in range(args.epochs):

            try:
                # Train epoch and valuation loss
                train_loss, model_output = graph_utils.train_loop(graph, gat, loss_func, optimizer)
                val_loss = loss_func(model_output[graph.val_idx], graph.y[graph.val_idx].float()).item()
                
                # Rewrite the labels from vectors to integers
                y_pred_train, y_train = graph_utils.rewrite_labels(model_output[graph.train_idx].sigmoid().round()).long(), torch.sum(graph.y[graph.train_idx], dim=-1).long()
                y_pred_val, y_val = graph_utils.rewrite_labels(model_output[graph.val_idx].sigmoid().round()).long(), torch.sum(graph.y[graph.val_idx], dim=-1).long()

                # Train and valuation accuracy
                train_acc = graph_utils.accuracy(acc, y_pred_train, y_train)
                val_acc = graph_utils.accuracy(acc, y_pred_val, y_val)

                # Train and valuation macro recall
                train_macro_recall = graph_utils.macro_recall(macro_recall, y_pred_train, y_train)
                val_macro_recall = graph_utils.macro_recall(macro_recall, y_pred_val, y_val)

                # Train and valuation macro recall
                train_macro_precision = graph_utils.macro_recall(macro_precision, y_pred_train, y_train)
                val_macro_precision = graph_utils.macro_recall(macro_precision, y_pred_val, y_val)

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
                # Print train and valuation accuracy
                print(f"\ttrain accuracy: {train_acc.item()}\n\tval accuracy: {val_acc.item()}")
                # Print train and valuation macro recall
                print(f"\ttrain macro recall: {train_macro_recall.item()}\n\tval macro recall: {val_macro_recall.item()}")
                # Print train and valuation macro precision
                print(f"\ttrain macro precision: {train_macro_precision.item()}\n\tval macro precision: {val_macro_precision.item()}")
                # Print train and valuation binary accuracy
                print(f"\ttrain binary recall: {train_binary_recall.item()}\n\tval binary recall: {val_binary_recall.item()}")

                if val_macro_recall.item() > best_recall:
                    best_recall = val_macro_recall.item()
                    best_i = i         
                    save = {
                        "state_dict": gat.state_dict(),
                        }
                
                write_to_summary(writer, config, i, 
                             train_loss, val_loss, 
                             val_acc.item(),
                             val_macro_recall.item(),
                             val_macro_precision.item(),
                             val_binary_recall.item())
                
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print('\t| WARNING: ran out of memory')
                    print("\t", e, "\n")
                else:
                    raise e
                
                error = True
                # don't continue for this configuration
                break
        if not error:
            torch.save(save, path_join(args.output_dir, f"{config}_{best_i}.pt"))
    writer.close()