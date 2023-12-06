import torch
import torch.nn.functional as F
from torch_geometric.utils import remove_isolated_nodes
from torcheval.metrics import MultilabelAccuracy, MultilabelAUPRC, MeanSquaredError
from torchmetrics.classification import MultilabelRecall

import graph_utils

from os.path import join as path_join
from itertools import product
import numpy as np

from torch.utils.tensorboard import SummaryWriter

EPOCHS = 5
MANUAL = True
BATCH = None
OUTPUT = "weights/" # where to save model weights to
DATA = "data/"

def write_to_summary(writer, model_config, i, train_loss, val_loss, val_metric, val_acc, val_mse):
    writer.add_scalar(f"Loss/train_{model_config}", train_loss, i)
    writer.add_scalar(f"Loss/val_{model_config}", val_loss, i)
    writer.add_scalar(f"AUPRC/{model_config}", val_metric, i)
    writer.add_scalar(f"Accuracy/{model_config}", val_acc, i)
    writer.add_scalar(f"MSE/{model_config}", val_mse, i)


if __name__ == "__main__":
    hids = [32, 16, 8] # nodes per hidden layer
    nhids = [1] # number of hidden layers, useful if MANUAL=False
    in_heads = [2]
    out_heads = [1]
    dropouts = [0., 0.2, 0.4]
    activations = [F.relu, F.leaky_relu, None] # activation function used in GAT
    v2 = [False] # use GATv2Conv or GATConv
    optimizers = ["SGD", "Adam"]
    learning_rates = [1e-2, 1e-3]
    combinations = product(hids, nhids, in_heads, out_heads, dropouts, activations,
                           v2, optimizers, learning_rates)

    # for reproducibility
    torch.manual_seed(42)
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # load the graph
    graph = torch.load(path_join(DATA, "graph.pt"), map_location=device)
    graph.to(device)

    in_channels = graph.num_features
    out_channels = graph.num_classes

    # removing isolated nodes
    isolated = (remove_isolated_nodes(graph['edge_index'])[2] == False).sum(dim=0).item()
    print(f'Number of isolated nodes = {isolated}\n')

    # use batches if indicated
    if BATCH is not None:
        number_of_nodes = graph.x.shape[0]
        batch_indices = torch.arange(number_of_nodes) // BATCH 
    else:
        batch_indices = None

    # cross entropy loss -- w/ logits
    loss_func = torch.nn.BCEWithLogitsLoss()

    # validation metrics
    metric = MultilabelAUPRC(num_labels=3)
    acc = MultilabelAccuracy()
    mse = MeanSquaredError()
    macrorecall = MultilabelAccuracy(num_labels=3, average="macro")

    # for tensorboard support
    writer = SummaryWriter()

    for hid, nhid, in_head, out_head, dropout, activation, use_v2, optimizer, lr in combinations:
        # for every new configuration, empty cache
        torch.cuda.empty_cache()

        if activation is not None:
            act_name = activation.__name__
        else:
            act_name = None

        config = f"hid_{hid}_dropout_{dropout}_activation_{act_name}_v2_{use_v2}_optimizer_{optimizer}_lr_{lr}"
        print(f"Configuration:\n\t{config}")

        gat = graph_utils.get_model(in_channels, out_channels, hidden_channels=hid, activation=activation,
              v2=use_v2, in_head=in_head, out_head=out_head, dropout=dropout, manual=MANUAL, num_layers=nhid)
        gat.to(device)

        optimizer = graph_utils.get_optimizer(optimizer, gat, lr)

        best_val = float("inf")
        error = False

        for i in range(EPOCHS):

            try:
                train_loss, model_output = graph_utils.train_loop(graph, gat, loss_func, optimizer, batch_indices)
                val_loss, val_metric, val_acc, val_mse = graph_utils.val_loop(graph, model_output, loss_func, metric, acc, mse)

            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print('\t| WARNING: ran out of memory')
                    print("\t", e, "\n")
                    error = True
                else:
                    raise e
                
                # don't continue for this configuration
                break

            write_to_summary(writer, config, i, train_loss, val_loss, val_metric, val_acc, val_mse)

            # MSE, so lower is better!
            if best_val > val_mse.item():
                best_val = val_mse.item()
                best_i = i
                save = {
                'state_dict': gat.state_dict(),
                }
                freq_matrix = graph_utils.frequency(graph, model_output)
        
        if not error:
            # only save best model of each configuration
            print(f"\tBest validation MSE at epoch {best_i}:", val_mse.item())
            print("\tTotal train samples:", len(graph.y[graph.train_idx]))
            print("\tTotal frequencies", int(torch.sum(freq_matrix).detach().cpu()))
            print("\t", freq_matrix, "\n")
            torch.save(save, path_join(OUTPUT, f"{config}_{best_i}.pt"))
    
    writer.close()