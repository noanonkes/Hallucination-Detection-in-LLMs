import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GAT
from torch_geometric.utils import remove_isolated_nodes
from torcheval.metrics import MultilabelAccuracy, MultilabelAUPRC, MeanSquaredError

import graph_utils

from os.path import join as path_join
from itertools import product
import numpy as np

from torch.utils.tensorboard import SummaryWriter

EPOCHS = 100
OUTPUT = "weights/"
BATCH = 100 # no batches -> BATCH = None

def get_optimizer(optimizer, model, lr):
    if optimizer == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    elif optimizer == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    else:
        # uhh, that should be impossible
        ...
    return optimizer


def write_to_summary(writer, model_config, i, train_loss, val_loss, val_metric, val_acc, val_mse):
    writer.add_scalar(f"Loss/train/{model_config}", train_loss, i)
    writer.add_scalar(f"Loss/val/{model_config}", val_loss, i)
    writer.add_scalar(f"AUPRC/val/{model_config}", val_metric, i)
    writer.add_scalar(f"Accuracy/val/{model_config}", val_acc, i)
    writer.add_scalar(f"MSE/val/{model_config}", val_mse, i)


if __name__ == "__main__":
    hids = [8, 16, 32, 64] # nodes per hidden layer
    n_hids = [1, 2, 3, 4] # amount of hidden layers
    dropouts = [0.8, 0.6, 0.4, 0.2, 0.]
    activations = [F.elu, F.relu, F.leaky_relu] # activation function used in GAT
    v2 = [True, False] # use GATv2Conv or GATConv
    optimizers = ["SGD", "Adam"]
    learning_rates = [1e-2, 1e-3, 1e-4]
    combinations = product(hids, n_hids, dropouts, 
                           activations, v2, optimizers, 
                           learning_rates)
    
    # for reproducibility
    torch.manual_seed(42)
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # load the graph
    graph = torch.load("data/graph.pt", map_location=device)
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

    # for tensorboard support
    writer = SummaryWriter()

    for hid, n_hid, dropout, activation, use_v2, optimizer, lr in combinations:
        config = f"hid_{hid}_nhid_{n_hid}_dropout_{dropout}_activation_{activation.__name__}_v2_{use_v2}_optimizer_{optimizer}_lr_{lr}"
        print(config)
        
        gat = GAT(in_channels, hid, n_hid, out_channels, dropout=dropout, act=activation, v2=use_v2)
        gat.to(device)

        optimizer = get_optimizer(optimizer, gat, lr)

        best_val = 0.

        for i in range(EPOCHS):

            train_loss, model_output = graph_utils.train_loop(graph, gat, loss_func, optimizer, batch_indices)
            val_loss, val_metric, val_acc, val_mse = graph_utils.val_loop(graph, model_output, loss_func, metric, acc, mse)

            write_to_summary(writer, config, i, train_loss, val_loss, val_metric, val_acc, val_mse)

            if best_val < val_mse.item():
                best_val = val_mse.item()
                best_i = i
                save = {
                'state_dict': gat.state_dict(),
                }
        
        # only save best model of each configuration
        print(f"Best validation MSE at epoch {best_i}:", val_mse.item(), "\n")
        torch.save(save, path_join(OUTPUT, f"{config}_{best_i}.pt"))