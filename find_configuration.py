import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, GATConv

from torcheval.metrics import MultilabelAccuracy, MultilabelAUPRC, MeanSquaredError

import graph_utils

from os.path import join as path_join
import numpy as np

from torch.utils.tensorboard import SummaryWriter

EPOCHS = 100
OUTPUT = "weights/"

def get_optimizer(optimizer, model, lr):
    if optimizer == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    elif optimizer == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        # uhh, that should be impossible
        ...
    return optimizer

def get_model(hid, in_head, dropout, activation, layer):
    if layer == "GATv2Conv":
        layer = GATv2Conv
    else:
        layer = GATConv
    class GAT(torch.nn.Module):
        def __init__(self, n_in=768, out_head=1, n_classes=3):
            super(GAT, self).__init__()        
            
            self.conv1 = layer(n_in, hid, heads=in_head, dropout=dropout)
            self.conv2 = layer(hid*in_head, n_classes, concat=False,
                                heads=out_head, dropout=dropout)

        def forward(self, data):
            x, edge_index = data.x, data.edge_index
            if dropout: 
                x = F.dropout(x, p=dropout, training=self.training)
            x = self.conv1(x, edge_index)
            if activation is not None:
                x = activation(x)
            if dropout: 
                x = F.dropout(x, p=dropout, training=self.training)
            x = self.conv2(x, edge_index)
            
            return x
    
    model = GAT()
    return model


if __name__ == "__main__":

    args = {"hid": [128, 64, 32, 16],
            "in_head": [4, 2],
            "dropout": [0.8, 0.6, 0.4, 0.2, 0.],
            "activation": [F.elu, F.relu, F.leaky_relu, None],
            "layer": ["GATv2Conv", "GATConv"],
            "optimizer": ["SGD", "Adam"],
            "lr": [1e-2, 1e-3, 1e-4]}
    
    # for reproducibility
    torch.manual_seed(42)
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # load the graph
    graph = torch.load("data/small_graph.pt", map_location=device)
    # necessary when working with small graph
    graph.train_idx = torch.tensor(np.arange(0, 70))
    graph.val_idx = torch.tensor(np.arange(70,121))

    graph.to(device)

    # cross entropy loss -- w/ logits
    loss_func = torch.nn.BCEWithLogitsLoss()

    # validation metrics
    metric = MultilabelAUPRC(num_labels=3)
    acc = MultilabelAccuracy()
    mse = MeanSquaredError()

    writer = SummaryWriter()

    # lol 
    for hid in args["hid"]:
        for in_head in args["in_head"]:
            for dropout in args["dropout"]:
                for activation in args["activation"]:
                    for layer in args["layer"]:
                        for optimizer_n in args["optimizer"]:
                            for lr in args["lr"]:
                                model_config = f"hid_{hid}_inhead_{in_head}_dropout_{dropout}_activation_{activation}_layer_{layer}_optimizer_{optimizer_n}_lr_{lr}"
                                print(model_config)

                                model = get_model(hid, in_head, dropout, activation, layer)
                                model.to(device)

                                optimizer = get_optimizer(optimizer_n, model, lr)

                                best_val = 0.

                                for i in range(EPOCHS):

                                    train_loss = graph_utils.train_loop(graph, model, loss_func, optimizer)
                                    val_loss, val_metric, val_acc, val_mse = graph_utils.val_loop(graph, model, loss_func, metric, acc, mse)

                                    writer.add_scalar(f"Loss/train/{model_config}", train_loss, i)
                                    writer.add_scalar(f"Loss/val/{model_config}", val_loss, i)
                                    writer.add_scalar(f"AUPRC/val/{model_config}", val_metric, i)
                                    writer.add_scalar(f"Accuracy/val/{model_config}", val_acc, i)
                                    writer.add_scalar(f"MSE/val/{model_config}", val_mse, i)

                                    print(f'Epoch: {i}\n\ttrain: {train_loss}\n\tval: {val_loss}')
                                    print('Val AUPRC:\n\t', val_metric.item(), '\nVal accuracy:\n\t', val_acc.item(), '\nVal MSE:\n\t', val_mse.item(), "\n")

                                    if best_val < val_mse.item():
                                        best_val = val_mse.item()
                                        best_i = i
                                        save = {
                                        'state_dict': model.state_dict(),
                                        }
                                
                                torch.save(save, path_join(OUTPUT, f"{model_config}_{best_i}.pt"))