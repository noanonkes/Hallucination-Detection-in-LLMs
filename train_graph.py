import torch, argparse
from torcheval.metrics import MultilabelAccuracy, MultilabelAUPRC
from torchmetrics.classification import MultilabelRecall
from torchmetrics import MeanSquaredError
from torch_geometric.utils import remove_isolated_nodes

import graph_utils

from os.path import join as path_join
torch.set_printoptions(profile="full")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # command line args for specifying the situation
    parser.add_argument('--use-cuda', action='store_true', default=False,
                        help='Use GPU acceleration if available')
    parser.add_argument('--path', type=str, default='data/',
                        help="Path to the data folder")
    parser.add_argument('--output_dir', type=str, default='weights/',
                        help="Path to save model weights to")
    parser.add_argument('--num-workers', type=int, default=4,
                        help="Number of cores to use when loading the data")
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs to train the model')
    parser.add_argument('--optimizer', type=str, default="Adam",
                        choices=["SGD", "Adam"],
                        help='Which optimizer to use for training')
    parser.add_argument('--learning-rate', type=float, default=1e-3,
                        help='Learning rate for the optimizer')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Batch size for graph training')
    parser.add_argument('--own-gat', action='store_true', default=False,
                        help='Use own designed GAT in GAT.py folder')
    args = parser.parse_args()

    # for reproducibility
    torch.manual_seed(42)
    
    print("STARTING...  setup:")
    print(args)
    print("-" * 120)
    print("\n" * 2)

    # some paramaters
    if args.use_cuda:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    else:
        device = torch.device('cpu')

    # load graph
    graph = torch.load(args.path + "graph.pt", map_location=device)
    graph.to(device)

    # removing isolated nodes
    isolated = (remove_isolated_nodes(graph['edge_index'])[2] == False).sum(dim=0).item()
    print(f'Number of isolated nodes = {isolated}\n')

    # define model
    in_channels = graph.num_features
    out_channels = graph.num_classes
    hidden_channels = 32
    activation = None
    v2 = False # use GATConv or GATv2Conv
    num_layers = 2 # only if manual = False
    in_head = 2 # only if manual = True
    out_head = 1 # only if manual = True
    dropout = 0. # only if manual = True
    manual = args.own_gat # use manually defined GAT

    gat = graph_utils.get_model(in_channels, out_channels, hidden_channels, activation, v2, in_head,
                                out_head, dropout, manual, num_layers)
    gat.to(device)

    # if batch size is given, get batch indices
    if args.batch_size is not None:
        batch_size = args.batch_size # because their data.x.shape[0] is almost prime with the exception of 3
        number_of_nodes = graph.x.shape[0]
        batch_indices = torch.arange(number_of_nodes) // batch_size
    else:
        batch_indices = None

    # cross entropy loss -- w/ logits
    loss_func = torch.nn.BCEWithLogitsLoss()

    # we needed to use this metric, probably only in validation
    metric = MultilabelAUPRC(num_labels=3)
    acc = MultilabelAccuracy()
    mse = MeanSquaredError(num_outputs=3)
    macrorecall = MultilabelRecall(num_labels=3, average="macro")

    optimizer = graph_utils.get_optimizer(args.optimizer, gat, args.learning_rate)

    # MSE, so lower is better! Recall, so higher is better!
    best_val = 0.
    # best_val = float("inf")
    for i in range(args.epochs):

        train_loss, model_output = graph_utils.train_loop(graph, gat, loss_func, optimizer, batch_indices)
        val_loss, val_metric, val_acc, val_mse, val_recall = graph_utils.val_loop(graph, model_output, loss_func, metric, acc, mse, macrorecall)

        print(f'Epoch: {i}\n\ttrain loss: {train_loss}\n\tval loss: {val_loss}')
        print('\tval AUPRC: ', val_metric.item(), '\n\tval accuracy: ', val_acc.item(), '\n\tval MSE: ', val_mse, '\n\tval recall: ', val_recall.item())

        # MSE, so lower is better!
        # if best_val > val_MSE.item():
        # Recall, so higher is better!
        if best_val < val_recall.item():
            best_val = val_recall.item()
            best_i = i
            best_model_output = model_output
            save = {
            'state_dict': gat.state_dict(),
            }
    
    freq_matrix = graph_utils.frequency(graph, best_model_output)
    print("\nTotal train samples:", len(graph.y[graph.train_idx]))
    print("Total frequencies", int(torch.sum(freq_matrix).detach().cpu()))
    print(freq_matrix)
    # save best model only
    torch.save(save, path_join(args.output_dir, f"GAT_{best_i}.pt"))