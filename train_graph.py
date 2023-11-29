import torch, argparse
from torcheval.metrics import MultilabelAccuracy, MultilabelAUPRC, MeanSquaredError

import graph_utils
from GAT import GAT

from os.path import join as path_join

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
    parser.add_argument('--no-progress-bar', action='store_true',
                        help='Hide the progress bar during training loop')
    parser.add_argument('--optimizer', type=str, default="Adam",
                        choices=["SGD", "Adam"],
                        help='Which optimizer to use for training')
    parser.add_argument('--learning-rate', type=float, default=1e-3,
                        help='Learning rate for the optimizer')
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

    use_tqdm = not args.no_progress_bar

    model = GAT()
    model.to(device)

    graph = torch.load(args.path + "graph.pt", map_location=device)
    graph.to(device)

    # cross entropy loss -- w/ logits
    loss_func = torch.nn.BCEWithLogitsLoss()

    # we needed to use this metric, probably only in validation
    metric = MultilabelAUPRC(num_labels=3)
    acc = MultilabelAccuracy()
    mse = MeanSquaredError()

    # optimizer; change
    if args.optimizer == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)
    elif args.optimizer == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    else:
        # uhh, that should be impossible
        ...

    best_val = 0.

    for i in range(args.epochs):

        train_loss = graph_utils.train_loop(graph, model, loss_func, optimizer)
        val_loss, val_metric, val_acc, val_mse = graph_utils.val_loop(graph, model, loss_func, metric, acc, mse)

        print(f'Epoch: {i}\n\ttrain: {train_loss}\n\tval: {val_loss}')
        print('Val metric: ', val_metric.item(), '\tVal accuracy: ', val_acc.item(), '\tVal MSE: ', val_mse.item())

        if best_val < val_mse.item():
            best_val = val_mse.item()
            save = {
            'state_dict': model.state_dict(),
            }
            torch.save(save, path_join(args.output_dir, f"GAT_{i}.pt"))