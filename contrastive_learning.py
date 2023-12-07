import torch, argparse
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import graph_utils
from os.path import join as path_join
from itertools import product
import numpy as np

class SimCLR(nn.Module):
    """
    Based on code from Deep Learning 1 tutorial by Phillip Lippe.
    https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial17/SimCLR.html
    """
    def __init__(self, device, temperature=0.07):
        super(SimCLR, self).__init__()
        self.temperature = temperature
        self.device = device

    def forward(self, feats, labels):
        # Calculate cosine similarity
        cos_sim = F.cosine_similarity(feats[:,None,:], feats[None,:,:], dim=-1)
        # Mask out cosine similarity to itself
        self_mask = torch.eye(cos_sim.shape[0], dtype=torch.bool, device=self.device)
        cos_sim.masked_fill_(self_mask, 9e-15)
        
        # Find positive example -> where the labels are equal
        pos_mask = torch.eq(labels, labels.T).to(self.device)

        # InfoNCE loss
        cos_sim = cos_sim / self.temperature
        nll = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)
        nll = nll.mean()

        return nll

class SimpleDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def train_loop(dataloader, model,loss_func, optimizer, device):
    model.train()
    epoch_loss = 0.
    for j, (embeddings, labels) in enumerate(dataloader):
        optimizer.zero_grad()

        embeddings, labels = embeddings.to(device), labels.to(device)

        labels = torch.sum(labels, axis=1)

        feats = model(embeddings)
        train_loss = loss_func(feats, labels)

        train_loss.backward()
        optimizer.step()
        
        epoch_loss += train_loss

    return epoch_loss.item() / j


def val_loop(dataloader, model, loss_func, device):
    model.eval()
    total_loss = 0.
    with torch.no_grad():
        for j, (embeddings, labels) in enumerate(dataloader):
            embeddings, labels = embeddings.to(device), labels.to(device)

            labels = torch.sum(labels, axis=1)

            with torch.no_grad():
                feats = model(embeddings)
                val_loss = loss_func(feats, labels)
                total_loss += val_loss

    return total_loss.item() / j


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
    graph = torch.load(path_join(args.path, "graph.pt"), map_location=device)

    # create a simple dataset in order to easily batch from
    train_dataset = SimpleDataset(graph.x[graph.train_idx], graph.y[graph.train_idx])
    val_dataset = SimpleDataset(graph.x[graph.val_idx], graph.y[graph.val_idx])
    print("Train dataset size:", len(train_dataset))
    print("Validate dataset size:", len(val_dataset))


    # find configuration that has lowest loss
    nlayers = [1, 2] # number of layers
    activations = [nn.LeakyReLU(), nn.ReLU(), None]
    optimizers = ["SGD", "Adam"]
    learning_rates = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
    batch_sizes = [32, 64, 128, 256]
    temps = [0.07, 0.1, 0.2]
    combinations = product(nlayers, activations,
                           optimizers, learning_rates, batch_sizes,
                           temps)
    
    for combi, (n, activation, optimizer, lr, batch_size, temp) in enumerate(combinations):
        # do not run if n layers = 1 -> no activation function
        if n == 1 and activation is not None:
            continue
        
        # for every new configuration, empty cache
        torch.cuda.empty_cache()

        # contrastive loss function with specific temperature
        loss_func = SimCLR(device=device, temperature=temp)
        
        # test different batch sizes
        train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, drop_last=True)
        val_loader = DataLoader(val_dataset, shuffle=True, batch_size=batch_size, drop_last=True)

        if activation is not None:
            act_name = activation._get_name()
        else:
            act_name = None

        config = f"n_{n}_act_{act_name}_opt_{optimizer}_lr_{lr}_bs_{batch_size}_t_{temp}"
        print(f"Configuration:\n\t{config}")

        # create model according to the amount of layers and activation function
        layers = [nn.Linear(768, 768)]
        if activation is not None and n > 1:
            layers.append(activation)
        if n > 1:
            layers.append(nn.Linear(768, 768))

        model = nn.Sequential(*layers)
        model.to(device)

        # different optimizers with different learning rates
        optimizer = graph_utils.get_optimizer(optimizer, model, lr)
        
        best_val = float("inf")
        l_train, l_val = [], []
        for i in range(args.epochs):
            train_loss = train_loop(train_loader, model, loss_func, optimizer, device)
            val_loss = val_loop(val_loader, model, loss_func, device)

            l_train.append(train_loss)
            l_val.append(val_loss)
            if best_val > val_loss:
                best_val = val_loss
                best_train = train_loss
                best_i = i
                save = {
                'state_dict': model.state_dict(),
                }

        print(f'\tBest epoch: {best_i}\n\t\ttrain loss: {best_train}\n\t\tval loss: {best_val}')
        worst_i = np.argmax(l_val)
        print(f'\tWorst epoch: {worst_i}\n\t\ttrain loss: {l_train[worst_i]}\n\t\tval loss: {l_val[worst_i]}')
        torch.save(save, path_join(args.output_dir, f"{config}_{best_i}.pt"))                 