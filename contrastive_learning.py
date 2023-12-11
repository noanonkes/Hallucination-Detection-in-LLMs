import torch, argparse
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from os.path import join as path_join
from collections import defaultdict
import graph_utils

class SimCLR(nn.Module):
    """
    Based on code from SupContrast by the original authors.
    https://github.com/HobbitLong/SupContrast/blob/master/losses.py
    
    And code from Deep Learning 1 tutorial by Phillip Lippe.
    https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial17/SimCLR.html
    """
    def __init__(self, device, temperature=0.07):
        super(SimCLR, self).__init__()
        self.temperature = temperature
        self.device = device

    def forward(self, feats, labels, logs, mode="train"):
        # Calculate cosine similarity
        cos_sim = F.cosine_similarity(feats[:,None,:], feats[None,:,:], dim=-1)
        
        # Find positive example -> where the labels are equal
        pos_mask = torch.eq(labels, labels.T).to(self.device)
        # Mask out cosine similarity to itself
        pos_mask.fill_diagonal_(False)

        # InfoNCE loss
        cos_sim = cos_sim / self.temperature
        nll = -cos_sim + torch.logsumexp(cos_sim, dim=-1)
        nll = (pos_mask * nll).sum(-1) / pos_mask.sum(-1)
        nll = nll.mean()

        # some extra metrics
        metric = torch.empty((cos_sim.shape[0], ))
        for row, sim in enumerate(cos_sim):
            # positive examples for this row
            row_mask = pos_mask[row]

            sim_copy = sim.clone().detach()
            pos_examples = sim_copy[row_mask]
            
            # don't make own examples similar to itself
            sim_copy[row_mask] = -1.
            comb_sim = torch.cat([pos_examples, sim_copy])
            max_pos = pos_examples.argmax()

            # get the position of the most similar positive pair
            sim_argsort = torch.where(comb_sim.argsort(descending=True) == max_pos)[0][0]
            metric[row] = sim_argsort
        
        # Logging ranking metrics
        logs[mode+"_acc_top1"].append((metric == 0).float().mean().item())
        logs[mode+"_acc_top2"].append((metric < 2).float().mean().item())
        logs[mode+"_acc_top5"].append((metric < 5).float().mean().item())
        logs[mode+"_acc_mean_pos"].append(1+metric.float().mean().item())

        return nll, logs

class SimpleDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def train_loop(dataloader, model, loss_func, optimizer, lr_scheduler, logs, device):
    model.train()
    epoch_loss = 0.
    for j, (embeddings, labels) in enumerate(dataloader):

        embeddings, labels = embeddings.to(device), labels.to(device)

        labels = torch.sum(labels, axis=1, keepdim=True)

        feats = model(embeddings)
        train_loss, logs = loss_func(feats, labels, logs, mode="train")

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        
        epoch_loss += train_loss
    lr_scheduler.step()

    return epoch_loss.item() / j, logs


def val_loop(dataloader, model, loss_func, val_logs, device):
    model.eval()
    total_loss = 0.
    with torch.no_grad():
        for j, (embeddings, labels) in enumerate(dataloader):
            embeddings, labels = embeddings.to(device), labels.to(device)

            labels = torch.sum(labels, axis=1, keepdim=True)

            with torch.no_grad():
                feats = model(embeddings)
                val_loss, val_logs = loss_func(feats, labels, val_logs, mode="val")
                total_loss += val_loss

    return total_loss.item() / j, val_logs


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
    parser.add_argument("--epochs", type=int, default=1000,
                        help="Number of epochs to train the model")
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

    # create a simple dataset in order to easily batch from
    train_dataset = SimpleDataset(graph.x[graph.train_idx], graph.y[graph.train_idx])
    val_dataset = SimpleDataset(graph.x[graph.val_idx], graph.y[graph.val_idx])
    print("Train dataset size:", len(train_dataset))
    print("Validate dataset size:", len(val_dataset))
    # batch size is set to 256
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=256, drop_last=True)
    val_loader = DataLoader(val_dataset, shuffle=True, batch_size=256, drop_last=True)
    
    activation = nn.ReLU()
    lr = 1e-4 # optimizer
    weight_decay = 1e-4 # lr scheduler
    temp = 0.07 # loss

    layers = [nn.Linear(768, 768), activation, nn.Linear(768, 128)]
    model = nn.Sequential(*layers)
    model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(),
                                lr=lr,
                                weight_decay=weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                            T_max=args.epochs,
                                                            eta_min=lr/50)
    
    train_logs = defaultdict(list)
    val_logs = defaultdict(list)
    
    # empty cache
    torch.cuda.empty_cache()

    # contrastive loss function with specific temperature
    loss_func = SimCLR(device=device, temperature=temp)
    
    config = f"act_{activation._get_name()}_opt_AdamW_lr_{lr}_bs_{256}_t_{temp}"
    print(f"Configuration:\n\t{config}")

    for i in range(args.epochs):
        train_loss, train_logs = train_loop(train_loader, model, loss_func, optimizer, lr_scheduler, train_logs, device)
        val_loss, val_logs = val_loop(val_loader, model, loss_func, val_logs, device)

        print(f"Epoch {i}\n\ttrain loss: {train_loss}\n\tval loss: {val_loss}")
        for (t_key, t_value), (v_key, v_value) in zip(train_logs.items(), val_logs.items()):
            print()
            print(f"\t{t_key}: {torch.mean(torch.tensor(t_value[-60:]))}")
            print(f"\t{v_key}: {torch.mean(torch.tensor(v_value[-60:]))}")
        print()
        
        save = {
        "state_dict": model.state_dict(),
        }

        # Save every epoch!
        torch.save(save, path_join(args.output_dir, f"embedder_{config}_{i}.pt"))                 