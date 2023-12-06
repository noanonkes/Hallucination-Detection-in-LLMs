import torch, argparse
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import graph_utils
from os.path import join as path_join

class SimpleDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
class SelfSupervisedContrastiveLoss(nn.Module):
    def __init__(self, device, temperature=0.07,
                 base_temperature=0.07):
        super(SelfSupervisedContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.device = device
        self.base_temperature = base_temperature

    def forward(self, features, labels):

        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(self.device)
        anchor_dot_contrast = torch.div(torch.matmul(features, features.T), self.temperature)

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=0, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # mask out self-contrast cases
        mask.fill_diagonal_(0.)
        # mask[j] = 0

        # compute log_prob
        exp_logits = torch.exp(logits) * (1. - mask)
        # print("1", exp_logits)

        log_prob = logits - torch.log(exp_logits.sum(1, keepdims=True))
        # print("2", log_prob)
        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        # print("3", mean_log_prob_pos)

        # loss
        loss = mean_log_prob_pos
        loss = loss.mean()

        return loss



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

    train_dataset = SimpleDataset(graph.x[graph.train_idx], graph.y[graph.train_idx])
    val_dataset = SimpleDataset(graph.x[graph.val_idx], graph.y[graph.val_idx])
    print("Train dataset size:", len(train_dataset))
    print("Validate dataset size:", len(val_dataset))

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=128, drop_last=True)
    val_loader = DataLoader(val_dataset, shuffle=True, batch_size=128, drop_last=True)

    linear = nn.Linear(768, 64)
    linear.to(device)

    # loss_func = ContrastiveLoss()
    loss_func = SelfSupervisedContrastiveLoss(device=device, temperature=0.5)

    optimizer = graph_utils.get_optimizer(args.optimizer, linear, args.learning_rate)

    best_val = float("inf")
    for i in range(args.epochs):
        linear.train()
        batch_loss = 0
        n_batch = 0
        optimizer.zero_grad()
        for j, (features, labels) in enumerate(train_loader):
            features, labels = features.to(device), labels.to(device)
            labels = torch.sum(labels, axis=1)

            out = linear(features)
            train_loss = loss_func(out, labels)
            
    
            batch_loss += train_loss
            n_batch += 1

        batch_loss.backward()
        optimizer.step()

        print(f'Epoch: {i}\n\ttrain loss: {batch_loss.item() / n_batch}')

        linear.eval()
        batch_loss = 0
        n_batch = 0
        for j, (features, labels) in enumerate(val_loader):
            features, labels = features.to(device), labels.to(device)

            labels = torch.sum(labels, axis=1)

            with torch.no_grad():
                out = linear(features)
                val_loss = loss_func(out, labels)

                batch_loss += val_loss.item()
                n_batch += 1


        if best_val > batch_loss / n_batch:
            best_val = batch_loss / n_batch
            best_i = i
            save = {
            'state_dict': linear.state_dict(),
            }

    torch.save(save, path_join(args.output_dir, f"linear_{best_i}.pt"))