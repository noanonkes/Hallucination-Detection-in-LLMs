import numpy as np
import torch, argparse
from torcheval.metrics import MultilabelAccuracy, MultilabelAUPRC
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertModel

import utils
from baselines import MisinformationMLP, MisinformationCrossEncoder, MisinformationPCA
from dataloader import SentenceLabelDataset

from os.path import join as path_join

def set_seed(seed):
    """
    Function for setting the seed for reproducibility.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main(data_dir, output_dir, model_name, model_embed, optimizer_name, lr, batch_size, epochs, use_cuda, seed, use_tqdm):
    # Set the seed for reproducibility
    set_seed(seed)
    # Use cuda if cuda is available
    device = torch.device("cuda") if use_cuda and torch.cuda.is_available() else torch.device("cpu")

    # Instantiate model
    if model_name == "mlp":
        model = MisinformationMLP(768, 3)
        reduce = False
    elif model_name == "cross_encoder": 
        model = MisinformationCrossEncoder()
        reduce = False
    elif model_name == "pca":
        model  = MisinformationPCA()
        reduce = True
    else:
        raise ValueError("Unsupported model type.")
    model.to(device)

    # Load pre-trained BERT model and tokenizer
    tokenizer = BertTokenizer.from_pretrained(model_embed)
    embed = BertModel.from_pretrained(model_embed)
    embed.to(device)

    # Load the data from root folder
    full_dataset = SentenceLabelDataset(data_dir)
    train_size = int(len(full_dataset) * 0.7)
    val_size = int(len(full_dataset) * 0.1)
    test_size = len(full_dataset) - train_size - val_size

    train_data, val_data, test_data = torch.utils.data.random_split(full_dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_data, batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_data, batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_data, batch_size, shuffle=False, num_workers=4)
    
    # cross entropy loss -- w/ logits
    loss_func = torch.nn.BCEWithLogitsLoss()

    # we needed to use this metric, probably only in validation
    metric = MultilabelAUPRC(num_labels=3)
    acc = MultilabelAccuracy()

    # optimizer; change
    if optimizer_name == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    elif optimizer_name == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        raise ValueError("Unsupported optimizer type. Please choose between 'SGD' and 'Adam'.")

    best_val = -np.inf

    for i in range(epochs):
        train_loss = utils.train_loop(train_loader, model, embed, tokenizer, loss_func, optimizer, device, use_tqdm, reduce)
        val_loss, val_metric, val_acc = utils.val_loop(val_loader, model, embed, tokenizer, loss_func, device, metric, acc, use_tqdm, reduce)
        print(f'Epoch: {i}\n\ttrain: {train_loss}\n\tval: {val_loss}')
        print('Val metric: ', val_metric.item(), '\tVal accuracy: ', val_acc.item())

        if best_val < val_metric.item():
            best_val = val_metric.item()
            # save the model - with chosen model name!
            save = {
            'state_dict': model.state_dict(),
            }
            torch.save(save, path_join(output_dir, f"{model_name}.pt"))
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--data_dir', type=str, default='data/',
                        help="Path to the data folder")
    parser.add_argument('--output_dir', type=str, default='weights/',
                        help="Path to save model weights to")

    parser.add_argument('--model_name', type=str, default='mlp',
                        choices=["mlp", "cross_encoder", "pca"],
                        help='Model type to train and evaluate.')
    parser.add_argument("--model_embed", type=str, default="bert-base-uncased",
                        help="Name of model used to embed sentences")

    parser.add_argument('--optimizer_name', type=str, default="SGD",
                        choices=["SGD", "Adam"],
                        help='Which optimizer to use for training')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate for the optimizer')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Number of sentences to use in a batch')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs to train the model')

    parser.add_argument('--use_cuda', action='store_true', default=True,
                        help='Use GPU acceleration if available')
    parser.add_argument('--seed', default=42, type=int,
                        help='Seed to use for reproducing results')
    parser.add_argument('--use_tqdm', action='store_true', default=True,
                        help='Hide the progress bar during training loop')
    
    args = parser.parse_args()
    kwargs = vars(args)
    
    print("STARTING...  setup:")
    print(args)
    print("-" * 120)
    print("\n" * 2)
    
    main(**kwargs)