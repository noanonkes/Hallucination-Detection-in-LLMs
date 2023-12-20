import torch, argparse
import numpy as np

from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertModel
from os.path import join as path_join, exists

from utils import compute_embeddings, set_seed, train, validate, EvalManager
from baselines import MisinformationMLP, MisinformationCrossEncoder, MisinformationPCA
from dataloader import SentenceLabelDataset, EmbeddingsDataset

def initialize_model(model, pretrained, n_components):
    """
    Initialize the specified model.
    """
    if model == "mlp":
        return MisinformationMLP(768, 3), 0

    elif model == "ce":
        return MisinformationCrossEncoder(pretrained, 3), 0

    elif model == "pca":
        return MisinformationPCA(n_components, 3), n_components

    else:
        raise ValueError(f"{model} is not supported!")

def get_optimizer(optimizer, model, lr):
    """
    Get the optimizer based on the provided name.
    """
    if optimizer== "SGD":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    elif optimizer == "Adam":
        return torch.optim.Adam(model.parameters(), lr=lr)

    else:
        raise ValueError("Unsupported optimizer type. Please choose between 'SGD' and 'Adam'.")

def precompute_embeddings(loader, tokenizer, pretrained, device):
    """
    Precompute BERT embeddings for the dataset.
    """
    all_embeddings, all_targets = [], []
    for pairs, targets in tqdm(loader, ncols=100, desc='Precomputing Embeddings'):
        embeddings = compute_embeddings(pairs, tokenizer, pretrained, device)
        all_embeddings.append(embeddings)
        all_targets.append(targets)

    all_embeddings = torch.cat(all_embeddings).cpu()
    all_targets = torch.cat(all_targets)
    
    return all_embeddings, all_targets

def get_dataloaders(data_dir, model, batch_size, tokenizer, pretrained, device):
    embedding_file = "./embeddings/bert-base-uncased.pt"
    
    if exists(embedding_file) and model != 'ce':
        # If the embedding file exists, load the embeddings
        print("Embedding file exists. Loading embeddings...")
        loaded_data = torch.load(embedding_file)
        all_embeddings, all_targets = loaded_data['embeddings'], loaded_data['targets']
        
    elif model != 'ce':
        print("No embeddings found. Creating embeddings...")
        dataset = SentenceLabelDataset(data_dir)
        loader = DataLoader(dataset, 512, shuffle=False, num_workers=4)

        # Precompute bert embeddings
        all_embeddings, all_targets = precompute_embeddings(loader, tokenizer, pretrained, device)
        
        # Save the embeddings for future use
        print("Saving embeddings...")
        torch.save({'embeddings': all_embeddings, 'targets': all_targets}, embedding_file)

    if model != 'ce':
        # Create an EmbeddingsDataset directly from the collected embeddings and targets
        dataset = EmbeddingsDataset(all_embeddings, all_targets)
    else:
        dataset = SentenceLabelDataset(data_dir)
    
    # Split the dataset into train, validation, and test sets
    train_size = int(len(dataset) * 0.7)
    val_size = int(len(dataset) * 0.1)
    test_size = len(dataset) - train_size - val_size

    train_data, val_data, test_data = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

    # Create DataLoaders for train, validation, and test sets
    train_loader= DataLoader(train_data, batch_size, shuffle=True, num_workers=4, drop_last=True)
    val_loader  = DataLoader(val_data, batch_size, shuffle=False, num_workers=4, drop_last=True)
    test_loader = DataLoader(test_data, batch_size, shuffle=False, num_workers=4, drop_last=True)

    return train_loader, val_loader, test_loader

def main(args):
    # Set the seed for reproducibility
    set_seed(args.seed)

    # Use cuda if cuda is available
    device = torch.device("cuda") if args.use_cuda and torch.cuda.is_available() else torch.device("cpu")

    # Instantiate model
    model, reduce = initialize_model(args.model, args.pretrained, args.n_components)
    model.to(device)

    # Load pre-trained BERT model and tokenizer
    tokenizer = BertTokenizer.from_pretrained(args.pretrained)
    pretrained = BertModel.from_pretrained(args.pretrained)
    pretrained.to(device)

    # Loading the data
    train_loader, val_loader, test_loader = get_dataloaders(args.data_dir, args.model, args.batch_size, tokenizer, pretrained, device)

    # Cross-entropy loss -- w/ logits
    criterion = torch.nn.BCEWithLogitsLoss()
    # criterion = SimCLR(device, 0.07)

    # Initialize metrics for validation
    metrics = EvalManager(device)

    # Instantiate optimizer
    optimizer = get_optimizer(args.optimizer, model, args.lr)

    best_val = -np.inf
    for i in tqdm(range(args.epochs), ncols=100, desc=f'Training'):
        train_loss = train(model, train_loader, criterion, optimizer, device, reduce)
        val_loss, val_metrics = validate(model, val_loader, criterion, metrics, device, reduce)
        print(f'Epoch: {i}\n\ttrain: {train_loss}\n\tval: {val_loss}')
        print('Val mse: ', val_metrics['MSE'], '\tVal accuracy: ', val_metrics['Accuracy'])

        if best_val < val_loss:
            best_val = val_loss
            # Save the state of the best model
            best_model_state = {'state_dict': model.state_dict()}
            torch.save(best_model_state, path_join(args.output_dir, f"{args.model}.pt"))

    # Load the best model
    if best_model_state:
        model.load_state_dict(best_model_state['state_dict'])
        test_loss, test_metrics = validate(model, test_loader, criterion, metrics, device, reduce)
        print('Test Loss: ', test_loss)
        print('Test MSE: ', test_metrics['MSE'], '\tTest Accuracy: ', test_metrics['Accuracy'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--data_dir', type=str, default='../data/',
                        help="Path to the data folder.")
    parser.add_argument('--output_dir', type=str, default='../weights/',
                        help="Path to save model weights to.")

    parser.add_argument('--model', type=str, default='mlp',
                        choices=["mlp", "ce", "pca"],
                        help='Model type to train and evaluate.')
    parser.add_argument("--pretrained", type=str, default="bert-base-uncased",
                        help="Model for pretrained embeddings.")

    parser.add_argument('--optimizer', type=str, default="SGD",
                        choices=["SGD", "Adam"],
                        help='Optimizer to use for training.')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate for the optimizer.')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Number of items in a batch.')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='Number of epochs to train the model.')
    parser.add_argument('--n_components', type=int, default=0,
                        help='Number of components for PCA.')

    parser.add_argument('--use_cuda', action='store_true', default=True,
                        help='Use GPU acceleration if available.')
    parser.add_argument('--seed', default=42, type=int,
                        help='Seed to use for reproducing results')

    # Parse arguments
    args = parser.parse_args()

    # Print setup information
    print("STARTING...  setup:")
    print(args)
    print("-" * 120)
    print("\n" * 2)

    # Execute main function
    main(args)
