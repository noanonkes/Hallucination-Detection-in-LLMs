import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertModel
from torch_geometric.data import Data

import argparse
import numpy as np
from dataloader import SentenceLabelDataset
from graph_utils import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--use-cuda", action="store_true", default=False,
                        help="Use GPU acceleration if available")
    parser.add_argument("--path", type=str, default="data/",
                        help="Path to data")
    parser.add_argument("--model_name", type=str, default="bert-base-uncased",
                        help="Name of model used to embed sentences")
    parser.add_argument('--num-workers', type=int, default=4,
                        help="Number of cores to use when loading the data")
    args = parser.parse_args()

    # for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # use cuda if cuda is available
    if args.use_cuda:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    else:
        device = torch.device("cpu")

    # load the data from root folder
    full_dataset = SentenceLabelDataset(args.path)
    dataloader = DataLoader(full_dataset, batch_size=1,
                            shuffle=True, num_workers=args.num_workers)

    # get train/val/holdout/test sizes -> 70 10 15 5
    train_size = int(len(full_dataset) * 0.7)
    val_size = int(len(full_dataset) * 0.1)
    holdout_size = int(len(full_dataset) * 0.15)
    test_size = len(full_dataset) - train_size - val_size - holdout_size

    # Load pre-trained BERT model and tokenizer
    tokenizer = BertTokenizer.from_pretrained(args.model_name)
    model = BertModel.from_pretrained(args.model_name)
    model.to(device)

    print("Calculating node features...")
    node_features = get_embeddings(model, tokenizer, dataloader, device)
    print("Node features shape:", node_features.shape, "\n")

    print("Getting labels...")
    labels = get_labels(dataloader).to(device)
    print("Labels shape:", labels.shape, "\n")

    print("Calculating edges...")
    distances, edge_index = get_edge_index(node_features)
    edge_index = edge_index.t().contiguous().to(device)
    print("Edge index shape:", edge_index.shape)

    # 11 answers per questions
    idx = np.arange(len(full_dataset)) * 11

    train_idx = np.array([np.arange(i, i+11) for i in idx[:train_size]]).ravel()
    val_idx = np.array([np.arange(i, i+11) for i in idx[train_size:train_size + val_size]]).ravel()
    holdout_idx = np.array([np.arange(i, i+11) for i in idx[train_size + val_size:train_size + val_size + holdout_size]]).ravel()
    test_idx = np.array([np.arange(i, i+11) for i in idx[train_size + val_size + holdout_size:]]).ravel()

    data = Data(x=node_features, y=labels, edge_index=edge_index)
    data.train_idx = torch.tensor(train_idx, dtype=torch.long)
    data.val_idx = torch.tensor(val_idx, dtype=torch.long)
    data.holdout_idx = torch.tensor(holdout_idx, dtype=torch.long)
    data.test_idx = torch.tensor(test_idx, dtype=torch.long)

    print("Final dataloader:", data)

    torch.save(data, args.path + 'graph.pt')
    torch.save(distances, args.path + 'distances.pt')