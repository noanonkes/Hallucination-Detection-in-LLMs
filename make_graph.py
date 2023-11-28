import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertModel
from torch_geometric.data import Data

import argparse
from dataloader import SentenceLabelDataset
from graph_utils import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--use-cuda", action="store_true", default=False,
                        help="Use GPU acceleration if available")
    parser.add_argument("--path", type=str, default="data/generated/full_train.csv",
                        help="CSV file containing the data")
    parser.add_argument("--model_name", type=str, default="bert-base-uncased",
                        help="Name of model used to embed sentences")
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
    train_size = int(len(full_dataset) * 0.6)
    val_size = int(len(full_dataset) * 0.1)
    holdout_size = int(len(full_dataset) * 0.15)
    test_size = len(full_dataset) - train_size - val_size - holdout_size

    dataloader = DataLoader(full_dataset, batch_size=1,
                            shuffle=True, num_workers=args.num_workers)
        
    # Load pre-trained BERT model and tokenizer
    tokenizer = BertTokenizer.from_pretrained(args.model_name)
    model = BertModel.from_pretrained(args.model_name)
    model.to(device)

    node_features = get_embeddings(model, tokenizer, dataloader, device)
    labels = get_labels(dataloader).to(device)
    edge_index = get_edge_index(node_features).t().contiguous().to(device)

    # 11 answers per questions
    idx = np.arange(len(full_dataset)) * 11

    train_idx = np.array([np.arange(i, i+11) for i in idx[:train_size]]).ravel()
    val_idx = np.array([np.arange(i, i+11) for i in idx[train_size:train_size + val_size]]).ravel()
    holdout_idx = np.array([np.arange(i, i+11) for i in idx[train_size + val_size:train_size + val_size + holdout_size]]).ravel()
    test_idx = np.array([np.arange(i, i+11) for i in idx[train_size + val_size + holdout_size:]]).ravel()

    data = Data(x=node_features, y=labels, edge_index=edge_index)
    data.train_idx = train_idx
    data.val_idx = val_idx
    data.test_idx = test_idx
    print(data)