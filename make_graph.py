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

    # use cuda if cuda is available
    if args.use_cuda:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    else:
        device = torch.device("cpu")

    dataset = SentenceLabelDataset(args.path)
    dataloader = DataLoader(dataset, batch_size=1,
                            shuffle=False, num_workers=16)
    
    # Load pre-trained BERT model and tokenizer
    tokenizer = BertTokenizer.from_pretrained(args.model_name)
    model = BertModel.from_pretrained(args.model_name)
    model.to(device)

    node_features = get_embeddings(model, tokenizer, dataloader, device)
    labels = get_labels(dataloader).to(device)
    edge_index = get_edge_index(node_features).t().contiguous().to(device)

    data = Data(x=node_features, y=labels, edge_index=edge_index)
    print(data)