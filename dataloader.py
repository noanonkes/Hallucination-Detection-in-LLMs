from torch.utils.data import Dataset
import torch
import pandas as pd
import numpy as np

class SentenceLabelDataset(Dataset):
    """Dataset."""

    def __init__(self, csv_file, limit=2):
        """
        Arguments:
            csv_file (string): Path to the csv file with labels.
            limit (int): Number of samples to use (for debugging).
        """
        self.data = pd.read_csv(csv_file).head(limit)

    def rewrite_label(self, idx):
        cat2vec = np.array([[0,0,0],
                              [1,0,0],
                              [1,1,0],
                              [1,1,1]])
        return cat2vec[idx]
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sentence = self.data.iloc[idx]["sentence"]
        cat = self.data.iloc[idx]["label"]
        label = self.rewrite_label(cat)

        return sentence, label