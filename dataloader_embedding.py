from torch.utils.data import Dataset
import torch
import pandas as pd
import numpy as np


class EmbeddingLabelDataset(Dataset):
    """
    A PyTorch dataset to handle sentences embeddings and their associated labels from a CSV file.

    This dataset loads sentence embeddings and their corresponding labels from a CSV file. 
    It can handle various tasks such as classification, where labels are transformed into encoded vectors.

    Args:
        csv_file (str): Path to the CSV file containing sentence embeddings and labels.
        limit (int, optional): Number of samples to use from the dataset (for debugging or limiting dataset size).

    Attributes:
        data (pandas.DataFrame): Loaded CSV data containing sentence embeddings and labels.

    Methods:
        rewrite_label(idx): Transforms a categorical label index into an encoded vector representation.
    """


    def __init__(self, csv_file, limit=0):
        """
        Initializes the SentenceLabelDataset with the provided CSV file path and optional limit for dataset size.

        Args:
            csv_file (str): Path to the CSV file containing sentence embeddings and labels.
            limit (int, optional): Number of samples to use from the dataset (for debugging or limiting dataset size).
        """
        self.data = pd.read_csv(csv_file)
        if limit:
            self.data = self.data.head(limit)


    def rewrite_label(self, idx):
        """
        Transforms a categorical label index into an encoded vector representation.

        Args:
            idx (int): Categorical index representing a label.

        Returns:
            numpy.ndarray: Encoded vector representation of the label.
        """
        cat2vec = np.array([[0,0,0],
                              [1,0,0],
                              [1,1,0],
                              [1,1,1]])
        return cat2vec[idx]
    

    def __len__(self):
        """
        Returns the total number of samples in the dataset.

        Returns:
            int: Number of samples in the dataset.
        """
        return len(self.data)


    def __getitem__(self, idx):
        """
        Retrieves a specific sample from the dataset by index.

        Args:
            idx (int or slice): Index or slice indicating the sample embedding(s) to retrieve.

        Returns:
            tuple: A tuple containing the embeddings and its corresponding label.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        embedding = self.data.iloc[idx]["embedding"]
        cat = self.data.iloc[idx]["label"]
        label = self.rewrite_label(cat)

        return embedding, label