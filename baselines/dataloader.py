import torch
import pandas as pd
import numpy as np

from torch.utils.data import Dataset
from tqdm import tqdm


class SentenceLabelDataset(Dataset):
    """
    A PyTorch dataset to handle sentences and their associated labels from a CSV file.

    This dataset loads sentences and their corresponding labels from a CSV file. 
    It can handle various tasks such as classification, where labels are transformed into encoded vectors.

    Args:
        path (str): Directory path where CSV files are located.
        limit (int, optional): Number of samples to use from the dataset (for debugging or limiting dataset size).

    Attributes:
        pairs (list): List of (query, answer) pairs.
        labels (list): List of labels associated with the pairs.
    """
    def __init__(self, path, limit=0):
        """
        Initializes the SentenceLabelDataset with the provided CSV file path and optional limit for dataset size.

        Args:
            path (str): Directory path where CSV files are located.
            limit (int, optional): Number of samples to use from the dataset (for debugging or limiting dataset size).
        """
        self.pairs = []
        self.labels = []

        try:
            q_data = pd.read_json(f"{path}/sampled_data.json", lines=True)
            nc_data = pd.read_csv(f"{path}/generated/no_context.csv")
            wc_data = pd.read_csv(f"{path}/generated/with_context.csv")

        except FileNotFoundError as e:
            print(f"File not found: {e.filename}")
            raise

        iter = len(q_data) if limit == 0 else min(len(q_data), limit)
        with tqdm(total=iter, desc='Loading dataset') as pbar:
            for index, row in q_data.iterrows():
                query = row['data']['paragraphs'][0]['qas'][0]['question']
                answer = row['data']['paragraphs'][0]['qas'][0]['answers'][0]['text']
                
                self._add_pairs(query, answer, 3)
                self._add_generated(index, query, nc_data, wc_data)

                if limit != 0 and len(self.pairs) >= limit:
                    break
    
    def _add_pairs(self, query, answer, label):
        self.pairs.append((query, answer))
        self.labels.append(label)

    def _add_generated(self, index, query, nc_data, wc_data):
        nc_ans = nc_data[nc_data['qid'] == index]['ans'].tolist()
        nc_label = nc_data[nc_data['qid'] == index]['label'].tolist()

        wc_ans = wc_data[wc_data['qid'] == index]['ans'].tolist()
        wc_label = wc_data[wc_data['qid'] == index]['label'].tolist()

        for (ans, label) in zip(nc_ans, nc_label):
            self._add_pairs(query, ans, label)

        for (ans, label) in zip(wc_ans, wc_label):
            self._add_pairs(query, ans, label)

    def rewrite_label(self, idx):
        """
        Transforms a categorical label index into an encoded vector representation.

        Args:
            idx (int): Categorical index representing a label.

        Returns:
            numpy.ndarray: Encoded vector representation of the label.
        """
        cat2vec = np.array([[0., 0., 0.],
                            [1., 0., 0.],
                            [1., 1., 0.],
                            [1., 1., 1.]])

        return torch.tensor(cat2vec[idx])

    def __len__(self):
        """
        Returns the total number of samples in the dataset.

        Returns:
            int: Number of samples in the dataset.
        """
        return len(self.pairs)

    def __getitem__(self, idx):
        """
        Retrieves a specific sample from the dataset by index.

        Args:
            idx (int or slice): Index or slice indicating the sample(s) to retrieve.

        Returns:
            tuple: A tuple containing the sentence, its corresponding label, and the encoded label vector.
        """
        query, answer = self.pairs[idx]
        labels= self.labels[idx]

        return (query, answer), self.rewrite_label(labels)
    
    
class EmbeddingsDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Retrieves a specific sample from the dataset by index.

        Args:
            idx (int or slice): Index or slice indicating the sample(s) to retrieve.

        Returns:
            tuple: A tuple containing the sample data and its corresponding target label.
        """
        embeddings = self.data[idx]
        labels = self.targets[idx]
        
        return embeddings, labels
