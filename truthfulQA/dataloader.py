import torch
import pandas as pd
import numpy as np

from torch.utils.data import Dataset


class SentenceLabelDataset(Dataset):
    """
    A PyTorch dataset to handle sentences and their associated labels from a CSV file.

    This dataset loads sentences and their corresponding labels from a CSV file. 
    It can handle various tasks such as classification, where labels are transformed into encoded vectors.

    Args:
        path (str): Directory path where CSV files are located.

    Attributes:
        pairs (list): List of (query, answer) pairs.
        labels (list): List of labels associated with the pairs.
    """
    def __init__(self, path):
        """
        Initializes the SentenceLabelDataset with the provided CSV file path.

        Args:
            path (str): Directory path where CSV files are located.
        """
        self.answers = []
        self.labels = []

        try:
            check_data = pd.read_csv(f"{path}/generated/truthfulqa.csv")

        except FileNotFoundError as e:
            print(f"File not found: {e.filename}")
            raise

        for _, row in check_data.iterrows():
            answer = row["ans"]
            label = row["label"]
            self.answers.append(answer)
            self.labels.append(label)

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
        return len(self.answers)

    def __getitem__(self, idx):
        """
        Retrieves a specific sample from the dataset by index.

        Args:
            idx (int or slice): Index or slice indicating the sample(s) to retrieve.

        Returns:
            tuple: A tuple containing the sentence, its corresponding label, and the encoded label vector.
        """
        answer = self.answers[idx]
        label = self.labels[idx]

        return answer, self.rewrite_label(label)
