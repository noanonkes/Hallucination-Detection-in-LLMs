from torch.utils.data import Dataset
import torch
import pandas as pd
import numpy as np


class SentenceLabelDataset(Dataset):
    """
    A PyTorch dataset to handle sentences and their associated labels from a CSV file.

    This dataset loads sentences and their corresponding labels from a CSV file. 
    It can handle various tasks such as classification, where labels are transformed into encoded vectors.

    Args:
        csv_file (str): Path to the CSV file containing sentences and labels.
        limit (int, optional): Number of samples to use from the dataset (for debugging or limiting dataset size).

    Attributes:
        data (pandas.DataFrame): Loaded CSV data containing sentences and labels.

    Methods:
        rewrite_label(idx): Transforms a categorical label index into an encoded vector representation.
    """


    def __init__(self, path, limit=0):
        """
        Initializes the SentenceLabelDataset with the provided CSV file path and optional limit for dataset size.

        Args:
            csv_file (str): Path to the CSV file containing sentences and labels.
            limit (int, optional): Number of samples to use from the dataset (for debugging or limiting dataset size).
        """
        self.q_data = pd.read_json(path + 'sampled_data.json', lines=True)
        self.nc_data = pd.read_csv(path + 'generated/no_context.csv')
        self.wc_data = pd.read_csv(path + 'generated/with_context.csv')

    def rewrite_label(self, idx):
        """
        Transforms a categorical label index into an encoded vector representation.

        Args:
            idx (int): Categorical index representing a label.

        Returns:
            numpy.ndarray: Encoded vector representation of the label.
        """
        cat2vec = np.array([[0.,0.,0.],
                            [1.,0.,0.],
                            [1.,1.,0.],
                            [1.,1.,1.]])
        return cat2vec[idx]
    

    def __len__(self):
        """
        Returns the total number of samples in the dataset.

        Returns:
            int: Number of samples in the dataset.
        """
        return len(self.q_data)


    def __getitem__(self, idx):
        """
        Retrieves a specific sample from the dataset by index.

        Args:
            idx (int or slice): Index or slice indicating the sample(s) to retrieve.

        Returns:
            tuple: A tuple containing the sentence and its corresponding label.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        # Get the query and true answer from q_data at the given index
        query = self.q_data.iloc[idx]['data']['paragraphs'][0]['qas'][0]['question']
        answer = self.q_data.iloc[idx]['data']['paragraphs'][0]['qas'][0]['answers'][0]['text']
        
        # Find answers in nc_data and wc_data corresponding to the query qid
        nc_answers_df = self.nc_data.loc[self.nc_data['qid'] == idx]
        wc_answers_df = self.wc_data.loc[self.wc_data['qid'] == idx]
        
        # Get answers and labels from nc_data
        nc_answers = nc_answers_df['ans'].tolist()
        nc_labels = nc_answers_df['label'].tolist()
        
        # Get answers and labels from wc_data
        wc_answers = wc_answers_df['ans'].tolist()
        wc_labels = wc_answers_df['label'].tolist()

        all_answers = [answer] + nc_answers + wc_answers
        all_labels = [3] + nc_labels + wc_labels

        return query, all_answers, self.rewrite_label(all_labels)