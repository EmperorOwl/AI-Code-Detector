import pandas as pd
import torch
from torch.utils.data import Dataset


class CodeDataset(Dataset):
    """ Custom dataset for transformer models """

    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        if 'input_ids' not in row or 'attention_mask' not in row:
            raise KeyError("input_ids or attention_mask not found in row")

        input_ids = row['input_ids']
        attention_mask = row['attention_mask']
        label = row['Label']

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(label, dtype=torch.long)
        }
