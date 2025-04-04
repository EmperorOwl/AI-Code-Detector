import torch
from torch.utils.data import Dataset


class CodeDataset(Dataset):
    """ Represents a dataset of code snippets. """

    def __init__(self, tokenizer, samples: list):
        """ Initializes the dataset. """
        self.tokenizer = tokenizer
        self.samples = samples

    def __len__(self):
        """ Returns the length of the dataset. """
        return len(self.samples)

    def __getitem__(self, index: int):
        """ Returns a sample from the dataset in the format required for it
        to be processed by the model.
        """
        code, label = self.samples[index]
        inputs = self.tokenizer.encode_plus(code,
                                            padding='max_length',
                                            max_length=512,
                                            truncation=True)
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(label, dtype=torch.long)
        }
