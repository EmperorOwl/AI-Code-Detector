import torch
from torch.utils.data import Dataset

from src.pre_processing.sample import Sample


class CodeDataset(Dataset):

    def __init__(self, tokenizer, samples: list[Sample]):
        self.tokenizer = tokenizer
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int):
        """ Returns a sample from the dataset in the format required for it
        to be processed by the model.
        """
        sample = self.samples[index]

        inputs = self.tokenizer.encode_plus(sample.code,
                                            padding='max_length',
                                            max_length=512,
                                            truncation=True)
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(sample.label, dtype=torch.long)
        }
