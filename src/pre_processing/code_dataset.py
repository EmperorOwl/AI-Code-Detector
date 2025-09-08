import torch
from torch.utils.data import Dataset

from src.pre_processing.sample import Sample
from src.pre_processing.ast import get_ast_representation


class CodeDataset(Dataset):

    def __init__(self,
                 tokenizer,
                 samples: list[Sample],
                 use_ast: bool = False,
                 max_length: int = 512):
        self.tokenizer = tokenizer
        self.samples = samples
        self.tokenized_samples = []
        self.use_ast = use_ast
        self.max_length = max_length
        self.truncated_count = 0
        self.tokenize()

    def __len__(self):
        return len(self.samples)

    def tokenize(self):
        for sample in self.samples:
            text_input = sample.code
            if self.use_ast:
                text_input = get_ast_representation(
                    sample.code,
                    sample.language,
                    include_tokens=True
                )

            inputs = self.tokenizer.encode_plus(text_input,
                                                padding='max_length',
                                                max_length=self.max_length,
                                                truncation=True)

            input_ids = inputs['input_ids']
            attention_mask = inputs['attention_mask']

            if sum(attention_mask) == self.max_length:
                self.truncated_count += 1

            self.tokenized_samples.append({
                'input_ids': torch.tensor(input_ids, dtype=torch.long),
                'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
                'labels': torch.tensor(sample.label, dtype=torch.long)
            })

    def __getitem__(self, index: int):
        return self.tokenized_samples[index]
