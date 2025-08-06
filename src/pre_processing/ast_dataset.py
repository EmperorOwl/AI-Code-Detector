from torch.utils.data import Dataset
from src.pre_processing.ast_node import AstParser


class AstDataset(Dataset):
    """ Represents a dataset of code snippets with AST representations. """

    def __init__(self, samples: list):
        """ 
        Initializes the dataset with samples containing code, label, and language.
        Each sample should be a tuple of (code, label, language).
        """
        self.samples = []
        self.valid_samples = 0
        
        # Process samples and extract AST representations
        for code, label, language in samples:
            ast_repr = AstParser.get_ast_representation(code, language)
            if ast_repr is not None:
                self.samples.append((code, label, ast_repr, language))
                self.valid_samples += 1

    def __len__(self):
        """ Returns the length of the dataset. """
        return len(self.samples)

    def __getitem__(self, index: int):
        """ Returns a sample from the dataset. """
        return self.samples[index]
    
    def get_valid_count(self):
        """ Returns the number of valid samples (with parseable AST). """
        return self.valid_samples
