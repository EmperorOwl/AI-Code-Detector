import os
import pandas as pd


class Sample:
    """ Represents a sample of source code with its label and programming 
    language it was written in (needed to parse into AST code embedding)

    Label 0 means human-written code
    Label 1 means AI-generated code
    """

    def __init__(self, code: str, label: int, language: str):
        self.code = code
        self.label = label
        self.language = language


def clean_code(code: str) -> str:
    """ Replaces Windows-style line endings with Unix-style line endings and
    replaces tabs with spaces.
    """
    code = code.replace('\r\n', '\n')
    code = code.replace('\t', '    ')
    return code


def load_samples_from_dir(directory: str, language: str) -> list[Sample]:
    """ Load samples from under a directory. """
    samples = []
    for filename in os.listdir(directory):
        label = filename.split('_')[0]
        # Reverse GPTSniffer's labels
        if label == '1':
            label = 0
        elif label == '0':
            label = 1
        else:
            raise ValueError(f"Invalid label: {label}")
        filepath = os.path.join(directory, filename)
        with open(filepath, 'r', encoding='utf-8') as file:
            code = file.read()
            samples.append(Sample(code, label, language))
    return samples


def load_samples_from_csv(path: str, language: str) -> list[Sample]:
    """ Load samples from a CSV file. """
    samples = []
    df = pd.read_csv(path)
    for _, row in df.iterrows():
        code = row['code']
        if pd.isna(code):
            continue
        code = clean_code(code)
        if row['label'] == 'lm':
            label = 1
        elif row['label'] == 'human':
            label = 0
        else:
            raise ValueError(f"Invalid label: {row['label']}")
        samples.append(Sample(code, label, language))
    return samples
