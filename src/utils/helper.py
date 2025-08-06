import os
import pandas as pd


def clean_code(code: str) -> str:
    """ Replaces Windows-style line endings with Unix-style line endings and
    replaces tabs with spaces.
    """
    code = code.replace('\r\n', '\n')
    code = code.replace('\t', '    ')
    return code


def load_samples_from_dir(directory: str, 
                          language: str) -> list[tuple[str, int, str]]:
    """ Load samples from under a directory.
    Label 0 is AI-written, label 1 is human-written.
    """
    samples = []
    for filename in os.listdir(directory):
        label = int(filename.split('_')[0])
        filepath = os.path.join(directory, filename)
        with open(filepath, 'r', encoding='utf-8') as file:
            code = file.read()
            samples.append((code, label, language))
    return samples


def load_samples_from_csv(path: str, 
                          language: str) -> list[tuple[str, int, str]]:
    """ Load samples from a CSV file. """
    samples = []
    df = pd.read_csv(path)
    for index, row in df.iterrows():
        code = row['code']
        if pd.isna(code):
            continue
        code = clean_code(code)
        label = 0 if row['label'] == 'lm' else 1
        samples.append((code, label, language))
    return samples
