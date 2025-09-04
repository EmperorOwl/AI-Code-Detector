import os
from logging import Logger

from sklearn.model_selection import train_test_split

from src.pre_processing.sample import (load_samples_from_dir,
                                       load_samples_from_csv,
                                       Sample)
from src.utils.random import get_random_samples


RANDOM_STATE = 42
DATASET_PATH = "./datasets"
DATASETS = [
    # name, language, ai, path
    ('gptsniffer', 'java', 'chatgpt', 'gptsniffer'),
    ('humaneval', 'java', 'gpt_4', 'humaneval_chatgpt4_java_merged.csv'),
    ('humaneval', 'java', 'chatgpt', 'humaneval_chatgpt_java_merged.csv'),
    ('humaneval', 'java', 'gemini_pro', 'humaneval_gemini_java_merged.csv'),
    ('humaneval', 'python', 'gpt_4', 'humaneval_chatgpt4_python_merged.csv'),
    ('humaneval', 'python', 'chatgpt', 'humaneval_chatgpt_python_merged.csv'),
    ('humaneval', 'python', 'gemini_pro', 'humaneval_gemini_python_merged.csv'),
    ('mbpp', 'python', 'gpt_4', 'mbpp_chatgpt4_python_merged.csv'),
    ('mbpp', 'python', 'chatgpt', 'mbpp_chatgpt_python_merged.csv'),
    ('mbpp', 'python', 'gemini_pro', 'mbpp_gemini_python_merged.csv'),
    ('codenet', 'python', 'gemini_flash', 'codenet_gemini_python.csv')
]


class Dataset:
    def __init__(self, name: str, language: str, ai: str, samples: list[Sample]):
        self.name = name
        self.language = language
        self.ai = ai
        self.samples = samples


def load_datasets() -> list[Dataset]:
    datasets = []

    for name, language, ai, path in DATASETS:
        full_path = os.path.join(DATASET_PATH, language, path)
        if full_path.endswith('.csv'):
            samples = load_samples_from_csv(full_path, language)
        else:
            samples = load_samples_from_dir(full_path, language)
        datasets.append(Dataset(name, language, ai, samples))

    return datasets


Splits = tuple[list[Sample], list[Sample], list[Sample]]


def split_dataset(dataset: Dataset,
                  validation_size,
                  test_size,
                  logger: Logger,
                  max_sample_count: int | None = None) -> Splits:
    """Split samples into train, validation, and test sets."""
    samples = dataset.samples
    if max_sample_count is not None:
        samples = get_random_samples(samples, max_sample_count)

    if test_size == 1:
        train_split = []
        val_split = []
        test_split = samples
    elif test_size == 0:
        train_split, val_split = train_test_split(
            samples,
            test_size=validation_size,
            random_state=RANDOM_STATE
        )
        test_split = []
    else:
        # First split: separate test set
        train_val_split, test_split = train_test_split(
            samples,
            test_size=test_size,
            random_state=RANDOM_STATE
        )

        # Second split: separate validation from remaining training data
        # Calculate validation size relative to the remaining data
        val_size_adjusted = validation_size / (1 - test_size)
        train_split, val_split = train_test_split(
            train_val_split,
            test_size=val_size_adjusted,
            random_state=RANDOM_STATE
        )

    logger.info(f"{dataset.name.ljust(15)}"
                f"{dataset.language.ljust(13)}"
                f"{dataset.ai.ljust(17)}"
                f"{str(len(samples)).ljust(10)}"
                f"{str(len(train_split)).ljust(10)}"
                f"{str(len(val_split)).ljust(10)}"
                f"{str(len(test_split)).ljust(10)}")

    return train_split, val_split, test_split


def split_datasets(logger: Logger,
                   test_size: float | None = None,
                   val_size: float | None = None,
                   language_filter: str | None = None,
                   config: dict | None = None,) -> Splits:
    all_datasets = load_datasets()

    if config:
        logger.info(f"Preparing datasets (custom configuration) ...")
    elif test_size and val_size:
        training_size = 1 - (val_size + test_size)
        logger.info(
            f"Preparing datasets (split: {int(training_size*100)}/"
            f"{int(val_size*100)}/{int(test_size*100)}) ..."
        )
    else:
        raise ValueError()

    logger.info(f"{'Dataset'.ljust(15)}"
                f"{'Language'.ljust(13)}"
                f"{'AI'.ljust(17)}"
                f"{'Total'.ljust(10)}"
                f"{'Train'.ljust(10)}"
                f"{'Val'.ljust(10)}"
                f"{'Test'.ljust(10)}")
    logger.info("-" * 80)

    train_samples, val_samples, test_samples = [], [], []
    for dataset in all_datasets:
        if (config and dataset.name not in config
                or language_filter and language_filter != dataset.language):
            continue  # Skip this dataset

        if config:
            val_size = config[dataset.name].get('val_size', 0)
            test_size = config[dataset.name].get('test_size', 0)
            max_sample_count = config[dataset.name].get('max_sample_count')

        train, val, test = split_dataset(
            dataset,
            val_size,
            test_size,
            logger,
            max_sample_count=max_sample_count
        )
        train_samples.extend(train)
        val_samples.extend(val)
        test_samples.extend(test)

    train_size = len(train_samples)
    val_size = len(val_samples)
    test_size = len(test_samples)
    total_size = train_size + val_size + test_size
    logger.info("-" * 80)
    logger.info(f"{'Total'.ljust(45)}"
                f"{str(total_size).ljust(10)}"
                f"{str(train_size).ljust(10)}"
                f"{str(val_size).ljust(10)}"
                f"{str(test_size).ljust(10)}")
    logger.info("")

    return train_samples, val_samples, test_samples
