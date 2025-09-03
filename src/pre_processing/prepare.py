import os
from logging import Logger

from sklearn.model_selection import train_test_split

from src.pre_processing.sample import load_samples_from_dir, load_samples_from_csv


RANDOM_STATE = 42
DATASET_PATH = "./datasets"
DATASETS = [
    ('java', 'GPTSniffer ChatGPT', 'gptsniffer'),
    ('java', 'HumanEval GPT-4', 'humaneval_chatgpt4_java_merged.csv'),
    ('java', 'HumanEval ChatGPT', 'humaneval_chatgpt_java_merged.csv'),
    ('java', 'HumanEval Gemini Pro', 'humaneval_gemini_java_merged.csv'),
    ('python', 'HumanEval GPT-4', 'humaneval_chatgpt4_python_merged.csv'),
    ('python', 'HumanEval ChatGPT', 'humaneval_chatgpt_python_merged.csv'),
    ('python', 'HumanEval Gemini Pro', 'humaneval_gemini_python_merged.csv'),
    ('python', 'MBPP GPT-4', 'mbpp_chatgpt4_python_merged.csv'),
    ('python', 'MBPP ChatGPT', 'mbpp_chatgpt_python_merged.csv'),
    ('python', 'MBPP Gemini Pro', 'mbpp_gemini_python_merged.csv'),
    ('python', 'CodeNet Gemini Flash', 'codenet_gemini_python.csv')
]


def load_samples(datasets: list,
                 validation_size: float,
                 test_size: float,
                 logger: Logger) -> tuple[list, list, list]:
    """Load samples by splitting each dataset individually into train, 
    validation, and test datasets. 
    """
    training_size = 1 - (validation_size + test_size)

    logger.info(
        f"Preparing datasets (split: {int(training_size*100)}/"
        f"{int(validation_size*100)}/{int(test_size*100)}) ..."
    )
    logger.info(f"{'Dataset'.ljust(25)}"
                f"{'Language'.ljust(15)}"
                f"{'Total'.ljust(10)}"
                f"{'Train'.ljust(10)}"
                f"{'Val'.ljust(10)}"
                f"{'Test'.ljust(10)}")
    logger.info("-" * 80)

    # Split each dataset individually and combine the splits
    train_samples = []
    validation_samples = []
    test_samples = []
    for dataset_langauge, dataset_name, dataset_path in datasets:
        # Load samples from the dataset
        path = os.path.join(DATASET_PATH, dataset_langauge, dataset_path)
        if path.endswith('.csv'):
            samples = load_samples_from_csv(path, dataset_langauge)
        else:
            samples = load_samples_from_dir(path, dataset_langauge)

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

        train_samples.extend(train_split)
        validation_samples.extend(val_split)
        test_samples.extend(test_split)
        logger.info(f"{dataset_name.ljust(25)}"
                    f"{dataset_langauge.ljust(15)}"
                    f"{str(len(samples)).ljust(10)}"
                    f"{str(len(train_split)).ljust(10)}"
                    f"{str(len(val_split)).ljust(10)}"
                    f"{str(len(test_split)).ljust(10)}")

    train_size = len(train_samples)
    validation_size = len(validation_samples)
    test_size = len(test_samples)
    total_size = train_size + validation_size + test_size
    logger.info("-" * 80)
    logger.info(f"{'Total'.ljust(40)}"
                f"{str(total_size).ljust(10)}"
                f"{str(train_size).ljust(10)}"
                f"{str(validation_size).ljust(10)}"
                f"{str(test_size).ljust(10)}")
    logger.info("")

    return train_samples, validation_samples, test_samples
