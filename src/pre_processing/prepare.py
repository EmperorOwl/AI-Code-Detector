import os

from sklearn.model_selection import train_test_split

from src.config import Config
from src.utils.helper import load_samples_from_dir, load_samples_from_csv


def load_samples_by_split(datasets: list,
                          test_size: float) -> tuple[list, list]:
    """Load samples by splitting each dataset individually."""
    print("Preparing datasets (split mode)...")
    print(f"TEST_SIZE: {int(test_size * 100)}%")
    print(f"{'Dataset'.ljust(25)}"
          f"{'Language'.ljust(15)}"
          f"{'Total'.ljust(10)}"
          f"{'Train'.ljust(10)}"
          f"{'Test'.ljust(10)}")
    print("-" * 70)

    # Split each dataset individually and combine the splits
    train_samples = []
    test_samples = []
    for dataset_langauge, dataset_name, dataset_path in datasets:
        # Load samples from the dataset
        path = os.path.join(Config.DATASET_PATH,
                            dataset_langauge, dataset_path)
        if path.endswith('.csv'):
            samples = load_samples_from_csv(path, dataset_langauge)
        else:
            samples = load_samples_from_dir(path, dataset_langauge)

        # Split the samples into train and test sets
        train_split, test_split = train_test_split(
            samples,
            test_size=test_size,
            random_state=Config.RANDOM_STATE
        )

        train_samples.extend(train_split)
        test_samples.extend(test_split)
        print(f"{dataset_name.ljust(25)}"
              f"{dataset_langauge.ljust(15)}"
              f"{str(len(samples)).ljust(10)}"
              f"{str(len(train_split)).ljust(10)}"
              f"{str(len(test_split)).ljust(10)}")

    print("-" * 70)
    print(f"{'Total'.ljust(40)}"
          f"{str(len(train_samples) + len(test_samples)).ljust(10)}"
          f"{str(len(train_samples)).ljust(10)}"
          f"{str(len(test_samples)).ljust(10)}")
    print("\n")

    return train_samples, test_samples


def load_samples_by_assignment(train_datasets: list,
                               test_datasets: list) -> tuple[list, list]:
    """Load samples by assigning specific datasets to train and test sets."""
    print("Preparing datasets...")
    print(f"{'Dataset'.ljust(25)}"
          f"{'Language'.ljust(15)}"
          f"{'Split'.ljust(10)}"
          f"{'Count'.ljust(10)}")
    print("-" * 60)

    train_samples = []
    test_samples = []

    # Load training datasets
    for dataset_langauge, dataset_name, dataset_path in train_datasets:
        path = os.path.join(Config.DATASET_PATH,
                            dataset_langauge, dataset_path)
        if path.endswith('.csv'):
            samples = load_samples_from_csv(path, dataset_langauge)
        else:
            samples = load_samples_from_dir(path, dataset_langauge)

        train_samples.extend(samples)
        print(f"{dataset_name.ljust(25)}"
              f"{dataset_langauge.ljust(15)}"
              f"{'Train'.ljust(10)}"
              f"{str(len(samples)).ljust(10)}")

    # Load testing datasets
    for dataset_langauge, dataset_name, dataset_path in test_datasets:
        path = os.path.join(Config.DATASET_PATH,
                            dataset_langauge, dataset_path)
        if path.endswith('.csv'):
            samples = load_samples_from_csv(path, dataset_langauge)
        else:
            samples = load_samples_from_dir(path, dataset_langauge)

        test_samples.extend(samples)
        print(f"{dataset_name.ljust(25)}"
              f"{dataset_langauge.ljust(15)}"
              f"{'Test'.ljust(10)}"
              f"{str(len(samples)).ljust(10)}")

    print("-" * 60)
    print(f"{'Total Train'.ljust(40)}"
          f"{str(len(train_samples)).ljust(10)}")
    print(f"{'Total Test'.ljust(40)}"
          f"{str(len(test_samples)).ljust(10)}")
    print(f"{'Grand Total'.ljust(40)}"
          f"{str(len(train_samples) + len(test_samples)).ljust(10)}")
    print("\n")

    return train_samples, test_samples
