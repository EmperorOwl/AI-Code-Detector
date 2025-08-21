import argparse
import sys

from src.config import Config
from src.pre_processing.prepare import (load_samples_by_split,
                                        load_samples_by_assignment)
from src.utils.dual_output import DualOutput


def filter_datasets(dataset_choices: list[int]):
    if not dataset_choices:
        return Config.DATASETS  # No filter required
    datasets = []
    for idx in dataset_choices:
        if 0 <= idx < len(Config.DATASETS):
            datasets.append(Config.DATASETS[idx])
        else:
            print(f"Warning: Dataset index {idx} is out of range. Skipping.")
    return datasets


def get_samples(dataset_choices, test_size, train_datasets, test_datasets):
    if train_datasets and test_datasets:
        return load_samples_by_assignment(train_datasets, test_datasets)

    if train_datasets and not test_datasets:
        raise Exception("--test-datasets not set")

    if test_datasets and not train_datasets:
        raise Exception("--train-datasets not set")

    datasets = filter_datasets(dataset_choices)
    test_size = Config.TEST_SIZE if test_size is None else test_size
    return load_samples_by_split(datasets, test_size)


def parse_dataset_choices(val):
    if val.lower() == 'all':
        return 'all'
    try:
        return int(val)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid dataset value: {val}")


def main():
    parser = argparse.ArgumentParser(description='AI Code Detector CLI')
    parser.add_argument('--train-codebert', action='store_true',
                        help='Train the CodeBERT model')
    parser.add_argument('--train-ast', action='store_true',
                        help='Train the AST-based model')
    parser.add_argument('--train-both', action='store_true',
                        help='Train both models sequentially')

    parser.add_argument('--view-datasets', action='store_true',
                        help='View available datasets')

    parser.add_argument('--datasets',
                        type=int,
                        nargs='+',
                        help="List of datasets (e.g. 1 2 3) omit for all")
    parser.add_argument('--test-size', type=float, default=None)

    parser.add_argument('--train-datasets',
                        type=int,
                        nargs='+',
                        help='List of datasets for training')
    parser.add_argument('--test-datasets',
                        type=int,
                        nargs='+',
                        help='List of datasets for testing')

    args = parser.parse_args()

    if args.view_datasets:
        for i, dataset in enumerate(Config.DATASETS):
            print(f"{i}: {dataset[1]} ({dataset[0]})")

    elif args.train_codebert or args.train_ast or args.train_both:
        # Setup dual output for logging to log file
        dual_output = DualOutput()
        sys.stdout = dual_output

        # Load samples
        train_samples, test_samples = get_samples(args.datasets,
                                                  args.test_size,
                                                  args.train_datasets,
                                                  args.test_datasets)
        # Lazy import models
        from src.train import train_model

        # Train models
        if args.train_codebert or args.train_both:
            train_model('codebert', train_samples, test_samples, dual_output)
        if args.train_ast or args.train_both:
            train_model('ast', train_samples, test_samples, dual_output)

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
