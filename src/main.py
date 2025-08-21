import argparse

from src.train import train_codebert, train_ast
from src.pre_processing.prepare import (load_samples_by_split,
                                        load_samples_by_assignment)


def main():
    parser = argparse.ArgumentParser(description='AI Code Detector CLI')
    parser.add_argument('--train-codebert', action='store_true',
                        help='Train the CodeBERT model')
    parser.add_argument('--train-ast', action='store_true',
                        help='Train the AST-based model')
    parser.add_argument('--train-both', action='store_true',
                        help='Train both models sequentially')
    parser.add_argument(
        '--dataset-mode',
        choices=['split', 'assign'],
        default='split',
        help='Dataset loading mode: "split" to split each dataset, "assign" to use predefined train/test sets'
    )

    args = parser.parse_args()

    # Choose loading function based on dataset mode
    if args.dataset_mode == 'assign':
        train_samples, test_samples = load_samples_by_assignment()
    else:
        train_samples, test_samples = load_samples_by_split()

    if args.train_codebert:
        train_codebert(train_samples, test_samples)
    elif args.train_ast:
        train_ast(train_samples, test_samples)
    elif args.train_both:
        train_codebert(train_samples, test_samples)
        train_ast(train_samples, test_samples)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
