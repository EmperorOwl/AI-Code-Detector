import argparse


from src.train import train_codebert, train_ast
from src.pre_processing.prepare import load_samples


def main():
    parser = argparse.ArgumentParser(description='AI Code Detector CLI')
    parser.add_argument('--train-codebert', action='store_true',
                        help='Train the CodeBERT model')
    parser.add_argument('--train-ast', action='store_true',
                        help='Train the AST-based model')
    parser.add_argument('--train-both', action='store_true',
                        help='Train both models sequentially')

    args = parser.parse_args()

    if args.train_codebert:
        train_codebert(*load_samples())
    elif args.train_ast:
        train_ast(*load_samples())
    elif args.train_both:
        train_samples, test_samples = load_samples()
        train_codebert(train_samples, test_samples)
        train_ast(train_samples, test_samples)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
