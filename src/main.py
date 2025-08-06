import argparse

from src.models.codebert_model import CodeBertModel
from src.models.ast_model import AstModel


def train_codebert():
    print("Training CodeBERT model...")
    model = CodeBertModel(use_saved=False)
    model.train()
    model.evaluate()

    if input("Save model (y/n): ").strip().lower() == "y":
        model.save()
        print("CodeBERT model saved!")


def train_ast():
    print("Training AST model...")
    model = AstModel(use_saved=False)
    model.train()
    model.evaluate()

    if input("Save model (y/n): ").strip().lower() == "y":
        model.save()
        print("AST model saved!")


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
        train_codebert()
    elif args.train_ast:
        train_ast()
    elif args.train_both:
        print("Training both models...")
        train_codebert()
        print("\n" + "="*50)
        train_ast()
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
