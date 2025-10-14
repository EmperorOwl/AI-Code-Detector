import argparse

from src.models.transformer import CodeBertModel, UniXcoderModel
from src.models.classifiers import EmbeddingModel
from src.trials.helper import run_trial


def eval_codebert(trial_name, use_ast):
    run_trial(trial_name, CodeBertModel, use_ast)


def eval_unixcoder(trial_name, use_ast):
    run_trial(trial_name, UniXcoderModel, use_ast)


def eval_embedding(trial_name, use_ast):
    run_trial(trial_name, EmbeddingModel, use_ast)


def main():
    # Create argument parser
    parser = argparse.ArgumentParser(
        description='Evaluate transformer models for AI code detection'
    )

    # Trial name argument
    parser.add_argument(
        '--trial',
        type=str,
        choices=['same_sources', 'independent_sources'],
        required=True,
        help='Trial to run',
    )

    # Model selection argument
    parser.add_argument(
        '--model',
        type=str,
        choices=['codebert', 'unixcoder', 'embedding'],
        help='Model to evaluate',
    )

    # Use AST argument
    parser.add_argument(
        '--use-ast',
        action='store_true',
        help='Use AST representation for tokenization',
        default=False
    )

    # Parse arguments
    args = parser.parse_args()
    trial_name = args.trial
    model = args.model
    use_ast = args.use_ast

    if model == 'codebert':
        eval_codebert(trial_name, use_ast)
    elif model == 'unixcoder':
        eval_unixcoder(trial_name, use_ast)
    elif model == 'embedding':
        eval_embedding(trial_name, use_ast)


if __name__ == "__main__":
    main()
