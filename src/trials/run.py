import argparse

from src.models.transformer import CodeBertModel, UniXcoderModel
from src.trials.helper import run_trial


def eval_codebert(trial_name):
    run_trial(trial_name, CodeBertModel)


def eval_unixcoder(trial_name):
    run_trial(trial_name, UniXcoderModel)


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
        choices=['codebert', 'unixcoder', 'all'],
        help='Model to evaluate',
        default='all'
    )

    # Parse arguments
    args = parser.parse_args()
    trial_name = args.trial
    model = args.model

    if model == 'codebert':
        eval_codebert(trial_name)
    elif model == 'unixcoder':
        eval_unixcoder(trial_name)
    elif model == 'all':
        eval_codebert(trial_name)
        eval_unixcoder(trial_name)


if __name__ == "__main__":
    main()
