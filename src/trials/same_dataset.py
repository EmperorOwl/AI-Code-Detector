import argparse

from src.models.transformer import CodeBertModel, UniXcoderModel
from src.trials.helper import run_trial
from src.utils.config import CONFIG, TEST_CONFIG


def eval_codebert(trial_name, config):
    run_trial(trial_name,
              CodeBertModel,
              config['SAMPLING_REQUIREMENTS'],
              config['EVAL_BATCH_SIZE'])


def eval_unixcoder(trial_name, config):
    run_trial(trial_name,
              UniXcoderModel,
              config['SAMPLING_REQUIREMENTS'],
              config['EVAL_BATCH_SIZE'])


def main():
    TRIAL_NAME = 'same_dataset'

    # Create argument parser
    parser = argparse.ArgumentParser(
        description='Evaluate transformer models for AI code detection'
    )

    # Configuration selection argument
    parser.add_argument(
        '--test',
        action='store_true',
        help='Use test configuration (default: use full configuration)'
    )

    # Parse arguments
    args = parser.parse_args()

    # Select configuration based on --test flag
    config = TEST_CONFIG if args.test else CONFIG

    eval_codebert(TRIAL_NAME, config)
    eval_unixcoder(TRIAL_NAME, config)


if __name__ == "__main__":
    main()
