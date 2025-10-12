import time
import argparse


from src.dataset_processing.dataset_helper import DatasetHelper
from src.dataset_processing.dataset_tokenizer import DatasetTokenizer
from src.models.transformer import CodeBertModel, UniXcoderModel
from src.models.transformer.transformer_model import TransformerModel
from src.utils.logger import get_logger
from src.utils import config


def train_model(model_class: type[TransformerModel], use_ast: bool) -> None:
    # Start timer
    start_time = time.time()

    # Configure
    model_name = model_class.MODEL_NAME.lower()
    path_name = model_name + ('-ast' if use_ast else '')

    # Create logger
    log_file_path = f"{config.TRAINING_DIR}/{path_name}.log"
    logger = get_logger(model_class.MODEL_NAME, log_file_path)

    # Log start info
    logger.info(f"Log: {log_file_path}\n")

    # Load and sample datasets
    helper = DatasetHelper(logger)
    df = helper.load_dataset_from_csv(f'{config.DROID_PATH}')
    df = helper.sample_dataset(df, config.SAMPLING_REQUIREMENTS)

    # Tokenize dataset
    tokenizer = DatasetTokenizer(logger, model_class, use_ast)
    df = tokenizer.tokenize_code_samples(df)
    tokenizer.analyze_tokenization(df)

    # Split dataset
    train_df, val_df, _ = helper.split_dataset(df)
    helper.log_dataset_splits_table(df, train_df, val_df)

    # Initialise model
    model = model_class(logger)

    # Train model
    model.train(
        train_df,
        val_df,
        config.NUM_TRAIN_EPOCHS,
        config.TRAIN_BATCH_SIZE
    )

    # Save model
    model.save(dir_name=path_name)

    # Log runtime
    end_time = time.time()
    seconds = end_time - start_time
    logger.info(f"Runtime: {seconds:.2f} seconds ({seconds / 60:.2f} minutes)")
    logger.info("")


def main():
    # Create argument parser
    parser = argparse.ArgumentParser(
        description='Train transformer models for AI code detection'
    )

    # Model selection argument
    parser.add_argument(
        '--model',
        type=str,
        choices=['codebert', 'unixcoder'],
        required=True,
        help='Model to train (codebert or unixcoder)'
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

    # Select model class
    model_classes = {
        'codebert': CodeBertModel,
        'unixcoder': UniXcoderModel
    }
    model_class = model_classes[args.model]

    # Train the model
    train_model(model_class, args.use_ast)


if __name__ == "__main__":
    main()
