import time
import argparse

from src.dataset_processing.dataset_helper import DatasetHelper
from src.models.classifiers.embedding_model import EmbeddingModel
from src.models.classifiers.xgboost_model import XGBoostEmbeddingModel
from src.utils.logger import get_logger
from src.utils import config


def train_embedding_model(model_class: type[EmbeddingModel | XGBoostEmbeddingModel]) -> None:
    """
    Train the embedding-based classifier.
    UniXcoder is always frozen - only the classifier is trained.

    Args:
        classifier_type (str): Type of classifier to use ("simple" or "xgboost")
    """
    # Start timer
    start_time = time.time()

    # Configure based on classifier type
    model_name = model_class.MODEL_NAME.lower()

    # Create logger
    log_file_path = f"{config.TRAINING_DIR}/{model_name}.log"
    logger = get_logger(model_class.MODEL_NAME, log_file_path)

    # Log start info
    logger.info(f"Log: {log_file_path}\n")

    # Load and sample datasets
    helper = DatasetHelper(logger)
    df = helper.load_dataset_from_csv(f'{config.DROID_PATH}')
    df = helper.sample_dataset(df, config.SAMPLING_REQUIREMENTS)

    # Split dataset
    train_df, val_df, _ = helper.split_dataset(df)
    helper.log_dataset_splits_table(df, train_df, val_df)

    # Initialize model
    model = model_class(logger)

    # Train model
    model.train(
        train_df,
        val_df,
        config.NUM_TRAIN_EPOCHS,
        config.TRAIN_BATCH_SIZE
    )

    # Save model
    model.save(dir_name=model_name)

    # Log runtime
    end_time = time.time()
    seconds = end_time - start_time
    logger.info(f"Runtime: {seconds:.2f} seconds ({seconds / 60:.2f} minutes)")
    logger.info("")


def main():
    """Main function with argument parsing."""
    # Create argument parser
    parser = argparse.ArgumentParser(
        description='Train embedding-based classifier for AI code detection'
    )

    # Classifier type argument
    parser.add_argument(
        '--classifier',
        type=str,
        choices=['simple', 'xgboost'],
        default='simple',
        help='Type of classifier to use (simple or xgboost)'
    )

    # Parse arguments
    args = parser.parse_args()

    # Select model class
    model_classes = {
        'simple': EmbeddingModel,
        'xgboost': XGBoostEmbeddingModel
    }
    model_class = model_classes[args.classifier]

    # Train the model
    train_embedding_model(model_class)


if __name__ == "__main__":
    main()
