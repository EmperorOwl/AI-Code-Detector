import time
import argparse

from src.dataset_processing.dataset_helper import DatasetHelper
from src.models.classifiers.embedding_model import EmbeddingModel
from src.utils.logger import get_logger
from src.utils import config


def train_embedding_model() -> None:
    """
    Train the EmbeddingModel classifier.
    UniXcoder is always frozen - only the classifier is trained.
    """
    # Start timer
    start_time = time.time()

    # Configure
    model_name = "embedding"

    # Create logger
    log_file_path = f"{config.TRAINING_DIR}/{model_name}.log"
    logger = get_logger("EmbeddingModel", log_file_path)

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
    model = EmbeddingModel(logger)

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
        description='Train EmbeddingModel classifier for AI code detection'
    )

    # Parse arguments (no additional arguments needed)
    args = parser.parse_args()

    # Train the model
    train_embedding_model()


if __name__ == "__main__":
    main()
