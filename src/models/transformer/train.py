import time
import argparse


from src.dataset_processing.dataset_helper import DatasetHelper
from src.dataset_processing.dataset_tokenizer import DatasetTokenizer
from src.models.transformer import CodeBertModel, UniXcoderModel
from src.models.transformer.transformer_model import TransformerModel
from src.utils.logger import get_logger
from src.utils.config import TRAINING_DIR, CONFIG, TEST_CONFIG, DATASET_DIR


def train_model(model_class: type[TransformerModel],
                sampling_requirements: dict,
                num_train_epochs: int,
                batch_size: int) -> None:
    # Start timer
    start_time = time.time()

    # Create logger
    model_name = model_class.MODEL_NAME.lower()
    log_file_path = f"{TRAINING_DIR}/{model_name}.log"
    logger = get_logger(model_class.MODEL_NAME, log_file_path)

    # Log start info
    logger.info(
        f"Timestamp: {time.strftime('%d/%m/%Y %I:%M %p',
                                    time.localtime(start_time))}"
    )
    logger.info(f"Log: {log_file_path}\n")

    # Load and sample datasets
    helper = DatasetHelper(logger)
    df = helper.load_dataset_from_csv(f'{DATASET_DIR}/droid_dataset.csv')
    df = helper.sample_dataset(df, sampling_requirements)

    # Tokenize dataset
    tokenizer = DatasetTokenizer(logger, model_class)
    df = tokenizer.tokenize_code_samples(df)
    tokenizer.analyze_tokenization(df)

    # Split dataset
    train_df, val_df, _ = helper.split_dataset(df)
    helper.log_dataset_splits_table(df, train_df, val_df)

    # Initialise model
    model = model_class(logger)

    # Train model
    model.train(train_df,
                val_df=val_df,
                num_train_epochs=num_train_epochs,
                batch_size=batch_size
                )

    # Save model
    model.save(dir_name=model_name)

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

    # Configuration selection argument
    parser.add_argument(
        '--test',
        action='store_true',
        help='Use test configuration (default: use full configuration)'
    )

    # Parse arguments
    args = parser.parse_args()

    # Select model class
    model_classes = {
        'codebert': CodeBertModel,
        'unixcoder': UniXcoderModel
    }
    model_class = model_classes[args.model]

    # Select configuration based on --test flag
    config = TEST_CONFIG if args.test else CONFIG

    # Train the model
    train_model(
        model_class=model_class,
        sampling_requirements=config['SAMPLING_REQUIREMENTS'],
        num_train_epochs=config['NUM_TRAIN_EPOCHS'],
        batch_size=config['TRAIN_BATCH_SIZE']
    )


if __name__ == "__main__":
    main()
