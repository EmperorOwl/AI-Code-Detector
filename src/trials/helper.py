import time
import os

from src.dataset_processing.dataset_helper import DatasetHelper
from src.dataset_processing.dataset_tokenizer import DatasetTokenizer
from src.models.transformer.transformer_model import TransformerModel
from src.utils.analysis import save_predictions
from src.utils.logger import get_logger
from src.utils.config import OUTPUT_DIR, DATASET_DIR


def run_trial(trial_name: str,
              model_class: type[TransformerModel],
              sampling_requirements: dict,
              eval_batch_size: int) -> None:
    model_name = model_class.MODEL_NAME

    # Start timer
    start_time = time.time()

    # Create output directory
    trial_dir = f"{OUTPUT_DIR}/{trial_name}"
    os.makedirs(trial_dir, exist_ok=True)

    # Create logger
    log_file_path = f"{trial_dir}/{model_name.lower()}.log"
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
    train_df, val_df, test_df = helper.split_dataset(df)
    helper.log_dataset_splits_table(df, train_df, val_df, test_df)

    # Load saved model
    loaded_model = model_class(logger,
                               load_from_saved_path=model_name.lower())

    # Test model
    output_df = loaded_model.predict(
        test_df=test_df,
        batch_size=eval_batch_size
    )

    # Save predictions
    predictions_file_path = f"{trial_dir}/predictions.csv"
    save_predictions(logger, output_df, model_name, predictions_file_path)

    # Log runtime
    end_time = time.time()
    seconds = end_time - start_time
    logger.info(f"Runtime: {seconds:.2f} seconds ({seconds / 60:.2f} minutes)")
    logger.info("")
