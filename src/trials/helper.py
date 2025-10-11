import time
import os
from logging import Logger
from typing import Literal

import pandas as pd

from src.dataset_processing.dataset_helper import DatasetHelper
from src.dataset_processing.dataset_tokenizer import DatasetTokenizer
from src.models.transformer.transformer_model import TransformerModel
from src.utils.analysis import save_predictions
from src.utils.logger import get_logger
from src.utils.config import (OUTPUT_DIR,
                              CONFIG,
                              DatasetPaths)


def get_test_dataset(
    logger: Logger,
    model_class: type[TransformerModel],
    trial_name: str,
    is_test_run: bool = False
) -> pd.DataFrame:

    helper = DatasetHelper(logger)
    tokenizer = DatasetTokenizer(logger, model_class)

    if trial_name == 'same_sources':
        df = helper.load_dataset_from_csv(DatasetPaths.DROID)

        if is_test_run:
            df = helper.sample_dataset(df,
                                       CONFIG['dev']['SAMPLING_REQUIREMENTS'])

        train_df, val_df, test_df = helper.split_dataset(df)
        helper.log_dataset_splits_table(df, train_df, val_df, test_df)

    elif trial_name == 'independent_sources':
        droid_df = helper.load_dataset_from_csv(DatasetPaths.DROID)
        aig_df = helper.load_dataset_from_csv(DatasetPaths.AIG)
        sniffer_df = helper.load_dataset_from_csv(DatasetPaths.SNIFFER)

        if is_test_run:
            droid_df = helper.sample_dataset(
                droid_df,
                CONFIG['dev']['SAMPLING_REQUIREMENTS']
            )
            aig_df = helper.sample_dataset(aig_df, {
                ('Python', 'Gemini Flash'): 10,
                ('Python', 'Human'): 10,
            })
            sniffer_df = helper.sample_dataset(sniffer_df, {
                ('Java', 'ChatGPT'): 10,
                ('Java', 'Human'): 10,
            })

        df = pd.concat([droid_df, aig_df, sniffer_df], ignore_index=True)

        train_df, val_df, _ = helper.split_dataset(droid_df)
        test_df = pd.concat([aig_df, sniffer_df], ignore_index=True)
        helper.log_dataset_splits_table(df, train_df, val_df, test_df)

    else:
        raise ValueError(f"Invalid trial name: {trial_name}")

    test_df = tokenizer.tokenize_code_samples(test_df)
    tokenizer.analyze_tokenization(test_df)

    return test_df


def run_trial(trial_name: str,
              model_class: type[TransformerModel],
              mode: Literal['dev', 'prod'] = 'prod') -> None:
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

    # Get test dataset for this trial
    test_df = get_test_dataset(
        logger,
        model_class,
        trial_name,
        is_test_run=mode == 'dev'
    )

    # Load saved model
    loaded_model = model_class(logger,
                               load_from_saved_path=model_name.lower())

    output_df = loaded_model.predict(
        test_df=test_df,
        batch_size=CONFIG[mode]['EVAL_BATCH_SIZE']
    )

    # Save predictions
    predictions_file_path = f"{trial_dir}/predictions.csv"
    save_predictions(logger, output_df, model_name, predictions_file_path)

    # Log runtime
    end_time = time.time()
    seconds = end_time - start_time
    logger.info(f"Runtime: {seconds:.2f} seconds ({seconds / 60:.2f} minutes)")
    logger.info("")
