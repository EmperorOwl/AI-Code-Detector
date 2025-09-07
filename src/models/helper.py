import time

from src.models.codebert.base import BaseTransformerModel
from src.models.ast.base import AstModel
from src.pre_processing.dataset import split_datasets
from src.utils.logger import get_logger


def train_and_evaluate_model(model_class: type[BaseTransformerModel | AstModel],
                             model_kwargs: dict,
                             dataset_kwargs: dict,
                             log_file_name: str):
    model_name = type(model_class).__name__

    # Start timer
    start_time = time.time()

    # Create logger
    log_file_path = model_class.LOG_DIR + "/" + log_file_name
    logger = get_logger(model_name, log_file_path)

    # Log start info
    logger.info(
        f"Timestamp: {time.strftime('%d/%m/%Y %I:%M %p',
                                    time.localtime(start_time))}"
    )
    logger.info(f"Log: {log_file_path}\n")

    # Load and split datasets
    train, validation, test = split_datasets(logger, **dataset_kwargs)

    # Initialise model
    model = model_class(logger, **model_kwargs)

    # Train model
    model.train(train, validation)

    # Test model
    model.predict(test)

    # Log runtime
    end_time = time.time()
    seconds = end_time - start_time
    logger.info(f"Runtime: {seconds:.2f} seconds ({seconds / 60:.2f} minutes)")
