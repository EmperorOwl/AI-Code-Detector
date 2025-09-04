import time

from src.models.codebert.base import CodeBertModel
from src.pre_processing.prepare import split_datasets
from src.utils.logger import get_logger

start_time = time.time()

# Create logger
logger = get_logger('codebert_model', 'outputs/codebert_model/z_temp.log')

# Load datasets
config = {
    'gptsniffer': {'test_size': 1},
    'humaneval': {'val_size': 0.2},
    'mbpp': {'val_size': 0.2},
    'codenet': {'test_size': 1, 'max_sample_count': 1480}
}
train, validation, test = split_datasets(logger, config=config)

# train, validation, test = split_datasets(
#     logger,
#     test_size=0.1,
#     validation_size=0.1,
#     language_filter='python'
# )

# Initialize model
model = CodeBertModel(
    num_train_epochs=3,
    batch_size=16,
    logger=logger
)

# Train model
model.train(train, validation)

# Evaluate model
model.predict(test)

# Output runtime
end_time = time.time()
seconds = end_time - start_time
logger.info(f"Runtime: {seconds:.2f} seconds ({seconds / 60:.2f} minutes)")
