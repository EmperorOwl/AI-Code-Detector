import time

from src.models.codebert.base import CodeBertModel
from src.pre_processing.prepare import load_samples, DATASETS
from src.utils.logger import get_logger

start_time = time.time()

# Create logger
logger = get_logger('codebert', 'outputs/codebert_model.log')

# Load datasets
# train, validation, test = load_samples([DATASETS[1]], 0.1, 0.1, logger)
train, validation, test = load_samples(DATASETS, 0.1, 0.1, logger)

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
