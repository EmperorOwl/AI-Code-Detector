import time

# from src.models.ast.base import AstModel
from src.pre_processing.prepare import split_datasets
from src.utils.logger import get_logger

start_time = time.time()

# Create logger
logger = get_logger('ast_model', 'outputs/ast_model/z_temp.log')

config = {
    # dataset: [val_split, test_split]
    'gptsniffer': [0, 1],
    'humaneval': [0.2, 0],
    'mbpp': [0.2, 0],
    'codenet': [0, 1]
}

# Load datasets
train, validation, test = split_datasets(logger, config=config)

train, validation, test = split_datasets(
    logger, 
    test_size=0.1, 
    validation_size=0.1,
    language_filter='python'
)

# # Initialize model
# model = AstModel(
#     use_scaler=False,
#     add_structural_features=False,
#     max_iterations=1000,
#     logger=logger
# )

# # Train model
# model.train(train, validation)

# # Evaluate model
# model.predict(test)

# # Output runtime
# end_time = time.time()
# seconds = end_time - start_time
# logger.info(f"Runtime: {seconds:.2f} seconds ({seconds / 60:.2f} minutes)")
