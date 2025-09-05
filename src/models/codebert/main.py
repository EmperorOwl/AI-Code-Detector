from src.models.codebert.base import CodeBertModel
from src.models.helper import train_and_evaluate_model
from src.config import CONFIG

# Run configurations with base model
for config in CONFIG:
    model_kwargs = {
        'epochs': 3,
        'batch_size': 16
    }

    train_and_evaluate_model(
        model_class=CodeBertModel,
        model_kwargs=model_kwargs,
        dataset_kwargs=config['dataset_config'],
        log_file_name='base_' + config['log_file_suffix']
    )
