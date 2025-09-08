from src.models.codebert.base import CodeBertModel
from src.models.codebert.base import UniXcoderModel
from src.models.helper import train_and_evaluate_model
from src.config import CONFIG

# Run configurations with base model
for config in CONFIG:
    model_kwargs = {
        'num_train_epochs': 3,
        'batch_size': 16
    }

    train_and_evaluate_model(
        model_class=CodeBertModel,
        model_kwargs=model_kwargs,
        dataset_kwargs=config['dataset_config'],
        log_file_name='base_' + config['log_file_suffix']
    )


# Run configurations with candidate model
for config in CONFIG:
    model_kwargs = {
        'num_train_epochs': 3,
        'batch_size': 16,
        'use_ast': True
    }

    train_and_evaluate_model(
        model_class=UniXcoderModel,
        model_kwargs=model_kwargs,
        dataset_kwargs=config['dataset_config'],
        log_file_name=config['log_file_suffix']
    )
