from src.models.ast.base import AstModel
from src.models.helper import train_and_evaluate_model

from src.config import CONFIG

# Run configurations with base model
for config in CONFIG:
    model_kwargs = {}

    train_and_evaluate_model(
        model_class=AstModel,
        model_kwargs=model_kwargs,
        dataset_kwargs=config['dataset_config'],
        log_file_name='base_' + config['log_file_suffix']
    )


# Run configurations with improved model
for config in CONFIG:
    model_kwargs = {
        'use_scaler': True,
        'add_structural_features': True
    }

    train_and_evaluate_model(
        model_class=AstModel,
        model_kwargs=model_kwargs,
        dataset_kwargs=config['dataset_config'],
        log_file_name='improved_' + config['log_file_suffix']
    )
