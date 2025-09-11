from src.models.codebert.base import CodeBertModel
from src.models.ast.base import AstModel
from src.models.ensemble.base import EnsembleModel
from src.models.helper import train_and_evaluate_model
from src.config import CONFIG
from src.utils.logger import get_logger


# Run configurations with ensemble model
for config in CONFIG:

    logger = get_logger('temp', './outputs/ensemble_model/z_temp.log')

    model_kwargs = {
        'transformer_model': CodeBertModel(
            logger,
            num_train_epochs=3,
            batch_size=32,
        ),
        'ast_model': AstModel(
            logger,
            use_scaler=True,
        ),
        'transformer_weight': 0.6,
        'ast_weight': 0.4,
    }

    train_and_evaluate_model(
        model_class=EnsembleModel,
        model_kwargs=model_kwargs,
        dataset_kwargs=config['dataset_config'],
        log_file_name=config['log_file_suffix']
    )
