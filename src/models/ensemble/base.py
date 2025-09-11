from logging import Logger

import numpy as np

from src.pre_processing.sample import Sample
from src.utils.results import log_results


class EnsembleModel:
    """ 
    Ensemble model that combines a transformer model and an AST model using 
    weighted averaging.
    """
    LOG_DIR = './outputs/ensemble_model'

    def __init__(self,
                 logger: Logger,
                 transformer_model,
                 ast_model,
                 transformer_weight: float = 0.6,
                 ast_weight: float = 0.4):

        self.logger = logger
        self.transformer_model = transformer_model
        self.ast_model = ast_model
        self.transformer_weight = transformer_weight
        self.ast_weight = ast_weight

        # Validate weights
        if abs(transformer_weight + ast_weight - 1.0) > 1e-6:
            raise ValueError("Weights must sum to 1.0")

        self.logger.info(
            f"Ensemble Model initialized with weighted averaging "
            f"(transformer weight: {transformer_weight}, "
            f"AST weight: {ast_weight})\n"
        )

    def train(self,
              train_samples: list[Sample],
              validation_samples: list[Sample]):
        self.logger.info("Training Ensemble model...")
        self.transformer_model.train(train_samples, validation_samples)
        self.ast_model.train(train_samples, validation_samples)

    def predict(self, test_samples: list[Sample]):
        self.logger.info("Evaluating Ensemble model...")

        # Get predictions from both models
        _, transformer_probs = self.transformer_model.predict(
            test_samples
        )
        _, ast_probs = self.ast_model.predict(test_samples)

        # Convert to numpy arrays
        transformer_probs = np.array(transformer_probs)
        ast_probs = np.array(ast_probs)

        # Weighted average of probabilities
        weighted_probs = (transformer_probs * self.transformer_weight +
                          ast_probs * self.ast_weight)
        ensemble_preds = np.argmax(weighted_probs, axis=1)

        # Log results
        y_true = [sample.label for sample in test_samples]
        log_results(y_true, ensemble_preds, self.logger)

        return ensemble_preds, weighted_probs
