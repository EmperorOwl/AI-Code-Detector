from logging import Logger

import torch

import numpy as np
from transformers import T5EncoderModel, RobertaTokenizer
from sklearn.linear_model import LogisticRegression

from src.pre_processing.code_dataset import CodeDataset, Sample
from src.utils.results import log_results


class AstModel:
    """ AST-based model using Logistic Regression classifier with Code T5+ 
    embeddings
    """
    MODEL_NAME = "Salesforce/codet5p-110m-embedding"
    SAVED_MODEL_PATH = "./saved/ast_model"

    def __init__(self,
                 max_iterations: int,
                 logger: Logger) -> None:
        self.max_iterations = max_iterations

        # Set up tokenizer, embedding model and classifier
        self.tokenizer = RobertaTokenizer.from_pretrained(self.MODEL_NAME)
        self.embedding_model = T5EncoderModel.from_pretrained(self.MODEL_NAME)
        self.classifier = LogisticRegression(
            max_iter=self.max_iterations,
            random_state=42,
            class_weight='balanced'
        )

        # Set up device
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.embedding_model.to(self.device)  # type: ignore

        # Set up logger
        self.logger = logger

    def train(self,
              train_samples: list[Sample],
              validation_samples: list[Sample]):
        train_dataset = AstDataset(train_samples)
        validation_dataset = AstDataset(validation_samples)

        x_train = []
        y_train = []
        for embedding, label in self.train_dataset:
            x_train.append(embedding)
            y_train.append(label)

        x_train = np.array(x_train)
        y_train = np.array(y_train)
        self.classifier.fit(x_train, y_train)

        iterations = self.classifier.n_iter_[0]
        self.logger.info(
            f"Iterations Performed: {iterations}/{self.max_iterations}"
        )
        self.logger.info("")

        x_val = []
        y_val = []
        for embedding, label in validation_dataset:
            x_val.append(embedding)
            y_val.append(label)

        self.classifier.predict(x_val)
        log_results(x_val, y_val, self.logger)

    def predict(self, test_samples: list[Sample]):
        pass
