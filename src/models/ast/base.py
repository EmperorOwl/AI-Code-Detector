from logging import Logger

import torch

import numpy as np
from transformers import T5EncoderModel, RobertaTokenizer
from sklearn.linear_model import LogisticRegression

from src.pre_processing.sample import Sample
from src.pre_processing.ast import get_ast_representation
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
        self.logger.info(
            f"CodeT5+ Embedding model initialized using device: {self.device}\n"
            f"Logistic Regression Classifier initialized using device: cpu\n"
        )

    def _get_code_embedding(self, sample: Sample):
        ast_str = get_ast_representation(sample.code, sample.language)

        inputs = self.tokenizer(
            ast_str,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)

        with torch.no_grad():
            outputs = self.embedding_model(**inputs)
            # Use mean pooling of the last hidden states
            embeddings = outputs.last_hidden_state.mean(dim=1)

        return embeddings.cpu().numpy().flatten()

    def train(self,
              train_samples: list[Sample],
              validation_samples: list[Sample]):
        # Prepare training dataset
        x_train = []
        y_train = []
        for sample in train_samples:
            x_train.append(self._get_code_embedding(sample))
            y_train.append(sample.label)

        x_train = np.array(x_train)
        y_train = np.array(y_train)

        # Train the classifier
        self.logger.info("Training the classifier...")
        self.classifier.fit(x_train, y_train)
        iterations = self.classifier.n_iter_[0]
        self.logger.info(
            f"Iterations Performed: {iterations}/{self.max_iterations}\n"
        )

        # Prepare validation samples
        x_val = []
        y_val = []
        for sample in validation_samples:
            x_val.append(self._get_code_embedding(sample))
            y_val.append(sample.label)

        # Evaluate the classifier
        y_pred = self.classifier.predict(x_val)

        # Save results
        log_results(y_val, y_pred, self.logger)

    def predict(self, test_samples: list[Sample]):
        # Prepare test dataset
        x_test = []
        y_test = []
        for sample in test_samples:
            x_test.append(self._get_code_embedding(sample))
            y_test.append(sample.label)

        # Make predictions
        y_pred = self.classifier.predict(x_test)

        # Save results
        log_results(y_test, y_pred, self.logger)
