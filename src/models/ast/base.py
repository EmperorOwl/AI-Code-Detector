from logging import Logger

import torch

import numpy as np
from transformers import T5EncoderModel, RobertaTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.pre_processing.sample import Sample
from src.pre_processing.ast import (get_ast_representation,
                                    extract_structural_features)
from src.utils.results import log_results


class AstModel:
    """ AST-based model using Logistic Regression classifier with Code T5+ 
    embeddings
    """
    MODEL_NAME = "Salesforce/codet5p-110m-embedding"
    SAVED_MODEL_PATH = "./saved/ast_model"

    def __init__(self,
                 use_scaler: bool,
                 add_structural_features: bool,
                 max_iterations: int,
                 logger: Logger) -> None:
        self.use_scaler = use_scaler
        self.add_structural_features = add_structural_features
        self.max_iterations = max_iterations

        # Set up tokenizer, embedding model and classifier
        self.tokenizer = RobertaTokenizer.from_pretrained(self.MODEL_NAME)
        self.embedding_model = T5EncoderModel.from_pretrained(self.MODEL_NAME)
        self.classifier = LogisticRegression(
            max_iter=self.max_iterations,
            random_state=42,
            class_weight='balanced',
            # C=0.1,  # Regularization to prevent overfitting
            # solver='liblinear'  # Better for small datasets
        )
        self.scaler = StandardScaler()

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
        ast_repr = get_ast_representation(sample.code, sample.language)

        inputs = self.tokenizer(
            ast_repr,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,  # Increased from 512 for better coverage
            add_special_tokens=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.embedding_model(**inputs)
            # Use mean pooling of the last hidden states
            embeddings = outputs.last_hidden_state.mean(dim=1)

        embeddings = embeddings.cpu().numpy().flatten()

        if self.add_structural_features:
            structural_features = extract_structural_features(ast_repr)
            # Combine embeddings with structural features
            embeddings = np.concatenate((embeddings, structural_features))

        return embeddings

    def train(self,
              train_samples: list[Sample],
              validation_samples: list[Sample]):
        # Prepare training dataset
        self.logger.info(
            f"Generating AST code embeddings (add_structural_features: "
            f"{str(self.add_structural_features).lower()})...\n"
        )
        x_train = []
        y_train = []
        for sample in train_samples:
            x_train.append(self._get_code_embedding(sample))
            y_train.append(sample.label)

        x_train = np.array(x_train)
        y_train = np.array(y_train)

        # Scaler
        if self.use_scaler:
            self.logger.info("Scaling code embeddings...\n")
            x_train = self.scaler.fit_transform(x_train)

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

        # Scaler
        if self.use_scaler:
            x_val = self.scaler.transform(x_val)

        # Evaluate the classifier
        self.logger.info("Validating the classifier...")
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

        # Scaler
        if self.use_scaler:
            x_test = self.scaler.transform(x_test)

        # Make predictions
        self.logger.info("Evaluating AST model...")
        y_pred = self.classifier.predict(x_test)

        # Save results
        log_results(y_test, y_pred, self.logger)
