import os
import pickle
import torch
import numpy as np
import pandas as pd
from transformers import logging
from transformers import T5EncoderModel, RobertaTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

from src.config import Config, AstModelConfig
from src.pre_processing.ast_node import AstParser
from src.pre_processing.ast_dataset import AstDataset


class AstModel:
    """ AST-based model using Logistic Regression with Code T5+ embeddings """

    def __init__(self,
                 use_saved: bool = False,
                 train_samples: list | None = None,
                 test_samples: list | None = None):
        self.train_samples = train_samples
        self.test_samples = test_samples
        # Model
        self.tokenizer = None
        self.embedding_model = None
        self.classifier = None
        self.device = None
        # Datasets
        self.train_dataset = None
        self.test_dataset = None
        # Call setup to initialize
        self.setup(use_saved)

    def setup(self, use_saved: bool):
        logging.set_verbosity_error()
        # Load saved model if it exists
        if use_saved:
            if (not os.path.exists(AstModelConfig.SAVED_MODEL_PATH)
                    or not os.listdir(AstModelConfig.SAVED_MODEL_PATH)):
                raise FileNotFoundError("Saved model not found")

            with open(os.path.join(AstModelConfig.SAVED_MODEL_PATH,
                                   'classifier.pkl'), 'rb') as f:
                self.classifier = pickle.load(f)
            print("Loaded classifier from saved model")
            return
        else:
            self.classifier = LogisticRegression(
                max_iter=1000,
                random_state=Config.RANDOM_STATE,
                class_weight='balanced'
            )
            print("Initialized new Logistic Regression classifier")

        # Load Code T5+ embedding model
        self.tokenizer = RobertaTokenizer.from_pretrained(
            AstModelConfig.MODEL_NAME
        )
        self.embedding_model = T5EncoderModel.from_pretrained(
            AstModelConfig.MODEL_NAME
        )

        # Use GPU if available
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.embedding_model.to(self.device)  # type: ignore
        print(f"Using device: {self.device}")

        # Prepare datasets
        if not self.train_samples or not self.test_samples:
            raise ValueError("Train and test samples must be provided")
        self.train_dataset = AstDataset(self.train_samples)
        self.test_dataset = AstDataset(self.test_samples)

    def get_code_embedding(self, ast_representation: str) -> np.ndarray:
        if not self.tokenizer or not self.embedding_model:
            raise ValueError("Model not initialized - call setup() first")

        inputs = self.tokenizer(
            ast_representation,
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

    def train(self):
        if not self.classifier or not self.train_dataset:
            raise ValueError("Model not initialized - call setup() first")

        print("Training AST Model...")

        X_train = []
        y_train = []
        for code, label, ast_repr, language in self.train_dataset:
            embedding = self.get_code_embedding(ast_repr)
            X_train.append(embedding)
            y_train.append(label)

        X_train = np.array(X_train)
        y_train = np.array(y_train)

        self.classifier.fit(X_train, y_train)
        print("\n")

    def evaluate(self):
        if not self.classifier or not self.test_dataset:
            raise ValueError("Model not trained - call setup() first")

        print("Evaluating AST Model...")

        X_test = []
        y_test = []
        for code, label, ast_repr, language in self.test_dataset:
            embedding = self.get_code_embedding(ast_repr)
            X_test.append(embedding)
            y_test.append(label)

        X_test = np.array(X_test)
        y_test = np.array(y_test)

        # Make predictions
        y_pred = self.classifier.predict(X_test)

        target_names = ['AI', 'Human']
        print(classification_report(y_test, y_pred, target_names=target_names))

        cm = confusion_matrix(y_test, y_pred)
        cm_df = pd.DataFrame(
            cm,
            index=['Actual AI', 'Actual Human'],
            columns=['Predicted AI', 'Predicted Human']
        )
        print(cm_df.to_string())

    def save(self):
        if not self.classifier:
            raise ValueError("Model not trained - call train() first")

        path = AstModelConfig.SAVED_MODEL_PATH
        os.makedirs(path, exist_ok=True)

        with open(os.path.join(path, 'classifier.pkl'), 'wb') as f:
            pickle.dump(self.classifier, f)

        print(f"AST Model saved to {path}")

    def classify_code(self, code_snippet: str, language: str) -> float:
        if not self.classifier:
            raise ValueError("Model not trained - call train() first")

        # Get AST representation
        ast_repr = AstParser.get_ast_representation(code_snippet, language)
        if ast_repr is None:
            raise ValueError(
                "Could not parse code into AST. Code may have syntax errors."
            )

        # Get embedding
        embedding = self.get_code_embedding(ast_repr)
        embedding = embedding.reshape(1, -1)

        # Get prediction probability
        probabilities = self.classifier.predict_proba(embedding)
        ai_probability = probabilities[0][0]  # Probability of class 0 (AI)

        return ai_probability * 100  # Return as percentage
