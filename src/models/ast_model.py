import os
import pickle
import torch
import numpy as np
import pandas as pd
from transformers import logging
from transformers import T5EncoderModel, RobertaTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

from src.config import Config
from src.pre_processing.ast_node import AstParser
from src.pre_processing.ast_dataset import AstDataset


class AstModel:
    """ AST-based model using Logistic Regression with Code T5+ embeddings """
    MODEL_NAME = "Salesforce/codet5p-110m-embedding"
    SAVED_MODEL_PATH = "./saved/ast_model"

    def __init__(self,
                 max_iterations: int,
                 use_enhanced_features: bool,
                 use_saved: bool = False,
                 train_samples: list | None = None,
                 test_samples: list | None = None) -> None:
        self.train_samples = train_samples
        self.test_samples = test_samples
        self.max_iterations = max_iterations
        self.use_enhanced_features = use_enhanced_features
        # Model
        self.tokenizer = None
        self.embedding_model = None
        self.classifier = None
        self.scaler = None  # For feature scaling
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
            if (not os.path.exists(AstModel.SAVED_MODEL_PATH)
                    or not os.listdir(AstModel.SAVED_MODEL_PATH)):
                raise FileNotFoundError("Saved model not found")

            with open(os.path.join(AstModel.SAVED_MODEL_PATH,
                                   'classifier.pkl'), 'rb') as f:
                self.classifier = pickle.load(f)

            # Load scaler if it exists
            scaler_path = os.path.join(AstModel.SAVED_MODEL_PATH, 'scaler.pkl')
            if os.path.exists(scaler_path):
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)

            print("Loaded classifier from saved model")
            return
        else:
            # Configure LogisticRegression with better settings for enhanced features
            if self.use_enhanced_features:
                # For enhanced features, use liblinear solver which handles high-dimensional data better
                self.classifier = LogisticRegression(
                    max_iter=self.max_iterations,
                    random_state=Config.RANDOM_STATE,
                    class_weight='balanced',
                    solver='liblinear',  # Better for high-dimensional sparse data
                    C=1.0  # Regularization strength
                )
                print(
                    "Initialized Logistic Regression with enhanced features configuration")
            else:
                # Standard configuration for basic features
                self.classifier = LogisticRegression(
                    max_iter=self.max_iterations,
                    random_state=Config.RANDOM_STATE,
                    class_weight='balanced'
                )
                print(
                    "Initialized Logistic Regression with basic features configuration")

            # Initialize scaler for feature scaling
            self.scaler = StandardScaler()

        # Load Code T5+ embedding model
        self.tokenizer = RobertaTokenizer.from_pretrained(AstModel.MODEL_NAME)
        self.embedding_model = T5EncoderModel.from_pretrained(
            AstModel.MODEL_NAME
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
        # print(f"Train dataset size: {len(self.train_dataset.samples)} samples")
        self.test_dataset = AstDataset(self.test_samples)
        # print(f"Test dataset size: {len(self.test_dataset.samples)} samples")

    def get_code_embedding(self, ast_representation: str) -> np.ndarray:
        """Get basic code embedding using mean pooling (original method)."""
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

    def get_enhanced_features(self, ast_representation: str) -> np.ndarray:
        """Extract enhanced features combining embeddings and structural features."""
        if not self.tokenizer or not self.embedding_model:
            raise ValueError("Model not initialized - call setup() first")

        # Get Code T5+ embeddings with multiple pooling strategies
        inputs = self.tokenizer(
            ast_representation,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)

        with torch.no_grad():
            outputs = self.embedding_model(**inputs)
            hidden_states = outputs.last_hidden_state

            # Multiple pooling strategies
            mean_pooling = hidden_states.mean(dim=1)  # Original method
            max_pooling = hidden_states.max(dim=1)[0]  # Max pooling
            # CLS token (first token) - though T5 doesn't have CLS, use first token
            cls_pooling = hidden_states[:, 0, :]

            # Combine different pooling strategies
            combined_embedding = torch.cat(
                [mean_pooling, max_pooling, cls_pooling], dim=1)

        # Extract structural features from AST
        structural_features = self.extract_structural_features(
            ast_representation)

        # Combine embedding and structural features
        embedding_features = combined_embedding.cpu().numpy().flatten()
        all_features = np.concatenate(
            [embedding_features, structural_features])

        return all_features

    def extract_structural_features(self, ast_repr: str) -> np.ndarray:
        """Extract structural features from AST representation."""
        features = []

        # 1. AST Depth estimation (count nested parentheses)
        max_depth = 0
        current_depth = 0
        for char in ast_repr:
            if char == '(':
                current_depth += 1
                max_depth = max(max_depth, current_depth)
            elif char == ')':
                current_depth -= 1
        features.append(max_depth)

        # 2. Node type counts - common programming constructs
        # Python-specific nodes
        python_nodes = [
            'function_definition', 'class_definition', 'if_statement',
            'for_statement', 'while_statement', 'assignment', 'call',
            'import_statement', 'import_from_statement', 'return_statement',
            'expression_statement', 'with_statement', 'try_statement',
            'lambda', 'list_comprehension', 'dictionary_comprehension'
        ]

        # Java-specific nodes
        java_nodes = [
            'method_declaration', 'class_declaration', 'variable_declarator',
            'if_statement', 'for_statement', 'while_statement', 'assignment_expression',
            'method_invocation', 'import_declaration', 'return_statement',
            'expression_statement', 'try_statement', 'catch_clause',
            'synchronized_statement', 'switch_statement'
        ]

        # Count occurrences of common node types
        all_nodes = list(set(python_nodes + java_nodes))
        for node_type in all_nodes:
            count = ast_repr.count(node_type)
            features.append(count)

        # 3. AST size and complexity metrics
        features.append(len(ast_repr))  # Total AST string length
        # Number of nodes (opening parentheses)
        features.append(ast_repr.count('('))
        features.append(ast_repr.count(' '))  # Rough token count (spaces)

        # 4. Identifier and literal patterns
        # Variable/function names
        features.append(ast_repr.count('identifier'))
        features.append(ast_repr.count('string'))  # String literals
        features.append(ast_repr.count('integer'))  # Integer literals
        features.append(ast_repr.count('float'))  # Float literals

        # 5. Control flow complexity indicators
        features.append(ast_repr.count('if_statement') +
                        ast_repr.count('elif_clause'))
        features.append(ast_repr.count('for_statement') +
                        ast_repr.count('while_statement'))
        features.append(ast_repr.count('try_statement') +
                        ast_repr.count('except_clause'))

        # 6. Code style indicators
        features.append(ast_repr.count('comment'))  # Comments
        features.append(ast_repr.count('pass_statement'))  # Empty statements

        return np.array(features, dtype=np.float32)

    def get_features(self, ast_representation: str) -> np.ndarray:
        if self.use_enhanced_features:
            return self.get_enhanced_features(ast_representation)
        else:
            return self.get_code_embedding(ast_representation)

    def train(self):
        if not self.classifier or not self.train_dataset:
            raise ValueError("Model not initialized - call setup() first")

        print("Preparing embeddings...")
        X_train = []
        y_train = []
        for code, label, ast_repr, language in self.train_dataset:
            features = self.get_features(ast_repr)
            X_train.append(features)
            y_train.append(label)

        X_train = np.array(X_train)
        y_train = np.array(y_train)

        print(f"Feature vector size: {X_train.shape[1]}")

        # Apply feature scaling
        if self.scaler is not None:
            print("Applying feature scaling...")
            X_train_scaled = self.scaler.fit_transform(X_train)
        else:
            X_train_scaled = X_train

        print("Training classifier...")
        self.classifier.fit(X_train_scaled, y_train)
        print(
            f"Iterations Performed: {self.classifier.n_iter_}/{self.max_iterations}"
        )
        print("\n")

    def evaluate(self):
        if not self.classifier or not self.test_dataset:
            raise ValueError("Model not trained - call setup() first")

        print("Evaluating AST Model...")

        X_test = []
        y_test = []
        for code, label, ast_repr, language in self.test_dataset:
            features = self.get_features(ast_repr)
            X_test.append(features)
            y_test.append(label)

        X_test = np.array(X_test)
        y_test = np.array(y_test)

        # Apply the same scaling as used in training
        if self.scaler is not None:
            X_test_scaled = self.scaler.transform(X_test)
        else:
            X_test_scaled = X_test

        # Make predictions
        y_pred = self.classifier.predict(X_test_scaled)

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

        path = AstModel.SAVED_MODEL_PATH
        os.makedirs(path, exist_ok=True)

        with open(os.path.join(path, 'classifier.pkl'), 'wb') as f:
            pickle.dump(self.classifier, f)

        # Save scaler if it exists
        if self.scaler is not None:
            with open(os.path.join(path, 'scaler.pkl'), 'wb') as f:
                pickle.dump(self.scaler, f)

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

        # Get features using the appropriate method
        features = self.get_features(ast_repr)
        features = features.reshape(1, -1)

        # Apply the same scaling as used in training
        if self.scaler is not None:
            features = self.scaler.transform(features)

        # Get prediction probability
        probabilities = self.classifier.predict_proba(features)
        ai_probability = probabilities[0][0]  # Probability of class 0 (AI)

        return ai_probability * 100  # Return as percentage
