import os
import ast
import pickle

import torch
import pandas as pd
import numpy as np
from transformers import logging
from transformers import T5EncoderModel, RobertaTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

from src.config import Config, AstModelConfig
from src.pre_processing.ast_node import AstNode
from src.pre_processing.code_dataset import CodeDataset


class AstModel:
    """ AST-based model using Logistic Regression with Code T5+ embeddings """

    def __init__(self,
                 train_samples: list,
                 test_samples: list,
                 use_saved: bool = False):
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

    def parse_python_ast(self, code: str) -> Optional[ASTNode]:
        """Parse Python code into AST representation."""
        try:
            parsed = ast.parse(code)
            return self._convert_ast_node(parsed)
        except (SyntaxError, ValueError) as e:
            # Return None for code with syntax errors
            return None

    def parse_java_ast(self, code: str) -> Optional[ASTNode]:
        """Parse Java code into simplified AST representation.

        Note: This is a simplified parser since full Java parsing requires
        additional dependencies. In a production system, you would use
        a proper Java parser like javalang or tree-sitter.
        """
        # Simplified Java parsing - extract basic structural elements
        lines = code.split('\n')
        tokens = []

        for line in lines:
            line = line.strip()
            if not line or line.startswith('//') or line.startswith('/*'):
                continue

            # Extract basic Java constructs
            if 'class ' in line:
                tokens.append('ClassDeclaration')
            elif 'interface ' in line:
                tokens.append('InterfaceDeclaration')
            elif 'public ' in line or 'private ' in line or 'protected ' in line:
                if '(' in line and ')' in line:
                    tokens.append('MethodDeclaration')
                else:
                    tokens.append('FieldDeclaration')
            elif 'if(' in line or 'if (' in line:
                tokens.append('IfStatement')
            elif 'for(' in line or 'for (' in line:
                tokens.append('ForStatement')
            elif 'while(' in line or 'while (' in line:
                tokens.append('WhileStatement')
            elif 'try' in line:
                tokens.append('TryStatement')
            elif 'catch' in line:
                tokens.append('CatchClause')
            elif '=' in line and ';' in line:
                tokens.append('Assignment')

        if not tokens:
            return None

        # Create a simplified AST structure
        root = ASTNode('CompilationUnit')
        for token in tokens:
            root.children.append(ASTNode(token))

        return root

    def _convert_ast_node(self, node) -> ASTNode:
        """Convert Python ast node to our ASTNode representation."""
        node_type = type(node).__name__
        children = []

        for child in ast.iter_child_nodes(node):
            children.append(self._convert_ast_node(child))

        return ASTNode(node_type, children)

    def get_ast_representation(self, code: str, language: str) -> Optional[str]:
        """Get AST string representation for a code snippet."""
        if language == 'python':
            ast_node = self.parse_python_ast(code)
        elif language == 'java':
            ast_node = self.parse_java_ast(code)
        else:
            raise ValueError(f"Unsupported language: {language}")

        if ast_node is None:
            return None

        return ast_node.to_string()

    def get_code_embedding(self, ast_representation: str) -> np.ndarray:
        """Get Code T5+ embedding for AST representation."""
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

    def prepare(self):
        """Prepares the dataset for training."""
        print(f"{'Dataset'.ljust(25)}"
              f"{'Language'.ljust(15)}"
              f"{'Total'.ljust(10)}"
              f"{'Valid AST'.ljust(15)}"
              f"{'Train'.ljust(10)}"
              f"{'Test'.ljust(10)}")
        print("-" * 85)

        # Process each dataset and collect valid samples
        all_samples = []
        for language in DATASETS:
            for dataset_name, dataset_path in DATASETS[language]:
                # Load samples from the dataset
                path = os.path.join(DATASET_PATH, language, dataset_path)
                if path.endswith('.csv'):
                    samples = load_samples_from_csv(path)
                else:
                    samples = load_samples_from_dir(path)

                # Filter samples that can be parsed into AST
                valid_samples = []
                for code, label in samples:
                    ast_repr = self.get_ast_representation(code, language)
                    if ast_repr is not None:
                        valid_samples.append((code, label, ast_repr, language))

                all_samples.extend(valid_samples)

                # Split the valid samples for reporting
                if valid_samples:
                    train_split, test_split = train_test_split(
                        valid_samples,
                        test_size=TEST_SIZE,
                        random_state=RANDOM_STATE
                    )
                else:
                    train_split, test_split = [], []

                print(f"{dataset_name.ljust(25)}"
                      f"{language.ljust(15)}"
                      f"{str(len(samples)).ljust(10)}"
                      f"{str(len(valid_samples)).ljust(15)}"
                      f"{str(len(train_split)).ljust(10)}"
                      f"{str(len(test_split)).ljust(10)}")

        # Split all samples into train and test
        self.train_samples, self.test_samples = train_test_split(
            all_samples,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE
        )

        print("-" * 85)
        print(f"{'Total'.ljust(40)}"
              f"{str(len(all_samples)).ljust(15)}"
              f"{str(len(self.train_samples)).ljust(10)}"
              f"{str(len(self.test_samples)).ljust(10)}")

    def train(self):
        if not self.classifier or not self.train_dataset:
            raise ValueError("Model not initialized - call setup() first")

        X_train = []
        y_train = []
        for i, (code, label, ast_repr, language) in enumerate(self.train_samples):
            embedding = self.get_code_embedding(ast_repr)
            X_train.append(embedding)
            y_train.append(label)

        X_train = np.array(X_train)
        y_train = np.array(y_train)

        print("Training logistic regression classifier...")
        self.classifier.fit(X_train, y_train)
        print("Training completed!")

    def evaluate(self):
        if not self.classifier or not self.test_dataset:
            raise ValueError("Model not trained - call setup() first")

        print(f"TEST_SIZE: {int(Config.TEST_SIZE * 100)}%")

        X_test = []
        y_test = []
        for i, (code, label, ast_repr, language) in enumerate(self.test_samples):
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

        print(f"Model saved to {path}")

    def classify_code(self, code_snippet: str, language: str) -> float:
        if not self.classifier:
            raise ValueError("Model not trained - call train() first")

        # Get AST representation
        ast_repr = self.get_ast_representation(code_snippet, language)
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
