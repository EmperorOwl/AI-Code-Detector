import os
from logging import Logger

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import joblib

from src.utils.results import log_results
from src.utils import config


class SVMEmbeddingModel:
    """
    SVM classifier using UniXcoder embeddings for AI code detection.
    Good balance between simplicity and performance on high-dimensional data.
    """
    MODEL_NAME = "SVM-Embedding"
    PRETRAINED_MODEL_NAME = "microsoft/unixcoder-base"
    MAX_LENGTH = 1024

    def __init__(self,
                 logger: Logger,
                 load_from_saved_path: str | None = None) -> None:
        """
        Initialize the SVM embedding model.

        Args:
            logger (Logger): Logger instance for logging
            load_from_saved_path (str | None): Path to load saved classifier
        """
        self.logger = logger

        # Model configuration
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )

        # Initialize frozen UniXcoder for embedding extraction
        self.logger.info("Loading UniXcoder for embedding extraction...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.PRETRAINED_MODEL_NAME
        )
        self.encoder = AutoModel.from_pretrained(self.PRETRAINED_MODEL_NAME)
        
        # Freeze the encoder - we only use it for feature extraction
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        self.encoder.to(self.device)
        self.encoder.eval()  # Always in eval mode

        # Initialize SVM classifier with RBF kernel
        self.classifier = SVC(
            kernel='rbf',           # RBF kernel for non-linear patterns
            C=1.0,                  # Regularization parameter
            gamma='scale',          # Kernel coefficient (auto-scaled)
            probability=True,       # Enable probability estimates for confidence
            random_state=42,
            class_weight='balanced' # Handle class imbalance if present
        )
        
        # Scaler for normalizing embeddings (important for SVM)
        self.scaler = StandardScaler()

        if load_from_saved_path:
            self._load_saved_classifier(load_from_saved_path)

        self.logger.info(
            f"{self.MODEL_NAME} initialized with frozen UniXcoder "
            f"(device: {self.device})\n"
        )

    def _load_saved_classifier(self, load_from_saved_path: str):
        """Load a saved SVM classifier and scaler."""
        path = f"{config.TRAINING_DIR}/{load_from_saved_path}"
        self.classifier = joblib.load(f"{path}/svm_classifier.pkl")
        self.scaler = joblib.load(f"{path}/scaler.pkl")
        self.logger.info(f"Existing SVM classifier loaded from {path}")

    def extract_embeddings(self, code_samples: list[str]) -> np.ndarray:
        """
        Extract embeddings from code samples using frozen UniXcoder.
        
        Args:
            code_samples (list[str]): List of code samples
            
        Returns:
            np.ndarray: Embeddings array of shape (batch_size, hidden_size)
        """
        embeddings = []
        
        with torch.no_grad():
            for code in tqdm(code_samples, desc="Extracting embeddings"):
                # Tokenize
                inputs = self.tokenizer(
                    code,
                    padding='max_length',
                    truncation=True,
                    max_length=self.MAX_LENGTH,
                    return_tensors='pt'
                ).to(self.device)
                
                # Get embeddings
                outputs = self.encoder(**inputs)
                # Use mean pooling over sequence dimension
                embedding = outputs.last_hidden_state.mean(dim=1).squeeze()
                embeddings.append(embedding.cpu().numpy())
        
        return np.array(embeddings)

    def train(self,
              train_df: pd.DataFrame,
              val_df: pd.DataFrame,
              num_train_epochs: int,  # Not used for SVM, kept for interface compatibility
              batch_size: int):       # Not used for SVM, kept for interface compatibility
        """
        Train the SVM classifier on embeddings extracted from UniXcoder.

        Args:
            train_df (pd.DataFrame): Training dataset
            val_df (pd.DataFrame): Validation dataset
            num_train_epochs (int): Not used (SVM doesn't use epochs)
            batch_size (int): Not used (SVM processes all data at once)
        """
        self.logger.info(f"Training {self.MODEL_NAME} classifier\n")

        # Extract embeddings for training and validation
        self.logger.info("Extracting training embeddings...")
        X_train = self.extract_embeddings(train_df['Code'].tolist())
        y_train = train_df['Label'].values

        self.logger.info("Extracting validation embeddings...")
        X_val = self.extract_embeddings(val_df['Code'].tolist())
        y_val = val_df['Label'].values

        # Scale the embeddings (crucial for SVM performance)
        self.logger.info("Scaling embeddings...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)

        # Train SVM
        self.logger.info("Training SVM classifier...")
        self.classifier.fit(X_train_scaled, y_train)

        # Validation predictions
        val_pred = self.classifier.predict(X_val_scaled)
        val_proba = self.classifier.predict_proba(X_val_scaled)
        val_acc = accuracy_score(y_val, val_pred)

        # Log training results
        self.logger.info(f"\nSVM Training Complete:")
        self.logger.info(f"  - Kernel: {self.classifier.kernel}")
        self.logger.info(f"  - C parameter: {self.classifier.C}")
        self.logger.info(f"  - Gamma: {self.classifier.gamma}")
        self.logger.info(f"  - Support vectors: {self.classifier.n_support_}")
        self.logger.info(f"  - Validation Accuracy: {val_acc:.4f}")

        # Log class distribution in support vectors
        total_sv = sum(self.classifier.n_support_)
        self.logger.info(f"  - Total support vectors: {total_sv} "
                        f"({100 * total_sv / len(X_train):.1f}% of training data)")
        
        self.logger.info("")

    def predict(self, test_df: pd.DataFrame, batch_size: int) -> pd.DataFrame:
        """
        Make predictions on test samples.

        Args:
            test_df (pd.DataFrame): Test dataset
            batch_size (int): Not used (kept for interface compatibility)

        Returns:
            pd.DataFrame: 
                DataFrame with original columns plus 'Predicted_Label' 
                and 'Confidence' columns
        """
        self.logger.info(f"Evaluating {self.MODEL_NAME} model")

        # Extract embeddings for test data
        self.logger.info("Extracting test embeddings...")
        X_test = self.extract_embeddings(test_df['Code'].tolist())
        y_test = test_df['Label'].values

        # Scale embeddings using fitted scaler
        X_test_scaled = self.scaler.transform(X_test)

        # Make predictions
        predictions = self.classifier.predict(X_test_scaled)
        probabilities = self.classifier.predict_proba(X_test_scaled)

        # Calculate confidence as the probability of the predicted class
        confidence_scores = np.max(probabilities, axis=1)

        # Log results
        self.logger.info("")
        log_results(y_test, predictions, self.logger)

        # Create output DataFrame with predictions and confidence
        output_df = test_df.copy()
        output_df['Predicted_Label'] = predictions
        output_df['Confidence'] = (confidence_scores * 100).round(2)

        return output_df

    def save(self, dir_name: str):
        """ 
        Save the trained SVM classifier and scaler
        """
        path = f"{config.TRAINING_DIR}/{dir_name}"
        os.makedirs(path, exist_ok=True)
        
        # Save SVM model and scaler
        joblib.dump(self.classifier, f"{path}/svm_classifier.pkl")
        joblib.dump(self.scaler, f"{path}/scaler.pkl")
        
        # Save config
        config_dict = {
            'model_name': self.MODEL_NAME,
            'pretrained_model_name': self.PRETRAINED_MODEL_NAME,
            'max_length': self.MAX_LENGTH,
            'encoder_hidden_size': self.encoder.config.hidden_size,
            'svm_params': self.classifier.get_params()
        }
        
        import json
        with open(f"{path}/config.json", 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        self.logger.info(f"{self.MODEL_NAME} classifier saved to {path}")
        self.logger.info("")
