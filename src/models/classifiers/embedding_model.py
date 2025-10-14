import os
from logging import Logger

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm

from src.utils.results import log_results
from src.utils import config


class SimpleClassifier(nn.Module):
    """
    Simple neural network classifier for code embeddings.

    Architecture:
    - UniXcoder embeddings (768) -> hidden layers -> binary classification (2)
    
    Example: 768 -> 256 -> 128 -> 2
    - 768: UniXcoder's embedding dimension (fixed by pre-trained model)
    - 256: First compression layer (3x reduction, preserves most information)
    - 128: Second compression layer (2x reduction, creates bottleneck)
    - 2: Binary output (Human=0, AI=1)
    """

    def __init__(self,
                 input_size: int,
                 hidden_sizes: list[int] = [256, 128],
                 dropout_rate: float = 0.3,
                 num_labels: int = 2):
        super().__init__()

        # Build layers dynamically
        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_size = hidden_size

        # Output layer (no dropout after final layer)
        layers.append(nn.Linear(prev_size, num_labels))

        self.classifier = nn.Sequential(*layers)

    def forward(self, embeddings):
        return self.classifier(embeddings)


class EmbeddingModel:
    """
    UniXcoder based model for detecting AI-generated code.
    Uses frozen UniXcoder to extract embeddings, then trains a simple classifier.
    """
    MODEL_NAME = "Simple Embedding"
    PRETRAINED_MODEL_NAME = "microsoft/unixcoder-base"
    MAX_LENGTH = 1024

    def __init__(self,
                 logger: Logger,
                 load_from_saved_path: str | None = None) -> None:
        """
        Initialize the EmbeddingModel.

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

        # Initialize classifier
        encoder_hidden_size = self.encoder.config.hidden_size
        self.classifier = SimpleClassifier(encoder_hidden_size)
        self.classifier.to(self.device)

        if load_from_saved_path:
            self._load_saved_classifier(load_from_saved_path)

        self.logger.info(
            f"{self.MODEL_NAME} model initialized with frozen UniXcoder "
            f"(device: {self.device})\n"
        )

    def _load_saved_classifier(self, load_from_saved_path: str):
        """Load a saved classifier."""
        path = f"{config.TRAINING_DIR}/{load_from_saved_path}"
        self.classifier.load_state_dict(torch.load(
            f"{path}/classifier.pth",
            map_location=self.device
        ))
        self.logger.info(f"Existing classifier loaded from {path}")

    def extract_embeddings(self, code_samples: list[str]) -> torch.Tensor:
        """
        Extract embeddings from code samples using frozen UniXcoder.

        Args:
            code_samples (list[str]): List of code samples

        Returns:
            torch.Tensor: Embeddings tensor of shape (batch_size, hidden_size)
        """
        embeddings = []

        with torch.no_grad():
            for code in tqdm(code_samples, desc="Progress: "):
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
                embeddings.append(embedding.cpu())

        return torch.stack(embeddings)

    def train(self,
              train_df: pd.DataFrame,
              val_df: pd.DataFrame,
              num_train_epochs: int,
              batch_size: int):
        """
        Train the classifier on embeddings extracted from UniXcoder.

        Args:
            train_df (pd.DataFrame): Training dataset
            val_df (pd.DataFrame): Validation dataset
            num_train_epochs (int): Number of training epochs
            batch_size (int): Batch size for training and validation
        """
        self.logger.info(
            f"Training {self.MODEL_NAME} classifier ("
            f"epochs: {num_train_epochs}, "
            f"batch_size: {batch_size})\n"
        )

        # Extract embeddings for training and validation
        self.logger.info("Extracting training embeddings...")
        train_embeddings = self.extract_embeddings(train_df['Code'].tolist())
        train_labels = torch.tensor(train_df['Label'].values, dtype=torch.long)

        self.logger.info("Extracting validation embeddings...")
        val_embeddings = self.extract_embeddings(val_df['Code'].tolist())
        val_labels = torch.tensor(val_df['Label'].values, dtype=torch.long)

        # Create data loaders
        train_dataset = TensorDataset(train_embeddings, train_labels)
        val_dataset = TensorDataset(val_embeddings, val_labels)

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False)

        # Setup optimizer and loss function
        optimizer = torch.optim.Adam(self.classifier.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        # Training loop
        self.logger.info("")
        self.logger.info(f"{'Epoch'.ljust(10)}"
                         f"{'Train Loss'.ljust(15)}"
                         f"{'Val Loss'.ljust(15)}"
                         f"{'Val Acc'.ljust(15)}")
        self.logger.info("-" * 55)

        best_val_loss = float('inf')
        patience = 3
        patience_counter = 0

        for epoch in range(num_train_epochs):
            # Training phase
            self.classifier.train()
            train_loss = 0.0

            for embeddings, labels in train_loader:
                embeddings, labels = embeddings.to(
                    self.device), labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.classifier(embeddings)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            # Validation phase
            self.classifier.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for embeddings, labels in val_loader:
                    embeddings, labels = embeddings.to(
                        self.device), labels.to(self.device)

                    outputs = self.classifier(embeddings)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()

                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()

            # Calculate averages
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            val_acc = 100 * val_correct / val_total

            # Log progress
            self.logger.info(f"{epoch+1:>9}"
                             f"{train_loss:>14.4f}"
                             f"{val_loss:>14.4f}"
                             f"{val_acc:>13.2f}%")

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    self.logger.info(f"Early stopping at epoch {epoch+1}")
                    break

        self.logger.info("")

    def predict(self, test_df: pd.DataFrame, batch_size: int) -> pd.DataFrame:
        """
        Make predictions on test samples.

        Args:
            test_df (pd.DataFrame): Test dataset
            batch_size (int): Batch size for prediction

        Returns:
            pd.DataFrame: 
                DataFrame with original columns plus 'Predicted_Label' 
                and 'Confidence' columns
        """
        self.logger.info(
            f"Evaluating {self.MODEL_NAME} model ("
            f"batch_size: {batch_size})"
        )

        # Extract embeddings for test data
        self.logger.info("Extracting test embeddings...")
        test_embeddings = self.extract_embeddings(test_df['Code'].tolist())
        test_labels = torch.tensor(test_df['Label'].values, dtype=torch.long)

        # Create data loader
        test_dataset = TensorDataset(test_embeddings, test_labels)
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False)

        # Make predictions
        self.classifier.eval()
        all_predictions = []
        all_probabilities = []
        all_true_labels = []

        with torch.no_grad():
            for embeddings, labels in test_loader:
                embeddings = embeddings.to(self.device)

                outputs = self.classifier(embeddings)
                probabilities = torch.softmax(outputs, dim=-1)
                predictions = torch.argmax(outputs, dim=-1)

                all_predictions.extend(predictions.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                all_true_labels.extend(labels.numpy())

        # Convert to numpy arrays
        predictions = np.array(all_predictions)
        probabilities = np.array(all_probabilities)
        true_labels = np.array(all_true_labels)

        # Calculate confidence as the probability of the predicted class
        confidence_scores = np.max(probabilities, axis=-1)

        # Log results
        self.logger.info("")
        log_results(true_labels, predictions, self.logger)

        # Create output DataFrame with predictions and confidence
        output_df = test_df.copy()
        output_df['Predicted_Label'] = predictions
        output_df['Confidence'] = (confidence_scores * 100).round(2)

        return output_df

    def save(self, dir_name: str):
        """ 
        Save the trained classifier
        """
        path = f"{config.TRAINING_DIR}/{dir_name}"
        os.makedirs(path, exist_ok=True)

        # Save classifier weights
        torch.save(
            self.classifier.state_dict(),
            f"{path}/classifier.pth"
        )

        # Save config
        config_dict = {
            'model_name': self.MODEL_NAME,
            'pretrained_model_name': self.PRETRAINED_MODEL_NAME,
            'max_length': self.MAX_LENGTH,
            'encoder_hidden_size': self.encoder.config.hidden_size
        }

        import json
        with open(f"{path}/config.json", 'w') as f:
            json.dump(config_dict, f, indent=2)

        self.logger.info(f"{self.MODEL_NAME} classifier saved to {path}")
        self.logger.info("")
