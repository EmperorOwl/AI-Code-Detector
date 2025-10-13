import os
from logging import Logger

import numpy as np
import pandas as pd
import torch
from transformers import (
    AutoModelForSequenceClassification,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments
)

from src.models.transformer.code_dataset import CodeDataset
from src.utils.callback import MetricsCallback
from src.utils.results import log_results
from src.utils import config


class TransformerModel:
    """
    Base transformer model for detecting AI-generated code.
    """
    MODEL_NAME = ""
    PRETRAINED_MODEL_NAME = ""
    MAX_LENGTH = -1

    def __init__(self,
                 logger: Logger,
                 load_from_saved_path: str | None = None) -> None:
        """
        Initialize the CodeBERT model.

        Args:
            logger (Logger): Logger instance for logging
            load_saved (bool): Whether to load a saved model
        """
        self.logger = logger

        # Model configuration
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )

        if load_from_saved_path:
            path = f"{config.TRAINING_DIR}/{load_from_saved_path}"
            self.model = AutoModelForSequenceClassification.from_pretrained(
                path
            )
            log_msg = f"Existing {self.MODEL_NAME} model loaded from {path}"
        else:
            # Initialize new model
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.PRETRAINED_MODEL_NAME,
                num_labels=2,  # Binary classification: Human (0) vs AI (1)
                output_attentions=False,
                output_hidden_states=False
            )
            log_msg = f"New {self.MODEL_NAME} model initialized"

        # Move model to device
        self.model.to(self.device)  # type: ignore

        # Log model ready
        self.logger.info(log_msg + f" (device: {self.device})\n")

    def train(self,
              train_df: pd.DataFrame,
              val_df: pd.DataFrame,
              num_train_epochs: int,
              batch_size: int):
        """
        Train the transformer model using HuggingFace Trainer.

        Args:
            train_df (pd.DataFrame): Training dataset
            val_df (pd.DataFrame): Validation dataset
            num_train_epochs (int): Number of training epochs
            batch_size (int): Batch size for training and validation
        """
        # Create datasets
        train_dataset = CodeDataset(train_df)
        val_dataset = CodeDataset(val_df)

        # Calculate warm up and logging steps
        steps_per_epoch = len(train_dataset) // batch_size
        total_steps = steps_per_epoch * num_train_epochs
        warmup_steps = total_steps // 10  # 10% of total steps
        logging_steps = steps_per_epoch // 4  # 25% of epoch steps

        # Log start training message
        self.logger.info(
            f"Training {self.MODEL_NAME} model ("
            f"epochs: {num_train_epochs}, "
            f"batch_size: {batch_size})\n"
        )

        # Log header
        self.logger.info(f"{'Epoch'.ljust(15)}"
                         f"{'Train Loss'.ljust(15)}"
                         f"{'Val Loss'.ljust(15)}"
                         f"{'Learning Rate'.ljust(15)}")
        self.logger.info("-" * 60)

        # Set up training arguments
        train_args = TrainingArguments(
            # Best Model
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,  # Lower loss is better
            save_total_limit=5,
            eval_strategy="steps",
            save_strategy="steps",
            # Steps
            eval_steps=logging_steps,
            save_steps=logging_steps,
            logging_steps=logging_steps,
            warmup_steps=warmup_steps,
            # Config
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            weight_decay=0.01,
            optim='adamw_torch',
            learning_rate=5e-5,
            # File
            output_dir='./results',
        )

        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=train_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )

        # Add metrics callback
        trainer.callback_handler.add_callback(MetricsCallback(self.logger))

        # Train the model
        trainer.train()

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
        # Create dataset
        test_dataset = CodeDataset(test_df)

        # Create trainer for prediction with specified batch size
        predict_args = TrainingArguments(
            per_device_eval_batch_size=batch_size,
        )

        trainer = Trainer(
            model=self.model,
            args=predict_args
        )

        # Log start evaluating message
        self.logger.info(
            f"Evaluating {self.MODEL_NAME} model ("
            f"batch_size: {batch_size})"
        )

        # Make predictions
        predictions_output = trainer.predict(test_dataset)

        # Get raw logits and convert to probabilities using softmax
        logits = predictions_output.predictions
        probabilities = torch.softmax(torch.tensor(logits), dim=-1).numpy()

        # Get predicted class labels
        predictions = np.argmax(logits, axis=-1)

        # Calculate confidence as the probability of the predicted class
        confidence_scores = np.max(probabilities, axis=-1)

        true_labels = predictions_output.label_ids

        # Log results
        self.logger.info("")
        log_results(true_labels, predictions, self.logger)

        # Create output DataFrame with predictions and confidence
        # HuggingFace Trainer shouldn't shuffle during prediction
        output_df = test_df.copy()
        output_df['Predicted_Label'] = predictions
        output_df['Confidence'] = (confidence_scores * 100).round(2)

        return output_df

    def save(self, dir_name: str):
        """ 
        Save the trained model
        """
        path = f"{config.TRAINING_DIR}/{dir_name}"
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        self.logger.info(f"{self.MODEL_NAME} model saved to {path}")
        self.logger.info("")


class CodeBertModel(TransformerModel):
    MODEL_NAME = 'CodeBERT'
    PRETRAINED_MODEL_NAME = 'microsoft/codebert-base'
    MAX_LENGTH = 512


class UniXcoderModel(TransformerModel):
    MODEL_NAME = "UniXcoder"
    PRETRAINED_MODEL_NAME = "microsoft/unixcoder-base"
    MAX_LENGTH = 1024
