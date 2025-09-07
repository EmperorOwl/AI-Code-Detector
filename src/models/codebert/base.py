import os
from logging import Logger

import torch
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from transformers import Trainer, TrainingArguments

from src.pre_processing.code_dataset import CodeDataset, Sample
from src.utils.callback import MetricsCallback
from src.utils.results import log_results


class BaseTransformerModel:
    """ Fine-tuned CodeBERT-compatible model (supports CodeBERT, UniXcoder) """
    MODEL_NAME = ""
    PRETRAINED_NAME = ""
    SAVED_MODEL_DIR = "./saved/"
    LOG_DIR = "./outputs/"
    MAX_LENGTH = 512

    def __init__(self,
                 logger: Logger,
                 num_train_epochs: int,
                 batch_size: int,
                 use_ast: bool = False):
        self.num_train_epochs = num_train_epochs
        self.batch_size = batch_size
        self.use_ast = use_ast

        # Set up tokenizer and model
        self.tokenizer = RobertaTokenizer.from_pretrained(self.PRETRAINED_NAME)
        self.model = RobertaForSequenceClassification.from_pretrained(
            self.PRETRAINED_NAME,
            num_labels=2
        )

        # Set up device
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)  # type: ignore

        # Set up logger
        self.logger = logger
        self.logger.info(
            f"{self.MODEL_NAME} model initialized using device: {self.device}\n"
        )

    def train(self,
              train_samples: list[Sample],
              validation_samples: list[Sample]):
        # Prepare datasets
        train_dataset = CodeDataset(
            self.tokenizer,
            train_samples,
            use_ast=self.use_ast,
            max_length=self.MAX_LENGTH
        )
        self.logger.info(
            "Train Dataset Truncated Count:", train_dataset.truncated_count
        )
        validation_dataset = CodeDataset(
            self.tokenizer,
            validation_samples,
            use_ast=self.use_ast,
            max_length=self.MAX_LENGTH
        )
        self.logger.info(
            "Validation Dataset Truncated Count:",
            validation_dataset.truncated_count
        )

        # Calculate warm up and logging steps
        steps_per_epoch = len(train_dataset) // self.batch_size
        total_steps = steps_per_epoch * self.batch_size
        warmup_steps = total_steps // 10  # 10% of total steps
        logging_steps = steps_per_epoch // 4  # 25% of epoch steps

        self.logger.info(
            f"Training {self.MODEL_NAME} Model ("
            f"epochs: {self.num_train_epochs}, "
            f"batch_size: {self.batch_size}) ..."
        )
        self.logger.info(f"{'Epoch'.ljust(15)}"
                         f"{'Train Loss'.ljust(15)}"
                         f"{'Val Loss'.ljust(15)}"
                         f"{'Learning Rate'.ljust(15)}")
        self.logger.info("-" * 60)

        train_args = TrainingArguments(
            # Best Model
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            save_total_limit=5,
            eval_strategy="steps",
            # Steps
            eval_steps=logging_steps,
            save_steps=logging_steps,
            logging_steps=logging_steps,
            warmup_steps=warmup_steps,
            # Config
            num_train_epochs=self.num_train_epochs,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            weight_decay=0.01,
            optim='adamw_torch',
            learning_rate=5e-5,
            # File
            output_dir='./results',
        )
        trainer = Trainer(
            model=self.model,
            args=train_args,
            train_dataset=train_dataset,
            eval_dataset=validation_dataset,
        )

        trainer.callback_handler.callbacks.pop()
        trainer.callback_handler.add_callback(MetricsCallback(self.logger))

        trainer.train()
        self.logger.info("")

    def predict(self, test_samples: list[Sample]):
        # Prepare test dataset
        test_dataset = CodeDataset(
            self.tokenizer,
            test_samples,
            use_ast=self.use_ast,
            max_length=self.MAX_LENGTH
        )
        self.logger.info(
            "Test Dataset Truncated Count:",
            test_dataset.truncated_count
        )
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False
        )

        # Evaluate model
        self.logger.info(f"Evaluating {self.MODEL_NAME} Model...")
        self.model.eval()
        y_true = []
        y_pred = []
        with torch.no_grad():
            for batch in test_dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                outputs = self.model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=1)
                y_true += labels.tolist()
                y_pred += predictions.tolist()

        # Save results
        log_results(y_true, y_pred, self.logger)

    def save(self):
        path = self.SAVED_MODEL_DIR
        os.makedirs(path, exist_ok=True)
        self.tokenizer.save_pretrained(path)
        self.model.save_pretrained(path)
        self.logger.info(f"{self.MODEL_NAME} model saved to {path}")


class CodeBertModel(BaseTransformerModel):
    MODEL_NAME = "CodeBERT"
    PRETRAINED_NAME = "microsoft/codebert-base"
    SAVED_MODEL_DIR = "./saved/codebert"
    LOG_DIR = "./outputs/codebert_model"


class UniXcoderModel(BaseTransformerModel):
    MODEL_NAME = "UniXcoder"
    PRETRAINED_NAME = "microsoft/unixcoder-base"
    SAVED_MODEL_DIR = "./saved/unixcoder"
    LOG_DIR = "./outputs/unixcoder_model"
    MAX_LENGTH = 1024
