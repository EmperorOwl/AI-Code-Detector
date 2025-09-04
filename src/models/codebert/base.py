import os
from logging import Logger

import torch
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from transformers import Trainer, TrainingArguments

from src.pre_processing.code_dataset import CodeDataset, Sample
from src.utils.callback import MetricsCallback
from src.utils.results import log_results


class CodeBertModel:
    """ Fine-tuned CodeBERT model """
    MODEL_NAME = "microsoft/codebert-base"
    SAVED_MODEL_PATH = "./saved/codebert_model"

    def __init__(self,
                 num_train_epochs: int,
                 batch_size: int,
                 logger: Logger):
        self.num_train_epochs = num_train_epochs
        self.batch_size = batch_size

        # Set up tokenizer and model
        self.tokenizer = RobertaTokenizer.from_pretrained(self.MODEL_NAME)
        self.model = RobertaForSequenceClassification.from_pretrained(
            self.MODEL_NAME,
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
            f"CodeBERT model initialized using device: {self.device}\n"
        )

    def train(self,
              train_samples: list[Sample],
              validation_samples: list[Sample]):
        # Prepare datasets
        train_dataset = CodeDataset(self.tokenizer, train_samples)
        validation_dataset = CodeDataset(self.tokenizer, validation_samples)

        # Calculate warm up and logging steps
        steps_per_epoch = len(train_dataset) // self.batch_size
        total_steps = steps_per_epoch * self.batch_size
        warmup_steps = total_steps // 10  # 10% of total steps
        logging_steps = steps_per_epoch // 4  # 25% of epoch steps

        self.logger.info(
            f"Training CodeBERT Model (epochs: {self.num_train_epochs}, "
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
        test_dataset = CodeDataset(self.tokenizer, test_samples)
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False
        )

        # Evaluate model
        self.logger.info("Evaluating CodeBERT Model...")
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
        path = self.SAVED_MODEL_PATH
        os.makedirs(path, exist_ok=True)
        self.tokenizer.save_pretrained(path)
        self.model.save_pretrained(path)
        self.logger.info(f"CodeBERT model saved to {path}")
