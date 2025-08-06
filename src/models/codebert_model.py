import os

import torch
import pandas as pd
from transformers import logging
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from transformers import Trainer, TrainingArguments
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix

from src.config import Config, CodeBertConfig
from src.pre_processing.code_dataset import CodeDataset


class CodeBertModel:
    """ Fine-tuned CodeBERT model """

    def __init__(self,
                 train_samples: list,
                 test_samples: list,
                 use_saved: bool = False,
                 num_train_epochs: int = CodeBertConfig.NUM_TRAIN_EPOCHS,
                 batch_size: int = CodeBertConfig.BATCH_SIZE) -> None:
        self.train_samples = train_samples
        self.test_samples = test_samples
        # Config
        self.num_train_epochs = num_train_epochs
        self.batch_size = batch_size
        # Model
        self.tokenizer = None
        self.model = None
        self.device = None
        # Datasets
        self.train_dataset = None
        self.test_dataset = None
        # Call setup to initialise 
        self.setup(use_saved)

    def setup(self, use_saved: bool):
        logging.set_verbosity_error()
        # Load model from saved if it exists
        if use_saved:
            if (not os.path.exists(CodeBertConfig.SAVED_MODEL_PATH)
                    or not os.listdir(CodeBertConfig.SAVED_MODEL_PATH)):
                raise FileNotFoundError("Saved model not found")

            self.tokenizer = RobertaTokenizer.from_pretrained(
                CodeBertConfig.SAVED_MODEL_PATH
            )
            self.model = RobertaForSequenceClassification.from_pretrained(
                CodeBertConfig.SAVED_MODEL_PATH
            )
            print("Loaded model from saved")
        else:
            self.tokenizer = RobertaTokenizer.from_pretrained(
                CodeBertConfig.MODEL_NAME
            )
            self.model = RobertaForSequenceClassification.from_pretrained(
                CodeBertConfig.MODEL_NAME,
                num_labels=2
            )
            print("Loaded model from Hugging Face")

        # Use GPU if available
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.model.to(self.device)  # type: ignore
        print(f"Using device: {self.device}")
        
        # Prepare datasets
        self.train_dataset = CodeDataset(self.tokenizer, self.train_samples)
        self.test_dataset = CodeDataset(self.tokenizer, self.test_samples)

    def train(self):
        if not self.model or not self.train_dataset:
            raise ValueError("Model not initialized - call setup() first")

        # Calculate warmup steps and logging steps
        steps_per_epoch = len(self.train_dataset) // CodeBertConfig.BATCH_SIZE
        total_steps = steps_per_epoch * CodeBertConfig.NUM_TRAIN_EPOCHS
        warmup_steps = total_steps // 10  # 10% of total steps
        logging_steps = steps_per_epoch // 4  # 25% of epoch steps

        print(f"{'EPOCHS'.rjust(13)}: {CodeBertConfig.NUM_TRAIN_EPOCHS}")
        print(f"{'BATCH_SIZE'.rjust(13)}: {CodeBertConfig.BATCH_SIZE}")
        print(f"{'TOTAL_STEPS'.rjust(13)}: {total_steps}")
        print(f"{'WARMUP_STEPS'.rjust(13)}: {warmup_steps}")
        print(f"{'LOGGING_STEPS'.rjust(13)}: {logging_steps}")

        train_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=CodeBertConfig.NUM_TRAIN_EPOCHS,
            per_device_train_batch_size=CodeBertConfig.BATCH_SIZE,
            per_device_eval_batch_size=CodeBertConfig.BATCH_SIZE,
            warmup_steps=warmup_steps,
            weight_decay=CodeBertConfig.WEIGHT_DECAY,
            logging_dir='./logs',
            logging_steps=logging_steps,
            optim='adamw_torch',
            learning_rate=CodeBertConfig.LEARNING_RATE,
        )
        trainer = Trainer(
            model=self.model,
            args=train_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.test_dataset
        )
        trainer.train()
        trainer.evaluate()

    def evaluate(self):
        if not self.model or not self.test_dataset:
            raise ValueError("Model not trained - call setup() first")

        test_dataloader = DataLoader(
            self.test_dataset,
            batch_size=CodeBertConfig.BATCH_SIZE,
            shuffle=False
        )
        print(f"TEST_SIZE: {int(Config.TEST_SIZE * 100)}%")

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

        target_names = ['AI', 'Human']
        print(classification_report(y_true, y_pred, target_names=target_names))
        
        cm = confusion_matrix(y_true, y_pred)
        cm_df = pd.DataFrame(
            cm,
            index=['Actual AI', 'Actual Human'],
            columns=['Predicted AI', 'Predicted Human']
        )
        print(cm_df.to_string())

    def save(self):
        if not self.model or not self.tokenizer:
            raise ValueError("Model not initialized - call setup() first")

        path = CodeBertConfig.SAVED_MODEL_PATH
        os.makedirs(path, exist_ok=True)

        self.tokenizer.save_pretrained(path)
        self.model.save_pretrained(path)
        print(f"Model saved to {path}")

    def classify_code(self, code_snippet: str):
        if not self.model or not self.tokenizer:
            raise ValueError("Model not trained - call train() first")

        inputs = self.tokenizer.encode_plus(
            code_snippet,
            padding='max_length',
            max_length=512,
            truncation=True,
            return_tensors='pt'
        )
        input_ids = inputs['input_ids'].to(self.model.device)
        attention_mask = inputs['attention_mask'].to(self.model.device)

        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)
            ai_probability = probabilities[0][0].item()

        return ai_probability * 100  # Return as a percentage
