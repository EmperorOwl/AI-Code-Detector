import os
import torch
from transformers import logging
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from transformers import Trainer, TrainingArguments
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from code_dataset import CodeDataset
from utils.utils import load_samples_from_dir, load_samples_from_csv

MODEL_NAME = "microsoft/codebert-base"
SAVED_MODEL_PATH = "./saved_model"

GPTSNIFFER_JAVA = "./data/java/gptsniffer"
MBPP_CHATGPT_PYTHON = "./data/python/mbpp_chatgpt_python_merged.csv"


class Model:

    def __init__(self, use_saved: bool = False):
        """ Initializes the model. """
        self.tokenizer = None
        self.model = None
        self.device = None
        self.train_dataset = None
        self.test_dataset = None
        self.setup(use_saved)
        self.prepare()

    def setup(self, use_saved: bool):
        """ Sets up the model. """
        logging.set_verbosity_error()
        # Load model from saved if it exists
        if use_saved:
            if not os.path.exists(SAVED_MODEL_PATH) or not os.listdir(SAVED_MODEL_PATH):
                raise FileNotFoundError("Saved model not found")

            self.tokenizer = RobertaTokenizer.from_pretrained(SAVED_MODEL_PATH)
            self.model = RobertaForSequenceClassification.from_pretrained(SAVED_MODEL_PATH)
            print("Loaded model from saved")
        else:
            self.tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)
            self.model = RobertaForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
            print("Loaded model from Hugging Face")

        # Use GPU if available
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.model.to(self.device)
        print(f"Using device: {self.device}")

    def prepare(self):
        """ Prepares the dataset for training. """
        samples = load_samples_from_dir(GPTSNIFFER_JAVA)
        samples += load_samples_from_csv(MBPP_CHATGPT_PYTHON)

        # Split the samples into train and test sets
        train_samples, test_samples = train_test_split(samples,
                                                       test_size=0.2,
                                                       random_state=42)
        print(f"Total samples: {len(samples)} (train: {len(train_samples)}, test: {len(test_samples)})")

        # Create the datasets for processing
        self.train_dataset = CodeDataset(self.tokenizer, train_samples)
        self.test_dataset = CodeDataset(self.tokenizer, test_samples)

    def train(self):
        """ Trains the model. """
        train_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=3,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            warmup_steps=15,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=25,
            optim='adamw_torch',
            learning_rate=5e-5,
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
        """ Prints out the classification report.
        Make sure to train the model first.
        """
        test_dataloader = DataLoader(self.test_dataset,
                                     batch_size=16,
                                     shuffle=False)

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

        target_names = ['ChatGPT', 'Human']
        print(classification_report(y_true, y_pred, target_names=target_names))

    def save(self):
        """ Saves the model and tokenizer. """
        self.tokenizer.save_pretrained(SAVED_MODEL_PATH)
        self.model.save_pretrained(SAVED_MODEL_PATH)

    def classify_code(self, code_snippet: str):
        """ Classifies a code snippet. """
        inputs = self.tokenizer.encode_plus(code_snippet,
                                            padding='max_length',
                                            max_length=512,
                                            truncation=True,
                                            return_tensors='pt')
        input_ids = inputs['input_ids'].to(self.model.device)
        attention_mask = inputs['attention_mask'].to(self.model.device)

        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)
            ai_probability = probabilities[0][0].item()

        return ai_probability * 100  # Return as a percentage


if __name__ == '__main__':
    """ Retrains the model from scratch. """
    import time

    model = Model()
    start_time = time.time()

    print("Training model...")
    model.train()
    print("Model trained")
    model.save()
    print("Model saved")

    print("Evaluating model...")
    model.evaluate()

    end_time = time.time()
    print(f"Runtime: {end_time - start_time} seconds")
