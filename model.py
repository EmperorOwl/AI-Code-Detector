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
DATASET_PATH = "./data"


class Model:

    def __init__(self, use_saved: bool = False):
        """ Initializes the model. """
        self.tokenizer = None
        self.model = None
        self.device = None
        self.train_dataset = None
        self.test_dataset = None
        self.setup(use_saved)

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
        datasets = {
            'java': [
                ('GPT Sniffer', 'gptsniffer'),
                ('HumanEval ChatGPT 4', 'humaneval_chatgpt4_java_merged.csv'),
                ('HumanEval ChatGPT', 'humaneval_chatgpt_java_merged.csv'),
                ('HumanEval Gemini', 'humaneval_gemini_java_merged.csv')
            ],
            'python': [
                ('HumanEval ChatGPT 4', 'humaneval_chatgpt4_python_merged.csv'),
                ('HumanEval ChatGPT', 'humaneval_chatgpt_python_merged.csv'),
                ('HumanEval Gemini', 'humaneval_gemini_python_merged.csv'),
                ('MBPP ChatGPT 4', 'mbpp_chatgpt4_python_merged.csv'),
                ('MBPP ChatGPT', 'mbpp_chatgpt_python_merged.csv'),
                ('MBPP Gemini', 'mbpp_gemini_python_merged.csv')
            ]
        }

        print(f"{'Dataset'.ljust(25)}"
              f"{'Language'.ljust(15)}"
              f"{'Total'.ljust(10)}"
              f"{'Train'.ljust(10)}"
              f"{'Test'.ljust(10)}")
        print("-" * 70)

        # Split each dataset individually and combine the splits
        train_samples = []
        test_samples = []
        for language in datasets:
            for dataset_name, dataset_path in datasets[language]:
                # Load samples from the dataset
                path = os.path.join(DATASET_PATH, language, dataset_path)
                if path.endswith('.csv'):
                    samples = load_samples_from_csv(path)
                else:
                    samples = load_samples_from_dir(path)

                # Split the samples into train and test sets
                train_split, test_split = train_test_split(samples,
                                                           test_size=0.2,
                                                           random_state=42)
                train_samples.extend(train_split)
                test_samples.extend(test_split)
                print(f"{dataset_name.ljust(25)}"
                      f"{language.ljust(15)}"
                      f"{str(len(samples)).ljust(10)}"
                      f"{str(len(train_split)).ljust(10)}"
                      f"{str(len(test_split)).ljust(10)}")

        print("-" * 70)
        print(f"{'Total'.ljust(40)}"
              f"{str(len(train_samples) + len(test_samples)).ljust(10)}"
              f"{str(len(train_samples)).ljust(10)}"
              f"{str(len(test_samples)).ljust(10)}")

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
            warmup_steps=100,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=116,
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

        target_names = ['AI', 'Human']
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

    print("Preparing datasets...")
    model.prepare()
    print("Datasets prepared\n")

    print("Training model...")
    model.train()
    print("Model trained")
    model.save()
    print("Model saved\n")

    print("Evaluating model...")
    model.evaluate()
    print("\n")

    end_time = time.time()
    print(f"Runtime: {end_time - start_time} seconds")
