import os
import torch
import pandas as pd
from transformers import logging
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from transformers import Trainer, TrainingArguments
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from src.utils.code_dataset import CodeDataset
from src.utils.helper import load_samples_from_dir, load_samples_from_csv


MODEL_NAME = "microsoft/codebert-base"
SAVED_MODEL_PATH = "./saved_model"
DATASET_PATH = "./data"
TEST_SIZE = 0.2
RANDOM_STATE = 42
NUM_TRAIN_EPOCHS = 1
BATCH_SIZE = 16
DATASETS = {
    'java': [
        ('GPTSniffer ChatGPT', 'gptsniffer'),
        ('HumanEval GPT-4', 'humaneval_chatgpt4_java_merged.csv'),
        ('HumanEval ChatGPT', 'humaneval_chatgpt_java_merged.csv'),
        ('HumanEval Gemini Pro', 'humaneval_gemini_java_merged.csv')
    ],
    'python': [
        ('HumanEval GPT-4', 'humaneval_chatgpt4_python_merged.csv'),
        ('HumanEval ChatGPT', 'humaneval_chatgpt_python_merged.csv'),
        ('HumanEval Gemini Pro', 'humaneval_gemini_python_merged.csv'),
        ('MBPP GPT-4', 'mbpp_chatgpt4_python_merged.csv'),
        ('MBPP ChatGPT', 'mbpp_chatgpt_python_merged.csv'),
        ('MBPP Gemini Pro', 'mbpp_gemini_python_merged.csv'),
        ('CodeNet Gemini Flash', 'codenet_gemini_python.csv')
    ]
}


class CodeBertModel:

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
            if (not os.path.exists(SAVED_MODEL_PATH)
                    or not os.listdir(SAVED_MODEL_PATH)):
                raise FileNotFoundError("Saved model not found")

            self.tokenizer = RobertaTokenizer.from_pretrained(SAVED_MODEL_PATH)
            self.model = RobertaForSequenceClassification.from_pretrained(
                SAVED_MODEL_PATH
            )
            print("Loaded model from saved")
        else:
            self.tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)
            self.model = RobertaForSequenceClassification.from_pretrained(
                MODEL_NAME,
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

    def prepare(self):
        """ Prepares the dataset for training. """
        print(f"{'Dataset'.ljust(25)}"
              f"{'Language'.ljust(15)}"
              f"{'Total'.ljust(10)}"
              f"{'Train'.ljust(10)}"
              f"{'Test'.ljust(10)}")
        print("-" * 70)

        # Split each dataset individually and combine the splits
        train_samples = []
        test_samples = []
        for language in DATASETS:
            for dataset_name, dataset_path in DATASETS[language]:
                # Load samples from the dataset
                path = os.path.join(DATASET_PATH, language, dataset_path)
                if path.endswith('.csv'):
                    samples = load_samples_from_csv(path)
                else:
                    samples = load_samples_from_dir(path)

                # Split the samples into train and test sets
                train_split, test_split = train_test_split(
                    samples,
                    test_size=TEST_SIZE,
                    random_state=RANDOM_STATE
                )
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
        if not self.model:
            raise ValueError("Model not initialized. Call setup() first.")
        if not self.train_dataset or not self.test_dataset:
            raise ValueError("Datasets not prepared. Call prepare() first.")

        # Calculate warmup steps and logging steps
        steps_per_epoch = len(self.train_dataset) // BATCH_SIZE
        total_steps = steps_per_epoch * NUM_TRAIN_EPOCHS
        warmup_steps = total_steps // 10  # 10% of total steps
        logging_steps = steps_per_epoch // 4  # 25% of epoch steps

        print(f"{'EPOCHS'.rjust(13)}: {NUM_TRAIN_EPOCHS}")
        print(f"{'BATCH_SIZE'.rjust(13)}: {BATCH_SIZE}")
        print(f"{'TOTAL_STEPS'.rjust(13)}: {total_steps}")
        print(f"{'WARMUP_STEPS'.rjust(13)}: {warmup_steps}")
        print(f"{'LOGGING_STEPS'.rjust(13)}: {logging_steps}")

        train_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=NUM_TRAIN_EPOCHS,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE,
            warmup_steps=warmup_steps,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=logging_steps,
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
        if not self.model:
            raise ValueError("Model not initialized. Call setup() first.")
        if not self.test_dataset:
            raise ValueError("Datasets not prepared. Call prepare() first.")

        test_dataloader = DataLoader(self.test_dataset,
                                     batch_size=BATCH_SIZE,
                                     shuffle=False)
        print(f"TEST_SIZE: {int(TEST_SIZE * 100)}%")

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
        cm_df = pd.DataFrame(cm,
                             index=['Actual AI', 'Actual Human'],
                             columns=['Predicted AI', 'Predicted Human'])
        print(cm_df.to_string())

    def save(self):
        """ Saves the model and tokenizer. """
        if not self.model or not self.tokenizer:
            raise ValueError("Model not initialized. Call setup() first.")
        self.tokenizer.save_pretrained(SAVED_MODEL_PATH)
        self.model.save_pretrained(SAVED_MODEL_PATH)
        print(f"Model saved to {SAVED_MODEL_PATH}")

    def classify_code(self, code_snippet: str):
        """ Classifies a code snippet. """
        if not self.model or not self.tokenizer:
            raise ValueError("Model not initialized. Call setup() first.")

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
    import sys
    import time
    from src.utils.dual_output import DualOutput

    dual_output = DualOutput()
    sys.stdout = dual_output

    model = CodeBertModel()
    start_time = time.time()

    print("\nPreparing datasets...")
    model.prepare()

    print("\nTraining model...")
    model.train()

    print("\nEvaluating model...")
    model.evaluate()

    end_time = time.time()
    print(f"\nRuntime: {end_time - start_time} seconds")

    sys.stdout = sys.__stdout__
    filename = input("Save output to (filename): ").strip()
    if filename != "":
        with open(f'outputs/{filename}.log', 'w') as file:
            file.write(dual_output.buffer.getvalue())
        print(f"Model saved to outputs/{filename}.log")
    else:
        print("Output not saved")
    print()

    if input("Save model (y/n): ").strip().lower() == "y":
        model.save()
    else:
        print("Model not saved")
