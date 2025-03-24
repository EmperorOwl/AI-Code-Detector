import os
import torch
from transformers import logging
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from transformers import Trainer, TrainingArguments

from code_dataset import CodeDataset


class Model:
    MODEL_NAME = "microsoft/codebert-base"
    TRAIN_DATASET = "./training_data"
    TEST_DATASET = "./testing_data"
    SAVED_MODEL_PATH = "./saved_model"

    def __init__(self):
        """ Initializes the model. """
        self.tokenizer = None
        self.model = None
        self.setup()

    def setup(self):
        """ Sets up the model. """
        logging.set_verbosity_error()
        # Load model from saved if it exists
        if os.path.exists(Model.SAVED_MODEL_PATH) and os.listdir(Model.SAVED_MODEL_PATH):
            self.tokenizer = RobertaTokenizer.from_pretrained(Model.SAVED_MODEL_PATH)
            self.model = RobertaForSequenceClassification.from_pretrained(Model.SAVED_MODEL_PATH)
            print("Loaded model from saved")
        else:
            self.tokenizer = RobertaTokenizer.from_pretrained(Model.MODEL_NAME)
            self.model = RobertaForSequenceClassification.from_pretrained(Model.MODEL_NAME, num_labels=2)
            print("Loaded model from Hugging Face")

        # Use GPU if available
        if torch.cuda.is_available():
            self.model.to(torch.device("cuda"))
            print("Using GPU")
        else:
            self.model.to(torch.device("cpu"))
            print("Using CPU")

    def train(self):
        """ Trains the model. """
        train_dataset = CodeDataset(self.tokenizer, Model.TRAIN_DATASET)
        test_dataset = CodeDataset(self.tokenizer, Model.TEST_DATASET)
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
            train_dataset=train_dataset,
            eval_dataset=test_dataset
        )
        trainer.train()
        trainer.evaluate()

    def save(self):
        """ Saves the model and tokenizer. """
        self.model.save_pretrained(Model.SAVED_MODEL_PATH)
        self.tokenizer.save_pretrained(Model.SAVED_MODEL_PATH)

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
            print(probabilities)
            ai_probability = probabilities[0][0].item()  # Assuming label 0 is AI-written

        return ai_probability * 100  # Return as a percentage


if __name__ == '__main__':
    """ Trains the model. """
    import time

    model = Model()
    start_time = time.time()

    print("Training model...")
    model.train()
    print("Model trained")
    model.save()
    print("Model saved")

    end_time = time.time()
    print(f"Runtime: {end_time - start_time} seconds")
