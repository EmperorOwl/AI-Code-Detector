import torch
from transformers import logging
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from transformers import Trainer, TrainingArguments

from code_dataset import CodeDataset


class Model:
    MODEL_NAME = "microsoft/codebert-base"
    TRAIN_DATASET = "./training_data"
    TEST_DATASET = "./testing_data"

    def __init__(self):
        """ Initializes the model. """
        logging.set_verbosity_error()
        self.tokenizer = RobertaTokenizer.from_pretrained(Model.MODEL_NAME)
        self.model = RobertaForSequenceClassification.from_pretrained(Model.MODEL_NAME, num_labels=2)
        self.setup()

    def setup(self) -> None:
        """ Use GPU if available. """
        if torch.cuda.is_available():
            self.model.to(torch.device("cuda"))
            print("Using GPU")

        else:
            self.model.to(torch.device("cpu"))
            print("Using CPU")

    def train(self):
        train_dataset = CodeDataset(self.tokenizer, Model.TRAIN_DATASET)
        test_dataset = CodeDataset(self.tokenizer, Model.TEST_DATASET)
        train_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=2,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=10,
            optim='adamw_torch',
            learning_rate=5e-5,
            save_total_limit=2,
        )
        trainer = Trainer(
            model=self.model,
            args=train_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset
        )
        trainer.train()

        trainer.evaluate()

    def evaluate(self):
        pass
