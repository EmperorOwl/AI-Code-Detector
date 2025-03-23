import time
import torch
from transformers import RobertaTokenizer, RobertaForMaskedLM, pipeline, logging

logging.set_verbosity_error()
model = RobertaForMaskedLM.from_pretrained("microsoft/codebert-base-mlm")
tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base-mlm")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

if __name__ == '__main__':

    while True:
        CODE = input("Enter code: ")
        fill_mask = pipeline('fill-mask', model=model, tokenizer=tokenizer)

        start_time = time.time()
        outputs = fill_mask(CODE)
        end_time = time.time()

        predicted_token = outputs[0]['token_str']
        print(predicted_token)
        print(f"Runtime: {end_time - start_time} seconds")
