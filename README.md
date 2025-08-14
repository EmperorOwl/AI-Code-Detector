# AI Code Detector

### References

- [GPTSniffer](https://github.com/MDEGroup/GPTSniffer)
- [CodeBERT](https://github.com/microsoft/CodeBERT)
- [Hugging Face CodeBERT](https://huggingface.co/microsoft/codebert-base)
- [Hugging Face RoBERTa](https://huggingface.co/docs/transformers/main/en/model_doc/roberta#roberta)

https://github.com/google-research/google-research/blob/master/mbpp/README.md


Labels
0 = AI
1 = Human

### Commands

```bash
# Train CodeBERT model
python -m src.main --train-codebert

# Train AST model
python -m src.main --train-ast

# Start Flask app
python -m src.app
```