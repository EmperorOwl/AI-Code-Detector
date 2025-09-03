# AI Code Detector

### References

- [GPTSniffer](https://github.com/MDEGroup/GPTSniffer)
- [CodeBERT](https://github.com/microsoft/CodeBERT)
- [Hugging Face CodeBERT](https://huggingface.co/microsoft/codebert-base)
- [Hugging Face RoBERTa](https://huggingface.co/docs/transformers/main/en/model_doc/roberta#roberta)
- [Tree Sitter](https://tree-sitter.github.io/tree-sitter/)

https://github.com/google-research/google-research/blob/master/mbpp/README.md


### Labels
0 = AI
1 = Human

### Commands

```bash
# View available datasets
python -m src.main --view-datasets

# Train CodeBERT model
python -m src.main --train-codebert
python -m src.main --train-codebert --train-datasets 0 --test-datasets 1 2 3

# Train AST model
python -m src.main --train-ast  # All datasets and 20% test size
python -m src.main --train-ast --datasets 0 1 2 3  # Java only
python -m src.main --train-ast --datasets 4 5 6 7 8 9 10  # Python only

# Test AST parsing
python -m src.pre_processing.tests.test_ast_node

# Start Flask app
python -m src.app
```