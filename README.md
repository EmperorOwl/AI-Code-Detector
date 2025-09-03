# AI Code Detector

### References

- [GPTSniffer](https://github.com/MDEGroup/GPTSniffer)
- [CodeBERT](https://github.com/microsoft/CodeBERT)
- [Hugging Face CodeBERT](https://huggingface.co/microsoft/codebert-base)
- [Hugging Face RoBERTa](https://huggingface.co/docs/transformers/main/en/model_doc/roberta#roberta)
- [Tree Sitter](https://tree-sitter.github.io/tree-sitter/)
- [AIGCodeSet](https://huggingface.co/datasets/basakdemirok/AIGCodeSet)

https://github.com/google-research/google-research/blob/master/mbpp/README.md


### Labels
0 = Human
1 = AI

### Commands

```bash
python -m src.models.codebert.main
python -m src.models.ast.main
```