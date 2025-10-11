# AI Code Detector

#### Labels

> 0 = Human

> 1 = AI

### Commands

```bash
# Prepare all datasets
python -m src.dataset_processing.prepare_all

# Prepare individual datasets
python -m src.dataset_processing.datasets.aig_dataset
python -m src.dataset_processing.datasets.droid_dataset
python -m src.dataset_processing.datasets.sniffer_dataset
```

### Enable GPU

- [PyTorch Version](https://pytorch.org/get-started/locally/)

```
nvidia-smi
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

#### Project Structure

```
src/
├── dataset_processing/
│   ├── dataset_helper.py       # Dataset helper utilities
│   ├── dataset_tokenizer.py    # Tokenize code for model training
│   └── droid_dataset.py        # Load and prepare DroidCollection dataset
├── models/
│   └── transformer/
│       ├── code_dataset.py     # Dataset class for transformer models
│       ├── train.py            # Training script for transformer models
│       └── transformer_model.py # Transformer model implementations
├── trials/
│   ├── helper.py               # Trial helper utilities
│   └── same_dataset.py         # Evaluation on same dataset
└── utils/
    ├── analysis.py             # Analysis utilities
    ├── config.py               # Configuration management
    ├── logger.py               # Logging utilities
    └── results.py              # Model evaluation utilities
```

#### Models

- [CodeBERT](https://github.com/microsoft/CodeBERT)
- [Hugging Face CodeBERT](https://huggingface.co/microsoft/codebert-base)
- [Hugging Face RoBERTa](https://huggingface.co/docs/transformers/main/en/model_doc/roberta#roberta)
- [sckit-learn Linear Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)

#### Tools

- [Tree Sitter](https://tree-sitter.github.io/tree-sitter/)

#### Existing Solutions

- [GPTSniffer](https://github.com/MDEGroup/GPTSniffer)
- [CodeGPTSensor](https://github.com/doriscullen/CodeGPTSensor)
- [AST AI-Detector](https://github.com/mahantaf/AI-Detector)

#### Datasets

- [DroidCollection](https://huggingface.co/datasets/project-droid/DroidCollection)
- [AIGCodeSet](https://huggingface.co/datasets/basakdemirok/AIGCodeSet)
- [MBPP](https://github.com/google-research/google-research/blob/master/mbpp/README.md)

#### References

- [CodeBERT AST](https://github.com/microsoft/CodeBERT/issues/187)
- [Hugging Face Understanding Learning Curves](https://huggingface.co/learn/llm-course/en/chapter3/5)
