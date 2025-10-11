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
python -m src.dataset_processing.datasets.humaneval_dataset
python -m src.dataset_processing.datasets.mbpp_dataset
python -m src.dataset_processing.datasets.sniffer_dataset

# Run experiment
export ENV=prod
./run
```

### Enable GPU

- [PyTorch Version](https://pytorch.org/get-started/locally/)

```
nvidia-smi
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu128
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
- [ChatGPT Versions](https://en.wikipedia.org/wiki/ChatGPT#Model_versions)
- [Gemini Versions](<https://en.wikipedia.org/wiki/Gemini_(language_model)#Model_versions>)
