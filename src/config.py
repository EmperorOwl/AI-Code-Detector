class Config:
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    DATASET_PATH = "./datasets"
    DATASETS = [
        ('java', 'GPTSniffer ChatGPT', 'gptsniffer'),
        ('java', 'HumanEval GPT-4', 'humaneval_chatgpt4_java_merged.csv'),
        ('java', 'HumanEval ChatGPT', 'humaneval_chatgpt_java_merged.csv'),
        ('java', 'HumanEval Gemini Pro', 'humaneval_gemini_java_merged.csv'),
        ('python', 'HumanEval GPT-4', 'humaneval_chatgpt4_python_merged.csv'),
        ('python', 'HumanEval ChatGPT', 'humaneval_chatgpt_python_merged.csv'),
        ('python', 'HumanEval Gemini Pro', 'humaneval_gemini_python_merged.csv'),
        ('python', 'MBPP GPT-4', 'mbpp_chatgpt4_python_merged.csv'),
        ('python', 'MBPP ChatGPT', 'mbpp_chatgpt_python_merged.csv'),
        ('python', 'MBPP Gemini Pro', 'mbpp_gemini_python_merged.csv'),
        ('python', 'CodeNet Gemini Flash', 'codenet_gemini_python.csv')
    ]


class CodeBertConfig:
    MODEL_NAME = "microsoft/codebert-base"
    SAVED_MODEL_PATH = "./saved/codebert_model"
    NUM_TRAIN_EPOCHS = 12
    BATCH_SIZE = 16
    WEIGHT_DECAY = 0.01
    LEARNING_RATE = 5e-5


class AstModelConfig:
    MODEL_NAME = "Salesforce/codet5p-110m-embedding"
    SAVED_MODEL_PATH = "./saved/ast_model"
