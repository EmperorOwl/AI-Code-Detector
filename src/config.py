RANDOM_STATE = 42
DATASET_PATH = "./datasets"
DATASETS = [
    # name, language, ai, path
    ('gptsniffer', 'java', 'chatgpt', 'gptsniffer'),
    ('humaneval', 'java', 'gpt_4', 'humaneval_chatgpt4_java_merged.csv'),
    ('humaneval', 'java', 'chatgpt', 'humaneval_chatgpt_java_merged.csv'),
    ('humaneval', 'java', 'gemini_pro', 'humaneval_gemini_java_merged.csv'),
    ('humaneval', 'python', 'gpt_4', 'humaneval_chatgpt4_python_merged.csv'),
    ('humaneval', 'python', 'chatgpt', 'humaneval_chatgpt_python_merged.csv'),
    ('humaneval', 'python', 'gemini_pro', 'humaneval_gemini_python_merged.csv'),
    ('mbpp', 'python', 'gpt_4', 'mbpp_chatgpt4_python_merged.csv'),
    ('mbpp', 'python', 'chatgpt', 'mbpp_chatgpt_python_merged.csv'),
    ('mbpp', 'python', 'gemini_pro', 'mbpp_gemini_python_merged.csv'),
    ('codenet', 'python', 'gemini_flash', 'codenet_gemini_python.csv'),

    ('hmcorp', 'python', 'chatgpt', 'hmcorp.jsonl'),
    ('hmcorp', 'java', 'chatgpt', 'hmcorp.jsonl')
]

DEFAULT_SPLIT = {
    'test_size': 0.1,
    'val_size': 0.1,
}
DATASET_CONFIG_ALL = {
    **DEFAULT_SPLIT,
    'max_sample_count': 5000  # Limit to 5000 randomly selected samples
}
DATASET_CONFIG_JAVA_ONLY = {
    **DATASET_CONFIG_ALL,
    'language_filter': 'java'
}
DATASET_CONFIG_PYTHON_ONLY = {
    **DATASET_CONFIG_ALL,
    'language_filter': 'python'
}
DATASET_CONFIG_UNSEEN = {
    'config': {
        'gptsniffer': {'test_size': 1},
        'humaneval': {'val_size': 0.2},
        'mbpp': {'val_size': 0.2},
        'hmcorp': {'val_size': 0.2, 'max_sample_count': 5000},
        'codenet': {'test_size': 1, 'max_sample_count': 1480}
    }
}

CONFIG_ALL = {
    'dataset_config': DATASET_CONFIG_ALL,
    'log_file_suffix': 'all.log'
}

CONFIG_PYTHON_ONLY = {
    'dataset_config': DATASET_CONFIG_PYTHON_ONLY,
    'log_file_suffix': 'python.log'
}

CONFIG_JAVA_ONLY = {
    'dataset_config': DATASET_CONFIG_JAVA_ONLY,
    'log_file_suffix': 'java.log'
}

CONFIG_UNSEEN = {
    'dataset_config': DATASET_CONFIG_UNSEEN,
    'log_file_suffix': 'unseen.log'
}

CONFIG = [
    CONFIG_ALL,
    CONFIG_JAVA_ONLY,
    CONFIG_PYTHON_ONLY,
    CONFIG_UNSEEN
]
