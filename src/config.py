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

    # ('hmcorp', 'python', 'chatgpt', 'hmcorp.jsonl'),
    # ('hmcorp', 'java', 'chatgpt', 'hmcorp.jsonl')
]

DEFAULT_SPLIT = {'test_size': 0.1, 'val_size': 0.1}
JAVA_ONLY_CONFIG = {**DEFAULT_SPLIT, 'language_filter': 'java'}
PYTHON_ONLY_CONFIG = {**DEFAULT_SPLIT, 'language_filter': 'python'}
UNSEEN_CONFIG = {
    'config': {
        'gptsniffer': {'test_size': 1, 'max_sample_count': 155},
        'humaneval': {'val_size': 0.2},
        'mbpp': {'val_size': 0.2},
        'codenet': {'test_size': 1, 'max_sample_count': 1091}
    }
}


CONFIG = [
    # All
    {
        'dataset_config': DEFAULT_SPLIT,
        'log_file_suffix': 'all.log'
    },
    # Java Only
    {
        'dataset_config': JAVA_ONLY_CONFIG,
        'log_file_suffix': 'java.log'
    },
    # Python Only
    {
        'dataset_config': PYTHON_ONLY_CONFIG,
        'log_file_suffix': 'python.log'
    },
    # Unseen
    {
        'dataset_config': UNSEEN_CONFIG,
        'log_file_suffix': 'unseen.log'
    },
]
