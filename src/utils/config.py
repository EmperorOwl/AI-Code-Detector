SAVE_DIR = 'saved_models'
OUTPUT_DIR = 'outputs'
TRAINING_DIR = OUTPUT_DIR + '/training'
DATASET_DIR = OUTPUT_DIR + '/dataset_processing'


class DatasetPaths:
    DROID = DATASET_DIR + '/droid_dataset.csv'
    AIG = DATASET_DIR + '/aig_dataset.csv'
    SNIFFER = DATASET_DIR + '/sniffer_dataset.csv'
    HUMANEVAL = DATASET_DIR + '/humaneval_dataset.csv'
    MBPP = DATASET_DIR + '/mbpp_dataset.csv'


CONFIG = {
    'dev': {
        'NUM_TRAIN_EPOCHS': 1,
        'TRAIN_BATCH_SIZE': 8,
        'EVAL_BATCH_SIZE': 8,
        'SAMPLING_REQUIREMENTS': {
            ('Java', 'GPT-4o'): 100,
            ('Java', 'Human'): 100,
        }
    },
    'prod': {
        'NUM_TRAIN_EPOCHS': 10,
        'TRAIN_BATCH_SIZE': 32,
        'EVAL_BATCH_SIZE': 32,
    }
}
