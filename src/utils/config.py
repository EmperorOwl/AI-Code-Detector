import os

ENV = os.getenv('ENV')


if ENV == 'prod':
    IS_TEST_RUN = False
    OUTPUT_DIR = 'outputs'
    NUM_TRAIN_EPOCHS = 10
    TRAIN_BATCH_SIZE = 32
    EVAL_BATCH_SIZE = 32
    SAMPLING_REQUIREMENTS = None
else:
    OUTPUT_DIR = 'outputs_test'
    IS_TEST_RUN = True
    NUM_TRAIN_EPOCHS = 1
    TRAIN_BATCH_SIZE = 8
    EVAL_BATCH_SIZE = 8
    SAMPLING_REQUIREMENTS = {
        ('Java', 'GPT-4o'): 100,
        ('Java', 'Human'): 100,
    }

TRAINING_DIR = OUTPUT_DIR + '/training'
DATASET_DIR = OUTPUT_DIR + '/dataset_processing'

DROID_PATH = DATASET_DIR + '/droid_dataset.csv'
AIG_PATH = DATASET_DIR + '/aig_dataset.csv'
SNIFFER_PATH = DATASET_DIR + '/sniffer_dataset.csv'
HUMANEVAL_PATH = DATASET_DIR + '/humaneval_dataset.csv'
MBPP_PATH = DATASET_DIR + '/mbpp_dataset.csv'
