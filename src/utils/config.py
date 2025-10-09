SAVE_DIR = 'saved_models'
OUTPUT_DIR = 'outputs'
TRAINING_DIR = OUTPUT_DIR + '/training'


TEST_CONFIG = {
    'NUM_TRAIN_EPOCHS': 1,
    'TRAIN_BATCH_SIZE': 8,
    'EVAL_BATCH_SIZE': 8,
    'SAMPLING_REQUIREMENTS': {
        ('Java', 'GPT-4o'): 100,
        ('Java', 'Human'): 100,
    }
}


CONFIG = {
    'NUM_TRAIN_EPOCHS': 10,
    'TRAIN_BATCH_SIZE': 32,
    'EVAL_BATCH_SIZE': 32,
    'SAMPLING_REQUIREMENTS': None
}
