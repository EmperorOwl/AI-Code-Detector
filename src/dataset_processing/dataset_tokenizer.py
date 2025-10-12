from logging import Logger

import pandas as pd
import numpy as np
from transformers import AutoTokenizer, PreTrainedTokenizer
from tqdm import tqdm

from src.models.transformer import TransformerModel
from src.dataset_processing.ast_processor import AstProcessor
from src.utils import config


class DatasetTokenizer:

    def __init__(self,
                 logger: Logger,
                 model_class: type[TransformerModel],
                 use_ast: bool = False) -> None:
        """
        Initialize the DatasetTokenizer.

        Args:
            logger (Logger): Logger instance for all operations
            model_class (type[TransformerModel]): Transformer model class
        """
        self.logger = logger
        self.model_name = model_class.PRETRAINED_MODEL_NAME
        self.max_length = model_class.MAX_LENGTH
        self.tokenizer = self.setup_tokenizer()
        self.use_ast = use_ast
        self.ast_processor = AstProcessor(logger) if use_ast else None

    def setup_tokenizer(self) -> PreTrainedTokenizer:
        """
        Initialize a tokenizer for a given model.

        Returns:
            PreTrainedTokenizer: Configured tokenizer
        """
        self.logger.info("Setting up tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.logger.info(f"✓ Tokenizer for model {self.model_name} loaded")

        self.logger.info(f"\nConfiguration:")
        self.logger.info(
            f"  - Max length: {self.max_length}/{tokenizer.model_max_length}"
        )
        self.logger.info(f"  - Vocab size: {tokenizer.vocab_size:,}")

        self.logger.info(f"\nSpecial tokens:")
        for token_name, token_value in tokenizer.special_tokens_map.items():
            self.logger.info(f"  - {token_name}: {token_value}")
        self.logger.info("")

        return tokenizer

    def tokenize_code_samples(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Tokenize all code samples in the dataset.

        Args:   
            df (pd.DataFrame): Dataset to tokenize

        Returns:
            pd.DataFrame:
                Dataset with added input_ids and attention_mask columns
        """
        self.logger.info(
            f"Tokenizing code samples (use_ast: {self.use_ast})..."
        )

        # Initialize lists to store tokenized data
        input_ids_list = []
        attention_mask_list = []

        # Tokenize with progress bar
        for index, code in tqdm(df['Code'].items(),
                                total=len(df),
                                desc="Progress"):

            # Check if we are using the AST representation
            if self.use_ast:
                to_encode = self.ast_processor.generate_ast_sequence(
                    code,
                    df['Language'].iloc[index].lower()
                )
            else:
                to_encode = code

            # Tokenize the AST sequenece or the code
            encoded = self.tokenizer(
                to_encode,
                padding='max_length',
                truncation=True,
                max_length=self.max_length,
            )

            input_ids_list.append(encoded['input_ids'])
            attention_mask_list.append(encoded['attention_mask'])

        # Add tokenized data to dataframe
        tokenized_df = df.copy()
        tokenized_df['input_ids'] = input_ids_list
        tokenized_df['attention_mask'] = attention_mask_list

        self.logger.info(f"✓ {len(tokenized_df):,} samples tokenized\n")

        return tokenized_df

    def analyze_tokenization(self, df: pd.DataFrame) -> None:
        """
        Log tokenization results.

        Args:
            df (pd.DataFrame): Tokenized dataset to analyze
        """
        # Convert input_ids to numpy array for vectorized operations
        input_ids_array = np.array(df['input_ids'].tolist())

        # Calculate token lengths using vectorized operations
        # Count non-padding tokens for each sample
        token_lengths = np.sum(
            input_ids_array != self.tokenizer.pad_token_id, axis=1
        )

        # Calculate statistics using numpy functions
        min_length = int(np.min(token_lengths))
        max_length = int(np.max(token_lengths))
        average_length = float(np.mean(token_lengths))
        median_length = int(np.median(token_lengths))

        self.logger.info(f"\nToken length statistics:")
        self.logger.info(f"  - Min length: {min_length}")
        self.logger.info(f"  - Max length: {max_length}")
        self.logger.info(f"  - Average length: {average_length:.1f}")
        self.logger.info(f"  - Median length: {median_length}")

        # Show truncation statistics
        max_seq_length = input_ids_array.shape[1]
        truncated_count = int(np.sum(token_lengths == max_seq_length))
        truncated_percentage = truncated_count / len(df) * 100
        not_truncated_count = len(df) - truncated_count
        not_truncated_percentage = not_truncated_count / len(df) * 100

        self.logger.info(f"\nTruncation statistics:")
        self.logger.info(f"  - Samples truncated: {truncated_count:,} "
                         f"({truncated_percentage:.1f}%)")
        self.logger.info(f"  - Samples not truncated: {not_truncated_count:,} "
                         f"({not_truncated_percentage:.1f}%)\n")


def main():
    """ Example usage """
    from src.utils.logger import get_logger
    from src.dataset_processing.dataset_helper import DatasetHelper
    from src.models.transformer import CodeBertModel, UniXcoderModel

    # Initialize logger
    log_file_path = f"{config.OUTPUT_DIR}/dataset_tokenizer.log"
    logger = get_logger("dataset_tokenizer", log_file_path)

    # Load dataset
    helper = DatasetHelper(logger)
    df = helper.load_dataset_from_csv(f'{config.DROID_PATH}')

    tokenizer = DatasetTokenizer(
        logger=logger,
        model_class=UniXcoderModel,
        use_ast=True
    )
    df = tokenizer.tokenize_code_samples(df)
    tokenizer.analyze_tokenization(df)


if __name__ == "__main__":
    main()
