import random
from logging import Logger

import pandas as pd


class DatasetHelper:

    def __init__(self, logger: Logger):
        """
        Initialize the DatasetHelper.

        Args:
            logger (Logger): Logger instance for all operations
        """
        self.logger = logger

    def load_dataset_from_csv(self,
                              file_path: str = 'dataset.csv') -> pd.DataFrame:
        """ 
        Load dataset from CSV file as a pandas DataFrame.

        Args:
            file_path (str): Path to the dataset CSV file

        Returns:
            pd.DataFrame: Loaded dataset
        """
        self.logger.info(f"Loading dataset...")
        df = pd.read_csv(file_path)
        self.logger.info(f"✓ Loaded {len(df):,} samples\n")

        self.logger.info(f"Dataset columns:")
        for column in df.columns:
            self.logger.info(f"  - {column}")
        self.logger.info("")

        return df

    def save_dataset_to_csv(self,
                            df: pd.DataFrame,
                            file_path: str = 'dataset.csv') -> None:
        """
        Save the dataset to a CSV file.

        Args:
            df (pd.DataFrame): Dataset to save
            file_path (str): Path to save the dataset
        """
        self.logger.info(f"Saving dataset...")
        df.to_csv(file_path, index=False)
        self.logger.info(f"✓ Dataset saved as '{file_path}'\n")

    def split_dataset(self,
                      df: pd.DataFrame,
                      train_ratio: float = 0.8,
                      val_ratio: float = 0.1,
                      test_ratio: float = 0.1,
                      random_state: int = 42
                      ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split dataset into train, validation, and test sets.
        Splits each language-model combination individually to maintain ratios.

        Args:
            df (pd.DataFrame): Dataset DataFrame
            train_ratio (float): Proportion for training set
            val_ratio (float): Proportion for validation set
            test_ratio (float): Proportion for test set
            random_state (int): Random seed for reproducibility

        Returns:
            tuple: (train_df, val_df, test_df)
        """
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("Split ratios must sum to 1.0")

        self.logger.info(f"Splitting dataset...")

        # Set random seed for reproducibility
        random.seed(random_state)

        # Get all unique combinations of language and model
        combinations = df[['Language', 'Model']].drop_duplicates()

        train_dfs = []
        val_dfs = []
        test_dfs = []

        # Split each combination individually
        for _, row in combinations.iterrows():
            language, model = row['Language'], row['Model']

            # Filter samples for this combination
            combo_mask = (df['Language'] == language) & (df['Model'] == model)
            combo_df = df[combo_mask].copy()

            # Shuffle samples for this combination
            combo_shuffled = combo_df.sample(
                frac=1, random_state=random_state
            ).reset_index(drop=True)

            total_combo = len(combo_shuffled)
            train_end = int(total_combo * train_ratio)
            val_end = train_end + int(total_combo * val_ratio)

            combo_train = combo_shuffled.iloc[:train_end]
            combo_val = combo_shuffled.iloc[train_end:val_end]
            combo_test = combo_shuffled.iloc[val_end:]

            train_dfs.append(combo_train)
            val_dfs.append(combo_val)
            test_dfs.append(combo_test)

        # Combine all splits
        train_df = pd.concat(train_dfs, ignore_index=True)
        val_df = pd.concat(val_dfs, ignore_index=True)
        test_df = pd.concat(test_dfs, ignore_index=True)

        total = len(train_df) + len(val_df) + len(test_df)
        self.logger.info(
            f"✓ Created train dataset with {len(train_df):,} samples "
            f"({len(train_df) / total:.1%})"
        )
        self.logger.info(
            f"✓ Created validation dataset with {len(val_df):,} samples "
            f"({len(val_df) / total:.1%})"
        )
        self.logger.info(
            f"✓ Created test dataset with {len(test_df):,} samples "
            f"({len(test_df) / total:.1%})"
        )
        self.logger.info("")

        return train_df, val_df, test_df

    def log_dataset_splits_table(self,
                                 df: pd.DataFrame,
                                 train_df: pd.DataFrame | None = None,
                                 val_df: pd.DataFrame | None = None,
                                 test_df: pd.DataFrame | None = None) -> None:
        """
        Print dataset statistics in a table format.

        Args:
            df (pd.DataFrame): Full dataset
            train_df (pd.DataFrame | None): Training dataset
            val_df (pd.DataFrame | None): Validation dataset
            test_df (pd.DataFrame | None): Test dataset
        """
        if len(df) == 0:
            self.logger.info("No samples to analyze")
            return

        # Create table header
        self.logger.info(
            f"{'Dataset':<10} | {'Language':<8} | {'Model':<12} | "
            f"{'Total':<8} | {'Train':<8} | {'Val':<8} | {'Test':<8}"
        )
        self.logger.info("-" * 80)

        # Get all unique combinations of language and model
        combinations = df[['Language', 'Model']].drop_duplicates()

        # Calculate statistics for each combination
        for _, row in combinations.iterrows():
            language, model = row['Language'], row['Model']

            # Filter samples for this combination
            combo_mask = (df['Language'] == language) & (df['Model'] == model)
            total_count = combo_mask.sum()

            # Calculate train/val/test counts if splits provided
            train_count = 0
            val_count = 0
            test_count = 0

            if train_df is not None:
                train_mask = (train_df['Language'] == language) & (
                    train_df['Model'] == model)
                train_count = train_mask.sum()

            if val_df is not None:
                val_mask = (val_df['Language'] == language) & (
                    val_df['Model'] == model)
                val_count = val_mask.sum()

            if test_df is not None:
                test_mask = (test_df['Language'] == language) & (
                    test_df['Model'] == model)
                test_count = test_mask.sum()

            # Format counts (show "-" if no split provided)
            train_str = f"{train_count:,}" if train_df is not None else "-"
            val_str = f"{val_count:,}" if val_df is not None else "-"
            test_str = f"{test_count:,}" if test_df is not None else "-"

            # Print row
            name = "Droid"
            self.logger.info(
                f"{name:<10} | {language:<8} | {model:<12} | {total_count:<8,} | "
                f"{train_str:<8} | {val_str:<8} | {test_str:<8}"
            )

        # Print totals row
        total_all = len(df)
        total_train = len(train_df) if train_df is not None else 0
        total_val = len(val_df) if val_df is not None else 0
        total_test = len(test_df) if test_df is not None else 0

        train_total_str = f"{total_train:,}" if train_df is not None else "-"
        val_total_str = f"{total_val:,}" if val_df is not None else "-"
        test_total_str = f"{total_test:,}" if test_df is not None else "-"

        self.logger.info("-" * 80)
        self.logger.info(
            f"{'TOTAL':<36} | {total_all:<8,} | "
            f"{train_total_str:<8} | {val_total_str:<8} | {test_total_str:<8}"
        )
        self.logger.info("")

    def sample_dataset(self,
                       dataset_df: pd.DataFrame,
                       sampling_requirements: dict | None,
                       random_state: int = 42) -> pd.DataFrame:
        """
        Sample a dataset according to specified requirements.

        Args:
            dataset_df (pd.DataFrame): Standardized dataset with columns:
                - Code: The actual code
                - Language: Programming language (e.g., 'Python', 'Java')
                - Model: Model name ('Human', 'GPT-4o', 'GPT-4o mini', etc)
                - Label: 0 for human, 1 for AI
            sampling_requirements (dict | None): Dictionary mapping 
                (language, model) tuples to required sample counts 
                (if None, no sampling is done)
            random_state (int): Random seed for reproducible sampling

        Returns:
            pd.DataFrame: Sampled dataset ready for ML training
        """
        if sampling_requirements is None:
            return dataset_df

        self.logger.info("Sampling dataset...")

        sampled_dfs = []

        for (language, model), required in sampling_requirements.items():
            # Create mask for this language-model combination
            if model == 'Human':
                mask = ((dataset_df['Model'] == model) &
                        (dataset_df['Language'] == language) &
                        (dataset_df['Label'] == 0))
            else:
                mask = ((dataset_df['Model'] == model) &
                        (dataset_df['Language'] == language) &
                        (dataset_df['Label'] == 1))

            subset = dataset_df[mask]
            available = len(subset)

            if available < required:
                raise ValueError(
                    f"Insufficient samples for {language} {model}: "
                    f"only {available:,} available, need {required:,}"
                )

            # Random sample
            sampled = subset.sample(n=required, random_state=random_state)
            sampled_dfs.append(sampled)
            percentage = (required / available) * 100
            self.logger.info(
                f"✓ Sampled {required:,}/{available:,} {language} {model} "
                f"samples ({percentage:.1f}%)"
            )

        # Combine all sampled data
        final_df = pd.concat(sampled_dfs, ignore_index=True)

        self.logger.info("")

        return final_df

    def add_id_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add an ID column as first column in the dataset.

        Args:
            df (pd.DataFrame): Dataset to add ID column to

        Returns:
            pd.DataFrame: Dataset with ID column added
        """
        df.insert(0, 'ID', range(len(df)))
        return df
