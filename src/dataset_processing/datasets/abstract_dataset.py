import time
from abc import ABC, abstractmethod

import pandas as pd

from src.dataset_processing.dataset_helper import DatasetHelper
from src.utils.logger import get_logger
from src.utils.config import DATASET_DIR


class AbstractDataset(ABC):
    DATASET_NAME = 'dataset'
    SAMPLING_REQUIREMENTS = {}

    def __init__(self):
        """
        Initialize the dataset with a logger and helper.
        """
        self.logger = get_logger(self.DATASET_NAME, self.get_log_filepath())
        self.helper = DatasetHelper(self.logger)

    @abstractmethod
    def load(self) -> pd.DataFrame:
        """
        Load the dataset from source.

        Returns:
            pd.DataFrame: Raw dataset
        """
        pass

    @abstractmethod
    def info(self, df: pd.DataFrame) -> None:
        """
        Log original dataset information.

        Args:
            df (pd.DataFrame): Raw dataset
        """
        pass

    @abstractmethod
    def filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter the dataset for target requirements.

        Args:
            df (pd.DataFrame): Raw dataset

        Returns:
            pd.DataFrame: Filtered dataset
        """
        pass

    @abstractmethod
    def standardize(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize the dataset columns and format.

        Args:
            df (pd.DataFrame): Filtered dataset

        Returns:
            pd.DataFrame: Standardized dataset
        """
        pass

    @abstractmethod
    def analyse(self, df: pd.DataFrame) -> None:
        """
        Analyze the dataset and log results.

        Args:
            df (pd.DataFrame): Dataset to analyze
        """
        pass

    def save(self, df: pd.DataFrame) -> None:
        """
        Save the dataset to CSV.

        Args:
            df (pd.DataFrame): Dataset to save
        """
        self.helper.save_dataset_to_csv(df, self.get_save_filepath())

    def prepare(self) -> pd.DataFrame:
        """
        Main workflow to prepare the dataset.
        """
        start_time = time.time()

        # Step 1: Load dataset
        raw_df = self.load()

        # Step 2: Log original dataset info
        self.info(raw_df)

        # Step 3: Filter dataset
        filtered_df = self.filter(raw_df)

        # Step 4: Standardize dataset
        standardized_df = self.standardize(filtered_df)

        # Step 5: Analyze standardized dataset
        self.analyse(standardized_df)

        # Step 6: Sample the dataset
        sampled_df = self.helper.sample_dataset(
            standardized_df,
            self.SAMPLING_REQUIREMENTS
        )

        # Step 7: Analyze sampled dataset
        self.analyse(sampled_df)

        # Step 8: Save the dataset
        self.save(sampled_df)

        # Calculate and print runtime
        end_time = time.time()
        seconds = end_time - start_time
        self.logger.info(f"Runtime: {seconds:.2f} seconds "
                         f"({seconds / 60:.2f} minutes)")

        return sampled_df

    def get_save_filepath(self) -> str:
        """
        Return the save filepath for the dataset.
        """
        return f"{DATASET_DIR}/{self.DATASET_NAME}.csv"

    def get_log_filepath(self) -> str:
        """
        Return the log filepath for the dataset.
        """
        return f"{DATASET_DIR}/{self.DATASET_NAME}.log"
