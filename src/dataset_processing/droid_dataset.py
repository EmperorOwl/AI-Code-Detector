import time
from collections import defaultdict
from logging import Logger

import pandas as pd
from datasets import load_dataset as load_dataset_from_huggingface

from src.dataset_processing.dataset_helper import DatasetHelper


class DroidDataset:
    PATH = "project-droid/DroidCollection"
    SAMPLING_REQUIREMENTS = {
        ('Java', 'Human'): 12500,
        ('Java', 'GPT-4o'): 4000,
        ('Java', 'GPT-4o mini'): 7000,
        ('Java', 'DeepSeek'): 1500,

        ('Python', 'Human'): 12500,
        ('Python', 'GPT-4o'): 4000,
        ('Python', 'GPT-4o mini'): 7000,
        ('Python', 'DeepSeek'): 1500,
    }

    def __init__(self, logger: Logger):
        """
        Initialize the DroidDataset.

        Args:
            logger (Logger): Logger instance for all operations
        """
        self.logger = logger
        self.helper = DatasetHelper(logger)

    def load(self) -> pd.DataFrame:
        """
        Load the DroidCollection dataset from Hugging Face.

        Returns:
            pd.DataFrame: Raw dataset from all splits combined
        """
        self.logger.info("Loading DroidCollection dataset...")

        # Load the dataset
        dataset = load_dataset_from_huggingface(DroidDataset.PATH)

        # Convert all splits to pandas DataFrames and combine them
        train_df = pd.DataFrame(dataset['train'])  # type: ignore
        dev_df = pd.DataFrame(dataset['dev'])  # type: ignore
        test_df = pd.DataFrame(dataset['test'])  # type: ignore

        # Combine all splits into one DataFrame
        df = pd.concat([train_df, dev_df, test_df], ignore_index=True)

        # Print original dataset information
        self.logger.info(f"\nDataset columns:")
        for column in df.columns.tolist():
            self.logger.info(f"  - {column}")

        self.logger.info(f"\nLanguages available:")
        for language in sorted(df['Language'].unique()):
            self.logger.info(f"  - {language}")

        self.logger.info(f"\nLabels available:")
        for label in sorted(df['Label'].unique()):
            self.logger.info(f"  - {label}")

        self.logger.info(f"\nModel families available:")
        for model_family in sorted(df['Model_Family'].unique()):
            self.logger.info(f"  - {model_family}")

        self.logger.info(f"\nGenerators available:")
        for generator in sorted(df['Generator'].unique()):
            self.logger.info(f"  - {generator}")
        self.logger.info("")

        return df

    def filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter the dataset for target languages, models, and labels.

        Args:
            df (pd.DataFrame): Raw dataset

        Returns:
            pd.DataFrame: Filtered dataset with standardized columns:
                - Code: The actual code
                - Language: Programming language ('Python', 'Java')
                - Model: Model name ('Human', 'GPT-4o', 'GPT-4o mini', etc)
                - Label: 0 for human, 1 for AI
        """
        self.logger.info("Filtering DroidCollection dataset...")
        self.logger.info(f"✓ Total samples in dataset: {len(df):,}")

        # Filter for specific languages
        language_filter = df['Language'].isin(['Python', 'Java'])
        filtered_df = df[language_filter]
        self.logger.info(
            f"✓ Samples after language filtering (Python/Java): "
            f"{len(filtered_df):,}"
        )

        # Filter for specific model families
        model_family_filter = filtered_df['Model_Family'].isin([
            'human', 'gpt-4o', 'gpt-4o-mini', 'deepseek-ai'
        ])
        filtered_df = filtered_df[model_family_filter]
        self.logger.info(
            f"✓ Samples after model filtering "
            f"(Human/GPT-4o/GPT-4o mini/DeepSeek): {len(filtered_df):,}"
        )

        # Filter for specific labels
        label_filter = filtered_df['Label'].isin([
            'HUMAN_GENERATED', 'MACHINE_GENERATED'
        ])
        filtered_df = filtered_df[label_filter]
        self.logger.info(
            f"✓ Samples after label filtering "
            f"(HUMAN_GENERATED/MACHINE_GENERATED): {len(filtered_df):,}"
        )

        # Create standardized dataframe
        standardized_df = pd.DataFrame()
        standardized_df['Code'] = filtered_df['Code']
        standardized_df['Language'] = filtered_df['Language']

        # Standardize model names
        model_mapping = {
            'gpt-4o': 'GPT-4o',
            'gpt-4o-mini': 'GPT-4o mini',
            'deepseek-ai': 'DeepSeek',
            'human': 'Human'
        }
        standardized_df['Model'] = filtered_df['Model_Family'].map(
            model_mapping)

        # Standardize Label column (0 for human, 1 for AI)
        standardized_df['Label'] = (
            filtered_df['Label'] == 'MACHINE_GENERATED'
        ).astype(int)

        self.logger.info("")

        return standardized_df

    def analyse(self, df: pd.DataFrame) -> None:
        """
        Analyze the dataset and print results in table format.

        Args:
            df (pd.DataFrame): Dataset to analyze
        """
        results = defaultdict(lambda: defaultdict(int))

        for _, row in df.iterrows():
            language = row['Language']
            model = row['Model']
            is_ai = row['Label'] == 1

            if is_ai:
                results[language][f"{model}_MACHINE_GENERATED"] += 1
            else:
                results[language]["Human_HUMAN_GENERATED"] += 1

        # Table header - exactly 80 characters wide
        self.logger.info(f"{'Language':<10} {'Human':<11} {'GPT-4o':<10} "
                         f"{'GPT-4o mini':<12} {'DeepSeek':<10} "
                         f"{'Total AI':<12} {'Total':<10}")
        self.logger.info("-" * 80)

        # Initialize grand totals
        grand_human = 0
        grand_gpt4o = 0
        grand_gpt4o_mini = 0
        grand_deepseek = 0
        grand_total_ai = 0
        grand_total = 0

        for language in sorted(results.keys()):
            # Human-written samples
            human_generated = results[language].get('Human_HUMAN_GENERATED', 0)

            # AI-generated samples
            gpt4o_ai = results[language].get('GPT-4o_MACHINE_GENERATED', 0)
            gpt4o_mini_ai = results[language].get(
                'GPT-4o mini_MACHINE_GENERATED', 0)
            deepseek_ai = results[language].get(
                'DeepSeek_MACHINE_GENERATED', 0)

            total_ai = gpt4o_ai + gpt4o_mini_ai + deepseek_ai
            total_samples = human_generated + total_ai

            self.logger.info(f"{language:<10} {human_generated:<11,} "
                             f"{gpt4o_ai:<10,} {gpt4o_mini_ai:<12,} "
                             f"{deepseek_ai:<10,} {total_ai:<12,} "
                             f"{total_samples:<10,}")

            # Add to grand totals
            grand_human += human_generated
            grand_gpt4o += gpt4o_ai
            grand_gpt4o_mini += gpt4o_mini_ai
            grand_deepseek += deepseek_ai
            grand_total_ai += total_ai
            grand_total += total_samples

        self.logger.info("-" * 80)
        self.logger.info(f"{'TOTAL':<10} {grand_human:<11,} {grand_gpt4o:<10,} "
                         f"{grand_gpt4o_mini:<12,} {grand_deepseek:<10,} "
                         f"{grand_total_ai:<12,} {grand_total:<10,}")

        # Add % of AI row
        if grand_total_ai > 0:
            gpt4o_pct = (f"{(grand_gpt4o / grand_total_ai) * 100:.1f}%"
                         if grand_gpt4o > 0 else "-")
            gpt4o_mini_pct = (f"{(grand_gpt4o_mini / grand_total_ai) * 100:.1f}%"
                              if grand_gpt4o_mini > 0 else "-")
            deepseek_pct = (f"{(grand_deepseek / grand_total_ai) * 100:.1f}%"
                            if grand_deepseek > 0 else "-")

            self.logger.info(f"{'% of AI':<10} {'':<11} {gpt4o_pct:<10} "
                             f"{gpt4o_mini_pct:<12} {deepseek_pct:<10} "
                             f"{'':<12} {'':<10}")

        self.logger.info("-" * 80)
        self.logger.info("")

    def prepare(self) -> None:
        """
        Main workflow to prepare the dataset.

        Steps:
        1. Load the DroidCollection dataset
        2. Filter the dataset
        3. Analyze the filtered dataset
        4. Sample according to requirements
        5. Analyze the sampled dataset
        6. Add ID column
        7. Save to CSV
        """
        start_time = time.time()

        # Step 1: Load dataset
        raw_df = self.load()

        # Step 2: Filter dataset
        filtered_df = self.filter(raw_df)

        # Step 3: Analyze filtered dataset
        self.analyse(filtered_df)

        # Step 4: Sample the droid dataset
        sampled_df = self.helper.sample_dataset(
            filtered_df,
            DroidDataset.SAMPLING_REQUIREMENTS
        )

        # Step 5: Analyze sampled dataset
        self.analyse(sampled_df)

        # Step 6: Add ID column
        sampled_df = self.helper.add_id_column(sampled_df)

        # Step 7: Save to CSV
        self.helper.save_dataset_to_csv(sampled_df)

        # Calculate and print runtime
        end_time = time.time()
        seconds = end_time - start_time
        self.logger.info(f"Runtime: {seconds:.2f} seconds "
                         f"({seconds / 60:.2f} minutes)")


def main():
    from src.utils.logger import get_logger

    logger = get_logger("droid_dataset", "outputs/droid_dataset.log")
    droid = DroidDataset(logger)
    droid.prepare()


if __name__ == "__main__":
    main()
