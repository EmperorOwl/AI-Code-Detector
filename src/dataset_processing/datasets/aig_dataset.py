import pandas as pd

from src.dataset_processing.datasets.abstract_dataset import AbstractDataset


class AIGDataset(AbstractDataset):
    DATASET_NAME = 'aig_dataset'
    PATH = "basakdemirok/AIGCodeSet"
    SAMPLING_REQUIREMENTS = {
        ('Python', 'Human'): 1500,
        ('Python', 'Gemini Flash'): 1500,
    }

    def load(self) -> pd.DataFrame:
        # Load the dataset from Hugging Face
        df = self.helper.load_dataset_from_huggingface(
            self.PATH,
            splits=['train', 'test']
        )
        return df

    def info(self, df: pd.DataFrame) -> None:
        self.logger.info(f"Dataset columns:")
        for column in df.columns.tolist():
            self.logger.info(f"  - {column}")

        self.logger.info(f"\nModels available:")
        for model in sorted(df['LLM'].unique()):
            self.logger.info(f"  - {model}")

        self.logger.info(f"\nLabels available:")
        for label in sorted(df['label'].unique()):
            self.logger.info(f"  - {label}")
        self.logger.info("")

    def filter(self, df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info("Filtering dataset...")
        self.logger.info(f"✓ Total samples in dataset: {len(df):,}")

        # Filter for specific models
        model_filter = df['LLM'].isin([
            'Human', 'GEMINI'
        ])
        filtered_df = df[model_filter]
        self.logger.info(
            f"✓ Samples after model filtering "
            f"(Human/GEMINI): {len(filtered_df):,}"
        )

        # Filter line count
        filtered_df = self.helper.add_line_count_column(
            filtered_df,
            code_column='code'
        )
        filtered_df = self.helper.filter_line_count(filtered_df)
        self.logger.info("")

        return filtered_df

    def standardize(self, df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info("Standardizing dataset...")

        # Create standardized dataframe
        df = df.reset_index(drop=True)
        standardized_df = pd.DataFrame()
        standardized_df['Dataset'] = ['AIG'] * len(df)
        standardized_df['Code'] = df['code']
        standardized_df['Line_Count'] = df['Line_Count']
        standardized_df['Language'] = 'Python'
        standardized_df['Model'] = df['LLM'].map({
            'Human': 'Human',
            'GEMINI': 'Gemini Flash'
        })  # Standardize model names
        standardized_df['Label'] = df['label']

        self.helper.add_id_column(standardized_df)

        self.logger.info(
            f"✓ Dataset standardized with columns "
            f"{', '.join(standardized_df.columns.tolist())}"
        )
        self.logger.info("")
        return standardized_df

    def analyse(self, df: pd.DataFrame) -> None:
        human_count = len(df[df['Model'] == 'Human'])
        gemini_count = len(df[df['Model'] == 'Gemini Flash'])
        total_count = len(df)

        # Table header
        self.logger.info(f"{'Language':<10} {'Human':<10} {'Gemini Flash':<15} "
                         f"{'Total':<10}")
        self.logger.info("-" * 50)

        # Python row
        self.logger.info(f"{'Python':<10} {human_count:<10,} "
                         f"{gemini_count:<15,} {total_count:<10,}")

        # Total row
        self.logger.info("-" * 50)
        self.logger.info(f"{'TOTAL':<10} {human_count:<10,} "
                         f"{gemini_count:<15,} {total_count:<10,}")
        self.logger.info("-" * 50)
        self.logger.info("")

        self.helper.analyse_line_count(df)


def main():
    dataset = AIGDataset()
    dataset.prepare()


if __name__ == "__main__":
    main()
