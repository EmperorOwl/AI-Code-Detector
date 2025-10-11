import os
import pandas as pd

from src.dataset_processing.datasets.abstract_dataset import AbstractDataset
from src.utils.config import DATASET_DIR


class SnifferDataset(AbstractDataset):
    DATASET_NAME = 'sniffer_dataset'
    PATH = "datasets/gptsniffer"
    SAMPLING_REQUIREMENTS = {
        ('Java', 'Human'): 500,
        ('Java', 'ChatGPT'): 500,
    }

    def load(self) -> pd.DataFrame:
        self.logger.info("Loading dataset from local directory...")

        dataset_path = os.path.join(self.PATH)
        files = os.listdir(dataset_path)

        data = []
        for filename in files:
            file_path = os.path.join(dataset_path, filename)

            if filename.startswith('0_'):
                label = 1
                model = 'ChatGPT'
            elif filename.startswith('1_'):
                label = 0
                model = 'Human'
            else:
                self.logger.warning(
                    f"Skipping file with unexpected prefix: {filename}"
                )
                continue

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    code = f.read()

                data.append({
                    'Dataset': 'Sniffer',
                    'Code': code,
                    'Language': 'Java',
                    'Model': model,
                    'Label': label
                })

            except Exception as e:
                self.logger.error(f"Error reading file {filename}: {str(e)}")
                continue

        df = pd.DataFrame(data)
        self.logger.info(f"✓ Loaded GPTSniffer dataset\n")
        return df

    def info(self, df: pd.DataFrame) -> None:
        pass

    def filter(self, df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info("Filtering GPTSniffer dataset...")
        self.logger.info(f"✓ Total samples in dataset: {len(df):,}")

        df = self.helper.add_line_count_column(df)
        df = self.helper.filter_line_count(df)
        self.logger.info("")

        return df

    def standardize(self, df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info("Standardizing dataset...")
        self.helper.add_id_column(df)
        df = df[['ID', 'Dataset', 'Code', 'Line_Count',
                 'Language', 'Model', 'Label']]
        self.logger.info(
            f"✓ Dataset standardized with columns "
            f"{', '.join(df.columns.tolist())}"
            f"\n"
        )
        return df

    def analyse(self, df: pd.DataFrame) -> None:
        human_count = len(df[df['Model'] == 'Human'])
        chatgpt_count = len(df[df['Model'] == 'ChatGPT'])
        total_count = len(df)

        self.logger.info(f"{'Language':<10} {'Human':<10} {'ChatGPT':<10} "
                         f"{'Total':<10}")
        self.logger.info("-" * 50)

        self.logger.info(f"{'Java':<10} {human_count:<10,} "
                         f"{chatgpt_count:<10,} {total_count:<10,}")

        self.logger.info("-" * 50)
        self.logger.info(f"{'TOTAL':<10} {human_count:<10,} "
                         f"{chatgpt_count:<10,} {total_count:<10,}")
        self.logger.info("-" * 50)
        self.logger.info("")

        self.helper.analyse_line_count(df)


def main():
    dataset = SnifferDataset()
    dataset.prepare()


if __name__ == "__main__":
    main()
