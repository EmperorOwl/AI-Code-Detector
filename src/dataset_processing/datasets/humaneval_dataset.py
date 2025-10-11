import os
import pandas as pd
from collections import defaultdict

from src.dataset_processing.datasets.abstract_dataset import AbstractDataset


class HumanEvalDataset(AbstractDataset):
    DATASET_NAME = 'humaneval_dataset'
    PATH = "datasets/humaneval"
    SAMPLING_REQUIREMENTS = {
        ('Java', 'Human'): 450,
        ('Java', 'ChatGPT'): 150,
        ('Java', 'GPT-4'): 150,
        ('Java', 'Gemini Pro'): 150,

        ('Python', 'Human'): 300,
        ('Python', 'ChatGPT'): 100,
        ('Python', 'GPT-4'): 100,
        ('Python', 'Gemini Pro'): 100,
    }

    def load(self) -> pd.DataFrame:
        self.logger.info("Loading dataset from CSV files...")

        dataset_path = os.path.join(self.PATH)
        files = os.listdir(dataset_path)
        csv_files = [f for f in files if f.endswith('.csv')]

        data = []
        for filename in csv_files:
            file_path = os.path.join(dataset_path, filename)

            parts = filename.replace('.csv', '').split('_')
            if len(parts) < 3:
                self.logger.warning(
                    f"Skipping file with unexpected format: {filename}"
                )
                continue

            model_name = parts[1]
            language = parts[2].capitalize()

            model_mapping = {
                'chatgpt': 'ChatGPT',
                'chatgpt4': 'GPT-4',
                'gemini': 'Gemini Pro'
            }

            model = model_mapping.get(model_name, model_name)

            try:
                df = pd.read_csv(file_path)

                for _, row in df.iterrows():
                    label_str = row['label']
                    if label_str == 'human':
                        label = 0
                        actual_model = 'Human'
                    elif label_str == 'lm':
                        label = 1
                        actual_model = model
                    else:
                        self.logger.warning(
                            f"Unknown label '{label_str}' in {filename}"
                        )
                        continue

                    data.append({
                        'Dataset': 'HumanEval',
                        'Code': row['code'],
                        'Language': language,
                        'Model': actual_model,
                        'Label': label
                    })

            except Exception as e:
                self.logger.error(f"Error reading file {filename}: {str(e)}")
                continue

        df = pd.DataFrame(data)
        self.logger.info(f"✓ Loaded HumanEval dataset\n")
        return df

    def info(self, df: pd.DataFrame) -> None:
        pass

    def filter(self, df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info("Filtering dataset...")
        self.logger.info(f"✓ Total samples in dataset: {len(df):,}")

        df = self.helper.add_line_count_column(df)
        df = self.helper.filter_line_count(df)
        self.logger.info("")

        return df

    def standardize(self, df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info("Standardizing dataset...")
        df = df[['Dataset', 'Code', 'Line_Count',
                 'Language', 'Model', 'Label']]
        self.logger.info(
            f"✓ Columns standardized: "
            f"{', '.join(df.columns.tolist())}"
            f"\n"
        )
        return df

    def analyse(self, df: pd.DataFrame) -> None:
        results = defaultdict(lambda: defaultdict(int))

        for _, row in df.iterrows():
            language = row['Language']
            model = row['Model']
            results[language][model] += 1

        self.logger.info(f"{'Language':<10} {'Human':<11} {'ChatGPT':<10} "
                         f"{'GPT-4':<10} {'Gemini Pro':<12} "
                         f"{'Total AI':<12} {'Total':<10}")
        self.logger.info("-" * 80)

        grand_human = 0
        grand_chatgpt = 0
        grand_gpt4 = 0
        grand_gemini = 0
        grand_total_ai = 0
        grand_total = 0

        for language in sorted(results.keys()):
            human_count = results[language].get('Human', 0)
            chatgpt_count = results[language].get('ChatGPT', 0)
            gpt4_count = results[language].get('GPT-4', 0)
            gemini_count = results[language].get('Gemini Pro', 0)

            total_ai = chatgpt_count + gpt4_count + gemini_count
            total_samples = human_count + total_ai

            self.logger.info(f"{language:<10} {human_count:<11,} "
                             f"{chatgpt_count:<10,} {gpt4_count:<10,} "
                             f"{gemini_count:<12,} {total_ai:<12,} "
                             f"{total_samples:<10,}")

            grand_human += human_count
            grand_chatgpt += chatgpt_count
            grand_gpt4 += gpt4_count
            grand_gemini += gemini_count
            grand_total_ai += total_ai
            grand_total += total_samples

        self.logger.info("-" * 80)
        self.logger.info(f"{'TOTAL':<10} {grand_human:<11,} "
                         f"{grand_chatgpt:<10,} {grand_gpt4:<10,} "
                         f"{grand_gemini:<12,} {grand_total_ai:<12,} "
                         f"{grand_total:<10,}")

        if grand_total_ai > 0:
            chatgpt_pct = (f"{(grand_chatgpt / grand_total_ai) * 100:.1f}%"
                           if grand_chatgpt > 0 else "-")
            gpt4_pct = (f"{(grand_gpt4 / grand_total_ai) * 100:.1f}%"
                        if grand_gpt4 > 0 else "-")
            gemini_pct = (f"{(grand_gemini / grand_total_ai) * 100:.1f}%"
                          if grand_gemini > 0 else "-")

            self.logger.info(f"{'% of AI':<10} {'':<11} {chatgpt_pct:<10} "
                             f"{gpt4_pct:<10} {gemini_pct:<12} "
                             f"{'':<12} {'':<10}")

        self.logger.info("-" * 80)
        self.logger.info("")

        self.helper.analyse_line_count(df)


def main():
    dataset = HumanEvalDataset()
    dataset.prepare()


if __name__ == "__main__":
    main()
