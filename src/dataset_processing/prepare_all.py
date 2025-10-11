from src.dataset_processing.datasets.droid_dataset import DroidDataset
from src.dataset_processing.datasets.aig_dataset import AIGDataset
from src.dataset_processing.datasets.sniffer_dataset import SnifferDataset
from src.dataset_processing.datasets.humaneval_dataset import HumanEvalDataset
from src.dataset_processing.datasets.mbpp_dataset import MBPPDataset


def main():
    droid_dataset = DroidDataset()
    droid_dataset.prepare()

    aig_dataset = AIGDataset()
    aig_dataset.prepare()

    sniffer_dataset = SnifferDataset()
    sniffer_dataset.prepare()

    humaneval_dataset = HumanEvalDataset()
    humaneval_dataset.prepare()

    mbpp_dataset = MBPPDataset()
    mbpp_dataset.prepare()


if __name__ == "__main__":
    main()
