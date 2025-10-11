from src.dataset_processing.datasets.droid_dataset import DroidDataset
from src.dataset_processing.datasets.aig_dataset import AIGDataset
from src.dataset_processing.datasets.sniffer_dataset import SnifferDataset


def main():
    droid_dataset = DroidDataset()
    droid_dataset.prepare()

    aig_dataset = AIGDataset()
    aig_dataset.prepare()

    sniffer_dataset = SnifferDataset()
    sniffer_dataset.prepare()


if __name__ == "__main__":
    main()
