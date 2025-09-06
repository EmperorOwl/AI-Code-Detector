import random
from sklearn.model_selection import train_test_split

from src.pre_processing.sample import Sample


def get_random_samples(samples: list[Sample], size: int) -> list:
    """ Randomly sample from the list of samples while maintaining the original 
    ratio of labels.
    """
    # Extract labels for stratification
    labels = [sample.label for sample in samples]

    # Calculate the fraction to sample
    sample_fraction = size / len(samples)

    # Use stratified sampling to maintain label distribution
    sampled_samples, _, _, _ = train_test_split(
        samples,
        labels,
        train_size=sample_fraction,
        stratify=labels,
        random_state=42
    )

    return sampled_samples
