import random


def get_random_samples(samples, size: int) -> list:
    rng = random.Random(42)
    sample = rng.sample(samples, size)
    return sample
