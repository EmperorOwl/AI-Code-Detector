python -m src.dataset_processing.datasets.droid_dataset
python -m src.dataset_processing.datasets.aig_dataset

python -m src.models.transformer.train --model codebert
python -m src.models.transformer.train --model unixcoder

python -m src.trials.run --trial same_sources
python -m src.trials.run --trial independent_sources