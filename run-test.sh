python -m src.dataset_processing.datasets.droid_dataset
python -m src.dataset_processing.datasets.aig_dataset

python -m src.models.transformer.train --model codebert --test
python -m src.models.transformer.train --model unixcoder --test

python -m src.trials.same_dataset --test
