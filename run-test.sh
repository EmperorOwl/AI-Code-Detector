python -m src.dataset_processing.prepare_all

python -m src.models.transformer.train --model codebert --test
python -m src.models.transformer.train --model unixcoder --test

python -m src.trials.run --trial same_sources --test
python -m src.trials.run --trial independent_sources --test