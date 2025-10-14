# Prepare datasets
python -m src.dataset_processing.prepare_all

# Train models
python -m src.models.transformer.train --model codebert
python -m src.models.transformer.train --model unixcoder
python -m src.models.classifiers.train

# Run trials
python -m src.trials.run --trial same_sources --model codebert
python -m src.trials.run --trial same_sources --model unixcoder
python -m src.trials.run --trial same_sources --model embedding
python -m src.trials.run --trial independent_sources --model codebert
python -m src.trials.run --trial independent_sources --model unixcoder
python -m src.trials.run --trial independent_sources --model embedding