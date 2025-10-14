# Prepare datasets
python -m src.dataset_processing.prepare_all

# Train models
python -m src.models.transformer.train --model codebert
python -m src.models.transformer.train --model unixcoder
python -m src.models.classifiers.train --classifier simple
python -m src.models.classifiers.train --classifier xgboost

# Run trials
python -m src.trials.run --trial same_sources --model codebert
python -m src.trials.run --trial same_sources --model unixcoder
python -m src.trials.run --trial same_sources --model simple
python -m src.trials.run --trial same_sources --model xgboost
python -m src.trials.run --trial independent_sources --model codebert
python -m src.trials.run --trial independent_sources --model unixcoder
python -m src.trials.run --trial independent_sources --model simple
python -m src.trials.run --trial independent_sources --model xgboost