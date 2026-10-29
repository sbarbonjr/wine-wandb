.PHONY: install data train sweep evaluate decision test

install:
	pip install -r requirements.txt && pip install -e .

data:
	python -c "from wine_wandb import data; print(data.download('data/winequality-red.csv'))"

train:
	python -m wine_wandb.train

sweep:
	python -m wine_wandb.sweep

# usage: make evaluate ARTIFACT=wine-pipeline:v3
evaluate:
	python -m wine_wandb.evaluate --artifact $(ARTIFACT)

decision:
	python -m wine_wandb.decision

test:
	WANDB_MODE=disabled pytest
