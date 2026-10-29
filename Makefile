.PHONY: install data train train-classifier sweep evaluate evaluate-premium figures test

install:
	pip install -r requirements.txt && pip install -e .

data:
	python -c "from wine_wandb import data; print(data.download('data/winequality-red.csv'))"

train:
	python -m wine_wandb.train

train-classifier:
	python -m wine_wandb.train_classifier

sweep:
	python -m wine_wandb.sweep

# usage: make evaluate ARTIFACT=wine-pipeline:v3
evaluate:
	python -m wine_wandb.evaluate --artifact $(ARTIFACT)

# usage: make evaluate-premium ARTIFACT=wine-premium:v0
evaluate-premium:
	python -m wine_wandb.evaluate --task premium --artifact $(ARTIFACT)

figures:
	python -m wine_wandb.figures

test:
	WANDB_MODE=disabled pytest
