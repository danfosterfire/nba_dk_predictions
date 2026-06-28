PYTHON := .venv/bin/python
PIP    := .venv/bin/pip

.PHONY: venv install fetch preprocess features train evaluate predict test clean

venv:
	/opt/homebrew/bin/python3.14 -m venv .venv

install: venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

fetch:
	$(PYTHON) -m src.data.fetch

preprocess:
	$(PYTHON) -m src.data.preprocess

features:
	$(PYTHON) -m src.features.encode

train:
	$(PYTHON) -m src.train

evaluate:
	$(PYTHON) -m src.evaluate

predict:
	$(PYTHON) -m src.predict

test:
	.venv/bin/pytest tests/ -v

# Run the full pipeline end-to-end
pipeline: fetch preprocess features train evaluate

clean:
	rm -rf data/raw/* data/processed/* data/features/* outputs/checkpoints/* outputs/predictions/*
