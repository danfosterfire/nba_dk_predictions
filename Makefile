PYTHON := .venv/bin/python
PIP    := .venv/bin/pip

.PHONY: venv install fetch preprocess features train evaluate predict test clean \
        season-matrix pca archetypes eda team-context component-targets context-value \
        opponent persistence aging target-profile feature-diagnostics dashboard

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

season-matrix:
	$(PYTHON) -m src.eda.season_matrix

pca:
	$(PYTHON) -m src.eda.pca

archetypes:
	$(PYTHON) -m src.eda.archetypes

team-context:
	$(PYTHON) -m src.features.team_context

component-targets:
	$(PYTHON) -m src.features.targets

context-value:
	$(PYTHON) -m src.eda.context_value

opponent:
	$(PYTHON) -m src.features.opponent

persistence:
	$(PYTHON) -m src.eda.persistence

aging:
	$(PYTHON) -m src.eda.aging

target-profile:
	$(PYTHON) -m src.eda.target

feature-diagnostics:
	$(PYTHON) -m src.eda.feature_diagnostics

# Full EDA sweep, in dependency order
eda: season-matrix pca archetypes team-context context-value opponent \
     component-targets persistence aging target-profile feature-diagnostics

dashboard:
	.venv/bin/streamlit run dashboard/app.py

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
