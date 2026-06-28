# NBA Deep Learning

Predicts a player's DraftKings fantasy points (dk_pts) for each game of the current season using their prior-season game logs as input. The model (LSTM or Transformer) reads the player's last N games from the prior season and outputs an expected dk_pts per game.

## Project layout

```
src/data/      fetch, preprocess, dataset
src/features/  rolling stats, matchup context, encoding
src/models/    lstm, transformer, xgboost baseline
src/train.py   training loop
src/evaluate.py test-split metrics
src/predict.py  inference entry point
configs/       default.yaml — all hyperparams and paths
data/          raw → processed → features pipeline
outputs/       checkpoints and prediction CSVs
```

## Pipeline (run in order)

```bash
python -m src.data.fetch           # pull raw game logs from nba_api
python -m src.data.preprocess      # clean + add season column → data/processed/game_logs.parquet
python -m src.train                # train model → outputs/checkpoints/best_model.pt
                                   #   also saves scaler/feature_cols to data/features/
python -m src.evaluate             # test MAE/RMSE → outputs/predictions/test_metrics.csv
python -m src.predict              # inference → outputs/predictions/next_season_predictions.csv
```

`src/features/encode.py` (rolling aggregate features) is retained for experimentation but is no longer part of the primary training pipeline.

## Python environment

Always use the project virtual environment (`.venv`) for running Python or installing packages — never the system Python.

```bash
# Create (first time only)
python -m venv .venv

# Activate
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

All `python`, `pip`, and `pytest` commands must be prefixed with the venv activation or run via the venv's binaries directly (`.venv/bin/python`, `.venv/bin/pip`, `.venv/bin/pytest`).

## Tests

```bash
.venv/bin/pytest tests/
```

## Key config knobs (configs/default.yaml)

- `model.type`: `lstm` or `transformer`
- `features.sequence_length`: how many prior-season games to use as input (default 10)
- `features.target_stats`: `[dk_pts]`
- `data.seasons`: which NBA seasons to pull (need ≥3 for a train/val/test split)

## Train / val / test split

Temporal walk-forward by season. With seasons [S1, S2, S3, S4]:
- Train:  predict S3 games using S2 stats (and any earlier pairs)
- Val:    predict S3 games using S2 stats (second-to-last pair)
- Test:   predict S4 games using S3 stats (last pair)

The scaler is fit only on the prior-season games that feed into training, preventing leakage.
