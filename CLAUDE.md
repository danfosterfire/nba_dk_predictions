# NBA Deep Learning

Predicts NBA player statistics (pts, reb, ast, min) for upcoming games using deep learning on historical game logs.

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
python -m src.data.preprocess      # clean → data/processed/game_logs.parquet
python -m src.features.encode      # engineer features → data/features/features.parquet
python -m src.train                # train model → outputs/checkpoints/best_model.pt
python -m src.evaluate             # test MAE/RMSE → outputs/predictions/test_metrics.csv
python -m src.predict              # inference → outputs/predictions/next_game_predictions.csv
```

## Tests

```bash
pytest tests/
```

## Key config knobs (configs/default.yaml)

- `model.type`: `lstm` or `transformer`
- `features.sequence_length`: how many past games to feed in (default 10)
- `features.target_stats`: list of stats to predict
- `data.seasons`: which NBA seasons to pull
