# NBA DK Predictions

**READ THIS FIRST**

This project is a data science effort to leverage historical NBA statistics 
and a few contemporaneous data scources to develop a strategy for drafting
fantay basketball teams in a draft kings NBA best ball contest. The core 
pipeline is:

1. Fetch and preprocess raw data (primarily historical stats from NBA.com)
2. Fit GLMs built in stan (bayesian models) to predict per-game per-player 
production in terms of games played, minutes played (if playing in a 
game), and various box-score rates (per minute played).
3. Use the fitted posterior parameter values to simulate players' performance 
across seasons, in order to back-test various drafting strategies under 
realistic tournament rules.
4. Communicate findings concisely via a dashboad.

## Where to find context

Context for this project is stored across various .md files in ./docs. Refer 
to the relevant docs for the task at hand. The docs are:

  - project-spec.md: Project overview and core rules for implementation. 
  **Always read this.**
  - adp-plan.md: Plan and notes for collecting average draft position ("adp") data
  - availability-plan.md: Plan and notes for modeling availability (games played 
  and minutes per game). Work completed and mostly archival.
  - dashboard-plan.md: Plan and notes for implementing a streamlit dashboard 
  presenting project findings. To be updated.
  - data-quirks.md: Notes and findings in exploring the raw data.
  - dk_best_ball_rules.md: Copy of the tournament rules for draft kings best 
  ball tournaments **always read this**.
  - eda-plan.md: Plan and notes for exploratory data analysis and feature 
  reduction. Work completed and mostly archival at this point.
  - facts-archive.md: Known facts, do not re-derive. Refer to this if we 
  re-open model design questions, but it is safe to ignore otherwise.
  - games-played-plan.md: Plan and notes for the games played model. Work 
  completed and mostly archival at this point.
  - injuries_paper.md: Copy of a study on workload contributing to achilles 
  tendon ruptures in basketball players. Archival.
  - minutes-composition-plan.md: Plan and notes for the production version of 
  the minutes-played model. Work completed and mostly archival at this point.
  - model-development-notes.md: Detailed notes and findings developed during
  the model selection and fitting processes. Archival unless we reopen model 
  selection questions.
  - old_readme.md: Archival project context. Superceded by @README.md.
  - pipeline.md: Lists and describes the make files used to implement the 
  project pipeline. 
  - predictions-plan.md: Plan and notes for the models for the various box-score 
  statistics that feed into the *dk_pts* figure. Work completed and mostly 
  archival.
  - provenance-plan.md: Plan for keeping documentation, dashboard, and 
  context up-to-date with latest findings and ensuring reproducibility of 
  context.
  - shot-attempt-basis-plan.md: Plan for updating one of the box-score statistics
  models (fg3a as a beta-binomial rate over the total fga, rather than fg2a and 
  fg3a as separate poisson counts). Work completed, archival.
  - simulations-plan.md: Plan and notes for building the simulations and backtesting 
  drafting strategies under the DK best ball rules.
  - train-validate-test-split.md: Plan and notes for revising the project to 
  respect the train/validate/test split.

## Conventions

- No shared config helper exists; the idiom is
  `cfg = yaml.safe_load(open("configs/default.yaml"))` in `__main__` blocks, duplicated in
  several modules. Match it — do not refactor it away.
- Entry points run as `python -m src.<module>` from the repo root, with a matching
  `Makefile` target listed in `.PHONY`.
- Reuse `src/data/preprocess.py::compute_dk_pts` verbatim; never reimplement DK scoring.
- Reuse `src/data/fetch.py::_slug` / `_season_start_year` for season-key handling.
- `Path(...).mkdir(parents=True, exist_ok=True)` before every write; print
  `f"... {n:,} ... → {dest}"` progress lines.
- Artifact save/load mirrors `src/features/encode.py::save_artifacts` / `load_artifacts`.
- When a module defines classes that get **pickled**, its `__main__` block must import
  `run` through the package path (`from src.eda.pca import run as _run`) — otherwise
  classes pickle as `__main__.Foo` and cannot be loaded from any other process.
- Tests use plain `assert` with synthetic builders, no fixtures or classes (mirroring
  `tests/test_preprocess.py`).
- **When a load-bearing decision is taken, reversed, or measured, add or update its entry in
  `dashboard/decisions.py`** alongside the `CLAUDE.md` / plan-doc edit. The registry is what
  the dashboard's decision log renders, and it carries `source` and `reviewed` so
  `make dashboard-audit` can flag entries whose source doc has moved since. Statuses come
  from a closed vocabulary — a reversal becomes `withdrawn` and keeps its entry rather than
  being deleted, because the reversals are the most useful thing on that page.
