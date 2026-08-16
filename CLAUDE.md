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
to the relevant docs for the task at hand.

**This section is a router, not a summary.** One line per doc saying what it covers and
when to open it — no findings, no figures, no dates. Detail belongs in the doc itself.
The docs are:

  - project-spec.md: Project overview and core rules for implementation. 
  **Always read this.**
  - adp-plan.md: Plan and notes for collecting average draft position ("adp") data
  - availability-plan.md: Plan and notes for modeling availability (games played 
  and minutes per game). Work completed and mostly archival.
  - availability-window-plan.md: The availability head's fitting window, season term, 
  dispersion, **likelihood**, absence-composition covariates and trials assumption, what 
  the shipped mixture is worth in the contest, how players the head has no row for are 
  treated, and the simulator's **availability layout** — where a player's missed games fall, 
  which the head cannot say. Read it before changing any of those, or before pricing a head 
  change with `make strategy-sweep`.
  - composition-quadrature-plan.md: Workplan for fitting the composition's 
  per-(player, season) effect by marginalizing the latents with per-unit quadrature, 
  with a role-graded σ. Read before touching `sigma_u`, the injection constant, or 
  `composition_glm.stan`'s effect blocks.
  - dashboard-plan.md: The dashboard's charter and pages. The dashboard is a 
  **data-visualization surface**, not a project walkthrough — read this before adding 
  or editing a view.
  - dashboard-revision-plan.md: The revision round after the dashboard expansion. It 
  inherits every rule `dashboard-plan.md` sets; read that one first.
  - data-quirks.md: Notes and findings in exploring the raw data.
  - dk_best_ball_rules.md: Copy of the tournament rules for draft kings best 
  ball tournaments **always read this**.
  - docs-audit.md: How the two documentation guards work — `make docs-audit` (a gate) 
  and `make dashboard-audit` (a report). Read this before editing a quoted figure or 
  adding a doc to the audit.
  - draw-time-calibration-plan.md: Workplan for grading the composition's injected 
  minutes σ by role at simulation time. Read before touching the injection constant.
  - eda-plan.md: Plan and notes for exploratory data analysis and feature 
  reduction. Work completed and mostly archival at this point.
  - entry-collection-plan.md: Runbook for the first real DK entries — 20 ADP-null 
  autodrafts plus 5–10 live drafts for pick-log capture, and the two-fit production 
  schedule they require. Read before entering any contest or fitting for one.
  - facts-archive.md: Known facts, do not re-derive. Refer to this if we 
  re-open model design questions, but it is safe to ignore otherwise.
  - games-played-plan.md: Plan and notes for the games played model. Work 
  completed and mostly archival at this point.
  - injuries_paper.md: Copy of a study on workload contributing to achilles 
  tendon ruptures in basketball players. Archival.
  - minutes-composition-plan.md: Plan and notes for the production version of 
  the minutes-played model. Work completed and mostly archival at this point.
  - minutes-window-plan.md: The marginal minutes head's fitting window and dispersion, 
  and what they do to the composition's injected σ. Read it before changing either, or 
  before re-opening whether the marginal head can be retired.
  - model-cards-plan.md: The contract between the fitted heads and the dashboard's 
  model detail pages — what `make model-cards` writes, the rules the emitter inherits, 
  and each head's declared `chain_role`. Read this before adding a `model_card_*` 
  artifact or changing a head's variant ladder.
  - model-development-notes.md: Detailed notes and findings developed during
  the model selection and fitting processes. Archival unless we reopen model 
  selection questions.
  - old_readme.md: Archival project context. Superceded by @README.md.
  - pipeline.md: Lists and describes the make files used to implement the 
  project pipeline. 
  - potential-to-dos.md: Parking lot for measurable ideas that are not yet 
  scheduled — what to compare, what evidence points at it, and what would 
  settle it. Not commitments and not results; deliberately outside 
  `make docs-audit`.
  - predictions-plan.md: Plan and notes for the models for the various box-score 
  statistics that feed into the *dk_pts* figure. Work completed and mostly 
  archival.
  - preseason-plan.md: The post-preseason problem-statement change — current-season 
  preseason games entering the existing heads (as feature columns on the availability, 
  minutes and component rate heads, and as the composition's prior share), the gates, 
  which head is opted out and why, and the October production runbook. Read before 
  touching preseason fetch, panel, features, or any head's preseason flag.
  - provenance-plan.md: Plan for keeping documentation, dashboard, and 
  context up-to-date with latest findings and ensuring reproducibility of 
  context.
  - shot-attempt-basis-plan.md: Plan for updating one of the box-score statistics
  models (fg3a as a beta-binomial rate over the total fga, rather than fg2a and 
  fg3a as separate poisson counts). Work completed, archival.
  - sim-inputs-plan.md: Workplan for the simulator's calibrated draw inputs — the 
  consumed-vs-diagnostic inventory, a documentation/dashboard accuracy pass (first), 
  and four measurement items on the copula, overdispersion and serial structure. 
  Read before editing how any simulator input is described or consumed.
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
  `dashboard/decisions.py`** alongside a plan-doc edit (create a new document 
  in /docs for new workflows). The registry is what
  the dashboard's decision log renders, and it carries `source` and `reviewed` so
  `make dashboard-audit` can flag entries whose source doc has moved since. Statuses come
  from a closed vocabulary — a reversal becomes `withdrawn` and keeps its entry rather than
  being deleted, because the reversals are the most useful thing on that page.
- Do not update @CLAUDE.md or @README.md with detailed findings. Instead, update 
  the relevant detailed plan doc in /docs, and include that document in the 
  list of docs above.
- Keep responses focused, brief, and concise. Keep disclaimers and caveats short, 
and spend most of the response on the main answer. When asked to explain 
something, give a high-level summary unless an 
in-depth explanation is specifically requested.