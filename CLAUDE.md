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
  - availability-ship-plan.md: **Ephemeral scaffolding**, written 2026-08-11 — a 
  six-session work plan with one prepared prompt per session for shipping the 
  2012-13 window and role-graded `rho` into `stan_availability.py` and propagating 
  it through posteriors, model cards, the simulator, the games-played floor and the 
  contest layer. Carries the three traps that would otherwise bite (do **not** filter 
  `availability_design` — six modules import it; the Stan change is a transplant of 
  `composition_glm.stan`'s `n_rho`/`rho_bin` pattern; every session must end with 
  `make docs-audit` green because it is a gate). Delete it when the round lands, as 
  `dashboard-build-prompts.md` was.
  - availability-window-plan.md: The fitting window, the season trend, and where the 
  beta-binomial's `rho` lives — opened 2026-08-11 when `model_card_ecdf.csv` showed the 
  availability head missing **both** ends of its own distribution in opposite directions, 
  invisible to every metric it had been gated on. Carries the era break test (a sup-F scan 
  against a Monte-Carlo null), the `make availability-window` ladder that crosses window × 
  season term × dispersion, and the three results it settled: a 2012-13 window plus a 
  role-graded `rho` is the shippable arm, a season trend is a **null** because it buys the 
  boundaries by wrecking the body, and `rho` is pooled across *players* rather than across 
  *seasons*. Read this before changing the availability head's fitting window or adding a 
  season term to it. Its last section carries the same question, measured but not yet 
  laddered, for the two minutes heads. **`make availability-weighting` (2026-08-11) 
  is its follow-up**: four ways to *spend* old seasons rather than keep or discard 
  them — `l2` x lookback, separate `beta`/`rho` windows, exponential season decay, 
  and per-coefficient-block windowing — selected on a rolling harness over the 
  fitting half and then confirmed once on validation. The measurement that stands 
  is that the drift is in the **level**, not the relationships: 5 of 20 columns 
  carry it, and windowing the age and absence blocks makes the head *worse*. 
  **None of the four beats the plain window on validation**, so nothing new ships, 
  and one mechanism explains all four — they lean on the COVID trough.
  - dashboard-plan.md: Plan and notes for the streamlit dashboard. As of 
  2026-08-08 the dashboard is a **data-visualization surface**, not a project 
  walkthrough — read this before adding or editing a view. Its nine-page 
  "expansion" shipped 2026-08-10 and the revision round below added a tenth page 
  the same day; the "Charter amendment 2026-08-10" subsection 
  sets the three bounds the one page of prose (the Overview) exists under, and 
  each step has a "Step N, as built" section. **Bound 1 was amended 2026-08-10** 
  when the Overview was rewritten as a paper — "opens above the fold, scrolls no 
  further than one screen more", a measured ceiling — and bound 2 gained a half: 
  a typed sentence on that page carries no digit at all. 
  `dashboard-build-prompts.md`, the 
  ten one-per-session build prompts, was ephemeral scaffolding and was deleted 
  when the expansion landed.
  - dashboard-revision-plan.md: The round *after* the expansion, planned 
  2026-08-10 — five one-per-session steps (appearance, which availability head 
  ships, DHARMa-style quantile residuals, a dk_pts page at the unit the contest is 
  decided at, and the Overview as a paper). It inherits every rule 
  `dashboard-plan.md` set; read that 
  one first. Three of the five items turned out to have different answers than the 
  request assumed, and the doc records why. **All five shipped 2026-08-10** and each 
  has an "as built" section: the appearance is now Streamlit's own setting, with 
  the page chrome generated into `.streamlit/config.toml` by `make 
  dashboard-config`; every head declares its role in the shipped chain; block 6 
  of the model pages is a scaled quantile residual rather than a raw one; the 
  dashboard has a tenth page, **Weekly scores**, which is Gate A at the scoring 
  period (`make weekly-scores`, `src/sim/weekly.py`); and the Overview is now a 
  four-section paper rather than five hero tiles, which cost a charter amendment. 
  Step 4's unit moved from the 
  tournament round to the **week** mid-session at the user's request, and the doc 
  records both the request and what the change cost. Its prompts appendix was 
  ephemeral scaffolding and was deleted when the round landed.
  - data-quirks.md: Notes and findings in exploring the raw data.
  - dk_best_ball_rules.md: Copy of the tournament rules for draft kings best 
  ball tournaments **always read this**.
  - docs-audit.md: How the two documentation guards work — `make docs-audit` 
  (re-derives every quoted result from its artifact, a gate; sampler timings are 
  presence-checked only) and `make 
  dashboard-audit` (registry drift, a report). Read this before editing a 
  quoted figure or adding a doc to the audit.
  - eda-plan.md: Plan and notes for exploratory data analysis and feature 
  reduction. Work completed and mostly archival at this point.
  - facts-archive.md: Known facts, do not re-derive. Refer to this if we 
  re-open model design questions, but it is safe to ignore otherwise.
  - games-played-plan.md: Plan and notes for the games played model. Work 
  completed and mostly archival at this point.
  - injuries_paper.md: Copy of a study on workload contributing to achilles 
  tendon ruptures in basketball players. Archival.
  - model-cards-plan.md: The contract between the fitted heads and the dashboard's 
  model detail pages — what `make model-cards` writes, and the four rules the emitter 
  inherits (`selection_split` only, the `train` posterior window, a build-time recipe 
  check that fails rather than writing a wrong artifact, and a predictive drawn through 
  each head's own `predict_samples` whose mean must reproduce that head's own). Read 
  this before adding a `model_card_*` artifact or changing a head's variant ladder. 
  Since 2026-08-10 every head also declares a **`chain_role`** — what the simulator 
  does with it, in a closed vocabulary, pinned against `src/sim/` by a test rather 
  than merely written down. Sixteen of the twenty heads are in the draw path; 
  `gp_entry`, `gp_exit`, `gp_onset` and the marginal `minutes` head are not. The 
  ninth artifact, `model_card_quantile.csv`, is DHARMa's scaled quantile residual 
  and **replaced** the calibration file's raw-residual panel; its KS distance is 
  reported and never thresholded.
  - minutes-composition-plan.md: Plan and notes for the production version of 
  the minutes-played model. Work completed and mostly archival at this point.
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