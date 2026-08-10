# Model Cards: the contract between the fitted heads and the model pages

This is the artifact contract `docs/dashboard-plan.md` promised when step 3 of the expansion
was built. It is a planning-and-contract doc, not a measurement report.

**All eight artifacts ship as of 2026-08-10.** Session 3a landed the index, the
coefficients, the features and the feature correlations; **session 3b landed the predictive
half** — `model_card_ecdf.csv`, `model_card_calibration.csv` and
`model_card_sample.parquet` — and added the fifth build-time check that goes with them.
**Session 4 added the eighth**, `model_card_feature_density.parquet`, which is the half of
`feature-correlation-not-pair-plots` the first two sessions left open: the correlation file
flags which pairs earn a joint density, and nothing had binned one.

---

## Why an emitter exists at all

The dashboard's binding rule is that it reads artifacts and never imports `src/`, pinned by
`test_the_dashboard_imports_nothing_from_src`. Against the model detail pages, almost
nothing the pages need is on disk: eighteen of twenty heads publish no coefficients
anywhere, and *nothing* publishes the features a head was fed.

Everything they need does exist in `data/features/posteriors/<window>/*.pkl`, and **that is
the one place the dashboard may not look.** Two reasons, and the second is the load-bearing
one:

- unpickling a `PosteriorArtifact` imports `src.models.posteriors`, which the `ast`-based
  purity guard cannot see, because it walks *static* imports only. The guard would pass
  while the rule broke;
- the object it hands back carries a fitted `StandardScaler` and the ordered design steps.
  That is precisely the capability to score an arbitrary frame — and a dashboard that can
  score is a dashboard that can silently disagree with the fit it is describing, which is
  the one failure the rule exists to prevent.

So `src/models/model_cards.py` stands between them. It reads the pickles and each head's
**own variant ladder**, and writes flat long-format CSVs the dashboard reads and nothing
else. Registered as `model-cards-emitter-not-posterior-pickles`.

---

## The three rules the emitter inherits

### 1 · No test row can reach an artifact

Every frame is carved by `held_out.selection_split`, which drops the last two target seasons
entirely rather than returning them and trusting nobody to read them.
`model_cards._split_pair` is the module's only door to a split, and
`test_the_emitter_reaches_a_split_only_through_selection_split` walks the import graph to
prove `split_seasons`, `final_split` and `unlocked` are not imported at all.

`SPLITS = ("train", "validation")` is the closed vocabulary, and `_check_splits` refuses
anything outside it on the way to disk. It cannot fire today; it exists because the failure
it guards is **silent by construction** — a page rendering a third split would look like a
feature, not a bug.

Every head's fit span ends at **2021-22**, and a test on the shipped index asserts it.

### 2 · The `train` posterior window, never `train_val`

At `train_val` the validation rows were in the fit, so a "validation" histogram or scatter
drawn from those coefficients is an in-sample picture wearing the wrong label. `WINDOW` is a
module constant rather than a `make` variable **on purpose**: there is no correct second
value, and a `WINDOW=train_val` knob is an invitation. `posteriors.require_window` is the
enforcement — it refuses an artifact fitted wider instead of letting the leak be discovered
in a picture, and it is the only guard that can, because a head fitted wider has read those
seasons *through its coefficients* and no frame-level split guard can see that.

### 3 · The recipe is verified at build time, and the check is bigger than `posteriors.py`'s

`posteriors.py` reproduces each head's design matrix and predictions on a 400-row probe and
fails the build rather than writing a wrong artifact. This module inherits that and needs
more, because it has a **second drift surface the probe check cannot see**: it re-derives
the frames themselves. A `build_design` that changed shape, a split that moved or a filter
that drifted would leave the coefficients describing one population and the histograms
describing another — and both files would look perfectly well-formed.

So `verify` runs four checks per head and raises on any of them, and the predictive half
adds a fifth in `check_predictive`:

| # | check | what it catches |
|---|---|---|
| 1 | **the population anchor** — the rebuilt fitting frame has exactly the row count and season span `posteriors.py` recorded in the provenance | a moved split, a changed builder, a drifted filter |
| 2 | **the recipe against the ladder** — the persisted recipe applied to the raw frame equals the head's own variant ladder put through the head's own scaler, both splits, to `posteriors.DESIGN_TOL` (1e-9) | an imputation mean, a log/logit scale, a spline knot set, an interaction or a bin edge that has moved |
| 3 | **the feature block is producible** — every column `recipe.features` names exists on the rebuilt frame | a ladder that stopped minting a column |
| 4 | **the artifact's own round-trip** — `PosteriorArtifact.roundtrip()` against the design matrix and predictions the *fitted head itself* stored | everything above, plus a wrong link function or a mis-thinned draw block |
| 5 | **the drawn predictive against the head's own mean** — `predictive_bias`, within 5% | a missing exposure, a trials column that moved, a link applied twice — none of which checks 1–4 can see, because all four pass on a design matrix that is then drawn from wrongly |

**Check 2 is tautological for the nine heads whose recipe carries no design steps** — their
raw frame *is* their design frame, so the comparison is an identity. The index says so per
head in `design_check` (`ladder` for the eleven that carry steps, `vacuous` for the nine
that do not), which is the difference between a gate and a green tick that means nothing.
Check 4 is non-vacuous for every head, which is why it is run rather than assumed redundant.

As shipped, every head passes **check 2 at exactly 0.0** and the worst round-trip prediction
error across twenty heads is 1.3e-15. The worst check-5 gap is **+1.20%** (`gp_exit`) against
its 5% bar, and nineteen of twenty heads are under 1%. Registered as
`model-cards-verify-against-the-ladder`.

---

## What is on disk

All eight under `outputs/predictions/`, all long-format, all keyed by `head`. The six CSVs
are written with six significant digits — the per-feature statistics repeat on every bin row
deliberately (a page groups by feature and gets the histogram *and* the summary table from
one read), and full float64 repr tripled the largest file for precision no histogram can
draw.

| artifact | grain | rows | size | what it feeds |
|---|---|---|---|---|
| `model_card_index.csv` | head | **20** | 11 KB | the head selector, and the *unit* every page must state |
| `model_card_coefficients.csv` | head × term | **311** | 45 KB | the sorted credible-interval panel (block 4) |
| `model_card_features.csv` | head × feature × split × bin | **14,892** | 2.1 MB | the small-multiple histograms and the n/mean/sd/missing table (block 2) |
| `model_card_feature_corr.csv` | head × split × feature × feature | **8,920** | 767 KB | the correlation heatmap and the pairs that earn a density (block 3) |
| `model_card_feature_density.parquet` | head × pair × split × 2-D bin | **93,608** | 536 KB | the on-demand joint density beside that heatmap (block 3) |
| `model_card_ecdf.csv` | head × split × grid point | **2,977** | 328 KB | the observed ECDF over the predictive ribbon (block 5) |
| `model_card_calibration.csv` | head × split × panel × 2-D bin | **28,709** | 2.4 MB | fitted-vs-observed and residual-vs-fitted, as density (block 6) |
| `model_card_sample.parquet` | head × split × row | **54,375** | 782 KB | the bounded scatter overlaid on that density (block 6) |

**8.6 MB in total** (`du`), against the 7.4 MB the seven-artifact contract cost. Twenty
heads across four classes — availability
(5), minutes (2), box-score components (11), game length (2) — carrying 268 coefficients, 20
intercepts and 23 dispersion terms over 92 distinct features. `make model-cards` runs in
about **10 seconds** and needs **no CmdStan**: it refits nothing and runs no sampler, and
imports the head modules for their variant ladders and their own predictive.

### `model_card_index.csv` — one row per head

Carries the head's identity (`head`, `label`, `model_class`, `class_label`), its
**specification** (`unit`, `family`, `likelihood`, `description`, `variant`, `response`,
`dispersion`, `coefficient_scale`), its **population** (`n_fit`, `n_validation`,
`n_frame_rows`, `row_filter`, `first_season`, `last_season`, `fit_window`, `n_draws`,
`n_features`, `n_terms`, `n_density_pairs`), its
**sampler provenance** (`max_rhat`, `divergences`, `converged`, `git_sha`, `built_at`), its
**verification** (`recipe_design_error`, `roundtrip_prediction_error`, `design_check`,
`verified`) and — added by session 3b — its **predictive** (`response_label`,
`predictive_draws`, `n_predictive_train`, `n_predictive_validation`,
`predictive_rows_capped`, `predictive_weighted`, `fitted_source`, `predictive_check`,
`predictive_bias`, `ecdf_band_mc`, `ecdf_band_gated`, `player_season_sigma`).

That last block exists so a page can state what it drew rather than implying it drew
everything. Three of those columns are load-bearing and none is derivable from the other
artifacts: **`n_predictive_*`** is 20,613 for a composition fitted on 631,158 rows,
**`predictive_weighted`** is why the depth head's ribbon is over 1,861 games when the head
fitted 4 rows, and **`fitted_source`** says whether the scatter's x-axis is the head's own
reported mean or the mean of its draws.

**`unit` is declared here rather than hard-coded in a view, and that is the point of the
file.** Every model page must state its own unit prominently — the component heads are
fitted season-collapsed on 8,630 player-season rows while the composition is per player-game
on 631,158, and a reader comparing an R² across those pages without knowing that is being
misled. This repo has already paid for that lesson at the season/team-game boundary
(`make minutes-unification`). A unit string typed into a view goes stale the first time a
head is refitted at a different grain; read from the artifact it cannot.

`description` is the same argument one step further: it is **specification only** — what the
head models and how — never a result, which is the one kind of typed prose
`docs/dashboard-plan.md` allows on a model page.

**`n_fit` and `n_frame_rows` differ on exactly four heads, and that is not an error.**
`StanConversion.fit` drops rows with no attempts *internally*, so the population the head
actually fitted is smaller than the frame `posteriors.py` recorded:

| head | `n_fit` | `n_frame_rows` | `row_filter` |
|---|---|---|---|
| `fg3m_given_fg3a` | 7,695 | 8,630 | `fg3a > 0` |
| `ftm_given_fta` | 8,506 | 8,630 | `fta > 0` |
| `fg2m_given_fg2a` | 8,610 | 8,630 | `fg2a > 0` |
| `fg3a_given_fga` | 8,622 | 8,630 | `fga > 0` |

Both numbers ship because both are true and a reader comparing this file to
`posteriors/train/manifest.csv` would otherwise find a discrepancy with no explanation.
`n_fit` is the population the histograms describe; `n_frame_rows` is what the anchor check
compares against.

### `model_card_coefficients.csv` — one row per term

`mean`, `sd`, `q2.5`, `q25`, `q50`, `q75`, `q97.5` and `p_positive` over the persisted
draws, plus `term_family`, `term_role`, `basis_index`, `scaler_center` and `scaler_scale`.

**`term_family` is what keeps a spline basis off the panel.** Nine heads carry a six-column
basis over one underlying quantity, and drawn as six independent bars it swamps every real
term in the head. So `log_ast_p36_lag1__s3` groups under `log_ast_p36_lag1` and carries
`basis_index = 3`; the panel can collapse the family to one row or expand it. Imputation
flags group under `missingness` rather than under the column they flag, because `__miss`
terms answer "what does not knowing cost", which is one question across the block rather
than one per feature.

**Every coefficient is on the standardized design scale**, and `coefficient_scale` in the
index says so. That is not incidental: it is what makes a sorted bar chart across terms a
legitimate comparison rather than a plot of measurement units, since every head fits a
`StandardScaler`'d matrix. The scaler's `mean_` and `scale_` ride along per term so a
consumer can unstandardize without the fitted object — the same arithmetic
`stan_game_length._unstandardized` does, and the whole point of the emitter: **the numbers
travel, the capability to score does not.** Registered as
`model-card-coefficients-are-standardized`.

`p_positive` is the share of draws above zero, which is the reading a 95% interval crossing
zero under-reports — a term at 0.94 and a term at 0.50 both "cross zero" and are not the
same claim.

The composition's graded arm fits one `rho` per prior-share bin, so its dispersion is a
vector and each bin gets its own term (`rho[1]`…`rho[4]`). Collapsing it to a mean would
erase the 2.07× fringe-to-star spread that is the arm's whole reason for existing.

### `model_card_features.csv` — the histograms and the summary table

**The histogrammed values are the head's own design columns, taken from its own variant
ladder** (post-transform, pre-standardization) — not the raw builder columns. "The features
it was fed" *is* the design matrix, and the two things most worth looking at, the spline
bases and the imputation flags, do not exist in the raw frame at all. Registered as
`model-card-features-are-the-design-columns`.

**Train and validation share one edge set per feature, computed on the pooled values.**
Comparing the two histograms is the block's entire job and two histograms drawn on their own
edges cannot be compared. A feature with ≤ 12 distinct values gets one bin per value
(`bin_kind = discrete`, 24 of 268 features), so an imputation flag reads as two bars at 0
and 1 rather than as a spike in the first of thirty linear bins; everything else gets 30
linear bins closed on the right, so the maximum lands in the top bar rather than nowhere.
`density` is the share of that split's finite values, so a 773-row validation histogram can
be drawn over an 8,630-row training one.

**`missing_share` resolves through the source column, not the feature's own name.** A design
column is usually two transforms away from the column whose missingness it inherits:
`logit_fg3m_pct_lag1__s3` is a spline basis over a logit over `fg3m_pct_lag1`, and only the
last of those three is what the builder produced or the head flagged. `source_columns` peels
the basis index, then `__sq`, then a `log_`/`logit_` prefix, and the share is read off the
head's **own** `__miss` flag wherever it minted one — its record of what it filled, rather
than a re-derivation of it. Without that walk-back, every spline basis in the project would
report a flat zero, which renders as a perfectly good-looking page.

An imputation flag is itself never missing and reports 0; its **mean** is the share being
asked about. That is how `docs/dashboard-plan.md`'s "imputation flags shown as their own
share" lands — the flag is a feature row like any other and its mean is the number.

Three heads carry real missingness today: `composition` (17.04% of train rows are rookies
with no prior season, 12.63% on validation — one `design_missing` indicator over the whole
18-column block), `fg3m_given_fg3a` (4.11% / 1.06%) and `ftm_given_fta` (0.02% on train and
nothing at all on validation, which is why one of its spline bases is constant there).

### `model_card_feature_corr.csv` — the heatmap and the pairs worth a density

The **whole square, including the diagonal**, so a heatmap is a reshape rather than a
reconstruction — and so a constant column shows up as an empty row instead of vanishing from
the axis. Pearson pairwise-complete via `DataFrame.corr`, which is what leaves a constant
column as `NaN` rather than as a spurious zero. 77 of 8,920 cells are `NaN`, all on
validation, and they are a finding rather than a defect: `logit_fg2m_pct_lag1__s0`,
`logit_ftm_pct_lag1__s0` and `ftm_pct_lag1__miss` are **identically constant on the
validation split** — no validation row sits in the leftmost knot span, and no validation
free-throw row was imputed.

`pair_rank` ranks the distinct off-diagonal pairs by |r| inside each head and split;
`top_pair` marks the top 20, which is 40 rows because both orientations of a pair carry the
same rank. Those flags are what the density below is binned over, per
`feature-correlation-not-pair-plots`.

### `model_card_feature_density.parquet` — the joint behind the heatmap

Added by session 4, and it closes the second half of that decision. The heatmap answers "is
anything in this block collinear" at a glance and cannot answer "what does the joint
actually look like"; a full pair-plot matrix answers the second at 150–400 panels nobody
reads. So the joint is precomputed for the pairs the first question points at, and the page
draws one of them at a time.

`head × pair × split × 2-D bin` on an **18 × 18** grid, empty cells dropped — 93,608 cells
over 720 panels, a median of 133 occupied cells each. Coarser than the feature histograms'
30 bins because a 2-D cell holds 1/n of the rows a 1-D bar does, and a finer grid buys
resolution nothing has the rows to fill. Every pair carries the `r` of the split it is drawn
on, so a validation panel is not labelled with a training correlation.

Three rules, each shared with an existing artifact for the same reason it was adopted there:

- **The pair set is `top_pair` on the training split**, and both panels are drawn for it.
  Ranking per split would give a menu that reshuffles when the reader flips the split, which
  breaks the only thing the two panels are side by side for. Selection is read back off the
  rows `correlation_rows` just produced rather than re-ranked, so the flag a page filters the
  menu on and the panel it can actually draw cannot disagree.
- **Both splits share one edge set per pair**, pooled — the rule the feature histograms and
  the calibration grid already follow.
- **A discrete column gets one cell per value**, through the same `bin_edges` the 1-D
  histograms use, so an imputation flag reads as two columns of cells rather than as a spike
  in the first of eighteen.

**It is the second parquet, for the opposite reason to the first.** `model_card_sample`
is binary because it is three float columns and nothing else; this one is two long feature
names restated on every cell, where dictionary encoding is the measured difference between
**10.5 MB** and **0.55 MB**. That is what makes twenty pairs a head affordable — as a CSV
this artifact alone would have outweighed the other seven. Registered as
`model-card-density-is-parquet-not-csv`.

Two heads carry no pairs and say so in the index's `n_density_pairs`: `game_length_depth`
has no features at all and `game_length_ot` has one, so there is no pair to bin. A page reads
that column rather than rendering an empty selector.

**Both splits are emitted, which is one column more than `docs/dashboard-plan.md`'s sketch
asked for.** The collinearity question is a training-frame question, but every other block
on these pages shows train beside validation, and a block that changed shape on validation
would be saying something worth seeing — the constant spline bases above are exactly that.
It costs 4,460 extra rows.

---

## The predictive half

Built by session 3b, 2026-08-10. All three artifacts are cut from **one** draw per head per
split — the ribbon, the density and the scatter are three readings of one predictive, and
drawing them separately would let a page show a ribbon and a scatter that disagree.

### Every head declares what its predictive is *of*

`RESPONSES` is `HeadSpec.unit`'s twin and exists for the same reason: the twenty heads share
no observable. It names the realized column, the trials denominator, the multiplicity weight
where the frame is collapsed cells, and the `label` a page writes on its axis — "games
played", "season minutes", "minutes in one team-game", "absence-spell length (games)". A
unit string typed into a view goes stale on the next refit; an axis label typed into a view
is worse, because it can be wrong about a head it was never written for.

**Two heads are drawn over more rows than they were fitted on, and that is not an error.**
`gp_duration` and `game_length_depth` fit *collapsed cells* carrying a multiplicity — the
depth head is four rows for 1,861 overtime games — so the card expands by that weight before
drawing. Without the expansion the depth ECDF would be four points each carrying a whole
cell's mass onto a single draw. `predictive_weighted` says so in the index.

### The rule: the head's own predictive, or its own parameters

**Thirteen of the twenty heads expose a `predict_samples`** — the seven counts, the four
conversions, the marginal minutes head and the composition — and those are rehydrated around
their persisted draws and called, exactly as `src/models/minutes_unification.py` does. The
minutes and composition heads reuse *that module's* rehydrators rather than a second copy,
so the composition is drawn **with the shipped `sim.minutes.player_season_sigma = 0.450`**
injected, which is what every other consumer gets. `player_season_sigma` is in the index
because a page comparing this card against `stan_composition_metrics.csv` has to know.

**The other seven never draw at all.** Availability, the three games-played binomial heads
and overtime onset score through an explicit pmf; the two beta-geometric heads score through
a log-likelihood. There is no `predict_samples` to call, so `family_draws` takes the per-draw
*parameters* from the artifact's own `mu_draws` and the head family's own shape function
(`stan_minutes.beta_shapes` for the `rho` beta-binomial, `games_played.beta_shapes` for the
`kappa` frailty) and writes only the sampling call — one line per family. Check 5 is what
keeps that honest.

### Check 5, and why the beta-geometrics get a sharper version

`predictive_bias` is the signed gap between the drawn mean and the head's own reported mean,
as a share of it. It catches what checks 1–4 structurally cannot: all four pass on a design
matrix that is then drawn from on the wrong scale. Worst across twenty heads is **+1.20%**
(`gp_exit`) against a 5% bar.

Two heads get a different reading and one gets none:

- the beta-geometrics are checked on **`P(T = 1)`**, because `mu` *is* `P(T = 1)` for that
  likelihood — a prediction rather than a restatement of the mean. `game_length_depth` reads
  −0.61%, `gp_duration` +0.06%;
- the **composition is not checked**, and `predictive_check = none` says so. Its `response`
  is `eta`, the linear predictor of one *step* in a sequential allocation, which is not on
  any scale the drawn minutes can be compared against. `fitted_source = predictive_mean`
  there and on the two beta-geometrics, whose mean is `E[1/p]` under a Beta frailty and is
  not the `mu` they report.

### The budget, and the check that it is enough

**200 draws over at most 20,000 rows a split.** The composition sets both: at 631,158 rows
times its 1,000 persisted draws the predictive alone is 631M numbers, and none of that buys a
better picture. The row cap is a subsample of the population, so it moves Monte Carlo error
rather than the estimand — and on the composition it is taken in **whole team-game blocks**
through `posteriors.team_game_probe`, because `ragged_arrays` rejects a frame cut through a
block. (Blocked frames round up to whole blocks, which is why the composition reports 20,613
rather than 20,000.)

**The draw budget is measured rather than assumed**, which the build prompt asked for
specifically. `band_stability` re-reads the 95% ribbon on two *interleaved* halves of the
draws — interleaved because they arrive chain-major — and reports the largest disagreement
across grid points, in ECDF units. Two independent D/2 readings differ by about **twice** the
standard error of the D-draw estimate they average to, so the statistic is a conservative
bound on the shipped ribbon, and the bar is set at 0.02 to hold that ribbon inside one ECDF
point. Measured, at 100 / 200 / 400 draws:

| head | 100 | **200 (shipped)** | 400 |
|---|---|---|---|
| `game_length_depth` | 0.0346 | **0.0145** | 0.0073 |
| `availability` | 0.0190 | **0.0135** | 0.0102 |
| `gp_entry` | 0.0148 | **0.0107** | 0.0054 |
| `minutes` | 0.0139 | **0.0074** | 0.0081 |
| `ast` | 0.0091 | **0.0071** | 0.0065 |
| `composition` | 0.0023 | **0.0020** | 0.0014 |

It falls as `1/sqrt(D)`, which is the confirmation that it is measuring Monte Carlo error and
not misfit. At the shipped budget the worst gated head sits at 0.0145 — the shipped ribbon is
good to roughly 0.007 ECDF, well under the resolution one of these panels is drawn at, and
400 draws would buy a third of a pixel for double the cost.

**`game_length_ot` is reported and not gated**, at 0.5000. Its validation split is two season
cells, so its ECDF takes three values and a half-sample gap of 0.5 is the frame rather than
the budget. `ecdf_band_gated` marks it, and heads under `BAND_MIN_ROWS = 500` are excluded
from the gate rather than from the file.

### `model_card_ecdf.csv` — the ribbon

`head × split × grid point`, with `observed` and seven quantiles of the per-draw ECDF
(`q2.5`, `q10`, `q25`, `q50`, `q75`, `q90`, `q97.5`), so the 50 / 80 / 95% bands the page
draws are pairs of columns rather than a reconstruction.

**One replicate dataset per posterior draw**, not the pooled predictive: pooling gives the
predictive's own CDF, which has no width and cannot be compared against a single realized
sample. **The grid is quantiles of the *observed***, 100 of them, or one point per value for
a response with fewer than that many distinct ones (`grid_kind`). Drawn from the observed and
not from the draws on purpose — an over-wide predictive then shows as a ribbon that has not
reached 1 at the last grid point, where a grid stretched to cover it would hide that in the
axis.

**Read the size of the miss, not a pass/fail.** At n ≈ 10⁴ the ribbon is ±1–2 ECDF points
wide and every head in the project falls outside it somewhere. On the training split the
observed curve sits inside the 95% band at 22% of grid points for `availability`, 10% for
`minutes` and 7% for the composition. That is what a posterior predictive check does at this
sample size rather than a defect, and the useful reading is the largest vertical distance
from `q50`: **0.037** for availability, **0.055** for minutes, **0.051** for the composition
and **0.080** for `gp_onset`. A page that renders in-or-out as a verdict will report that
every head fails.

### `model_card_calibration.csv` — the two density panels

`head × split × panel × 2-D bin` over a 30 × 30 grid, `panel` ∈ {`fitted_observed`,
`residual_fitted`}, **empty cells dropped** — 28,709 rows against the 72,000 a dense grid
would carry. Binned rather than per-row for the reason the whole contract is binned: the
composition's scatter is 631,158 points per panel, which is not an artifact but a copy of the
data.

Edges span the **pooled 0.5–99.5%** range of each axis with the tails **clipped into** the
end bins, so one heavy-tailed residual cannot collapse the grid to a single cell and nothing
is dropped — the per-panel counts sum to `n`, and a test pins that. Both splits share one
edge set per panel, for the same reason the feature histograms do.

### `model_card_sample.parquet` — the texture

`head × split × row`, capped at 2,000 rows per head and split — 54,375 rows, `float32`,
782 KB. Past a couple of thousand a scatter is a blob, so the cap is a legibility decision as
much as a size one, and the rows are taken by `thin` rather than at random so the overlay
spans the frame and does not move under a reader between builds. Parquet rather than CSV
because it is the one artifact that is three float columns and nothing else.

### What the first drawing of these heads already shows

Not a result this doc is claiming, but worth recording since nothing had ever drawn these
heads before and a page reader will see it: **the three games-played binomial heads
over-predict their own observable at the row mean** — `gp_onset` predicts 5.30 spell onsets
per player-season against 4.08 realized, `gp_exit` 6.80 against 5.83, `gp_entry` 4.51 against
4.27 — while `availability` is nearly unbiased (54.5 against 55.2 games). It is the head and
not the emitter: the drawn predictive reproduces the head's own reported mean to 0.06%, and
the posterior mean of `mu` agrees with the plug-in one to within 0.1%. The suspect is a
beta-binomial with heavy dispersion on a target with 64% zeros (`gp_exit`), whose *mean* is
not pinned to the empirical mean the way a binomial GLM's is. It is logged in
`docs/potential-to-dos.md` rather than chased here, because the composite the
tenure decomposition actually ships is `SpellProcess`, and whether this survives composition
is the measurement that would settle it.

---

## Where this sits in the pipeline

```
make stan          fits every head, writes metrics/diagnostics, THROWS THE DRAWS AWAY
make posteriors    refits once per head at its shipped variant, persists draws + recipe
make model-cards   ← here. Reads those pickles. No refit, no CmdStan, ~12 s
                     → outputs/predictions/model_card_*.csv, model_card_sample.parquet
                       and model_card_feature_density.parquet
dashboard          reads those artifacts and only those artifacts
```

`make model-cards` is downstream of `make posteriors` and nothing else. It fails loudly and
by name if the posteriors are missing, at the wrong window, no longer describe the frames the
builders produce, or can no longer be drawn from on the scale their own means are on.

**There is deliberately no rebuild-one-head flag**, which is the opposite of
`make posteriors --groups` and for a measured reason: that target is a day of sampler time,
where re-running one group and merging the manifest by head is the only sane workflow, and
this one is ten seconds for all twenty heads. A partial rebuild would buy nothing and would
quietly leave `model_card_index.csv` describing three heads. To debug one head's check,
`--check <heads>` builds and verifies **without writing** — including the predictive, which
is the expensive half:

```
.venv/bin/python -m src.models.model_cards --check composition
```

## Tests

`tests/test_model_cards.py`, plain `assert` with synthetic builders, **74 tests** (34 from
session 3a, 31 from 3b, 9 from session 4's density). The heads are real `PosteriorArtifact`s with their draws **injected**
rather than sampled — the same stance `tests/test_posteriors.py` takes, and for the same
reason: the emitter refits nothing either. The one exception is the rehydration test, which
builds a real `StanCount` around injected draws and asserts that `draw_predictive` returns
byte-identical output to the head's own `predict_samples`, because "no second implementation
of any head's predictive" is this half's load-bearing rule and a docstring cannot hold it.

The coverage is one case per way this module can be wrong *silently*, because every one of
those renders as a good-looking picture: a histogram drawn on the wrong edges, a
missing-share that resolves to zero because the flag sits under a third name, a correlation
that reports 0 for a constant column, a split label outside the vocabulary, a spline basis
drawn as six unrelated bars, a collapsed cell frame drawn one row per cell, a residual panel
that is a second copy of the observed one, an ECDF band pooled across draws instead of read
per draw, a calibration grid collapsed by one outlier, a joint density whose two splits are
binned on their own grids, a menu of pairs the density does not carry. The checks that would
fail loudly —
the population anchor, the design tolerance, the drawn-mean scale and the band's own stability
— get one test each for the raise.

**Thirteen tests read the shipped artifacts**, keeping a *derived* quantity honest against the
artifact it came from: no split outside the vocabulary, no fit span past 2021-22, every head
verified with `recipe_design_error ≤ 1e-9` and a declared unit, feature counts agreeing across
the index, the features file and the square of the correlation file, term counts agreeing
between the index and the coefficients — and, from 3b, every carded head appearing in all
three predictive artifacts, every shipped ECDF curve monotone under an ordered band, every
gated head's ribbon inside `ECDF_BAND_TOL`, every checkable head's drawn mean inside
`PREDICTIVE_BIAS_TOL`, every calibration panel counting all its rows, and the row and draw
budgets holding; and, from session 4, every head's density covering exactly its own flagged
pairs at the count the index reports, and every density panel counting all of its own rows. They skip rather than fail on a fresh checkout, since `make model-cards`
needs `make posteriors` first.

**Deliberately not in `make docs-audit`.** Most figures quoted here are re-derived by
`make model-cards` itself — the row counts, the design errors, the season spans, the band
statistic and the drawn-mean gap are all checked at build time and the build fails on
disagreement — so a docs-audit builder would be a second copy of a gate that already runs.
The 3b figures that the emitter does *not* verify are the ones in the two tables above: the
band at 100 and 400 draws, which are measurements of budgets that are not shipped, and the
ECDF-coverage readings. Those are the honest candidates for a presence check if this doc
starts being edited by hand.
