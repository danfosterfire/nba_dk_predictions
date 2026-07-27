# NBA DraftKings Points Prediction

Predict a player's DraftKings fantasy points (`dk_pts`) for **each game** of an upcoming
season, using only information available before the
season starts.

---

## The problem

### Target

DraftKings NBA scoring is a linear function of the box score, plus a nonlinear bonus:

```
dk_pts = 1.0·PTS + 0.5·FG3M + 1.25·REB + 1.5·AST + 2.0·STL + 2.0·BLK − 0.5·TOV + bonus
bonus  = 1.5 for a double-double, 4.5 for a triple-double
```

(implemented once, in `src/data/preprocess.py::compute_dk_pts` — never reimplement it)

### The prediction-time constraint

This is the defining constraint of the project. At prediction time — before the season
starts — we know:

1. The **season schedule** — who plays whom, when, home or away.
2. **Season-start rosters** — definitively which team each player is on, and each team's
   identity. Offseason moves are known.
3. The **previous-season stats** of every team and player.

We do **not** know: **within-season** roster changes (mid-season trades), current-season
minutes, injuries, or in-season form.

So the information set is a **cross-season join**: *current*-season roster membership
combined with *prior*-season statistics. Three consequences drive the design:

- **Team composition is built on the roster that will actually play**, not last season's
  roster. For player P in season S, his teammates are his season-S teammates, each
  described by their season S-1 stats. Likewise for opponents.
- **Team identity is known**, so team embeddings and team fixed effects are legitimate
  inputs — not just aggregated player stats.
- **Every feature except schedule-derived ones is still constant within a player-season**,
  since rosters are fixed at season start in our information set. The only per-game
  variation available is opponent, home/away, and rest.

Only **mid-season** churn is irreducible: 78 of 572 players (13.6%) appeared for more than
one team during 2023-24. Modelling that is out of scope for now.

Season-start rosters are derivable from data already on disk: a player's team in his
**earliest game of season S** is his season-start team. That is genuinely known before the
season, so using it in a backtest is not leakage.

Two knock-on details this creates:

- **Minutes weights must come from season S-1.** Season-S minutes are unknown, so any
  minutes-weighted roster aggregate weights teammates by their *prior-season* minutes.
- **Rookies and returnees have no S-1 stats** but *are* on the known roster. Roster
  aggregation must either impute them (draft position is in the matrix as
  `bio_draft_number`) or exclude them and renormalize — silently dropping them biases the
  aggregate toward veterans.

---

## Where the signal actually is (measured, not assumed)

Measured on 254,187 player-games across 2014-15 → 2023-24. Within-player-season residual
sd is **9.4 dk_pts** — this is the per-game noise a model must predict around a player's
own season mean.

| Source | Share of variance |
|---|---|
| Player-season identity (of total per-game variance) | **58.0%** |
| Own minutes played (of within-player residual) | **18.6%** — *not knowable in advance* |
| Opponent × season | 0.69% |
| Opponent × player-archetype × season, above a shuffled null | +0.31% to +0.97% |
| Home / away | 0.03% |

> The last two rows are **in-sample ANOVAs using contemporaneous opponent identity** —
> ceilings, not achievable gains. The interaction row is quoted as a range because it
> depends entirely on which marginal the null permutes (opponent-within-season gives
> +0.97%, archetype gives +0.31%, on identical data). An encoder restricted to the
> opponent's *prior* season delivers **0.368%** from the main effect and **+0.022%** from
> the interaction, held out. See "Opponent encoding" below.

Season-average opponent effect has sd **0.185 dk_pts** against an 11.48 dk_pts spread in
season-average dk_pts — 1.6%.

**Therefore: ~90% of attainable skill is getting each player's season-level rate right;
~1–2% is per-game modulation.** Effort should be budgeted accordingly. Team composition
is worth building, but it is not where the bulk of the accuracy lives.

Two refinements that change *how* to encode opponent:

- **Opponent effects partially cancel in the dk_pts sum.** Summing the per-component
  weighted opponent sd gives 1.105 dk_pts; measuring on dk_pts directly gives 0.785. A
  slow, stingy defense suppresses points while a fast one concedes rebound chances, so a
  single dk_pts head sees ~40% less opponent signal than the components do.
- **The matchup interaction earns its place per component, not on the aggregate.** Held
  out, it adds +0.203% on blocks — nearly doubling the main effect there — against +0.022%
  on dk_pts. An earlier version of this document claimed the interaction beat the main
  effect outright; that comparison was between an in-sample ceiling and an achievable
  gain, and it does not survive. See "Opponent encoding".

---

## Modeling design

### Predict components, not dk_pts directly

Because `dk_pts` is a linear sum, predict the parts and combine:

```
dk_pts ≈ [ P(play) · minutes ] × Σ wᵢ · rateᵢ  +  E[bonus]
```

- **Availability head** — `P(play)` and expected minutes. Minutes is the dominant common
  factor (18.6% of within-player residual), and the schedule features (rest,
  back-to-backs) act on it directly. For the *season total*, predicting **games played**
  is the single largest lever: 82 vs 60 games is a 27% swing.

  > **And it is the least predictable input in the project.** Year over year, with season
  > absorbed and minutes-weighted, **games played persists at r = 0.316** — against 0.779
  > for minutes per game, 0.869 for per-game `dk_pts`, and 0.90+ for the stickiest rate
  > columns. The largest lever on the season total is also the noisiest thing we can
  > condition on. Shrink this head hard toward a league/age baseline rather than toward
  > the player's own prior GP, and carry the age curve here (see "Where age acts").
  > Reproduce from `src/eda/persistence.py`'s helpers over
  > `season_matrix_roster_tierA.parquet`; `min`/`gp` are held out of `persistence.csv` as
  > volume columns.
- **Per-36 rate heads** — one per component (`pts, fg3m, reb, ast, stl, blk, tov`). Rates
  are markedly more stable across seasons than per-game totals. Use count-appropriate
  likelihoods: `stl` (~1/game) and `blk` (~0.5/game) are low counts and are misspecified
  under plain MSE.
- **Bonus** — `E[bonus] ≠ bonus(E[components])`. The double-double bonus is a threshold on
  five components at ≥10, so it needs the *joint* predictive distribution. A single dk_pts
  regression cannot represent it at all. Two implementations, both calibrated against
  11,627 player-seasons of realized bonus (mean 0.1268 dk_pts/game):
  - `src/features/targets.py::expected_bonus` — Monte Carlo with a shared per-game
    Gamma frailty inducing the positive correlation between components. Independent
    sampling is **23% too low** (0.098); `overdispersion=0.10` is unbiased (+0.001) and
    calibrates across minutes buckets.
  - `src/models/multihead.py::expected_bonus_analytic` — differentiable Poisson-binomial
    over the five categories, for use inside the loss. Correlates 0.9999 with the
    independent Monte Carlo; being independent it inherits the same downward bias.

Point estimates stay unbiased either way by linearity, so decomposing costs nothing on the
mean and gains on the bonus, the diagnostics, and the recovered opponent signal.

**The strongest measured argument for this structure is cross-component cancellation.**
Team-context features move the components hard and in opposite directions, then almost
entirely cancel in the DK-weighted sum: `teammate_assist_supply` shifts 2.11 dk_pts of
gross per-36 component movement into 0.25 net (8.3×), and the same holds for the opponent
main effect (1.105 gross vs 0.785 net). A single dk_pts head is fitting the residue.

An earlier version of this document argued instead that team context acts with *opposite
signs on rates and minutes*, so they cancel in per-game dk_pts. That was an artifact of
pooling 30 seasons without absorbing era — see the own-team section below. It does not
survive season fixed effects, and the cross-component argument replaces it.

### Dimensionality reduction — two different budgets

Treating this as one problem is the main trap. Effective sample size differs by an order
of magnitude:

| | Effective n | p | Approach |
|---|---|---|---|
| Player's own prior-season stats | 10,900 player-seasons | 150 | **Barely reduce.** 73:1 is comfortable. |
| Team composition | **892 team-seasons** | must be tiny | **Reduce hard** — 2–4 dims per side. |

**Player features:** feed within-season z-scored raw stats (era-neutral, so the model does
not spend capacity relearning the 3-point revolution). Keep `min`/`gp` unscaled and
separate — they are volume, not style. Treat PCA here as an *ablation*, not the default:
for gradient-boosted trees it typically hurts, and an NN's first linear layer already
subsumes it.

**Why linear DR adds nothing at the team level:** the minutes-weighted mean of PC scores is
*exactly* the PCA projection of the minutes-weighted mean stat line (verified to 3.6e-15) —
both operations are linear, so they commute:

```
Σ wᵢ Wᵀ(xᵢ − μ)  =  Wᵀ(Σ wᵢxᵢ − μ)
```

So a team represented by the mean of its players' PCs is informationally identical to its
average stat line, and the PCA bought only compression. All the value lives in the
**nonlinear** step — clustering, or similarity weighting.

**Why the mean is the wrong summary anyway:** 336 team-season pairs sit in the closest 1%
of mean-PC distance while allocating >55% of their minutes to different archetypes. A
balanced roster and a polarized one average to the same point — and that distinction is
exactly what matters for usage competition.

### Own-team encoding (leaving player P out)

Implemented in `src/features/team_context.py`. All features are leave-one-out and O(1) per
player given team aggregates (`Σ_{i≠P} wᵢxᵢ = S − w_P x_P`, renormalized by `W − w_P`; for
the max, precompute the top two and swap in the runner-up when P is the argmax).

The encoder aggregates the **season-S roster described by season S-1 stats**, weighting
teammates by their S-1 minutes. Season-start rosters come from the game logs: a player is
on team T's roster if his first appearance for T falls inside T's first 10 games. That
window recovers a ~15-man roster covering ~96% of realized team minutes; an opening-night
cut finds only 11 players and 84%, dropping anyone injured in October.

**Measured incremental value** (`make context-value`), on 8,038 player-season transitions,
after controlling for the player's own prior-season dk_pts, minutes and usage — **and for
season**:

| Feature | r vs next-season per-36 pts | r vs minutes | r vs per-game dk_pts | ΔR² |
|---|---|---|---|---|
| `teammate_usage_load` | **−0.198** | **−0.096** | **−0.160** | **+0.0066** |
| `teammate_usage_max` | −0.093 | −0.072 | −0.111 | +0.0032 |
| `teammate_assist_supply` | −0.012 | −0.067 | −0.102 | +0.0027 |
| `teammate_usage_sum` | −0.113 | −0.070 | −0.094 | +0.0023 |
| `roster_coverage` | +0.004 | −0.038 | −0.045 | +0.0005 |
| `team_pace` | +0.091 | −0.002 | +0.008 | +0.0000 |
| `role_crowding` | +0.048 | −0.000 | −0.005 | +0.0000 |
| `teammate_spacing` | +0.035 | −0.014 | +0.011 | +0.0000 |

The whole block adds **+0.0086 R²** (0.7431 → 0.7517) — **3.4× the +0.0025 measured under
the superseded construction**, and still squarely inside the 1–2% variance budget.

Four findings, three of which overturn what was previously recorded here:

- **Season fixed effects are mandatory, and their absence invented the previous result.**
  Pooled across 30 seasons, `teammate_spacing` correlates **+0.315** with next-season
  per-36 points and `team_pace` **+0.336**. Absorb season and they collapse to **+0.035**
  and **+0.091**. Floor spacing and pace both roughly doubled over the sample, and so did
  scoring, so any feature built from them tracks any rising outcome. The previously
  reported +0.096 and +0.087 were era trends in a team-context costume. Re-measuring the
  *superseded* construction with season effects gives `teammate_spacing` r = **+0.0007** —
  it was never there.
- **So "the rate/minutes split is where the value is" was wrong**, at least as argued from
  spacing. Once era is absorbed, every feature that matters pushes rates and minutes in the
  *same* direction: heavier teammate usage lowers both. Nothing cancels between them.
- **The real argument for component heads is cancellation *across components*, not between
  rate and minutes.** `teammate_assist_supply` is the clearest case: per sd it moves
  ast/36 by **−0.361**, reb/36 by **+0.218**, blk/36 by **+0.191** and tov/36 by −0.246,
  and almost none of it survives the DK sum. Weighting each component by its DK
  coefficient, it shifts **2.11 dk_pts of gross component movement into 0.25 net — an 8.3×
  cancellation**. `role_crowding` cancels 8.2×, `teammate_spacing` 5.0×, `team_pace` 4.7×.
  A single dk_pts head sees a small fraction of what the components see. This is the same
  structure already established for opponent effects (1.105 gross vs 0.785 net) and it is
  the surviving empirical case for the decomposition.
- **`teammate_usage_load` is created entirely by the correction.** Measured on the
  superseded S-1 roster it is nothing (r = +0.0006 vs per-36 pts, ΔR² = +0.0001); on the
  season-S roster it is the strongest own-team feature in the project (ΔR² = +0.0066 alone,
  more than the entire old block). That is exactly what the corrected construction was
  supposed to buy: "how much usage do my *new* teammates demand" is only a question you can
  ask about the roster that will actually play. On last season's roster it is mostly a
  restatement of P's own prior usage, which the controls already hold fixed.

  It is the minutes-weighted mean of teammates' prior usage scaled to a five-man lineup.
  The raw `teammate_usage_sum` is kept for continuity but grows with roster size, which is
  bookkeeping rather than basketball — `teammate_usage_load` is its roster-size-invariant
  form and measures ~3× better.

#### `role_crowding` is a settled null, and the old explanation was wrong

Under the corrected construction `role_crowding` still adds nothing: r = +0.048 vs
next-season per-36 points with season absorbed, −0.005 vs per-game dk_pts, and
ΔR² = +0.00001. Adding P's own prior per-36 style to the controls drops it to +0.020, so
even the residual is mostly "P is a scorer on a team of scorers" rather than crowding.

The previous explanation — coaches already resolved role redundancy when allocating
*prior-season* minutes, which is in the control set — **is now falsified**. Crowding on a
season-S lineup that did not exist in S-1 cannot have been absorbed by S-1 minutes, and it
is still a null. Two explanations remain, and they are not exclusive:

- Roster construction selects against true redundancy in the first place. Teams do not
  stockpile interchangeable players, so the variance to exploit is small.
- The mechanism is real but `teammate_usage_load` measures it directly and better. Cosine
  similarity between archetype membership vectors is a blunt proxy for "competes for my
  shots"; prior usage is the thing itself. Jointly, `role_crowding`, `teammate_spacing` and
  `team_pace` add **+0.0000** R² on top of the controls — the usage family carries the
  entire block (+0.0080 of the +0.0086).

Treat this as settled: archetype-similarity crowding is not a useful input. Usage is.

#### Roster members with no usable S-1 row

Quantified before building the aggregates, because dropping these players silently is the
largest bias available here. Weighted by **realized season-S minutes** — the exposure the
aggregate is failing to describe, and the only honest denominator, since weighting by the
S-1 minutes the aggregate actually uses would show a rookie contributing zero and every
roster looking fully covered:

| Population | Share of roster minutes | Share of roster head count |
|---|---|---|
| True rookies | 9.4% | 17.1% |
| Sub-threshold (played in S-1, failed `GP≥20 & MIN≥10`) | 5.4% | 12.6% |
| Returnees (played before, not in S-1) | 1.1% | 3.1% |
| **Total undescribed** | **15.9%** | **32.8%** |

Across 863 team-seasons the uncovered share has p50 = 14.1%, p90 = 32.9% and a maximum of
54.1% (PHI 2014-15, mid-Process). The README's prediction that the bias would be worst on
young, high-turnover rosters is confirmed by the worst-offender list: PHI 2014-15, CHA
2023-24, WAS 2024-25, HOU 2020-21. This is not a rounding error, so all three populations
are carried explicitly rather than filtered out:

- **Sub-threshold** — `season_matrix.py` now writes an unfiltered twin,
  `season_matrix_roster_tier*.parquet` (14,569 / 6,942 player-seasons vs 10,900 / 5,077
  qualified, +25%). Two consumers, two row sets: the `GP≥20 & MIN≥10` filter is a PCA
  concern, not a description of who was on the roster.
- **Shrinkage instead of exclusion.** Every prior-season description is multiplied by a
  reliability weight in z-space, which is exactly shrinkage toward the league average.
  `r(m) = 0.924 · m/(m+66)`, fitted to the observed season-to-season correlation of eight
  style stats across minutes bins on 11,272 consecutive-season pairs. Rates stabilize
  fast — the 200-minute qualification threshold already sits at r = 0.75 — which is why
  sub-threshold players are worth including rather than merely tolerable.
- **Returnees** fall back to their most recent season, decayed a further 0.96 per season of
  staleness (measured: lag-1…4 persistence of 0.921 / 0.881 / 0.853 / 0.826).
- **Rookies** are imputed from an **expanding-window** prior over historical rookies at the
  same draft-slot bucket (top-5 / lottery / late-first / second-round / undrafted), so a
  backtest never learns a slot's value from the future. Their reliability is 0 — the
  description is the pure prior — but they carry the bucket's expected *minutes*, which is
  what stops them from silently vanishing from the weighting.

  Their imputed membership vector is a **mean of row-normalized** memberships, not a
  normalized mean. Role crowding is a dot product of unit vectors, so
  `E[cos(m_P, m_rookie)] = m̂_P · E[m̂_rookie]` — averaging first is exact, and
  re-normalizing would claim a confident role we do not have.

Every row carries `stats_source` ∈ {`prior`, `stale`, `rookie`} and its `reliability`, and
every team-season carries `roster_coverage` — the share of roster weight that is observed
rather than imputed. Within the 10-game roster window: 84.1% `prior`, 14.1% `rookie`, 1.8%
`stale` by head count; `roster_coverage` averages 0.905 with p10 = 0.808. `roster_coverage`
is itself weakly predictive (ΔR² = +0.0005), so the model can discount thin aggregates
instead of being misled by them.

One residual look-ahead, inherent to deriving rosters from game logs: a player who misses
the team's first 10 games is not counted, and that is knowable only after the season
starts. It affects roster composition, not the outcome.

### Opponent encoding

Implemented in `src/features/opponent.py` (`make opponent`).

1. **Team-level defensive quality is free and already on disk.** The seven `team_stats_*`
   families and `team_estimated_metrics` are fetched, 30 seasons, one row per team, and
   were unused. `team_stats_opponent` is the one that matters: for team T its `OPP_*`
   columns are *what T's opponents recorded against T*, which is already the per-DK-
   component answer to "what does a player get against T?" — no player aggregation, and
   therefore none of the roster-coverage bias that would come with it.

   28 columns are put on a **per-100-possessions** basis, z-scored **within season**, and
   reduced to **5 PCs** (72% of profile variance), which come out interpretable:

   | | share | reads as |
   |---|---|---|
   | `opp_pc1` | 28.6% | overall defensive quality (`opp_pts`, `def_rating`, `opp_efg_pct`) |
   | `opp_pc2` | 14.6% | shot volume conceded vs fouls and turnovers forced |
   | `opp_pc3` | 11.8% | offensive rebounds conceded |
   | `opp_pc4` | 9.4% | 3-point profile conceded |
   | `opp_pc5` | 7.8% | 3-point volume conceded |

   Profiles are lagged one season and keyed on `team_id`, so relocations (SEA→OKC,
   NJN→BKN) stay joined to their own history. Because team identity is known at prediction
   time a **team embedding** is available as a complement — but prior-season *team* stats
   describe last season's roster, which may have turned over substantially.

   > ⚠️ **Basis quirk.** `fetch_team_stats` requests `PerMode="PerGame"` and the counting
   > stats do arrive per game, but `MIN` and `POSS` are **season totals** (`MIN` ≈ 3966 =
   > 48.4 × 82). The `stat / MIN * 36` rule that holds for the player families is wrong
   > here. Use `POSS / GP` for a per-100 basis.

2. **A low-rank matchup interaction.**

   ```
   matchup_k = (u_k · s_P)(v_k · o_T)        rank 1–2
   ```

   `s_P` is P's prior-season style (8 PCA dims), `o_T` the opponent's prior-season defensive
   profile (5 dims) — 26 parameters at rank 2, fitted by alternating least squares
   (biconvex, so each half-step is a closed-form ridge).

   It recovers exactly the predicted mechanism. For blocks, the fitted style axis
   correlates **+0.74 with height, +0.71 with blocks/36 and −0.55 with 3PA/36** — it is the
   big/guard axis, interacted with how much a defense concedes at the rim.

#### Measured: the main effect is the reliable part, the interaction lives in the components

Held out on 2024-25 and 2025-26 (39,709 player-games), as a share of within-player-season
residual variance. Per-36 rates are **minutes-weighted** — a per-36 rate from a 3-minute
appearance has ~6× the sd of one from a 30-minute game (`sd(pts_per36)` is 42.4 under 5
minutes against 6.6 above 24), and unweighted those rows swamp every rate regression:

| Outcome | Main effect | + interaction | Shuffled null | Interaction above null |
|---|---|---|---|---|
| `dk_pts` | **0.368%** | 0.388% | 0.366% | **+0.022%** |
| `blk_per36` | 0.237% | 0.440% | 0.237% | **+0.203%** |
| `pts_per36` | 0.357% | 0.421% | 0.355% | +0.066% |
| `reb_per36` | 0.168% | 0.218% | 0.168% | +0.051% |
| `ast_per36` | 0.361% | 0.411% | 0.363% | +0.049% |
| `fg3m_per36` | 0.091% | 0.111% | 0.088% | +0.023% |
| `stl_per36` | 0.215% | 0.228% | 0.216% | +0.012% |
| `tov_per36` | 0.399% | 0.383% | 0.399% | −0.016% |

**This reverses the guidance this section used to give.** The previous instruction was
"encode opponent as an interaction with player style, not as a standalone team-quality
term," on the strength of +1.00% for the interaction against 0.69% for the main effect.
Out of sample, using only what is knowable before the season, the **main effect beats the
interaction by ~17× on dk_pts** (0.368% vs +0.022%). Build the main effect properly; keep
the interaction because of what it does *per component*, not for the aggregate.

Three reasons the old comparison did not survive:

- **It compared an in-sample ceiling with an achievable gain.** "Opponent × season" and
  "opponent × archetype × season" are one-way ANOVAs using *contemporaneous* opponent
  identity. A usable encoder knows only the opponent's **prior** season, and the realized
  opponent-season effect correlates just **0.557** with its own previous value. Most of
  that ceiling is not reachable in principle.
- **The +1.00% depends on which null you pick.** On the original 2014-15→2023-24 window the
  same statistic is **+0.969%** if the null permutes opponent within season and **+0.311%**
  if it permutes the archetype label — same data, same cells, a 3× swing. With ~2,700 cells
  over 198k rows, expected chance R² is already ~1.4% against a raw statistic of 2.31%. The
  headline was mostly measuring cell count.
- **The interaction cancels across components, just like everything else here.** It is
  worth +0.203% on blocks — nearly doubling what the main effect gets — and +0.05–0.07% on
  points, rebounds and assists, but the signs disagree and only +0.022% survives into
  dk_pts. This is the third independent confirmation of the cross-component cancellation
  that motivates the multi-head design, and it is the reason to keep the interaction at all.

Rank barely matters: on dk_pts the above-null gain is +0.018% / +0.022% / +0.025% at ranks
1 / 2 / 3. At rank 2 the two style axes come out near mirror images of each other, so the
structure is effectively rank 1. Rank 2 is kept as the configured default; anything richer
is fitting noise against an effective support of 892 team-seasons.

The reproductions of the ceiling figures live in `variance_ceiling()`, so the numbers
quoted in the variance-budget table above have a runnable source.

### Which prior-season features are worth feeding

`make persistence` → `outputs/eda/persistence.csv`. Every column of the season matrix,
season *t* against *t+1* for the same player, **with season absorbed** and **minutes-weighted**
for the per-36 columns, over 11,272 consecutive-season pairs. This is the candidate feature
list for all three model families at once.

The expected outcome recorded here — "usage/rebound/assist rates are sticky, `FG3_PCT` and
`PLUS_MINUS` are noise" — is confirmed, but the useful cut is not the one that was
anticipated:

**It is not counts vs. percentages. It is *share* vs. *conversion*.**

| Sticky | *r* | Noise | *r* |
|---|---|---|---|
| `bas_reb` | 0.941 | `bas_fg3_pct` | 0.500 |
| `bas_oreb` | 0.924 | `bas_fg_pct` | 0.435 |
| `bas_ast` | 0.920 | `bas_ft_pct` | 0.365 |
| `bas_fg3a` | 0.908 | `adv_ts_pct` | 0.302 |
| `sco_pct_fga_3pt` | **0.886** | `adv_off_rating` | 0.285 |
| `usg_pct_fg3a` | **0.870** | `adv_efg_pct` | 0.281 |
| `usg_pct_reb` | **0.854** | `clu_*` (median) | 0.25 |
| `adv_ast_pct` | **0.845** | `adv_net_rating` | 0.180 |
| `adv_usg_pct` | 0.786 | `adv_def_rating` | 0.116 |

(bolded rows are *share* percentages, not counts)

Shot-mix and usage **shares** persist essentially as well as the counts they come from;
**conversion** percentages do not. "Shrink the percentages" applies only to the conversion
side — the share columns are first-class features and should not be shrunk with them.

Four further results, each of which changes what gets fed:

- **Whole families are noise.** Clutch (`clu_*`, every column ≤ 0.55), the four rating
  columns, `plus_minus` (0.477), `sl_backcourt_*` (≈ 0.01), and every tracking *conversion*
  column (`drv_drive_fg_pct` 0.162, `pu_pull_up_fg3_pct` 0.115). Drop them wholesale.
- **A pooled ranking promotes exactly those columns.** The largest pooled-vs-within gaps are
  `adv_e_pace` (0.574 → 0.212), `def_opp_pts_paint` (0.715 → 0.425), `adv_pace`
  (0.444 → 0.190) and `def_rating` (0.333 → 0.116). Pace and ratings drifted with the era;
  pooled, they look like signal. This is the same trap that produced the false
  `teammate_spacing` finding, now hit a second time in a different module.
- **Minutes weighting changes the ranking, not just the coefficients.** Median **+0.214** r
  across the 72 minutes-weighted Tier A columns. `stl` reads 0.743 weighted against 0.401
  unweighted, `tov` 0.790/0.456, `blk` 0.902/0.626. Unweighted, the low-count defensive
  stats look like noise when they are among the stickiest signals a player has.
- **Tier B's tracking families earn their 13-season cost.** `drv_drives` 0.922,
  `pu_pull_up_fga` 0.918, `pass_potential_ast` 0.913, `hus_contested_shots_2pt` 0.909 — the
  *volume* side of tracking is as persistent as the box score. Only the tracking
  percentages are noise, exactly as in the box score.

**Reliability is per column.** `r(m) = r_inf · m/(m+m0)` is refitted for every column rather
than reusing the single `0.924 · m/(m+66)` that `team_context.py` applies. Median `m0` is
**364** minutes (p10 112, p90 ≥ 2000), and minutes-for-r = 0.75 spans 192 (`fg3a`) to over
20,000 (clutch), median **1,416**. The global curve was fitted on eight *style* stats — the
fast-stabilizing end of the distribution — so it over-credits the median column. Lag decay
is column-specific too: `usg_pct_reb` runs 0.854 → 0.843 over lags 1–4 while `bas_pts` runs
0.861 → 0.698, so `team_context.py`'s flat 0.96/season staleness decay is too fast for
shares and too slow for volume.

### Head likelihoods, and the shape of each target

`make target-profile` → `outputs/eda/target_profile.csv`. One measurement, three consumers:
it picks a GLM family per component, a GBM objective per component, and validates
`multihead.py`'s Poisson-with-minutes-offset.

- **Poisson is correctly specified on the shot classes and badly misspecified on `pts` and
  `dk_pts`.** Within-player var/mean, per minutes bucket: `fgm` 0.94–1.03, `fg2m` 0.97–1.02,
  `fg3a` 0.86–0.97, `tov` 0.97–1.00, `stl` 1.02–1.04, `blk` 1.02–1.10 — against `pts`
  2.16–2.31 and `dk_pts` 2.26–2.58 in *every* bucket. The DK/point weighting is the entire
  source of the overdispersion; decomposing removes the misspecification rather than
  patching it with a negative binomial. Free throws are the exception (`ftm` 1.70–1.91,
  `fta` 1.95–2.08) because they arrive in pairs — model trips to the line and double them.
- **Zero-inflation is a minutes artifact, not a property of the target.** `dk_pts` is 0 in
  35.4% of sub-5-minute games and **0.0% above 18 minutes**. Conditional on minutes there is
  nothing to zero-inflate — the zeros are Poisson zeros, which is exactly what the
  availability × rate factorization already assumes. But in 30–48 minute games `blk` is
  still 0 in 58.6% of games, `fg3m` 42.2% and `stl` 33.8%: those three heads are genuinely
  low-count and misspecified under plain MSE at *any* minutes level.
- **Residual dispersion is worst at low *usage*, not low minutes.** `fga` var/mean is 1.62
  below 0.15 usage against 1.32 above 0.30; `reb` 1.72 vs 1.22; `pts` 3.28 vs 2.64. If a
  dispersion parameter is added, key it on usage.

**The running season total.** Extrapolating the first-k-game mean over the games actually
played — so this isolates the *rate* and holds availability known — settles the total
almost immediately:

| First k games | n | r | R² (extrapolated) | MAE (dk_pts) |
|---|---|---|---|---|
| 5 | 12,996 | 0.936 | **0.859** | 238 |
| 10 | 12,541 | 0.956 | 0.905 | 193 |
| 20 | 11,533 | 0.972 | 0.942 | 148 |
| 41 | 9,113 | 0.988 | 0.974 | 94 |

And the total splits cleanly into its two factors: `log(season total)` is **84.5%** explained
by `log(per-game rate)` alone and **73.4%** by `log(games played)` alone, with games played
carrying sd 20.8 on a mean of 56 (p10 23, p90 80). Both halves are large; the games half is
the one that barely persists (r = 0.316, above).

### Where age acts

`make aging` → `outputs/eda/aging_curves.csv`. Delta method (mean within-player change
between consecutive seasons), era-adjusted and minutes-weighted, indexed to age 23.

**The cross-sectional curve is worthless and the output proves it.** Mean per-36 DK-linear
reads 31.1 at age 19, 31.9 at 23, 31.6 at 30, 31.4 at 34 and **34.3 at 39** — flat, then
rising, because weak 34-year-olds are out of the league. The delta method recovers a real
arc from the same data. Both are written out side by side (`cross_sectional_mean` next to
`cumulative`) so the bias is visible rather than asserted.

| Age | 19 | 23 | 26 | 30 | 34 | 37 |
|---|---|---|---|---|---|---|
| per-36 DK-linear | 0.769 | 1.000 | **1.041** | 1.001 | 0.895 | 0.791 |
| minutes/game | 0.593 | 1.000 | 1.100 | 1.043 | 0.792 | **0.515** |
| games played | 0.890 | 1.000 | 1.039 | 1.045 | 0.952 | 0.870 |

**Aging is an availability story.** The rate arc is ±15% peak-to-34; the minutes arc is
−54% peak-to-37. Age belongs in the availability head, not spread across the rate heads.

Two refinements that a single curve would hide:

- **Components age differently.** At 34: `reb/36` 0.977, `blk/36` 1.011, `stl/36` 0.962 —
  essentially flat — against `pts/36` 0.844, `ast/36` 0.867, `fg3m/36` 0.872. Big-man
  counting stats barely age; skill and scoring do. A single `dk_pts` age curve averages two
  opposite shapes, which is one more argument for component heads.
- **Archetypes age differently.** The 3-point-heavy cluster peaks late (1.10 at 28–30) and
  is still 1.029 at 34; "low scoring volume, low usage" peaks at 26 and is 0.805 at 34.

The output is deliberately long and un-fitted — one row per (tier, metric, archetype, age) —
because a GLM wants a spline basis, a GBM wants raw age as a column, and the NN wants a
feature or embedding. `cumulative_ratio` is the age-aware replacement for
`team_context.py`'s flat 0.96/season decay.

### Model-family failure modes

`make feature-diagnostics` → `outputs/eda/feature_diagnostics.csv`. Not a signal ranking —
that is `persistence.py` — but the three ways choosing a model family can go wrong.

**1. The raw feature matrix is singular, not merely collinear.** Tier A has rank 144 of 149
columns, Tier B 277 of 285; the condition number is infinite. Sixteen Tier A columns have
*exactly* infinite VIF because families ship literal duplicates of each other:

```
adv_def_rating == def_def_rating      usg_pct_blk  == def_pct_blk
adv_dreb_pct   == def_dreb_pct        usg_pct_stl  == def_pct_stl
usg_pct_dreb   == def_pct_dreb        sco_pct_fga_2pt + sco_pct_fga_3pt == 1
```

So an unpenalized GLM or OLS on the raw matrix does not merely fit badly — it fails.
Ridge/elastic-net or a pruned set is mandatory, and that is a decision with evidence rather
than an assumption. 104 of 149 Tier A columns have VIF > 10 (median 92) and 60% have a
partner at |r| > 0.9. Pruning at |r| ≥ 0.95 collapses 74 Tier A columns into 22 clusters,
leaving **97 of 149**; Tier B collapses 152 into 50, leaving **183 of 285**. The largest
Tier A cluster is the entire 9-column rebound family (`bas_reb`, `bas_dreb`, `def_dreb`,
`adv_reb_pct`, `usg_pct_reb`, `adv_dreb_pct`, `usg_pct_dreb`, `def_dreb_pct`,
`def_pct_dreb`).

**2. Cardinality and cold start.** Key every team feature on `team_id`, never on
`team_abbreviation`:

| Categorical | Categories | Min n | Cold start |
|---|---|---|---|
| `team_id` | 30 | 317 | none |
| `team_abbreviation` | **36** | **24** | NOK, VAN, CHH, NOH, SEA — as few as 2 seasons |
| `archetype` | 9 (A) / 6 (B) | 540 / 377 | none |
| `draft_bucket` | 5 | 1,456 | none |

Relocations split their own history under the abbreviation and keep it under the id — which
is what `opponent.py` already does, now with the cardinality evidence behind it. `archetype`
and `draft_bucket` are comfortably large enough to one-hot; no target encoding is needed.

**3. The prior-season game sequence is worth ~+0.6 pp R².** Held out on the last two seasons
(10,215 train / 889 test), predicting next-season per-game `dk_pts`:

| Features | Held-out R² |
|---|---|
| Four season aggregates (mean dk, sd dk, mean minutes, games) | 0.7599 |
| + five order features (dk & minutes slope, lag-1 autocorr, last-10 gap, half-to-half gap) | **0.7657** |
| The same five order features on **shuffled** game order | 0.7584 |

`+0.0059` over aggregates and `+0.0074` above its own shuffled null — real, and about 0.8%
relative. **The LSTM/Transformer trunk over prior-season game logs is mostly re-deriving a
season mean that `season_matrix.py` already holds**, at much higher cost. That does not
forbid a sequence trunk, but it prices one.

**4. A reusable shuffled-null helper**, generalizing the one-off in
`opponent.py::variance_ceiling`. It reproduces the finding it was built from: +0.9619%
permuting opponent and +0.3080% permuting archetype, against the one-off's +0.9570% and
+0.3056% — gaps of 0.005 and 0.002 pp. Route any future GBM or NN importance ranking through
it so a score always ships with its own chance level.

**Bonus, from the archetype sweep** (`outputs/eda/archetype_sweep_tier*.csv`): k-means
silhouette decreases monotonically in k and peaks at 0.181 (Tier A) / 0.241 (Tier B) at
k = 4 — no k shows real separation. GMM BIC picks k = 9 and k = 6, which is what is fitted.
**Archetypes are a partition of a continuum, not discovered clusters.** Use the soft
membership vector, which is what `team_composition_*.parquet` already stores; a hard
archetype label claims structure the data does not have.

### Schedule features

The other knowable per-game lever: days rest, back-to-backs, games in the last 7 days,
road-trip position, home/away. `src/features/matchup.py::add_back_to_back_flag` exists.

### Leakage discipline

- **Never use same-game teammate stats.** Teammate minutes in game G are contemporaneous
  with P's performance in G — if someone is hurt, P's minutes rise and the feature encodes
  the outcome.
- **Own-team composition must exclude P.** Otherwise the feature partly encodes P's own
  style and the model can read part of the answer off its input.
- **Rookies have no prior season** — needs a fallback (`bio_draft_number` is in the matrix).
- Train/val/test is a **temporal walk-forward by season**; the scaler is fit only on the
  prior-season data feeding training.

### One honest caveat on the constraint — now measured

Prior-season-only is restrictive. Allowing games 1..k of the *current* season would beat
every team-composition feature described here by a wide margin. That was an assertion when
this document was written; `make target-profile` put a number on it.

**Five games of the current season settle 86% of the season total** (R² 0.859, MAE 238
dk_pts), and twenty settle 94%. Against that, the entire own-team context block is worth
+0.0086 R² and the opponent main effect 0.368% of within-player residual variance. The gap
is roughly two orders of magnitude.

Nothing in this repo is wasted by that — the prior-season features are exactly what a
before-the-season forecast has, and they are what makes game 1 predictable at all. But it
does set the honest frame: if the constraint is a hard product requirement, fine; if it is a
modeling convention, that is where the leverage is, and it is not close.

---

## Implementation plan

Staged so each step is inspectable. Stages 0–2 are complete.

| Stage | Scope | Status |
|---|---|---|
| **0** | Repair `fetch.py`; backfill 6 `player_stats_defense` seasons | ✅ done |
| **1** | `src/eda/season_matrix.py` — one row per (player, season) + coverage report | ✅ done |
| **2** | `src/eda/pca.py`, `src/eda/archetypes.py` — style axes and archetypes | ✅ done |
| **5** | `src/features/team_context.py` — LOO own-team encoder, roster(S) × stats(S-1) | ✅ done |
| **5** | `src/features/targets.py` — component targets + calibrated bonus | ✅ done |
| **6** | `src/models/multihead.py` — availability × per-36 rate heads | ✅ done |
| **5** | `src/eda/context_value.py` — what own-team context is worth | ✅ done |
| **7** | `src/features/opponent.py` — team defensive profile + low-rank matchup | ✅ done |
| **4** | `src/eda/persistence.py`, `aging.py`, `target.py`, `feature_diagnostics.py` | ✅ done |
| **3** | `dashboard/app.py` — 9-tab Streamlit explorer over every artifact above | ✅ done |
| **7** | Wire the multi-head model into `train.py` / `evaluate.py` | planned |

Stage 4's `persistence.py` was the most load-bearing for modeling, and its expected outcome
held: usage/rebound/assist rates are sticky (0.79–0.94) while `FG3_PCT` (0.500) and
`PLUS_MINUS` (0.477, and 0.154 unweighted) are noise. What it added beyond the expectation —
the share/conversion split, the noise families, and games played at r = 0.316 — is in
"Which prior-season features are worth feeding" above.

With Stage 4 and the dashboard landed, the remaining work is fitting: wire the multi-head
model into `train.py`/`evaluate.py`, then the GLM and GBM arms of the ensemble against the
same component targets. 242 tests pass (`.venv/bin/pytest tests/`).

See `docs/eda-plan.md` for the detailed EDA specification.

---

## Data

30 seasons, 1996-97 → 2025-26, ~284 MB of raw CSVs in `data/raw/`, ~30 season-level
families plus per-game logs.

| Artifact | Rows | Features |
|---|---|---|
| `season_matrix_tierA.parquet` | 10,900 player-seasons | 150 |
| `season_matrix_tierB.parquet` | 5,077 player-seasons | 287 |
| `season_matrix_roster_tierA.parquet` | 14,569 player-seasons | 150 |
| `season_matrix_roster_tierB.parquet` | 6,942 player-seasons | 287 |
| `team_composition_tierA.parquet` | 892 team-seasons | 9 archetype shares + aggregates |
| `team_context_tierA.parquet` | 11,928 player-seasons | 8 LOO own-team features |

The `season_matrix_roster_*` frames are the unfiltered twins of the qualified matrices —
same columns, every player who took the floor. The qualified frames serve the PCA; the
roster frames serve roster aggregation. See "Roster members with no usable S-1 row".

Human-readable analysis reports land in `outputs/eda/` (artifacts that later modeling
consumes go to `data/features/`):

| Report | Rows | What it holds |
|---|---|---|
| `persistence.csv` | 437 | Per column × tier: pooled / within-season / unweighted *r*, era gap, lags 1–4, fitted reliability curve |
| `aging_curves.csv` | 2,430 | Per (tier, metric, archetype, age): delta-method mean change, era-adjusted, cumulative curve, cross-sectional twin |
| `target_profile.csv` | 185 | Per component × minutes/usage bucket: var/mean, dispersion, zero share, skew — plus first-k-games → season total |
| `target_season_totals.parquet` | 13,895 | Per player-season: realized total and each first-k prefix mean |
| `feature_diagnostics.csv` | 449 | VIF, strongest-correlate and cluster per column; cardinality per categorical; null reproduction; sequence ablation |
| `feature_correlation_tier*.parquet` | 149 / 285 | The full within-season z-scored correlation matrices |
| `archetype_sweep_tier*.csv` | 11 | Silhouette and BIC over k = 4..14 |
| `team_context_value_tier*.csv` | 8 | Per own-team feature: ΔR² and partial *r* against every component |
| `opponent_matchup_tierA.csv` | 8 | Held-out main effect / interaction / shuffled null per outcome |

**Tier A** is 30 seasons of box-score-derived families. **Tier B** is 13 seasons
(2013-14+) adding the full tracking/hustle families, and is a strict column superset.

Coverage boundaries: core families 30 seasons; tracking and `pt_shot` 13 (2013-14+);
estimated metrics 12 (2014-15+); hustle 11 (2015-16+, and 2015-16 covers only 147 players
— a partial first-season rollout, not a fetch failure).

---

## Known gaps and potential to-dos

Not blocking, but each one costs some accuracy. Logged here so they don't get rediscovered.

### 1. Roster members with no usable prior-season stats — ✅ measured and handled

Quantified at **15.9% of roster minutes** and addressed in `src/features/team_context.py`;
see "Roster members with no usable S-1 row" above for the numbers and the treatment
(inclusive roster frame, minutes-proportional shrinkage, stale fallback, draft-slot rookie
prior, `roster_coverage`).

What remains open, in order of likely value:

- **The rookie prior is a draft-slot bucket mean and nothing more.** Rookies are 9.4% of
  roster minutes — the largest remaining population — and physicals, age and college
  production are all available or fetchable. NCAA/G-League stats need new fetching and are
  out of current scope.
- **Returnees get a flat 0.96/season decay, not an age adjustment.** The aging curves from
  `src/eda/aging.py` (Stage 4) are the natural source for a real one. Only 1.1% of minutes,
  so this is low priority.
- **The opponent encoder does not use any of this yet.** The same bias applies to any
  opponent aggregate built from player rows, though reading opponent quality from the
  team-level families (as planned) sidesteps it entirely.

### 2. Also logged

- **Mid-season trades** are not modelled: a traded player is attributed wholly to his last
  team, and ~13.6% of players appeared for 2+ teams in 2023-24. See the prediction-time
  constraint above.
- **2015-16 hustle covers only 147 players** (partial first-season rollout). Tier B's
  2013-14 and 2014-15 have no hustle/estimated columns at all, and the PCA imputes them to
  the season mean — a defensible but visible choice, surfaced in `coverage_report.csv`.
- **`bio_draft_*` is ~15% NaN** by construction (undrafted players), which is correct but
  means draft-slot priors need an explicit undrafted category rather than an imputed number.

## Quick start

```bash
make install          # create .venv and install requirements
make fetch            # pull raw data from nba_api (long)
make eda              # the full EDA sweep — see "Season-level EDA pipeline" in CLAUDE.md
make dashboard        # Streamlit explorer at http://localhost:8501
make test             # pytest
```

Always use `.venv` — never the system Python. See `CLAUDE.md` for conventions.

---

## Running the dashboard locally

A nine-tab Streamlit explorer over everything the EDA pipeline produces.

```bash
make dashboard
```

Then open **http://localhost:8501**. Stop it with `Ctrl-C`.

`make dashboard` is just `.venv/bin/streamlit run dashboard/app.py`, so anything
Streamlit accepts works too:

```bash
.venv/bin/streamlit run dashboard/app.py --server.port 8600   # different port
.venv/bin/streamlit run dashboard/app.py --server.address 0.0.0.0  # expose on the LAN
```

### Before the first run

The dashboard **only reads precomputed artifacts** — it never fetches, fits a PCA, or
trains anything, so it starts in about a second. That also means the artifacts have to
exist first:

```bash
make install    # once
make fetch      # once, and slow — pulls ~284 MB from nba_api
make eda        # builds data/features/ and outputs/eda/  (~2 minutes)
make dashboard
```

If an artifact is missing the affected tab says which `make` target produces it and the
rest of the app still works, so a partial pipeline is browsable rather than broken.

### The tabs

| Tab | What it answers |
|---|---|
| **Coverage** | Which raw families exist per season; the share of each roster described by real prior-season data |
| **PCA** | Scree, biplot with player hover, loadings, a single player's career through style space |
| **Archetypes** | Cluster profiles, representative rosters, team composition by season |
| **Persistence** | What survives a year — pooled vs season-absorbed *r*, the *t* vs *t+1* scatter, fitted reliability curves |
| **Aging** | Delta-method age curves against the cross-sectional curve, so survivorship bias is visible |
| **Target** | Per-component mean-variance by minutes (which likelihood each head needs), and the running season total |
| **Team context** | What own-team context is worth above a player's own prior season |
| **Opponent** | Defensive-profile axes, teams plotted in that space, the held-out matchup table |
| **Feature diagnostics** | Collinearity, cold-start cardinality, the shuffled-null check, sequence-vs-aggregate ablation |

**Sidebar controls** scope every tab at once: **Tier** (A = 30 seasons box-score, B = 13
seasons with tracking), **Era mode** (`within_season` is era-neutral and feeds modeling;
`pooled` makes era a visible axis), and **Appearance** (light/dark — chart colours are
selected per mode, not flipped). Every chart has a table-view twin in an expander beneath
it, so no value is reachable by colour alone.

### Troubleshooting

- **`make dashboard` exits with `Error 255` and an `Email:` prompt** — Streamlit's
  first-run onboarding. `.streamlit/config.toml` in the repo sets `headless = true` to
  suppress it; make sure you are running from the repo root so that file is picked up.
- **Port already in use** — an earlier instance is still running. `pkill -f "streamlit
  run"`, or use `--server.port` to pick another.
- **A tab warns that a file is missing** — run the `make` target it names. `make eda`
  rebuilds everything.
