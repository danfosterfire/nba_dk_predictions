# NBA Best Ball Points Prediction

Predict a player's fantasy score (`dk_pts`) for **each game** of an upcoming NBA season,
using only information available before the season starts — then use the full posterior to
draft better teams in DraftKings best-ball tournaments.

A chain of Bayesian models fitted in Stan turns 30 seasons of box scores into simulated
seasons; simulated seasons become draft boards, simulated snake drafts and tournament
backtests; and the same artifacts power a live draft-night recommender.

This file is the overview and deliberately quotes only headline figures. `CLAUDE.md` routes
to the detailed plan docs in [docs/](docs/), which hold the designs, the measurements and
the caveats; [dashboard/decisions.py](dashboard/decisions.py) is the audited decision log.

---

## 1. The problem

Scoring is a linear function of the box score plus a nonlinear bonus:

```
dk_pts = 1.0·PTS + 0.5·FG3M + 1.25·REB + 1.5·AST + 2.0·STL + 2.0·BLK − 0.5·TOV + bonus
bonus  = 1.5 for a double-double, 4.5 for a triple-double
```

implemented once, in [src/data/preprocess.py](src/data/preprocess.py) (`compute_dk_pts`).
The contest ([docs/dk_best_ball_rules.md](docs/dk_best_ball_rules.md)): 16 players drafted
in a snake draft, frozen for the season, the best 7 by slot scoring each week, and four
elimination rounds.

**The defining constraint is prediction time.** The draft happens after the preseason and
before the opener, so the model may know the schedule, the season-start rosters, everyone's
prior-season statistics and the current season's preseason box scores — and cannot know
within-season trades, regular-season minutes, injuries, or form. Every feature except the
schedule-derived ones is therefore constant within a player-season.

**Where the variance is** (`make variance-budget`): player-season identity is ~58% of total
per-game variance, and of the within-player remainder, 46.4% is the player's own minutes —
unknowable in advance. Per-game modulation (opponent, home/away, rest) is worth 1–2% at
most. So nearly all attainable skill is getting each player's season-level *distribution*
right, and the per-game matchup is close to noise.

And the contest pays for distribution *shape*, not means: the double-double bonus is a
simultaneous threshold on five components, a best-ball lineup is a weekly max over 16
players, and Round 1 of the payout is a zero-consolation knockout. That is why the
deliverable is a joint draw from a posterior rather than a sharper point estimate.

## 2. Architecture

The pipeline is a chain of `make` targets ([docs/pipeline.md](docs/pipeline.md)): raw data →
features → Stan heads → persisted posteriors → simulated seasons → drafts and tournaments →
dashboard.

### Data

30 seasons (1996-97 → 2025-26) pulled from `nba_api` by
[src/data/fetch.py](src/data/fetch.py) and cleaned into 731,906 regular-season player-games
by [src/data/preprocess.py](src/data/preprocess.py). Four capture programs feed the
availability and market side: box-score inactive/DNP status, the NBA's daily injury-report
PDFs, the ESPN status feed, and DraftKings / FantasyPros ADP boards. Three of those are not
backfillable, so `make daily-capture` stays on a cron ([docs/adp-plan.md](docs/adp-plan.md)).
Season-level feature matrices are assembled by the EDA pipeline in [src/eda/](src/eda/)
(`make eda`), one module per artifact.

### Models

Twelve quantities are modelled per player-game — availability, minutes, seven counts, the
three-point attempt share, and three conversion percentages — and `dk_pts` is reassembled
exactly rather than ever predicted directly
([src/features/targets.py](src/features/targets.py)). The generative structure is a chain of
conditionals:

```
availability → min | available → counts | min → makes | attempts
```

The parameter blocks are disjoint, so the joint posterior factorizes exactly and each head
is fitted separately — four `.stan` sources in [src/stan/](src/stan/) serve twenty-plus
heads (`make stan`), driven from [src/models/](src/models/):

- **Availability** — a beta-binomial mixture over games played out of team games, shrunk
  hard toward a league/age baseline. Games played is the largest lever on a season total
  and the least persistent quantity in the project, so this head gets the most structure:
  a disrupted season is its own mixture component, and the simulator lays absences out in
  spells rather than iid games ([docs/availability-window-plan.md](docs/availability-window-plan.md)).
- **Minutes** — two heads that ship together: a marginal head for the season-level spread,
  and a team-game *composition* that allocates each game's 5 × game-length minutes among
  the players who played, making the team total exact and teammate-absence redistribution
  a fitted quantity. A role-graded per-(player, season) effect injected at draw time gives
  the composition the season-level spread that iid per-game draws cannot produce
  ([docs/minutes-window-plan.md](docs/minutes-window-plan.md),
  [docs/draw-time-calibration-plan.md](docs/draw-time-calibration-plan.md)).
- **Components** — seven negative-binomial counts with minutes as exposure, the three-point
  share of attempts, and three conversion percentages, each gated against a no-fit
  carry-forward floor ([docs/predictions-plan.md](docs/predictions-plan.md)).
- **Game length** — overtime onset and depth, so a forward simulation can draw the
  48-versus-53-minute denominator it cannot look up.

Cross-component correlation enters at draw time, not fit time: one shared minutes draw per
player-game reaches every head as exposure, plus a Gaussian copula on the residual. Most
heads also carry current-season preseason features
([docs/preseason-plan.md](docs/preseason-plan.md)).

Selection runs on a temporal walk-forward split — train 1997-98 → 2021-22, validation
2022-23 and 2023-24, test 2024-25 and 2025-26 — and the test split is enforced by code
rather than discipline: [src/models/held_out.py](src/models/held_out.py) hands back guarded
frames that raise when read, and only `make final-evaluation` unlocks them
([docs/train-validate-test-split.md](docs/train-validate-test-split.md)).

### Simulation

`make simulate-season` ([src/sim/season.py](src/sim/season.py)) draws whole seasons from the
joint posterior — availability spells on the real schedule, game length, the minutes
allocation, component counts, the copula, the bonus — and writes a
`player × scoring_period × sim` tensor of `dk_pts` per season. Everything is a draw, never a
plugged-in mean, because thresholds make `E[bonus] ≠ bonus(E[x])`
([docs/simulations-plan.md](docs/simulations-plan.md)).

### Drafting and tournaments

The tensor becomes a draft ranking, executed against a market: `make adp` builds a
point-in-time-safe ADP panel, `make draft` runs snake drafts against an ADP-based field,
`make bracket` plays the four-round contest under the five real captured payout structures,
and `make strategy-sweep` backtests 24 strategies × 5 structures × 2 validation seasons at
500 simulated worlds each, plus a replay against realized box scores. Rake is expressed as a
break-even edge hurdle (+10.45% to +17.60%), so a measured edge has a unit it can be
compared in ([dashboard/economics.py](dashboard/economics.py)).

### Serving

`make dashboard` is a Streamlit surface over the artifacts — it reads artifacts only and
never refits. [dashboard/draft_room.py](dashboard/draft_room.py) is the live draft-night
recommender: one click per pick, pricing each candidate inside the roster already held; it
also runs standalone as `make draft-room`, because draft night is a thirty-second clock.

## 3. Results at a glance

Each figure is reproduced by the `make` target beside it; `make docs-audit` re-derives every
quoted figure here and in the plan docs from its artifact and fails the build on
disagreement.

- **Availability is the largest measured win** (`make season-total`): the head is worth
  −210 dk_pts of season-total MAE against assuming a full season (610.8 → 400.5), and an
  oracle on games played (MAE 214.4) beats an oracle on the scoring rate (261.9) — the
  availability distribution is where the effort belongs.
- **The rate side is nearly saturated by prior-season information**
  (`make stan-components`): a no-fit floor scores validation R² 0.81–0.95 on the count
  heads, and fitted heads clear it by small margins. Preseason features are the exception —
  on several heads the preseason block is worth more than fitting itself
  ([docs/preseason-plan.md](docs/preseason-plan.md)).
- **The minutes composition beats the independent draw on its own metric**
  (`make stan-composition`): −8.94% CRPS per team-game while hitting the team total
  exactly. At the season unit its raw spread is 4.58× too narrow, which the injected
  role-graded σ closes (`make minutes-unification`, `make minutes-role-sigma` — the full
  readout is [docs/minutes-window-plan.md](docs/minutes-window-plan.md) §6).
- **The drafting edge is large in simulation and unconfirmed in the realized replay — which
  is the result, not a caveat** (`make strategy-sweep`): the shipped strategy lifts Round-1
  advance probability by ~0.24 in the 600k tournament's simulated worlds and by ~0.20
  replayed against the two validation seasons' realized box scores, but with only two
  realized seasons the intervals select nothing.

## 4. Repository map

```
src/data/       fetch, preprocess, availability capture (box-score status, injury
                reports, ESPN feed), ADP capture, and the capture calendar
src/features/   component targets, game length, team context, opponent, availability,
                ADP, and scoring periods — the NBA week grid DK's rounds sit on
src/eda/        the season-level analysis pipeline — one module per artifact
src/models/     the Stan heads (availability, minutes, composition, components, game
                length, season terms), the sklearn references they are checked against,
                and posteriors.py, which persists thinned draws so nothing downstream
                refits; model_cards.py flattens them for the dashboard
src/stan/       four .stan sources for twenty-plus heads
src/sim/        the simulation and drafting layer — numpy over the posterior artifacts.
                season.py writes THE tensor, bracket.py the contest, draft.py the snake,
                strategy.py the sweep, draft_room.py the live recommender's engine
dashboard/      Streamlit views over the artifacts; reads artifacts only, never refits
docs/           the plan docs — designs, measurements, decision history (see CLAUDE.md)
```

## 5. Quick start

```bash
make install          # create .venv and install requirements
make fetch            # pull raw data from nba_api (long)
make eda              # the full season-level EDA sweep
make stan             # fit the availability, minutes and component heads
make dashboard        # the Streamlit views at http://localhost:8501
make test             # pytest
```

The Stan heads need a CmdStan toolchain, which pip does not manage:

```bash
.venv/bin/python -c "import cmdstanpy; cmdstanpy.install_cmdstan()"
```

Always use `.venv`, never the system Python. Two guards run over the documentation itself:
`make docs-audit` (a gate — quoted figures must match their artifacts) and
`make dashboard-audit` (a report on decision-registry drift). See `CLAUDE.md` for
conventions.
