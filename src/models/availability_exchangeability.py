"""The exchangeable-trials assumption — where it breaks, and at which unit it is visible.

`make availability-exchangeability` → `availability_exchangeability.csv`,
`availability_clustering.csv`.

## The item

`docs/availability-window-plan.md` §9 item 5, inherited from §7g. The availability head is
a beta-binomial on `gp` out of `team_games`, which asserts that — given the player-season's
frailty draw — the schedule's ~82 games are **exchangeable Bernoulli trials**. They are not.
Absences come in spells: one 40-game spell and forty single-game absences give the identical
`gp` and wildly different seasons. A beta-binomial absorbs the variance that clustering
inflates, through `rho`, but nothing in it carries the *shape*.

## Why no arm on the likelihood axis could have settled it, and why that is not a gap

**`gp` is invariant to the arrangement.** Permuting a player-season's played/missed vector
leaves `gp` exactly where it was, so no likelihood over `gp` — and no metric computed from a
`gp` predictive — can distinguish a clustered process from an exchangeable one except through
the *second-order* trace clustering leaves on the marginal. That trace is one number, and it
is already spent: with `n` trials, a within-cell clustering `C` and a between-cell frailty
`rho`, `games_played.variance_inflation` gives

    inflation = C + rho * (n - C)

so `C` and `rho` are **not separately identified from `gp` alone** — every point on that line
produces the same variance, and a fit will simply route clustering into `rho`. §7's five
frailty arms all vary the mixing distribution, which is the *other* term. That they did not
touch exchangeability is a property of the statistic, not an oversight.

And the experiment was already run, four times over, one head across.
`stan_games_played`'s arms are non-exchangeable **by construction** — an entry index, an exit
index and a within-tenure two-state chain with beta-geometric spells — and scored on the same
883 validation rows against the same floor, every one of them loses or ties at the `gp`
margin. `gp_margin_invariance` reads that table rather than re-running it. The `hybrid` row
is the proof rather than the evidence: it draws its count from the incumbent's own pmf and
only rearranges the absences, so it reproduces CRPS, PIT and the tail error **to every
decimal**. Maximal change in arrangement, zero change in every marginal metric.

## The unit that can see it

DraftKings scores **twenty scoring periods**, not one season total, and seats the best 7 of
16 in each. What a roster is exposed to is therefore not "how many games did he miss" but
"how many periods is he a guaranteed zero, and do they come in a row" — a question about the
arrangement and about nothing else. So the ladder here holds `gp` **fixed at its realized
value** and varies only the layout:

| arm | layout | what it is |
|---|---|---|
| `observed` | the real played/missed vector | the target |
| `clustered` | `games_played.allocate_spells` | **what ships** — beta-geometric spells at random starts |
| `exchangeable` | `gp` successes placed uniformly at random | what the head's likelihood literally asserts |

Holding `gp` fixed is what makes this a measurement of the assumption rather than of the
head: all three arms have identical games-played marginals by construction, so anything that
separates them is arrangement and cannot be anything else.

## Scope

Nothing here fits a model or changes one. It is an instrument for an assumption, and the
mitigation it prices — `allocate_spells` — already ships inside `sim/season.py`.

**Multi-team player-seasons are excluded from every population** for
`games_played.multi_team_seasons`' reason: the panel runs per (season, player, team) and a
traded player reads as having missed half the season twice over, which is a roster fact
wearing an availability costume. That is **593 of the 4,027 fitting rows (14.7%)**, and it
matters for reading the decomposition below — an ending tenure in this frame is **not**
followed by games for a new team, because a player who went on to play elsewhere is not
here. What is left is players who left the league, or arrived in it late.

**Half of what is left is still not an availability event**, and `missed_decomposition`
splits it rather than pooling it. See that function.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.eda.season_effects import ROLE_EDGES, ROLE_LABELS
from src.features.availability import absence_spells
from src.models.games_played import (MISSED_SHARE_EDGES, EdgeResampler, allocate_spells,
                                     edge_blocks, fit_beta_geometric, layout_tenure,
                                     missed_share_bin, single_team_panel,
                                     spell_lengths_from)
from src.models.availability import rung_zero
from src.models.held_out import selection_split
from src.models.stan_availability import (FIRST_SEASON, availability_design,
                                          restrict_window, role_bins)

#: Layout replicates per arm. The statistics below are means over player-seasons *and*
#: replicates, and the population is ~750 seasons x 20 periods, so the Monte Carlo error
#: at 25 replicates is already an order of magnitude under the gaps being read.
LAYOUT_REPS = 25

#: The consecutive-dead-period run length a roster slot is priced at. Three periods is
#: three weeks of a zero in a best-ball lineup that seats 7 of 16 — long enough that the
#: slot is not a slot, short enough to happen to a healthy star.
DEAD_RUN = 3

#: The panel `status` value meaning the player was not on an NBA roster for that game. It is
#: 99.98% covered on the 2012-13+ window and ~1% of interior spells, so on an edge block it
#: separates roster churn from injury cleanly. `docs/games-played-plan.md` measures it as
#: 98.5%-per-game persistent, which is why "absorbing" is the right idealization.
NOT_ROSTERED = "not_rostered"

#: How large the exchangeable arm's error has to be, relative to the observed value, before
#: a `recovered_share` is worth quoting. Below it the metric does not resolve the
#: arrangement and the ratio is noise over noise — see `_attach_gaps`.
ARRANGEMENT_TOL = 0.10

#: The `stan_games_played` arms that are non-exchangeable by construction, and what each
#: one changes relative to the incumbent beta-binomial. `hybrid` is the marginal-neutral
#: control and belongs at the end.
GP_ARMS = ("full_window", "three_state", "duration_covariates",
           "calibrated_fallback", "hybrid")

GP_METRICS = "stan_games_played_metrics.csv"

# ── Populations ───────────────────────────────────────────────────────────────

def role_frame(design: pd.DataFrame) -> pd.DataFrame:
    """`design` with its 1-based prior-MPG role bucket and that bucket's label."""
    out = design.copy()
    out["role_bin"] = role_bins(out)
    out["role"] = [ROLE_LABELS[b - 1] for b in out["role_bin"]]
    return out


# ── Analysis 1: where the non-exchangeability comes from ──────────────────────

def missed_decomposition(panel: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
    """Missed games split into **interior spells** and two kinds of **tenure edge block**.

    The three are different processes and only the first is what "absences come in spells"
    usually means. A late first appearance or an early last one is a single block at an end
    of the schedule — `games_played.py` establishes it is an absorbing hitting time rather
    than a low recovery rate — while an interior spell is an injury with a return.

    **A trade is not one of the three, because trades are not in the frame.**
    `single_team_panel` drops multi-team player-seasons, so an ending tenure here is not
    followed by games for a new team: the player left the league rather than the roster.

    **But half of what is left is still not an availability event, and the split is the
    point.** The panel carries `status` from 2006-07 (99.98% covered on this window), and an
    edge block is either `not_rostered` — the player was not on an NBA roster, so the game
    was never his to miss — or `inactive`/`dnp`, which is preseason and season-ending
    injury and is genuinely availability. The aggregate edge share hides that those two run
    in **opposite directions** across role, which is why they are emitted separately: the
    fringe bucket's edge blocks are mostly roster churn and a star's are mostly injury.

    The interior column is the control that makes the split readable — interior spells are
    ~1% `not_rostered`, so the status flag is picking out tenure and not noise.
    """
    sub = single_team_panel(panel, rows)
    keyed = rows.set_index(["season", "player_id"])["role_bin"]
    miss = sub[sub["played"] == 0].copy()
    miss["role_bin"] = [keyed.get((s, p), 0)
                        for s, p in zip(miss["season"], miss["player_id"])]
    edge = miss["in_appearance_window"] == 0
    not_rostered = miss["status"] == NOT_ROSTERED
    out = []
    for label, mask in [("all", pd.Series(True, index=miss.index))] + [
            (ROLE_LABELS[b - 1], miss["role_bin"] == b)
            for b in range(1, len(ROLE_LABELS) + 1)]:
        total = int(mask.sum())
        if not total:
            continue
        interior = int((mask & ~edge).sum())
        edge_off = int((mask & edge & not_rostered).sum())
        edge_on = int((mask & edge & ~not_rostered).sum())
        out.append({"analysis": "missed_decomposition", "population": label,
                    "missed_games": total, "interior_games": interior,
                    "edge_games": edge_off + edge_on,
                    "edge_not_rostered_games": edge_off,
                    "edge_still_rostered_games": edge_on,
                    "interior_share": interior / total,
                    "edge_share": (edge_off + edge_on) / total,
                    "edge_not_rostered_share": edge_off / total,
                    "edge_still_rostered_share": edge_on / total,
                    "not_rostered_share_of_edge":
                        edge_off / (edge_off + edge_on) if edge_off + edge_on else np.nan,
                    "interior_not_rostered_share":
                        int((mask & ~edge & not_rostered).sum()) / interior
                        if interior else np.nan})
    return pd.DataFrame(out)


def spell_shape_by_role(panel: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
    """Interior absence-spell **shape** and **rate**, per role bucket.

    The shipped layout takes one pooled `(mu, kappa)` for every player, which is only
    defensible if the shape is role-invariant. It is the rate that is not: `spells_per_season`
    is what separates a fringe player from a star, and the head already carries that through
    `gp`. Reported together so the pooling is a measured decision rather than a convenience.
    """
    spells = absence_spells(single_team_panel(panel, rows), "appearance")
    keyed = rows.set_index(["season", "player_id"])["role_bin"]
    spells = spells.assign(role_bin=[keyed.get((s, p), 0) for s, p
                                     in zip(spells["season"], spells["player_id"])])
    out = []
    groups = [("all", spells, len(rows))] + [
        (ROLE_LABELS[b - 1], spells[spells["role_bin"] == b],
         int((rows["role_bin"] == b).sum())) for b in range(1, len(ROLE_LABELS) + 1)]
    for label, group, cells in groups:
        t = group["spell_games"].to_numpy(dtype=float)
        fit = fit_beta_geometric(t)
        out.append({"analysis": "spell_shape", "population": label,
                    "player_seasons": cells, "spells": len(t),
                    "spells_per_season": len(t) / cells if cells else np.nan,
                    "mean_spell": float(t.mean()), "p_spell_1": float((t == 1).mean()),
                    "p_spell_ge3": float((t >= 3).mean()),
                    "p_spell_ge10": float((t >= 10).mean()),
                    "bg_mu": fit["mu"], "bg_kappa": fit["kappa"]})
    return pd.DataFrame(out)


def gp_margin_invariance(predictions_dir: Path) -> pd.DataFrame:
    """What the four non-exchangeable arms already fitted are worth at the `gp` margin.

    Read from `stan_games_played_metrics.csv` rather than re-run — the arms cost 47 minutes
    of sampler time and nothing about them has changed, which is the standing rule in
    `CLAUDE.md` about not refitting for record-keeping. `hybrid` is the row that carries the
    argument: identical marginal, maximally different arrangement.
    """
    path = predictions_dir / GP_METRICS
    if not path.exists():
        return pd.DataFrame(columns=["analysis", "arm"])
    metrics = pd.read_csv(path).set_index("variant")
    floor = metrics.loc["floor"]
    out = []
    for arm in GP_ARMS:
        if arm not in metrics.index:
            continue
        row = metrics.loc[arm]
        # The artifact's own column, not a difference against the shared `floor` row:
        # `three_state` and `within_tenure` score on 751 rows and carry their own floor,
        # and subtracting the 883-row one would silently compare two populations.
        out.append({"analysis": "gp_margin", "arm": arm,
                    "val_crps": float(row["val_crps"]),
                    "crps_vs_floor": float(row["crps_vs_floor"]),
                    "val_pit_ks": float(row["val_pit_ks"]),
                    "val_tail_error": float(row["val_tail_error"]),
                    "marginal_neutral": bool(
                        np.isclose(row["val_crps"], floor["val_crps"], atol=1e-9)
                        and np.isclose(row["val_pit_ks"], floor["val_pit_ks"], atol=1e-9))})
    return pd.DataFrame(out)


# ── The tenure factor ─────────────────────────────────────────────────────────

def edge_profile(cells: pd.DataFrame) -> pd.DataFrame:
    """What the edge share is conditional on, which is the choice `EdgeResampler` encodes.

    Emitted per `(role, missed-share)` cell and per margin, because the two keys are not
    equally load-bearing and the table is what says so. **The missed share is the axis**: the
    mean edge fraction roughly quadruples across its four bins while the four roles inside a
    bin span a few points. What role does carry is the *end* — `pre_share` runs the other way
    from `post_share` across the role range, which is §11b's two opposing processes (a fringe
    player signed late, a star whose season ends in March) showing up as position rather than
    as amount.
    """
    rows = cells[cells["missed"] > 0].copy()
    rows["share_bin"] = missed_share_bin(rows["missed"].to_numpy(),
                                         rows["team_games"].to_numpy())
    groups = [("all", "all", rows)]
    groups += [(ROLE_LABELS[b - 1], "all", rows[rows["role_bin"] == b])
               for b in range(1, len(ROLE_LABELS) + 1)]
    labels = [f"{MISSED_SHARE_EDGES[i]:.0%}-{MISSED_SHARE_EDGES[i + 1]:.0%}"
              for i in range(len(MISSED_SHARE_EDGES) - 1)]
    groups += [("all", labels[s - 1], rows[rows["share_bin"] == s])
               for s in range(1, len(labels) + 1)]
    groups += [(ROLE_LABELS[b - 1], labels[s - 1],
                rows[(rows["role_bin"] == b) & (rows["share_bin"] == s)])
               for b in range(1, len(ROLE_LABELS) + 1) for s in range(1, len(labels) + 1)]
    out = []
    for role, bucket, group in groups:
        if group.empty:
            continue
        edge_frac = (group["pre_frac"] + group["post_frac"]).to_numpy()
        out.append({"analysis": "edge_profile", "population": role,
                    "missed_share_bin": bucket, "player_seasons": len(group),
                    "mean_edge_frac": float(edge_frac.mean()),
                    "p_no_edge": float((edge_frac <= 0).mean()),
                    "p_all_edge": float((edge_frac >= 1 - 1e-9).mean()),
                    "mean_pre_frac": float(group["pre_frac"].mean()),
                    "mean_post_frac": float(group["post_frac"].mean()),
                    "mean_pre_games": float(group["pre"].mean()),
                    "mean_post_games": float(group["post"].mean())})
    return pd.DataFrame(out)


# ── Analysis 2: the layouts ───────────────────────────────────────────────────

def layout_exchangeable(gp: np.ndarray, team_games: np.ndarray,
                        seed: int = 0) -> np.ndarray:
    """`gp` played games placed uniformly at random — the head's own assertion, drawn.

    The mirror of `games_played.allocate_spells`: identical signature, identical output
    contract, identical realized `gp` on every row, and the *only* difference is where the
    absences fall. Conditional on `gp`, a beta-binomial's played/missed vector is exactly
    uniform over the arrangements — the frailty and the dispersion drop out of the
    conditional — so this needs no parameter from the head and there is none to get wrong.
    """
    rng = np.random.default_rng(seed)
    width = int(np.max(team_games))
    played = np.zeros((len(gp), width), dtype=np.int8)
    for i, (n, k) in enumerate(zip(team_games, gp)):
        n, k = int(n), int(k)
        if k >= n:
            played[i, :n] = 1
            continue
        if k > 0:
            played[i, rng.choice(n, size=k, replace=False)] = 1
    return played


def cell_index(panel: pd.DataFrame) -> dict:
    """The ragged player-season index every layout and statistic is evaluated over."""
    keys = ["season", "player_id"]
    grouped = panel.groupby(keys, sort=False)
    sizes = grouped.size().to_numpy()
    starts = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    played = panel["played"].to_numpy(np.int8)
    period = panel["period_index"].to_numpy(np.int64)
    return {"cells": grouped.size().reset_index()[keys],
            "sizes": sizes.astype(np.int64), "starts": starts,
            "played": played, "period": period,
            "gp": np.array([played[s:s + n].sum() for s, n in zip(starts, sizes)],
                           dtype=np.int64),
            "periods": [period[s:s + n] for s, n in zip(starts, sizes)]}


def observed_layout(index: dict) -> np.ndarray:
    """The realized played/missed vectors, in the same `(reps x rows x width)` shape."""
    width = int(index["sizes"].max())
    out = np.zeros((1, len(index["sizes"]), width), dtype=np.int8)
    for i, (s, n) in enumerate(zip(index["starts"], index["sizes"])):
        out[0, i, :n] = index["played"][s:s + n]
    return out


def period_statistics(layout: np.ndarray, index: dict,
                      select: np.ndarray | None = None,
                      dead_run: int = DEAD_RUN) -> dict:
    """Scoring-period exposure of a played/missed layout.

    Four statistics, all per player-season and then averaged over rows and replicates:

    - `p_dead_period` — the share of his team's scoring periods in which he plays **zero**
      games. A dead period is a guaranteed non-scorer in a lineup that seats 7 of 16.
    - `p_half_period` — the share in which he plays at most half the periods's games.
    - `longest_dead_run` — the longest run of consecutive dead periods, which is what a
      tournament round is actually exposed to.
    - `p_dead_run` — the share of player-seasons whose longest run reaches `dead_run`.

    Periods with no scheduled game for the player's team are dropped rather than counted as
    dead, because a bye is not an absence.
    """
    rows = np.arange(len(index["sizes"])) if select is None else np.flatnonzero(select)
    dead_shares, half_shares, runs = [], [], []
    for rep in range(layout.shape[0]):
        for i in rows:
            n = int(index["sizes"][i])
            pid = index["periods"][i]
            scheduled = np.bincount(pid)
            played = np.bincount(pid, weights=layout[rep, i, :n],
                                 minlength=len(scheduled))
            live = scheduled > 0
            dead = played[live] == 0
            dead_shares.append(float(dead.mean()))
            half_shares.append(float((played[live] <= 0.5 * scheduled[live]).mean()))
            runs.append(_longest_run(dead))
    runs = np.asarray(runs, dtype=float)
    return {"player_seasons": len(rows), "layouts": int(layout.shape[0]),
            "p_dead_period": float(np.mean(dead_shares)),
            "p_half_period": float(np.mean(half_shares)),
            "longest_dead_run": float(runs.mean()),
            "p_dead_run": float((runs >= dead_run).mean())}


def _longest_run(flags: np.ndarray) -> int:
    best = run = 0
    for flag in flags:
        run = run + 1 if flag else 0
        best = max(best, run)
    return best


def build_layouts(index: dict, roles: np.ndarray, mu: float, kappa: float,
                  edges: "EdgeResampler | None" = None, reps: int = LAYOUT_REPS,
                  seed: int = 0) -> dict[str, np.ndarray]:
    """Every arm's `(reps x rows x width)` played/missed tensor.

    **The drawn arms are a 2x2**, because the round found two mechanisms rather than one and
    they are not separable by inspection. One axis is the tenure factor — whether the edge
    blocks are laid at the ends from `EdgeResampler` or scattered as interior spells. The
    other is `allocate_spells`' overflow policy, which `overflow_incidence` measures firing
    on 41.28% of fringe rows and 2.32% of star rows and so is a role-shaped effect in its own
    right. `clustered` is the shipped corner and `tenure_merge` is the far one.
    """
    gp, team_games = index["gp"], index["sizes"]
    layouts = {"observed": observed_layout(index),
               "clustered": np.stack([allocate_spells(gp, team_games, mu, kappa,
                                                      seed=seed + rep)
                                      for rep in range(reps)]),
               "merge": np.stack([allocate_spells(gp, team_games, mu, kappa,
                                                  seed=seed + rep, overflow="merge")
                                  for rep in range(reps)]),
               "exchangeable": np.stack([layout_exchangeable(gp, team_games,
                                                             seed=seed + 5000 + rep)
                                         for rep in range(reps)])}
    if edges is not None:
        for arm, overflow in (("tenure", "collapse"), ("tenure_merge", "merge")):
            drawn = []
            for rep in range(reps):
                rng = np.random.default_rng(seed + 9000 + rep)
                pre, post = edges.draw(gp, team_games, roles, rng)
                drawn.append(layout_tenure(gp, team_games, pre, post, mu, kappa,
                                           seed=seed + 9000 + rep, overflow=overflow))
            layouts[arm] = np.stack(drawn)
    return layouts


def layout_spell_shape(layouts: dict[str, np.ndarray], index: dict,
                       roles: np.ndarray) -> pd.DataFrame:
    """The absence-spell lengths each arm actually **realizes**, against the observed ones.

    This is the falsification check `docs/potential-to-dos.md` item 6 asks for, and it is
    worth its own table because the shipped layout is not guaranteed to realize the
    distribution it draws from. `allocate_spells` truncates its last spell to make the missed
    total exact, and collapses to a *single block* whenever more spells were drawn than there
    are gaps to hold them — so on a heavily-absent row it manufactures long runs by accident.
    If that accident is what closes the fringe bucket, the defect is in the fitting loop
    rather than in the missing tenure factor, and the cheaper fix is to stop truncating.
    """
    out = []
    populations = [("all", np.ones(len(roles), dtype=bool))]
    populations += [(ROLE_LABELS[b - 1], roles == b) for b in range(1, len(ROLE_LABELS) + 1)]
    for label, select in populations:
        if not select.any():
            continue
        rows = np.flatnonzero(select)
        for arm, mat in layouts.items():
            lengths = np.concatenate([
                spell_lengths_from(mat[rep][rows], index["sizes"][rows])
                for rep in range(mat.shape[0])]) if len(rows) else np.zeros(0)
            if not len(lengths):
                continue
            out.append({"analysis": "layout_spell_shape", "population": label, "arm": arm,
                        "spells_per_season": len(lengths) / (len(rows) * mat.shape[0]),
                        "mean_spell": float(lengths.mean()),
                        "p_spell_ge10": float((lengths >= 10).mean()),
                        "p_spell_ge30": float((lengths >= 30).mean()),
                        "max_spell": int(lengths.max())})
    return pd.DataFrame(out)


def overflow_incidence(index: dict, roles: np.ndarray, mu: float, kappa: float,
                       edges: "EdgeResampler | None" = None, reps: int = LAYOUT_REPS,
                       seed: int = 0) -> pd.DataFrame:
    """How often the spell draw wants more spells than the schedule has gaps, by role.

    This is what makes the overflow policy a *modelling* choice rather than a defensive
    branch. It is not a rare guard: it fires on the fringe bucket an order of magnitude more
    often than on stars, so whichever policy is in force is a role-graded effect that nobody
    chose. The tenure arm's column is the interior residue's rate, which is the same
    condition after the edge blocks have been taken out of it.
    """
    gp, team_games = index["gp"], index["sizes"]
    arms = {"clustered": None}
    if edges is not None:
        arms["tenure"] = edges
    out = []
    populations = [("all", np.ones(len(gp), dtype=bool))]
    populations += [(ROLE_LABELS[b - 1], roles == b) for b in range(1, len(ROLE_LABELS) + 1)]
    for arm, resampler in arms.items():
        fired = np.zeros((reps, len(gp)), dtype=bool)
        for rep in range(reps):
            flags = np.zeros(len(gp), dtype=bool)
            if resampler is None:
                allocate_spells(gp, team_games, mu, kappa, seed=seed + rep,
                                overflow_out=flags)
            else:
                rng = np.random.default_rng(seed + 9000 + rep)
                pre, post = resampler.draw(gp, team_games, roles, rng)
                layout_tenure(gp, team_games, pre, post, mu, kappa,
                              seed=seed + 9000 + rep, overflow_out=flags)
            fired[rep] = flags
        for label, select in populations:
            if not select.any():
                continue
            out.append({"analysis": "overflow_incidence", "population": label, "arm": arm,
                        "player_seasons": int(select.sum()),
                        "overflow_rate": float(fired[:, select].mean())})
    return pd.DataFrame(out)


def layout_ladder(index: dict, roles: np.ndarray, mu: float, kappa: float,
                  edges: "EdgeResampler | None" = None, reps: int = LAYOUT_REPS,
                  seed: int = 0) -> pd.DataFrame:
    """Every layout arm at the scoring-period unit, by role.

    All of them carry the identical `gp` on every row — the observed one by definition, and
    the drawn ones because every layout places exactly `gp` played games. That equality is
    asserted rather than assumed, because it is the whole basis of the comparison: if the
    layouts moved `gp` at all, every gap below would be confounded with a marginal change.
    """
    gp, team_games = index["gp"], index["sizes"]
    layouts = build_layouts(index, roles, mu, kappa, edges=edges, reps=reps, seed=seed)
    for name, mat in layouts.items():
        realized = mat.sum(axis=2)
        if not np.array_equal(realized, np.broadcast_to(gp, realized.shape)):
            raise AssertionError(
                f"the {name!r} layout does not preserve games played on every row; the "
                f"period-unit comparison is only a measurement of arrangement if it does")
    populations = [("all", np.ones(len(gp), dtype=bool))]
    populations += [(ROLE_LABELS[b - 1], roles == b) for b in range(1, len(ROLE_LABELS) + 1)]
    out = []
    for label, select in populations:
        if not select.any():
            continue
        for arm, mat in layouts.items():
            out.append({"analysis": "period_layout", "population": label, "arm": arm,
                        **period_statistics(mat, index, select)})
    frame = pd.DataFrame(out)
    return _attach_gaps(frame)


def _attach_gaps(frame: pd.DataFrame) -> pd.DataFrame:
    """Each arm's error against `observed`, and how much of it `clustered` recovers.

    `recovered_share` is the fraction of the exchangeable arm's error that the shipped
    layout closes: 1.0 means it reaches the observed value, 0.0 that it is no better than
    assuming exchangeable trials, and above 1.0 that it overshoots. That ratio is the
    number this whole round exists to produce, because it is what says whether the
    assumption is a live defect or an already-paid one.

    **It is emitted only where the metric can see the arrangement at all.** `p_half_period`
    is nearly arrangement-invariant — the three arms agree on it to within 3% — and a share
    computed on a denominator that small reports the Monte Carlo error as a finding: on the
    star bucket the raw ratio is 32.5, which is a 0.0005 gap divided by itself. So a metric
    whose exchangeable error is under `ARRANGEMENT_TOL` of the observed value is flagged
    `arrangement_sensitive = False` and its share left blank. That a metric fails the flag
    is itself the result for that metric, not a hole in the table.

    **A fourth arm adds columns rather than rows.** `clustered` keeps the bare
    `recovered_share` name it has always had, because §11d's readings are audited under it;
    any further arm gets `<arm>_recovered_share` beside it on the same row, so the two are
    read against each other at a glance and no existing claim moves.
    """
    metrics = ["p_dead_period", "p_half_period", "longest_dead_run", "p_dead_run"]
    extra = [a for a in frame["arm"].unique()
             if a not in ("observed", "clustered", "exchangeable")]
    out = []
    for label, group in frame.groupby("population", sort=False):
        rows = group.set_index("arm")
        for metric in metrics:
            observed = float(rows.loc["observed", metric])
            exch = float(rows.loc["exchangeable", metric])
            clus = float(rows.loc["clustered", metric])
            gap = observed - exch
            sensitive = bool(observed and abs(gap) >= ARRANGEMENT_TOL * abs(observed))
            record = {"analysis": "period_gap", "population": label, "metric": metric,
                      "observed": observed, "clustered": clus, "exchangeable": exch,
                      "exchangeable_error": exch - observed,
                      "clustered_error": clus - observed,
                      "exchangeable_ratio": exch / observed if observed else np.nan,
                      "clustered_ratio": clus / observed if observed else np.nan,
                      "arrangement_sensitive": sensitive,
                      "recovered_share": (clus - exch) / gap if sensitive else np.nan}
            for arm in extra:
                value = float(rows.loc[arm, metric])
                record[arm] = value
                record[f"{arm}_error"] = value - observed
                record[f"{arm}_ratio"] = value / observed if observed else np.nan
                record[f"{arm}_recovered_share"] = ((value - exch) / gap if sensitive
                                                    else np.nan)
            out.append(record)
    return pd.concat([frame, pd.DataFrame(out)], ignore_index=True)


# ── Entry point ───────────────────────────────────────────────────────────────

def run(cfg: dict) -> dict[str, Path]:
    features_dir = Path(cfg["data"]["features_dir"])
    out_dir = Path(cfg["evaluation"]["predictions_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_head = cfg.get("stan", {}).get("availability", {})
    first_season = cfg_head.get("first_season", FIRST_SEASON)
    seed = int(cfg.get("features", {}).get("availability", {}).get("seed", 42))

    # Rung 0 only — the recorded decomposition and spell shapes are fitted quantities and
    # this round did not read §16's rows (`docs/availability-window-plan.md` §16j).
    design = rung_zero(availability_design(cfg))
    train, val = selection_split(design)
    fit_rows = role_frame(restrict_window(train, first_season))
    val_rows = role_frame(val)
    print(f"  role buckets from {ROLE_EDGES} on prior MPG; fitting rows "
          f"{len(fit_rows):,} ({first_season}+), validation rows {len(val_rows):,}")

    panel = pd.read_parquet(
        features_dir / "availability_panel.parquet",
        columns=["season", "player_id", "team_id", "game_id", "team_game_index",
                 "played", "in_appearance_window", "status"])

    # ── the decomposition and the pooled spell shape, on the FITTING rows only ─
    decomposition = missed_decomposition(panel, fit_rows)
    shape = spell_shape_by_role(panel, fit_rows)
    invariance = gp_margin_invariance(out_dir)
    fit_cells = edge_blocks(panel, fit_rows)
    edges = EdgeResampler(fit_cells)
    profile = edge_profile(fit_cells)
    clustering = pd.concat([decomposition, shape, invariance, profile], ignore_index=True)
    clustering_dest = out_dir / "availability_clustering.csv"
    clustering.to_csv(clustering_dest, index=False)

    edge = decomposition[decomposition["population"] == "all"].iloc[0]
    print(f"\n  Missed games on the fitting rows: {edge['missed_games']:,} — "
          f"{edge['interior_share']:.1%} interior spells, "
          f"{edge['edge_not_rostered_share']:.1%} edge blocks the player was NOT ROSTERED "
          f"for, {edge['edge_still_rostered_share']:.1%} edge blocks he was rostered "
          f"through")
    print(decomposition[["population", "missed_games", "interior_share",
                         "edge_not_rostered_share", "edge_still_rostered_share",
                         "not_rostered_share_of_edge"]].round(4).to_string(index=False))
    print(shape[["population", "spells_per_season", "mean_spell", "p_spell_ge10",
                 "bg_mu", "bg_kappa"]].round(4).to_string(index=False))
    print(f"\n  Edge blocks on the fitting rows: what the edge fraction is conditional on "
          f"({len(fit_cells):,} player-seasons, {int((fit_cells['missed'] > 0).sum()):,} "
          f"with a missed game)")
    print(profile[profile["population"] == "all"][
        ["missed_share_bin", "player_seasons", "mean_edge_frac", "p_no_edge", "p_all_edge",
         "mean_pre_frac", "mean_post_frac"]].round(4).to_string(index=False))
    print(profile[profile["missed_share_bin"] == "all"][
        ["population", "player_seasons", "mean_edge_frac", "mean_pre_games",
         "mean_post_games"]].round(4).to_string(index=False))
    if len(invariance):
        print("\n  Every non-exchangeable arm already fitted, at the gp margin:")
        print(invariance[["arm", "val_crps", "crps_vs_floor", "val_pit_ks",
                          "val_tail_error", "marginal_neutral"]]
              .round(4).to_string(index=False))

    # ── the period-unit ladder, on validation ─────────────────────────────────
    periods = pd.read_parquet(features_dir / "scoring_periods.parquet",
                              columns=["season", "game_id", "period_index"])
    periods["game_id"] = periods["game_id"].astype("int64")
    val_panel = (single_team_panel(panel, val_rows)
                 .merge(periods, on=["season", "game_id"], how="inner")
                 .sort_values(["season", "player_id", "team_game_index"])
                 .reset_index(drop=True))
    index = cell_index(val_panel)
    roles = (index["cells"].merge(val_rows[["season", "player_id", "role_bin"]],
                                  on=["season", "player_id"], how="left")["role_bin"]
             .to_numpy(np.int64))
    print(f"\n  Period ladder on {len(index['gp']):,} single-team validation "
          f"player-seasons of {len(val_rows):,}, {LAYOUT_REPS} layouts per arm")

    pooled = shape[shape["population"] == "all"].iloc[0]
    mu, kappa = float(pooled["bg_mu"]), float(pooled["bg_kappa"])
    ladder = layout_ladder(index, roles, mu, kappa, edges=edges, seed=seed)
    layouts = build_layouts(index, roles, mu, kappa, edges=edges, seed=seed)
    ladder = pd.concat([ladder, layout_spell_shape(layouts, index, roles),
                        overflow_incidence(index, roles, mu, kappa, edges=edges,
                                           seed=seed)], ignore_index=True)
    ladder_dest = out_dir / "availability_exchangeability.csv"
    ladder.to_csv(ladder_dest, index=False)
    print(ladder[ladder["analysis"] == "period_layout"][
        ["population", "arm", "p_dead_period", "p_half_period", "longest_dead_run",
         "p_dead_run"]].round(4).to_string(index=False))
    print("\n  Against `observed` — the exchangeable arm's error, and what each drawn "
          "layout recovers of it:")
    print(ladder[ladder["analysis"] == "period_gap"][
        ["population", "metric", "observed", "exchangeable", "recovered_share",
         "merge_recovered_share", "tenure_recovered_share",
         "tenure_merge_recovered_share"]].round(4).to_string(index=False))
    print("\n  How often the spell draw overflows the schedule's gaps, which is what makes "
          "the overflow policy a role-graded effect:")
    print(ladder[ladder["analysis"] == "overflow_incidence"][
        ["population", "arm", "player_seasons", "overflow_rate"]]
        .round(4).to_string(index=False))
    print("\n  The spell lengths each layout REALIZES, against the observed ones — the "
          "falsification check on `allocate_spells`' truncation:")
    print(ladder[ladder["analysis"] == "layout_spell_shape"][
        ["population", "arm", "spells_per_season", "mean_spell", "p_spell_ge10",
         "p_spell_ge30", "max_spell"]].round(4).to_string(index=False))
    print(f"\nWrote {len(clustering):,} rows → {clustering_dest}")
    print(f"Wrote {len(ladder):,} rows → {ladder_dest}")

    return {"availability_clustering": clustering_dest,
            "availability_exchangeability": ladder_dest}


if __name__ == "__main__":
    cfg = yaml.safe_load(open("configs/default.yaml"))
    run(cfg)
