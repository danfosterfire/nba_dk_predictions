"""`make final-evaluation` — the one place the held-out seasons are read.

Every sweep, ablation and gate in this project selects on **validation**. This module is
the counterpart: it takes the already-selected specification, refits it on train **plus**
validation, and scores it on the test seasons once. It is the only code permitted to unlock
`src/models/held_out.py`, and it exists so that unlocking is an act somebody performs rather
than a thing that happens.

## Why a refit rather than predicting test from the training fit

Predicting test from a train-only fit is cheaper and makes the validation and test columns
directly comparable. It is still the wrong number: at the point the final figure is taken
there is no reason to withhold the validation seasons from the model, and every reason to
report the performance of the thing that would actually be deployed. A train-only fit gives
a *lower bound* on deployed performance, not an estimate of it.

The cost of that choice is that a validation-versus-test comparison is confounded — the two
differ in training data as well as evaluation rows — which is precisely why the sweeps no
longer produce a test column at all. A val/test gap is not a replication check here, and the
project stopped pretending it was on 2026-08-05, after Gate D was decided on the test split
and reversed when re-decided on validation.

## What this is NOT for

Not for choosing between arms, tuning a hyperparameter, deciding a feature block, or
checking whether a validation result "held up". If a number from here changes a modelling
decision, the split has been spent and the estimate it produces is no longer unbiased. Run
it when the workflow is finished.

## What is registered

Three model heads — `availability`, `games_played`, `season_total` — and the `chain`, which
is not a head at all. The heads answer "how well does this quantity predict on seasons it
has not seen"; the chain answers the project's actual question, by building a board from the
deployed posterior, drafting the **shipped** strategy against an ADP field and scoring it on
the box scores that happened. It costs hours where the heads cost minutes, and it is the
only one that needs `make posteriors WINDOW=train_val` to have been run.

`final_evaluation.csv` is merged **by head** rather than replaced, so taking the chain does
not retract the heads.

Usage:
    python -m src.final_evaluation                # every registered head, chain included
    python -m src.final_evaluation games_played   # one of them
    python -m src.final_evaluation chain          # the deliverable-level readout
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.models.held_out import final_split, unlocked

REASON = "end-of-project evaluation of the whole workflow"


def _games_played(cfg: dict, out_dir: Path) -> pd.DataFrame:
    """Refit the games-played head's shipped arm on train+validation, score on test.

    Reads which arm shipped from `spell_process.csv` rather than re-deciding it here — the
    decision belongs to the validation sweep, and re-selecting at final-evaluation time
    would be the exact failure this module exists to prevent.

    ⚠️ **`arm` in that artifact is the arm the module profiled, not a promise that it
    ships.** `stan_games_played` sets it to the best fitted arm when *neither* candidate
    clears Gate D, and prints "the head does not ship; the incumbent stands" beside it. So
    the verdict is read from the `gate_d/passes` row, and when that is false the incumbent
    row below is not a baseline this evaluation happens to also carry — it is the complete
    held-out reading of what this part of the chain actually does.
    """
    from src.models.availability import BetaBinomialGLM, FEATURE_COLS
    from src.models.games_played import duration_rows, fit_beta_geometric
    from src.features.availability import spell_classes
    from src.models.season_total import FINAL_SPELL_PMF_FILE
    from src.models.stan_games_played import (CalibratedProcess, HybridProcess,
                                              games_played_design,
                                              measured_clustering, onset_rate_lags,
                                              score)

    process_path = out_dir / "spell_process.csv"
    if not process_path.exists():
        raise FileNotFoundError(
            f"no {process_path}; run `make stan-games-played` first — the final "
            f"evaluation scores the arm the VALIDATION sweep selected, it does not choose "
            f"one.")
    process = pd.read_csv(process_path)
    shipped = str(process["arm"].iloc[0])
    verdict = process[(process["component"] == "gate_d")
                      & (process["statistic"] == "passes")]["value"]
    gate_d_passed = bool(float(verdict.iloc[0])) if len(verdict) else False

    panel = pd.read_parquet(Path(cfg["data"]["features_dir"])
                            / "availability_panel.parquet")
    design = games_played_design(cfg, panel)
    design = design.merge(onset_rate_lags(design, cfg["data"]["seasons"]),
                          on=["season", "player_id"], how="left")
    full_train, test = final_split(design)
    max_games = int(design["team_games"].max())
    seed = int(cfg.get("stan", {}).get("seed", 42))

    print(f"  arm profiled by the validation sweep: {shipped}; Gate D "
          f"{'PASSED' if gate_d_passed else 'FAILED'}"
          + ("" if gate_d_passed else
             " — the spell process does not ship and the\n     incumbent availability "
             "head stands, so the incumbent row below IS this head's\n     held-out "
             "reading rather than a baseline beside it"))
    print(f"  refitting on {len(full_train):,} train+validation rows, scoring "
          f"{len(test):,} test rows "
          f"({', '.join(sorted(test['season'].unique()))})")

    incumbent = BetaBinomialGLM(1.0).fit(full_train)
    rows = [{"head": "games_played", "arm": "incumbent",
             **_keep(score(incumbent, test, max_games, seed))}]

    if not gate_d_passed:
        # 🔴 No pmf is written, and that is deliberate. `season_total`'s `spell_process`
        # treatment means "the games-played module's pmf, composed through the rate
        # model"; on validation that is the fitted spell process. Handing it the
        # INCUMBENT's pmf here would report the incumbent under the spell process's name
        # — two rows, one model, and a held-out Gate E comparing a model against itself.
        # `spell_process_pmf` returns `None` on a missing file and the treatment is
        # skipped loudly, which is the honest shape of "this arm does not ship".
        stale = out_dir / FINAL_SPELL_PMF_FILE
        if stale.exists():
            stale.unlink()
            print(f"  removed a stale {stale.name} — with Gate D failed there is no "
                  f"spell-process\n     pmf to compose, and the season total skips that "
                  f"treatment rather than scoring\n     the incumbent under its name")
        return pd.DataFrame(rows)

    _write_gp_pmf(incumbent, test, max_games, out_dir, "incumbent")

    if shipped in ("hybrid", "calibrated_fallback"):
        # Both are calibrations rather than fits, so the "refit" is of their inputs: the
        # incumbent above, and the spell shape / clustering measured WITHOUT the test
        # seasons even now — a simulator input calibrated on the rows it is scored against
        # would be leakage the split cannot catch (CLAUDE.md, simulator-inputs rule).
        held = set(test["season"].unique())
        train_panel = panel[~panel["season"].isin(held)]
        if shipped == "hybrid":
            interior = spell_classes(train_panel, "appearance")
            collapsed = duration_rows(interior)
            dur = fit_beta_geometric(collapsed["t"].to_numpy(),
                                     collapsed["w"].to_numpy())
            model = HybridProcess(incumbent, dur["mu"], dur["kappa"], seed=seed)
        else:
            model = CalibratedProcess(
                incumbent, measured_clustering(train_panel, full_train),
                "gp_calibrated", seed=seed)
        rows.append({"head": "games_played", "arm": shipped,
                     **_keep(score(model, test, max_games, seed))})
    else:
        print(f"  /!\\  {shipped} is a fitted arm; its final refit needs the Stan heads "
              f"and is not wired here yet.")
    return pd.DataFrame(rows)


def _keep(scored: dict) -> dict:
    return {k: scored[k] for k in
            ("crps", "mae", "r2", "pit_ks", "implied_overdispersion",
             "predicted_below_41", "observed_below_41",
             "predicted_below_60", "observed_below_60")}


def _write_gp_pmf(model, test: pd.DataFrame, max_games: int, out_dir: Path,
                  arm: str) -> Path:
    """The test-side games-played pmf, for `_season_total` to compose through.

    Written here and nowhere else, under a filename the validation-side sweep never uses.
    `season_total.spell_process_pmf` refuses partial coverage, so even if the two files
    were confused the key join would cover nothing and the treatment would be skipped
    loudly rather than answering a held-out question with validation rows.
    """
    from src.models.season_total import FINAL_SPELL_PMF_FILE

    pmf = model.predict_pmf(test, max_games)
    rows, cols = np.nonzero(pmf > 1e-9)
    frame = pd.DataFrame({
        "arm": arm,
        "season": test["season"].to_numpy()[rows],
        "player_id": test["player_id"].to_numpy()[rows],
        "team_games": test["team_games"].to_numpy()[rows],
        "gp": cols, "p": pmf[rows, cols]})
    dest = out_dir / FINAL_SPELL_PMF_FILE
    frame.to_csv(dest, index=False)
    print(f"  Saved {len(frame):,} held-out games-played pmf rows → {dest}")
    return dest


def _availability(cfg: dict, out_dir: Path) -> pd.DataFrame:
    """Refit the availability head on train+validation, score the three ports on test.

    There is nothing to select here — `stan_availability` compares one likelihood fitted
    two ways, and which of the three rows the simulator consumes was settled by the
    argument (the posterior, for the joint) rather than by a metric. So the final
    evaluation is the same measurement on the other frame, and it calls
    `stan_availability.fit_and_score` rather than reimplementing it.

    The board-correlation table comes back too, because "how much does my whole board move
    together" is the quantity the posterior exists to supply and a held-out reading of it
    is worth having beside the validation one it is calibrated from.
    """
    from src.models.availability import FEATURE_COLS
    from src.models.stan_availability import (FIRST_SEASON, ROLE_RHO,
                                              head_design, head_features, fit_and_score)

    cfg_av = cfg.get("features", {}).get("availability", {})
    cfg_stan = cfg.get("stan", {})
    cfg_head = cfg_stan.get("availability", {})
    seed = int(cfg_stan.get("seed", cfg_av.get("seed", 42)))
    l2 = float(cfg_av.get("glm_l2", 1.0))
    # The shipped window and dispersion grading, read the same way `stan_availability.run`
    # reads them — the final evaluation is that measurement on the other frame, so the
    # head's configuration must not be able to differ between the two.
    first_season = cfg_head.get("first_season", FIRST_SEASON)
    role_rho = bool(cfg_head.get("role_rho", ROLE_RHO))

    # `head_design`: the final reading has to score the head that SHIPS, preseason block
    # included, or it measures a model nobody runs.
    design = head_design(cfg)
    features = head_features()
    full_train, test = final_split(design, int(cfg_av.get("test_seasons", 2)))
    max_games = int(design["team_games"].max())

    print(f"  refitting on {len(full_train):,} train+validation rows, scoring "
          f"{len(test):,} test rows "
          f"({', '.join(sorted(test['season'].unique()))}), {len(features)} features "
          f"({len(features) - len(FEATURE_COLS)} of them the shipped preseason block)")

    scored = fit_and_score(full_train, test, max_games, cfg_stan, l2, seed,
                           first_season=first_season, role_rho=role_rho,
                           features=features)
    wide = (scored["metrics"][scored["metrics"]["group"] == "all"]
            .pivot_table(index="model", columns="metric", values="value"))
    rows = [{"head": "availability", "arm": name,
             "crps": float(wide.loc[name, "crps_games"]),
             "mae": float(wide.loc[name, "mae_games"]),
             "r2": float(wide.loc[name, "r2_gp_share"]),
             "pit_ks": float(wide.loc[name, "pit_ks_distance"]),
             "implied_overdispersion": float(wide.loc[name,
                                                      "implied_overdispersion"])}
            for name in wide.index]

    board = scored["board"]
    board.insert(0, "head", "availability")
    dest = out_dir / "final_evaluation_availability_board.csv"
    board.to_csv(dest, index=False)
    print(f"  Saved {len(board):,} held-out board-correlation rows → {dest}")
    return pd.DataFrame(rows)


def _season_total(cfg: dict, out_dir: Path) -> pd.DataFrame:
    """Refit every games-played treatment on train+validation, score on test.

    The deliverable-level number: `season_total = gp x dk_pts_per_game_played`, with the
    rate model held identical across rows so the contrast is the availability treatment
    and nothing else. `season_total.compare` is shared with the validation run for the
    same reason `_availability` shares `fit_and_score` — a held-out figure produced by a
    second copy of the composition would be a different measurement wearing the same name.
    """
    from src.models.season_total import (FINAL_SPELL_PMF_FILE, build_frame,
                                         compare, gate_e)

    frame = build_frame(cfg)
    full_train, test = final_split(
        frame, int(cfg.get("features", {}).get("availability", {})
                   .get("test_seasons", 2)))
    max_games = int(frame["team_games"].max())
    print(f"  refitting on {len(full_train):,} train+validation rows, scoring "
          f"{len(test):,} test rows "
          f"({', '.join(sorted(test['season'].unique()))})")

    metrics, predictions, rate_r2 = compare(full_train, test, max_games, out_dir,
                                            FINAL_SPELL_PMF_FILE)
    dest = out_dir / "final_evaluation_season_total.csv"
    metrics.to_csv(dest, index=False)
    print(f"  fixed rate model: held-out R2 {rate_r2:.4f} on dk_pts per game played")
    print(f"  Saved {len(metrics):,} held-out season-total rows → {dest}")

    wide = (metrics[metrics["group"] == "all"]
            .pivot_table(index="treatment", columns="metric", values="value"))
    gate = gate_e(wide)
    if gate["ran"]:
        print(f"  Gate E, held out: MAE {gate['mae']:.1f} against the incumbent's "
              f"{gate['incumbent_mae']:.1f}\n     — CONFIRMATION of the validation "
              f"verdict, never a re-decision of it.")
    return pd.DataFrame([{"head": "season_total", "arm": name,
                          "mae": float(wide.loc[name, "mae_dk_total"]),
                          "rmse": float(wide.loc[name, "rmse_dk_total"]),
                          "r2": float(wide.loc[name, "r2_dk_total"]),
                          "bias": float(wide.loc[name, "bias_dk_total"]),
                          "crps": float(wide.loc[name, "crps_dk_total"])
                          if "crps_dk_total" in wide.columns else float("nan")}
                         for name in wide.index])


# ── The chain — the workflow, not a head ──────────────────────────────────────

#: The fit window the held-out chain reads. `train_val` for exactly the reason every head
#: above refits before it scores: the figure should describe the model that would deploy.
#: `full` has already read the seasons it would be scored on, and `train` is a model
#: nobody would ship — so neither of them is the deployed chain.
CHAIN_WINDOW = "train_val"

#: What the shipped strategy is read against. `adp` is our own seat playing the market's
#: board — the "what if the model added nothing" control — and `model_mean` is the model
#: with neither the market blend nor the per-pick objective, which is what the shipped
#: strategy's two axes are worth measuring against. The analytic symmetric null rides on
#: every row already as `p_advance_null`.
CHAIN_BASELINES = ("adp", "model_mean")

#: The Gate A suffix the held-out simulation writes under. See `season.run`'s `label`.
CHAIN_LABEL = "_final"


def _adp_admissible(cfg: dict, season: str) -> dict:
    """How many of a season's draftable players carry a point-in-time-legal ADP.

    The contest replay needs a **market**, and `features/draft_pool.py` records that under
    `adp.training_rows` only five of nine ADP seasons survive point-in-time discipline —
    a board is admissible only if it was *observed* on or before the season's first game.
    2024-25 is one of the seasons that does not survive: every archived snapshot postdates
    the opener, so the board that season drafted on was never captured.

    That is a **capture** gap rather than a modelling one, and it lands on one of the two
    test seasons. It is checked here, before a field is drafted, because a field drafted
    from an all-`nan` board does not fail — it drafts something, and every number
    downstream of it would be a statement about that something.
    """
    pool = pd.read_parquet(Path(cfg["data"]["features_dir"]) / "draft_pool.parquet",
                           columns=["season", "player_id", "adp", "adp_dk_scale"])
    rows = pool[pool["season"] == season]
    return {"season": season, "n_pool": int(len(rows)),
            "n_priced": int(rows["adp_dk_scale"].notna().sum())}


def _ensure_tensor(cfg: dict, season: str, window: str = CHAIN_WINDOW) -> Path:
    """The season's tensor at the deployed window, simulated if it is not already there.

    A tensor from the **wrong** window is re-simulated rather than reused, and that is the
    whole reason this is a function: `sim_tensor_<season>.npz` is keyed by season alone, so
    a file left behind by a run at another window is indistinguishable from the right one
    until its stamp is read. `posteriors.require_window` makes the same check one layer
    down, on the coefficients.
    """
    from src.sim import season as sim_season

    dest = Path(cfg["data"]["features_dir"]) / f"sim_tensor_{season}.npz"
    if dest.exists():
        with np.load(dest, allow_pickle=False) as z:
            got = str(z["fit_window"])
        if got == window:
            print(f"  {season}: reusing the `{got}` tensor already on disk → {dest}")
            return dest
        print(f"  {season}: the tensor on disk was simulated at `{got}`, not `{window}` "
              f"— re-simulating")
    sim_season.run(cfg, seasons=[season], window=window, label=CHAIN_LABEL)
    return dest


def _chain(cfg: dict, out_dir: Path) -> pd.DataFrame:
    """The whole workflow on the held-out seasons — the deliverable rather than a head.

    The three heads above are the model's figures. This is the project's: a board built
    from the deployed posterior, drafted under the shipped strategy against an ADP field,
    and scored on the box scores that actually happened. `README.md`'s standing claim is
    that the drafting edge is large in simulation and unconfirmed in the realized replay,
    and this is that replay on seasons no part of the chain has seen.

    **Three things it deliberately does not do.**

    *It does not select.* The strategy comes out of `strategy_shipped.csv` — the rule
    `strategy.ship` was written for and `_games_played` already follows. Re-running the
    24-arm sweep here would turn the held-out reading into the largest selection event in
    the project.

    *It does not simulate a world.* `make strategy-sweep` reports both a simulated-truth
    sweep and a realized replay, and only the realized half belongs here: the simulated
    half needs Gate C's injection, which solves two nuisance parameters (`rho` from the
    market-minus-model skill gap, `g` from the season-total MAE) **against the realized
    season it is then scored on**. On validation that is a tuning surface and says so; on
    test it would be a held-out figure calibrated on the held-out answer.

    *It does not manufacture a market.* One of the two test seasons has no admissible ADP
    board at all — see `_adp_admissible` — so the contest is replayed on one season and the
    other is reported as the capture gap it is. Two test seasons was already the ceiling on
    this estimate; one is what the ADP capture actually left, and it is the reason the
    figure below is a readout rather than a result.
    """
    from src.models.posteriors import FIT_WINDOW, assert_same_specification
    from src.sim import draft_room
    from src.sim import strategy as strat
    from src.sim.bracket import split_frame

    cfg_sim = cfg.get("sim", {})
    cfg_strategy = cfg_sim.get("strategy", {})
    entered = cfg_sim.get("tournaments", {})
    # The sweep's own budget keys and its own seed, so the held-out replay and the
    # validation one differ in the seasons they score and in nothing else.
    n_sims = int(cfg_strategy.get("n_sims", strat.N_SIMS_SWEEP))
    n_field_drafts = int(cfg_strategy.get("field_drafts", strat.N_FIELD_DRAFTS))
    seed = strat.SEED

    design = split_frame(cfg)
    _, test = final_split(design, int(cfg.get("features", {}).get("availability", {})
                                      .get("test_seasons", 2)))
    seasons = sorted(str(x) for x in test["season"].unique())

    print(f"  the deployed chain is the `{CHAIN_WINDOW}` window — train PLUS validation, "
          f"the same refit rule the heads above follow")
    assert_same_specification(cfg, CHAIN_WINDOW, reference=FIT_WINDOW)

    shipped_path = out_dir / "strategy_shipped.csv"
    if not shipped_path.exists():
        raise FileNotFoundError(
            f"no {shipped_path}; run `make strategy-sweep` first — the held-out replay "
            f"scores the strategy the VALIDATION sweep selected, it does not choose one.")
    shipped = pd.read_csv(shipped_path)
    by_name = {s.name: s for s in strat.strategy_table()}
    names = list(dict.fromkeys(list(CHAIN_BASELINES)
                               + [str(n) for n in shipped["strategy"]]))
    unknown = [n for n in names if n not in by_name]
    if unknown:
        raise KeyError(f"{shipped_path} names strategies the table no longer carries: "
                       f"{unknown}. The shipped artifact and `strategy.strategy_table` "
                       f"have drifted apart; re-run `make strategy-sweep`.")
    strategies = [by_name[n] for n in names]
    ships = dict(zip(shipped["tournament"].astype(str),
                     shipped["strategy"].astype(str)))
    print(f"  selected strategy (from the validation sweep): "
          + ", ".join(f"{t} → {n}" for t, n in ships.items()))

    coverage = [_adp_admissible(cfg, season) for season in seasons]
    playable = [row["season"] for row in coverage if row["n_priced"] > 0]
    for row in coverage:
        mark = "" if row["n_priced"] else "   /!\\  NO admissible ADP — contest skipped"
        print(f"  {row['season']}: {row['n_priced']:,} of {row['n_pool']:,} draftable "
              f"players carry a point-in-time-legal ADP{mark}")
    # Written here rather than at the end, because it is the round's binding limitation
    # and it costs a millisecond: the hours of simulation below must not stand between a
    # reader and the reason one of the two seasons has no contest row.
    cover_dest = out_dir / "final_evaluation_adp_coverage.csv"
    pd.DataFrame(coverage).to_csv(cover_dest, index=False)
    print(f"  Saved {len(coverage):,} ADP-admissibility rows → {cover_dest}")

    # Every test season gets a tensor, including the one with no market: Gate A is the
    # simulator's own held-out reading — does a season drawn from the deployed posterior
    # land where the season that happened landed — and it needs no ADP at all. Only the
    # CONTEST needs a field to draft against.
    for season in seasons:
        print(f"\n  ── {season}: the tensor ──")
        _ensure_tensor(cfg, season)
    gate = _chain_gate_a(cfg, out_dir, seasons)

    rows = []
    for season in playable:
        print(f"\n  ── {season}: the contest ──")
        full = draft_room.load_room(cfg, season, n_sims=n_sims)
        if full.fit_window != CHAIN_WINDOW:
            raise AssertionError(
                f"{season}: the room reports the `{full.fit_window}` window, not "
                f"`{CHAIN_WINDOW}` — a held-out replay off a wider fit has read the "
                f"season it is being scored on, through the coefficients.")
        room, dropped = strat.priceable_room(cfg, full, seed, n_field_drafts)
        print(f"  board restricted to the {dropped['n_board']:,} players the tensor "
              f"prices: {dropped['n_dropped']:,} dropped, {dropped['n_dropped_priced']:,} "
              f"of them carrying ADP")

        portfolios = {}
        for spec_strategy in strategies:
            for tournament, spec in entered.items():
                n_entries = spec_strategy.n_entries or int(spec.get("entries", 1))
                rosters, _ = strat.draft_portfolio(
                    room, spec_strategy, tournament, n_entries,
                    np.random.default_rng(seed))
                portfolios[(spec_strategy.name, tournament)] = rosters
        print(f"  drafted {len(portfolios):,} portfolios "
              f"({len(strategies)} strategies x {len(entered)} tiers); replaying against "
              f"realized {season}")
        realized = strat.replay_realized(cfg, room, season, strategies, portfolios,
                                         seed, n_field_drafts)
        realized.insert(0, "shipped",
                        [n == ships.get(t) for n, t in zip(realized["strategy"],
                                                           realized["tournament"])])
        rows.append(realized)

    if not rows:
        print("\n  /!\\  no test season carries an admissible ADP board, so the contest "
              "cannot be replayed\n       on any of them. Gate A above still stands; the "
              "contest reading does not exist.")
        return gate
    realized = pd.concat(rows, ignore_index=True)
    realized.insert(0, "fit_window", CHAIN_WINDOW)
    dest = out_dir / "final_evaluation_chain.csv"
    realized.to_csv(dest, index=False)
    print(f"\n  Saved {len(realized):,} held-out contest rows over "
          f"{realized['season'].nunique()} season(s) → {dest}")

    _report_chain(realized, ships)
    hit = realized[realized["shipped"]]
    contest = pd.DataFrame([{"head": "chain", "arm": f"{row.tournament}/{row.strategy}",
                             "season": row.season,
                             "p_advance": row.p_advance,
                             "p_advance_null": row.p_advance_null,
                             "lift_vs_null": row.lift_vs_null,
                             "roi": row.roi,
                             "break_even_hurdle": row.break_even_hurdle}
                            for row in hit.itertuples()])
    return pd.concat([gate, contest], ignore_index=True)


def _report_chain(realized: pd.DataFrame, ships: dict) -> None:
    print("\n  The contest, replayed on the held-out season(s) — one world each, and the "
          "field\n  is the only thing that can be resampled. No interval here is a season "
          "interval.")
    for tournament in dict.fromkeys(realized["tournament"]):
        sub = realized[realized["tournament"] == tournament]
        null = float(sub["p_advance_null"].iloc[0])
        hurdle = float(sub["break_even_hurdle"].iloc[0])
        print(f"\n  {tournament}  (an ADP-drafted entry advances at {null:.6f}; "
              f"break-even hurdle {hurdle:+.2%})")
        for row in sub.itertuples():
            mark = "*" if row.strategy == ships.get(row.tournament) else " "
            print(f"  {mark} {row.season}  {row.strategy:<22} "
                  f"P(top 2 of 12) {row.p_advance:.4f}  lift {row.lift_vs_null:+.4f}  "
                  f"P(any of {int(row.n_entries)}) {row.p_any_advance:.4f}  "
                  f"ROI {row.roi:+.3f}")


def _chain_gate_a(cfg: dict, out_dir: Path, seasons: list[str]) -> pd.DataFrame:
    """The simulator's own held-out reading — no market required, so both test seasons.

    Gate A asks whether a season drawn from the deployed posterior lands where the season
    that happened landed, against the bars the validation run is measured on. It is read
    here rather than recomputed, from the labelled table `_ensure_tensor` wrote: a held-out
    run must not add rows to the pooled table `make docs-audit` re-derives its extremes
    from.

    Only the rows the simulator gates on come back. The diagnostics ride in the artifact —
    they are reported by `make simulate-season` itself and reading them here would put a
    row that nothing gates on into the one table that says what the workflow is worth.
    """
    path = out_dir / f"sim_season_gate_a{CHAIN_LABEL}.csv"
    if not path.exists():
        raise FileNotFoundError(f"no {path}; the held-out simulation did not write a "
                                f"Gate A table")
    gate = pd.read_csv(path)
    gate = gate[gate["season"].astype(str).isin(seasons)]
    scored = gate[gate["gate"] == "A"]

    print(f"\n  Gate A on the held-out seasons — the simulator against what happened, at "
          f"the bars\n  the validation run is measured on:")
    for row in scored.itertuples():
        parts = [f"{name} {getattr(row, name):.4f}"
                 for name in ("mae", "bias", "crps", "r2", "value")
                 if pd.notna(getattr(row, name, float("nan")))]
        bars = [f"bar {name} {getattr(row, 'bar_' + name):.4f}"
                for name in ("mae", "crps", "value")
                if pd.notna(getattr(row, "bar_" + name, float("nan")))]
        print(f"    {row.season}  {row.check:<22} " + "  ".join(parts)
              + ("   [" + ", ".join(bars) + "]" if bars else ""))

    return pd.DataFrame([{"head": "chain", "arm": f"gate_a/{row.check}",
                          "season": row.season,
                          "mae": getattr(row, "mae", float("nan")),
                          "bias": getattr(row, "bias", float("nan")),
                          "crps": getattr(row, "crps", float("nan")),
                          "r2": getattr(row, "r2", float("nan")),
                          "value": getattr(row, "value", float("nan"))}
                         for row in scored.itertuples()])


HEADS = {"availability": _availability,
         "games_played": _games_played,
         "season_total": _season_total,
         "chain": _chain}



def run(cfg: dict, heads: list[str] | None = None) -> Path:
    out_dir = Path(cfg["evaluation"]["predictions_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    chosen = heads or list(HEADS)
    unknown = [h for h in chosen if h not in HEADS]
    if unknown:
        raise ValueError(f"no final evaluation registered for {unknown}; "
                         f"known heads: {sorted(HEADS)}")

    print("Final evaluation — the ONE reading of the held-out seasons.")
    print("  Selection happened on validation. Nothing here may change a modelling "
          "decision;\n  if it does, the split is spent and this estimate is no longer "
          "unbiased.")

    frames = []
    with unlocked(REASON):
        for name in chosen:
            print(f"\n[{name}]")
            frames.append(HEADS[name](cfg, out_dir))

    built = pd.concat(frames, ignore_index=True)
    print("\nHeld-out performance:")
    print(built.round(4).to_string(index=False))

    # Merged over whatever is on disk **by head**, the idiom `posteriors.run`'s manifest
    # and `season.merge_gate` both already use. `python -m src.final_evaluation chain`
    # is the normal way to run this — the chain costs hours and the three heads do not —
    # and a partial run that replaced the file would silently retract the readings it did
    # not take. This is the one artifact in the project that cannot be re-derived on
    # demand: taking it again is taking it a second time.
    dest = out_dir / "final_evaluation.csv"
    out = built
    if dest.exists():
        existing = pd.read_csv(dest)
        if len(existing):
            out = pd.concat([existing[~existing["head"].isin(set(built["head"]))], built],
                            ignore_index=True)
    out.to_csv(dest, index=False)
    print(f"\nSaved {len(built):,} rows from this run, {len(out):,} in total → {dest}")
    print("  Heads with no registered final evaluation have NOT had their held-out number "
          "taken:\n  " + ", ".join(sorted(set(HEADS) - set(chosen)) or ["(none)"]))
    return dest


if __name__ == "__main__":
    cfg = yaml.safe_load(open("configs/default.yaml"))
    run(cfg, sys.argv[1:] or None)
