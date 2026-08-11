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

Usage:
    python -m src.final_evaluation                # every registered head
    python -m src.final_evaluation games_played   # one of them
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
    """
    from src.models.availability import BetaBinomialGLM, FEATURE_COLS
    from src.models.games_played import duration_rows, fit_beta_geometric
    from src.features.availability import spell_classes
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
    shipped = str(pd.read_csv(process_path)["arm"].iloc[0])

    panel = pd.read_parquet(Path(cfg["data"]["features_dir"])
                            / "availability_panel.parquet")
    design = games_played_design(cfg, panel)
    design = design.merge(onset_rate_lags(design, cfg["data"]["seasons"]),
                          on=["season", "player_id"], how="left")
    full_train, test = final_split(design)
    max_games = int(design["team_games"].max())
    seed = int(cfg.get("stan", {}).get("seed", 42))

    print(f"  selected arm (from the validation sweep): {shipped}")
    print(f"  refitting on {len(full_train):,} train+validation rows, scoring "
          f"{len(test):,} test rows "
          f"({', '.join(sorted(test['season'].unique()))})")

    incumbent = BetaBinomialGLM(1.0).fit(full_train)
    rows = [{"head": "games_played", "arm": "incumbent",
             **_keep(score(incumbent, test, max_games, seed))}]
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
                                              availability_design, fit_and_score)

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

    design = availability_design(cfg)
    full_train, test = final_split(design, int(cfg_av.get("test_seasons", 2)))
    max_games = int(design["team_games"].max())

    print(f"  refitting on {len(full_train):,} train+validation rows, scoring "
          f"{len(test):,} test rows "
          f"({', '.join(sorted(test['season'].unique()))}), {len(FEATURE_COLS)} features")

    scored = fit_and_score(full_train, test, max_games, cfg_stan, l2, seed,
                           first_season=first_season, role_rho=role_rho)
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


HEADS = {"availability": _availability,
         "games_played": _games_played,
         "season_total": _season_total}


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

    out = pd.concat(frames, ignore_index=True)
    print("\nHeld-out performance:")
    print(out.round(4).to_string(index=False))

    dest = out_dir / "final_evaluation.csv"
    out.to_csv(dest, index=False)
    print(f"\nSaved {len(out):,} final-evaluation rows → {dest}")
    print("  Heads with no registered final evaluation have NOT had their held-out number "
          "taken:\n  " + ", ".join(sorted(set(HEADS) - set(chosen)) or ["(none)"]))
    return dest


if __name__ == "__main__":
    cfg = yaml.safe_load(open("configs/default.yaml"))
    run(cfg, sys.argv[1:] or None)
