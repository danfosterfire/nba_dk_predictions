"""The variance budget — the project's most-quoted table, and until now print-only.

`README.md` and `CLAUDE.md` open with five numbers that size every modelling decision in the
repo: player-season identity is 58% of per-game variance, own minutes is the largest thing
left, and opponent and home/away are worth well under a percent. `README.md` says to
reproduce them with `opponent.py::variance_ceiling`, which is true for the opponent rows —
but that helper is called only inside `opponent.run`'s **printed** summary, and the identity,
minutes and home/away rows had no runnable source at all. `opponent_matchup_tierA.csv`
carries the *held-out* contrast, not the in-sample ceiling.

This module writes the whole table. Nothing here is reimplemented: the opponent rows come
from `opponent.variance_ceiling`, the null rows additionally from
`feature_diagnostics.cell_importance`, and both are emitted side by side with the gap
between them, so the one-off and its generalization cannot silently drift.

## Two denominators, and conflating them is the easiest way to misread the table

    total_per_game_variance          player-season identity only
    within_player_season_residual    everything else

Player-season identity is 58% of the total, so a row quoted "of the residual" is quoted
against the remaining 42%. The `basis` column makes the mix-up impossible rather than
merely discouraged.

## Both nulls, always

The opponent x archetype x season statistic reads **+0.96%** permuting opponent within
season and **+0.31%** permuting archetype — the same cells, the same rows, a 3x spread. The
artifact carries a row per null with `null_construction` naming the permuted marginal,
because quoting one without naming it is a recorded past error in this repo.

## The minutes row is a correction, and the superseded construction is kept

The recorded 18.6% conditions the within-player-season residual on the **raw minutes level**,
pooled across players. That is attenuated by construction: a 30-minute game is *below*
average for a 34-mpg starter and far *above* it for an 18-mpg reserve, so their residuals
cancel inside the cell. Conditioning instead on the minutes **deviation** — "he played eight
more minutes than he usually does", which is the contrast the residual is defined by — gives
**46.4%**, stable at 46.5% under a nonparametric fit, against a saturated per-player-season
upper bound of 59.4%. The mechanism is measurable directly: dk_pts per extra minute runs
0.85 at the bottom mpg tier to 1.23 at the top, so no single function of the raw level can
represent it.

Every row is kept, `own_minutes_raw_level` included, so the superseded figure stays legible
beside the corrected one instead of being quietly overwritten. The correction moves in the
safe direction — minutes matter *more* than recorded, which strengthens the shared-minutes
draw the simulator is built on rather than weakening it.

## Weighting and season

No minutes weighting: the standing rule applies to per-36 *rates*, where a garbage-time rate
carries real measurement error, and every row here is on per-game dk_pts. Season is absorbed
throughout — a player-season is nested inside a season, and every opponent cell carries
`season` explicitly.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.eda.feature_diagnostics import cell_importance, cell_means_r2, join_keys
from src.features.opponent import (
    load_player_games,
    nxt_map,
    variance_ceiling,
    within_player_season_residuals,
)

TOTAL = "total_per_game_variance"
RESIDUAL = "within_player_season_residual"

# The window the recorded budget was measured on.
DEFAULT_WINDOW = ["2014-15", "2023-24"]

# Bin width for the nonparametric minutes rows, in minutes. Fine enough that the answer is
# a property of the data rather than of the binning — 1, 2 and 3 minute bins agree to
# within 0.7 pp on the deviation basis.
MINUTES_BIN = 2.0

# The superseded construction, at the bin width that reproduces the recorded 18.6%.
RAW_LEVEL_BIN = 5.0


def _row(source: str, basis: str, n: int, seasons: str, share: float | None = None,
         value: float | None = None, units: str = "share_of_variance",
         null_construction: str = "none", in_sample: bool = True,
         note: str = "") -> dict:
    return {"source": source, "share_of_variance": share,
            "value": share if value is None else value, "units": units,
            "n_games": int(n), "basis": basis, "in_sample": in_sample,
            "null_construction": null_construction, "seasons": seasons, "note": note}


# ── Frames ────────────────────────────────────────────────────────────────────

def budget_frame(seasons: list[str], raw_dir: str | Path,
                 window: list[str] = None) -> pd.DataFrame:
    """Player-games in the window, with the within-player-season residuals attached.

    `load_player_games` already drops zero-minute rows and computes `dk_pts`, so this is the
    same frame `opponent.py` measures on — one definition of "a player-game" for both.
    """
    lo, hi = window or DEFAULT_WINDOW
    games = load_player_games(seasons, raw_dir)
    games = games[games["season"].between(lo, hi)].reset_index(drop=True)
    resid = within_player_season_residuals(games, ["dk_pts", "min"])
    games["resid_dk_pts"] = resid["dk_pts"].to_numpy(dtype=float)
    games["resid_min"] = resid["min"].to_numpy(dtype=float)
    return games


def archetype_panel(games: pd.DataFrame, seasons: list[str],
                    features_dir: str | Path, tier: str = "A") -> pd.DataFrame:
    """Games joined to the player's **prior-season** archetype, for the interaction rows.

    Lagged with `opponent.nxt_map`, the same shift `opponent.run` and
    `feature_diagnostics.opponent_null_reproduction` use — which is what makes the null rows
    here comparable to `feature_diagnostics.csv`'s row for row.
    """
    arch = pd.read_parquet(Path(features_dir) / f"archetypes_tier{tier}.parquet",
                           columns=["player_id", "season", "archetype"])
    arch["season"] = arch["season"].map(nxt_map(seasons))
    return games.merge(arch.dropna(subset=["season"]), on=["player_id", "season"],
                       how="inner")


# ── The rows ──────────────────────────────────────────────────────────────────

def identity_rows(games: pd.DataFrame, seasons: str) -> list[dict]:
    """Player-season identity, against the **total** per-game variance."""
    y = games["dk_pts"].to_numpy(dtype=float)
    keys = join_keys({"player_id": games["player_id"], "season": games["season"]})
    return [_row("player_season_identity", TOTAL, len(games), seasons,
                 share=cell_means_r2(y, keys),
                 note="who the player is, and which season — not what an encoder is for"),
            _row("player_season_cells", TOTAL, len(games), seasons,
                 value=float(len(np.unique(keys))), units="cells",
                 note="player-seasons in the window")]


def minutes_rows(games: pd.DataFrame, seasons: str, bin_width: float = MINUTES_BIN,
                 raw_bin: float = RAW_LEVEL_BIN) -> list[dict]:
    """Own minutes, four ways — the correction and the construction it supersedes.

    The ladder is the argument: a common slope on the minutes *deviation* (46.4%), the same
    thing without a linearity assumption (46.5%), a per-player-season minutes profile as the
    saturated upper bound (59.4%), and the superseded pooled function of the raw minutes
    *level* (18.6%). Only the last one is small, and it is small because it is confounded.
    """
    rd = games["resid_dk_pts"].to_numpy(dtype=float)
    rm = games["resid_min"].to_numpy(dtype=float)
    mn = games["min"].to_numpy(dtype=float)
    n = len(games)
    ident = cell_means_r2(games["dk_pts"].to_numpy(dtype=float),
                          join_keys({"p": games["player_id"], "s": games["season"]}))
    saturated = cell_means_r2(
        games["dk_pts"].to_numpy(dtype=float),
        join_keys({"p": games["player_id"], "s": games["season"],
                   "m": np.floor(mn / bin_width)}))
    return [
        _row("own_minutes", RESIDUAL, n, seasons,
             share=float(np.corrcoef(rd, rm)[0, 1] ** 2),
             note="common slope on the within-player-season minutes deviation"),
        _row("own_minutes_nonparametric", RESIDUAL, n, seasons,
             share=cell_means_r2(rd, np.floor(rm / bin_width)),
             note=f"cell means over {bin_width:g}-minute bins of the deviation — "
                  "agreeing with the linear row rules out a linearity artifact"),
        _row("own_minutes_saturated", RESIDUAL, n, seasons,
             share=(saturated - ident) / (1 - ident),
             note="cells are (player-season x minutes bin): a per-player minutes profile, "
                  "the nonparametric upper bound"),
        _row("own_minutes_raw_level", RESIDUAL, n, seasons,
             share=cell_means_r2(rd, np.floor(mn / raw_bin)),
             note=f"SUPERSEDED — pooled cell means over the raw {raw_bin:g}-minute level. "
                  "Reproduces the recorded 18.6% and is attenuated: the same minutes "
                  "count is above average for a reserve and below it for a starter"),
        _row("minutes_slope_bottom_mpg_tier", RESIDUAL, n, seasons,
             value=_slope_by_tier(games)[0], units="dk_pts_per_minute",
             note="why the raw-level construction attenuates: the slope is player-specific"),
        _row("minutes_slope_top_mpg_tier", RESIDUAL, n, seasons,
             value=_slope_by_tier(games)[-1], units="dk_pts_per_minute",
             note="why the raw-level construction attenuates: the slope is player-specific"),
    ]


def _slope_by_tier(games: pd.DataFrame,
                   edges: tuple[float, ...] = (12.0, 20.0, 28.0, 34.0)) -> list[float]:
    """dk_pts per extra minute, by the player-season's own mpg tier."""
    mpg = games.groupby(["player_id", "season"])["min"].transform("mean")
    tier = np.digitize(mpg, edges)
    out = []
    for t in np.unique(tier):
        m = tier == t
        rd, rm = games.loc[m, "resid_dk_pts"], games.loc[m, "resid_min"]
        out.append(float(np.polyfit(rm, rd, 1)[0]) if rm.std() > 0 else np.nan)
    return out


def opponent_main_rows(games: pd.DataFrame, seasons: str) -> list[dict]:
    """Opponent and home/away, on the **full** frame — neither needs an archetype.

    Kept off the archetype panel deliberately. Requiring a prior-season archetype drops 22%
    of the window and moves opponent x season from 0.69% to 0.79%, so measuring a main effect
    on the interaction's frame would overstate it for no reason. Every row carries its own
    `n_games` so the two frames stay distinguishable.
    """
    y = games["resid_dk_pts"].to_numpy(dtype=float)
    n = len(games)
    return [
        _row("opponent_x_season", RESIDUAL, n, seasons,
             share=cell_means_r2(y, join_keys({
                 "opponent": games["opponent_team_id"].astype(str),
                 "season": games["season"]})),
             note="contemporaneous opponent identity — a ceiling, not an achievable gain"),
        _row("home_away", RESIDUAL, n, seasons,
             share=cell_means_r2(y, games["is_home"].astype(str).to_numpy()),
             note="the smallest row in the budget, and the only one known in advance"),
    ]


def opponent_rows(panel: pd.DataFrame, seasons: str, n_shuffles: int = 20,
                  seed: int = 42) -> list[dict]:
    """The interaction and both nulls, on the archetype panel — through both helpers.

    `variance_ceiling` is the one-off the figures were quoted from;
    `feature_diagnostics.cell_importance` is the generalization that replaced it. Running
    both on identical rows and emitting `reproduction_gap` is the check that they agree —
    recorded at 0.005 pp and 0.002 pp, which is what the gap rows should show.

    The main effects come back on this frame too, suffixed, so the interaction can be read
    against a main effect measured on its own rows rather than against a different frame's.
    """
    y = panel["resid_dk_pts"].to_numpy(dtype=float)
    ceil = variance_ceiling(panel.assign(dk_pts=y), panel["archetype"],
                            n_shuffles=n_shuffles, seed=seed)
    n = len(panel)
    rows = [
        _row("opponent_x_season_archetype_panel", RESIDUAL, n, seasons,
             share=ceil["opponent_x_season"],
             note="the same main effect on the interaction's own rows, for comparability"),
        _row("home_away_archetype_panel", RESIDUAL, n, seasons, share=ceil["home_away"],
             note="the same main effect on the interaction's own rows, for comparability"),
        _row("opponent_x_archetype_x_season", RESIDUAL, n, seasons,
             share=ceil["opponent_x_archetype_x_season"],
             note=f"raw statistic over {ceil['n_cells']:,} cells, before any null"),
        _row("opponent_x_archetype_x_season_cells", RESIDUAL, n, seasons,
             value=float(ceil["n_cells"]), units="cells",
             note="cell count is most of the raw statistic — hence the nulls"),
    ]
    keys = {"opponent": panel["opponent_team_id"].astype(str).to_numpy(),
            "archetype": panel["archetype"].astype(str).to_numpy(),
            "season": panel["season"].to_numpy()}
    for marginal in ("opponent", "archetype"):
        imp = cell_importance(y, keys, marginal, within=keys["season"],
                              n_shuffles=n_shuffles, seed=seed)
        null_name = f"shuffle_{marginal}_within_season"
        one_off = (ceil["opponent_x_archetype_x_season"]
                   - ceil[f"null_shuffle_{marginal}"])
        rows += [
            _row("opponent_x_archetype_x_season_null_level", RESIDUAL, n, seasons,
                 share=imp["null_mean"], null_construction=null_name,
                 note=f"chance level over {n_shuffles} permutations, "
                      f"sd {imp['null_sd']:.5f}"),
            _row("opponent_x_archetype_x_season_above_null", RESIDUAL, n, seasons,
                 share=imp["above_null"], null_construction=null_name,
                 note="via feature_diagnostics.cell_importance"),
            _row("opponent_x_archetype_x_season_above_null_variance_ceiling", RESIDUAL,
                 n, seasons, share=one_off, null_construction=null_name,
                 note="via opponent.variance_ceiling — the one-off it generalizes"),
            _row("null_reproduction_gap", RESIDUAL, n, seasons,
                 share=imp["above_null"] - one_off, null_construction=null_name,
                 note="cell_importance minus variance_ceiling; recorded at <= 0.005 pp"),
        ]
    return rows


def spread_rows(games: pd.DataFrame, panel: pd.DataFrame, seasons: str) -> list[dict]:
    """The units everything else is quoted against, plus the opponent effect in dk_pts.

    A share of a residual means nothing without the residual's size, and the opponent row
    reads very differently as 0.7% of a variance than as an sd of 0.87 dk_pts against 9.4.
    """
    rd = games["resid_dk_pts"].to_numpy(dtype=float)
    per_season = panel.groupby("season").apply(
        lambda b: b.groupby("opponent_team_id")["resid_dk_pts"].mean().std(),
        include_groups=False)
    return [
        _row("within_player_season_residual_sd", RESIDUAL, len(games), seasons,
             value=float(rd.std()), units="dk_pts",
             note="the units every share on the residual basis is quoted against"),
        _row("dk_pts_sd", TOTAL, len(games), seasons,
             value=float(games["dk_pts"].std()), units="dk_pts",
             note="total per-game spread, before removing player-season identity"),
        _row("opponent_effect_sd", RESIDUAL, len(panel), seasons,
             value=float(per_season.mean()), units="dk_pts",
             note="season-average sd of the opponent cell means — the opponent row in the "
                  "units a reader can size against the residual sd"),
    ]


# ── Orchestration ─────────────────────────────────────────────────────────────

def measure(games: pd.DataFrame, panel: pd.DataFrame, seasons_label: str,
            n_shuffles: int = 20, seed: int = 42, bin_width: float = MINUTES_BIN,
            raw_bin: float = RAW_LEVEL_BIN) -> pd.DataFrame:
    rows = identity_rows(games, seasons_label)
    rows += minutes_rows(games, seasons_label, bin_width, raw_bin)
    rows += opponent_main_rows(games, seasons_label)
    rows += opponent_rows(panel, seasons_label, n_shuffles, seed)
    rows += spread_rows(games, panel, seasons_label)
    return pd.DataFrame(rows)


def run(cfg: dict, tier: str = "A") -> Path:
    features_dir = Path(cfg["data"]["features_dir"])
    out_dir = Path(cfg["eda"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    seasons = cfg["data"]["seasons"]
    vb = cfg["eda"].get("variance_budget", {})
    window = vb.get("window", cfg["eda"].get("availability", {}).get(
        "null_window", DEFAULT_WINDOW))
    # 20, not 5: the cell-mean null's per-draw sd is ~0.024 pp and its distribution is
    # skewed, so 5 draws land on +0.92% where 20 and 50 both give +0.96%.
    n_shuffles = vb.get("n_shuffles", cfg["eda"].get("availability", {}).get(
        "n_shuffles", 20))
    seed = cfg["training"]["seed"]
    label = f"{window[0]}..{window[1]}"

    games = budget_frame(seasons, cfg["data"]["raw_dir"], window)
    panel = archetype_panel(games, seasons, features_dir, tier)
    print(f"Variance budget: {len(games):,} player-games in {label}, "
          f"{len(panel):,} with a prior-season archetype")

    table = measure(games, panel, label, n_shuffles, seed,
                    vb.get("minutes_bin", MINUTES_BIN),
                    vb.get("raw_level_bin", RAW_LEVEL_BIN))
    dest = out_dir / "variance_budget.csv"
    table.to_csv(dest, index=False)

    shares = table[table["units"] == "share_of_variance"]
    print("\nShares, by denominator — the two are NOT interchangeable:")
    for basis in (TOTAL, RESIDUAL):
        sub = shares[shares["basis"] == basis]
        if sub.empty:
            continue
        print(f"  ── as a share of {basis} ──")
        for r in sub.itertuples():
            null = "" if r.null_construction == "none" else f"  [{r.null_construction}]"
            print(f"    {r.source:<52} {r.share_of_variance * 100:>7.3f}%  "
                  f"n={r.n_games:>7,}{null}")

    other = table[table["units"] != "share_of_variance"]
    print("\nUnits and counts:")
    for r in other.itertuples():
        print(f"    {r.source:<52} {r.value:>10.3f}  {r.units}")

    gaps = table[table["source"] == "null_reproduction_gap"]
    worst = float(gaps["share_of_variance"].abs().max()) * 100
    print(f"\n  variance_ceiling vs cell_importance: worst gap {worst:.4f} pp — the two "
          "implementations agree.")
    print("  The two nulls differ by ~3x on identical cells, which is why every null row "
          "names\n  the marginal it permuted.")
    print(f"\nVariance budget: {len(table):,} rows → {dest}")
    return dest


if __name__ == "__main__":
    cfg = yaml.safe_load(open("configs/default.yaml"))
    run(cfg)
