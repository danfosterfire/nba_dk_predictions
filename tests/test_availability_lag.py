"""§16's lag-recovery ladder on the availability head — the design, not the gate.

`docs/availability-window-plan.md` §16 widens `availability.build_design` from *is there a
lag-1 row* to *is there any usable prior season*, by filling a missing lag-1 block from the
nearest one that exists. The claim that lets the recovered rows be scored by a posterior
fitted before the ladder existed is narrow and mechanical — **rung 0 comes back
bit-identical** — so that is what these tests pin, alongside the rung boundaries, the shrink
arithmetic, and the two columns §16d says the design could not previously express.

Everything is deterministic: each synthetic player-season has an exact `gp` out of an exact
`team_games`, so a shrunk share can be checked against `(gp + k*a) / (n + k)` by arithmetic
rather than by tolerance.
"""

import numpy as np
import pandas as pd
import pytest

from src.models.availability import (AvailabilityLagLadder, LAG_COLS, STALENESS_COLS,
                                     build_design, lag_ladder, ladder_recovered)
from src.models.component_rates import LADDER_RUNGS, MIN_PRIOR_MINUTES

SEASONS = ["2017-18", "2018-19", "2019-20", "2020-21", "2021-22", "2022-23"]
TEAM_GAMES = 82

# Two prior-season volumes, either side of `classify`'s 200-minute test — which on THIS head
# only ever separates `returnee_lag2` from `returnee_thin`, never a rung from rung 0.
FULL, THIN = 1500.0, 100.0


def _season_row(player: int, season: str, gp: int, minutes: float) -> dict:
    """One played player-season, with every column `LAG_COLS` needs."""
    return {"season": season, "player_id": player, "gp": gp, "team_games": TEAM_GAMES,
            "gp_share": gp / TEAM_GAMES, "total_minutes": minutes,
            "minutes_per_game": minutes / max(gp, 1),
            "trailing_missed": 0.0, "end_play_rate": 1.0, "n_spells": 1.0,
            "longest_spell": 2.0, "playoff_games": 0.0, "playoff_mpg": 0.0,
            "playoff_minutes_share": 0.0, "career_minutes": minutes}


def _frame(careers: dict[int, dict[str, tuple[int, float]]]) -> pd.DataFrame:
    """`{player: {season: (gp, total_minutes)}}` -> the season-availability frame."""
    return pd.DataFrame([_season_row(p, s, gp, minutes)
                         for p, seasons in careers.items()
                         for s, (gp, minutes) in seasons.items()])


def _bios(tmp_path, players):
    from src.data.fetch import _slug, nbastats_dir
    dest = nbastats_dir(tmp_path)
    dest.mkdir(parents=True, exist_ok=True)
    for season in SEASONS:
        pd.DataFrame({"PLAYER_ID": list(players),
                      "AGE": [25 for _ in players]}).to_csv(
            dest / f"player_bio_stats_{_slug(season)}.csv", index=False)
    return tmp_path


def _starts() -> pd.Series:
    return pd.Series({s: pd.Timestamp(f"{s[:4]}-10-20") for s in SEASONS})


def _ladder(rungs=LADDER_RUNGS, k: float = 200.0, anchor: float = 0.25):
    """A ladder whose constants are chosen for arithmetic, not for realism."""
    return AvailabilityLagLadder(rungs=tuple(rungs), k_games=k, league_share=anchor)


# One player per rung, plus a veteran, a thin-lag-1 player and a true rookie. The target
# season is the last one in every case.
CAREERS: dict[int, dict[str, tuple[int, float]]] = {
    1: {s: (70, FULL) for s in SEASONS},                                  # veteran
    2: {"2021-22": (8, THIN), "2022-23": (70, FULL)},                     # thin lag-1
    3: {"2020-21": (70, FULL), "2022-23": (70, FULL)},                    # returnee, full
    4: {"2020-21": (8, THIN), "2022-23": (70, FULL)},                     # returnee, thin
    5: {"2019-20": (70, FULL), "2022-23": (70, FULL)},                    # away two seasons
    6: {"2022-23": (70, FULL)},                                           # true rookie
    7: {"2020-21": (60, FULL), "2021-22": (70, FULL),
        "2022-23": (70, FULL)},                                           # two consecutive
    8: {"2019-20": (60, FULL), "2021-22": (70, FULL),
        "2022-23": (70, FULL)},                                           # played S-1, S-3
}

#: What each player's target-season row must be labelled. `thin_prior` is deliberately
#: absent: on this head a thin lag-1 still HAS a `gp_share_lag1`, so player 2 is rung 0.
RUNG_OF = {1: "veteran", 2: "veteran", 3: "returnee_lag2", 4: "returnee_thin",
           5: "no_usable_lag", 7: "veteran", 8: "veteran"}


def _designs(tmp_path, rungs=LADDER_RUNGS, careers=None, **kw):
    careers = CAREERS if careers is None else careers
    frame, raw = _frame(careers), _bios(tmp_path, careers)
    plain = build_design(frame, SEASONS, raw, _starts())
    wide = build_design(frame, SEASONS, raw, _starts(), ladder=_ladder(rungs, **kw))
    return plain, wide


def _last(design: pd.DataFrame) -> pd.DataFrame:
    return design[design["season"] == SEASONS[-1]].set_index("player_id")


# ── Rung 0 does not move ──────────────────────────────────────────────────────

def test_rung_zero_comes_back_bit_identical(tmp_path):
    """The one claim §16f's non-regression requirement rests on.

    The recovered rows are legitimate only if the rows the shipped coefficients were fitted
    on did not move — not approximately, and not up to a tolerance. Every shared column is
    compared exactly, NaN for NaN.
    """
    plain, wide = _designs(tmp_path)
    key = ["player_id", "season"]
    a = plain.set_index(key).sort_index()
    b = wide[~ladder_recovered(wide)].set_index(key).sort_index()

    assert a.index.equals(b.index), "the ladder moved rung 0's row set"
    assert set(a.columns) <= set(b.columns), "the ladder dropped a shipped column"
    for col in a.columns:
        x, y = a[col], b[col]
        if x.dtype.kind in "fi":
            assert np.array_equal(x.to_numpy(float), y.to_numpy(float),
                                  equal_nan=True), col
        else:
            assert x.equals(y), col


def test_the_ladder_off_adds_no_column_and_no_row(tmp_path):
    """`None` is the pre-ladder builder exactly — seven modules import it and none of them
    asked for a wider frame or a `lag_rung` column they would have to ignore."""
    plain, wide = _designs(tmp_path)
    assert "lag_rung" not in plain.columns
    assert "gp_lag1" not in plain.columns and "team_games_lag1" not in plain.columns
    assert not any(c in plain.columns for c in STALENESS_COLS)
    assert not ladder_recovered(plain).any()
    assert len(wide) > len(plain)


def test_the_widening_can_never_shrink_the_unserved_pool(tmp_path):
    """§16d, as an assertion. The drop test has always been `gp_share_lag1` ALONE, so no
    rung can take a row the shipped design carries."""
    plain, wide = _designs(tmp_path)
    kept = set(map(tuple, wide[["player_id", "season"]].to_numpy()))
    assert set(map(tuple, plain[["player_id", "season"]].to_numpy())) <= kept


# ── The rung boundaries ───────────────────────────────────────────────────────

def test_each_rung_admits_the_population_it_names(tmp_path):
    """One player per rung, labelled by why the shipped design has no row for him."""
    _, wide = _designs(tmp_path)
    last = _last(wide)
    assert {p: last.loc[p, "lag_rung"] for p in RUNG_OF} == RUNG_OF


def test_a_thin_lag_one_is_rung_zero_here_and_not_a_recovered_rung(tmp_path):
    """The documented asymmetry with `component_rates`' ladder.

    That design's boundary is a 200-minute test, so a thin lag-1 fails it and `thin_prior`
    is a rung it can recover. This design's boundary is a PRESENCE test, so the same player
    has always been inside it — and `thin_prior` is therefore structurally empty here.
    """
    _, wide = _designs(tmp_path)
    last = _last(wide)
    assert last.loc[2, "lag_rung"] == "veteran"
    assert last.loc[2, "lag_source"] == "lag1"
    assert last.loc[2, "total_minutes_lag1"] == THIN < MIN_PRIOR_MINUTES
    assert "thin_prior" not in set(wide["lag_rung"])


def test_a_true_rookie_is_never_admitted_at_any_rung(tmp_path):
    """No own-availability feature exists at any lag, so the ladder must leave him to
    `no_design_availability`. §16c: the plug-in stays, whatever this round admits."""
    _, wide = _designs(tmp_path)
    assert 6 not in set(_last(wide).index)
    assert "true_rookie" not in set(wide["lag_rung"])


def test_only_the_configured_rungs_enter(tmp_path):
    """The gate is per rung and the config key is a list, so it has to bind."""
    _, one = _designs(tmp_path, rungs=("returnee_lag2",))
    last = _last(one)
    assert set(last["lag_rung"]) == {"veteran", "returnee_lag2"}
    assert set(last.index) == {1, 2, 3, 7, 8}


def test_an_unknown_rung_is_refused_by_name(tmp_path):
    """A typo in the config key must fail loudly rather than build a narrower design."""
    cfg = {"stan": {"availability": {"lag_ladder": ["returnee"]}},
           "evaluation": {"predictions_dir": str(tmp_path)}}
    with pytest.raises(ValueError, match="unknown rung"):
        lag_ladder(cfg)
    assert lag_ladder({"stan": {"availability": {"lag_ladder": []}}}) is None


# ── Provenance ────────────────────────────────────────────────────────────────

def test_every_row_says_where_its_block_came_from(tmp_path):
    """A scored unit whose provenance is not recorded cannot be audited on a board."""
    _, wide = _designs(tmp_path)
    last = _last(wide)
    assert last.loc[1, "lag_source"] == "lag1"        # veteran
    assert last.loc[3, "lag_source"] == "lag2"        # returnee
    assert last.loc[5, "lag_source"] == "lag3"        # away two seasons
    assert last.loc[1, "lag_minutes"] == FULL
    assert last.loc[3, "lag_minutes"] == FULL
    assert last.loc[4, "lag_minutes"] == THIN


def test_the_carried_block_is_the_source_season_not_the_missing_one(tmp_path):
    """A returnee's volume and context columns are what his most recent season actually
    was — they are not estimates of a rate, so they are carried raw."""
    _, wide = _designs(tmp_path)
    last = _last(wide)
    assert last.loc[3, "total_minutes_lag1"] == FULL
    assert last.loc[3, "gp_lag1"] == 70
    assert last.loc[3, "team_games_lag1"] == TEAM_GAMES
    assert np.isclose(last.loc[3, "minutes_per_game_lag1"], FULL / 70)


# ── The shrink ────────────────────────────────────────────────────────────────

def test_the_carried_share_is_the_empirical_bayes_blend(tmp_path):
    """`(gp + k*anchor) / (team_games + k)` — a proportion's reliability lives in its
    trials, which is `lag_recovery.recent_conversion`'s form and not a second one."""
    k, anchor = 200.0, 0.25
    _, wide = _designs(tmp_path, k=k, anchor=anchor)
    last = _last(wide)
    # The source season's own games, which for the thin returnee is 8 and not 70 — the
    # blend is on HIS trials, so a thin source lands nearer the anchor than a full one.
    for player, source_gp in ((3, 70), (4, 8), (5, 70)):
        assert np.isclose(last.loc[player, "gp_share_lag1"],
                          (source_gp + k * anchor) / (TEAM_GAMES + k)), player
    assert last.loc[4, "gp_share_lag1"] < last.loc[3, "gp_share_lag1"]
    # Rung 0 is untouched by the shrink, which is what bit-identity means column by column.
    assert np.isclose(last.loc[1, "gp_share_lag1"], 70 / TEAM_GAMES)


def test_the_shrink_reaches_both_ends_of_its_own_range(tmp_path):
    """`k = 0` is the raw carried share and a huge `k` is the anchor. The arm has to be able
    to express 'do not shrink', because that is the sensitivity §16 prices the constant
    against — the raw carry is the rest of the ladder with one number removed."""
    for k, expected in ((0.0, 70 / TEAM_GAMES), (1e9, 0.25)):
        _, wide = _designs(tmp_path, k=k, anchor=0.25)
        assert np.isclose(_last(wide).loc[3, "gp_share_lag1"], expected, atol=1e-5)


def test_the_anchor_is_below_the_veteran_level_and_pulls_downward(tmp_path):
    """§16b's measured direction, pinned. A carried season is biased HIGH by exactly the
    fact it erased, so the shrink has to move the level down — which is why the anchor is
    the recovered population's own rate and not the league's."""
    _, wide = _designs(tmp_path, k=200.0, anchor=0.25)
    last = _last(wide)
    assert last.loc[3, "gp_share_lag1"] < last.loc[1, "gp_share_lag1"]


# ── §16d's two columns ────────────────────────────────────────────────────────

def test_n_prior_seasons_counts_real_seasons_and_not_depth(tmp_path):
    """§16d's defect, fixed in the one column that could carry it.

    Player 3 played S-2 and nothing else; player 7 played S-1 and S-2. Under the old
    formula the recovered row would land on the healthy population's value and be absorbed
    into it. The count is bit-identical wherever lag-1 is present, which is every row the
    shipped design carries.
    """
    plain, wide = _designs(tmp_path)
    last, shipped = _last(wide), _last(plain)
    assert last.loc[7, "n_prior_seasons"] == 2       # two consecutive, unchanged
    assert last.loc[3, "n_prior_seasons"] == 1       # ONE real season, not two
    assert last.loc[5, "n_prior_seasons"] == 1
    for player in shipped.index:
        assert last.loc[player, "n_prior_seasons"] == shipped.loc[player,
                                                                 "n_prior_seasons"]


def test_the_interior_gap_column_sees_the_row_the_backfill_erases(tmp_path):
    """§16d's other half — the 130 rows the head already carries with a gap erased.

    Player 8 played S-1 and S-3 and missed S-2. The `lag2 := lag1` backfill below the
    `dropna` makes him a two-consecutive-season veteran on every other column, and this is
    the only one that disagrees. Zero on the healthy population, by construction.
    """
    _, wide = _designs(tmp_path)
    last = _last(wide)
    assert last.loc[8, "lag_interior_gaps"] == 1
    assert last.loc[8, "lag_rung"] == "veteran"
    assert last.loc[8, "n_prior_seasons"] == 2       # depth two, and the gap is real
    for player in (1, 3, 5, 7):
        assert last.loc[player, "lag_interior_gaps"] == 0


def test_the_staleness_block_is_inert_on_every_rung_zero_row(tmp_path):
    """Arm 2 adds three columns to the FIT, so what they say about the unchanged population
    has to be exactly 'nothing happened here' — `lag_interior_gaps` excepted, which is the
    population it was added to see."""
    _, wide = _designs(tmp_path)
    rung_zero = wide[~ladder_recovered(wide)]
    assert (rung_zero["lag_recovered"] == 0.0).all()
    assert (rung_zero["lag_gap_seasons"] == 0.0).all()
    recovered = wide[ladder_recovered(wide)].set_index("player_id")
    assert recovered.loc[3, "lag_gap_seasons"] == 1.0     # missed S-1
    assert recovered.loc[5, "lag_gap_seasons"] == 2.0     # missed S-1 and S-2
    assert (recovered["lag_recovered"] == 1.0).all()


def test_the_lag_columns_still_backfill_from_the_recovered_block(tmp_path):
    """The three lines under the `dropna` are unchanged, and they now run on a block the
    ladder wrote — which is why a recovered row survives the same way a one-season veteran
    does rather than needing a second exemption."""
    _, wide = _designs(tmp_path)
    last = _last(wide)
    for col in LAG_COLS:
        assert np.isfinite(last.loc[3, f"{col}_lag2"]), col
        assert np.isfinite(last.loc[3, f"{col}_lag3"]), col


# ── Turning the availability ladder on — §16j's second edit ──────────────────

def _cfgs():
    """`(ladder off, ladder on)` — the shipped config and the same with the key emptied."""
    import copy
    import yaml

    on = yaml.safe_load(open("configs/default.yaml"))
    off = copy.deepcopy(on)
    off["stan"]["availability"]["lag_ladder"] = []
    return off, on


def test_the_shipped_config_admits_the_one_rung_the_availability_gate_cleared():
    """§16i's verdict, as the key seven consumers read.

    Pinned because this key is a config list whose blast radius is the widest in the
    project: it reaches the minutes head, the composition, the spell process and the
    simulator through one choke point, and two of its three rungs were measured as losses.
    """
    import yaml

    cfg = yaml.safe_load(open("configs/default.yaml"))
    assert cfg["stan"]["availability"]["lag_ladder"] == ["returnee_lag2"]


def test_every_availability_fitting_path_cuts_to_rung_zero():
    """§16's imputation-only claim, pinned by parsing rather than by discipline.

    `availability_design` is the choke point seven consumers reach their rows through, and
    §16i's shipped arm scores the recovered rows with a posterior fitted BEFORE the ladder
    existed — the arm that admitted them to a fit was built, priced and rejected. So every
    module that fits from this design must cut to `rung_zero` first, and a new one that
    does not has to fail here rather than silently widen four heads' fitting populations.

    `StanAvailability.fitting_rows` is the head's own choke point and covers `run`,
    `fit_and_score`'s two point MLEs, `posteriors` and `final_evaluation` at once, which is
    why those four are not listed separately.
    """
    import pathlib

    for module in ("src/models/stan_availability.py",      # the head's own fitting_rows
                   "src/models/stan_minutes.py",
                   "src/models/stan_composition.py",
                   "src/models/stan_games_played.py",
                   "src/models/season_terms.py",
                   "src/models/availability_exchangeability.py",
                   "src/models/availability_no_prior.py",
                   "src/models/model_cards.py"):
        source = pathlib.Path(module).read_text()
        assert "rung_zero(" in source, (
            f"{module} reaches `availability_design` and does not cut to rung 0; §16's "
            f"ladder would enter its fitting population")


def test_the_ladder_widens_the_design_and_rung_zero_takes_it_back():
    """The design gains rows; the rows any head reads do not move.

    The claim is stated on the FEATURE MATRIX rather than on the frame, because the frame
    does move: `availability_preseason` writes a season-CENTRED twin of the preseason
    minutes level, and a season mean is a property of the frame, so 189 new rows shift it.
    That column is in no head's feature list and the matrix is what the coefficients see.
    """
    from src.models.availability import ladder_recovered, rung_zero
    from src.models.stan_availability import (PI_FEATURES, StanAvailability, head_design,
                                              head_features)

    off, on = _cfgs()
    d_off, d_on = head_design(off), head_design(on)
    assert len(d_on) > len(d_off)
    assert int(ladder_recovered(d_on).sum()) == len(d_on) - len(d_off)
    assert not ladder_recovered(d_off).any()

    head = StanAvailability(first_season="2012-13")
    f_off, f_on = head.fitting_rows(d_off), head.fitting_rows(d_on)
    assert len(f_off) == len(f_on)
    for block in (head_features(True), list(PI_FEATURES), ["gp", "team_games"]):
        np.testing.assert_array_equal(f_off[block].to_numpy(dtype=float),
                                      f_on[block].to_numpy(dtype=float))


def test_the_ladder_leaves_the_other_three_heads_bit_identical():
    """The minutes head, the composition and the spell process do not move at all.

    §16i measured the ladder on games played and on nothing else, so the three heads that
    merely *read* this design have to come back unchanged — and the composition is cut at
    its MERGE rather than at its split, so its scoring block is unmoved too. Without that
    cut, 5,710 player-game rows would flip `design_missing` from 1 to 0 and move `impute`'s
    train means, which is a different head and the one refit in this project measured in
    hours.
    """
    import pandas as pd
    from pathlib import Path

    from src.models.stan_composition import head_frame
    from src.models.stan_games_played import games_played_design
    from src.models.stan_minutes import head_design as minutes_head_design

    off, on = _cfgs()
    panel = pd.read_parquet(
        Path(on["data"]["features_dir"]) / "availability_panel.parquet")
    for label, builder in (("minutes", lambda c: minutes_head_design(c)),
                           ("composition", lambda c: head_frame(c)),
                           ("games_played", lambda c: games_played_design(c, panel))):
        a, b = builder(off).reset_index(drop=True), builder(on).reset_index(drop=True)
        assert len(a) == len(b), f"{label}: {len(a)} vs {len(b)}"
        differing = [c for c in a.columns if c in b.columns and not a[c].equals(b[c])]
        assert not differing, f"{label} moved on {differing}"
