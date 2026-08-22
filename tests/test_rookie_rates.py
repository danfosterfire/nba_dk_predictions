"""The rookie rate design and its no-fit floors — `docs/rookie-rates-plan.md` §5c.

Three properties carry this session and each one is a claim a later session leans on:

- **Disjointness** from the post-ladder veteran design. Session 6 makes the simulator's
  scorable units the *union* of the two families, so a player in both would enter the
  tensor twice — and the guarantee is not "the two builders happen to disagree", it is that
  they read one labelling (`lag_recovery.classify`) with complementary masks.
- **The shrinkage weight**, arithmetic rather than approximate: `w * (level - center)` at
  `w = min_pre / (min_pre + k)`, with both ends of the grid reachable.
- **Zero-recovery**, which is what lets the Stan L2-at-zero prior stand in for absent
  information. Every block must be *exactly* 0 where its information does not exist, so
  these are equality assertions and not tolerances.

Everything is deterministic: each synthetic player-season has an exact per-36 rate and an
exact preseason reading, so every expected value below is arithmetic.
"""

import numpy as np
import pandas as pd

from src.eda.preseason_value import MISSING_AGE_COLS
from src.features.team_context import UNDRAFTED_BUCKET
from src.models import rookie_rates as RR
from src.models.component_rates import (CONVERSION_HEADS, COUNT_HEADS, LADDER_RUNGS,
                                        LagLadder, build_design as veteran_design,
                                        rate_columns)

SEASONS = ["2017-18", "2018-19", "2019-20", "2020-21", "2021-22", "2022-23"]
TARGET = SEASONS[-1]

RATES = {"fga": 20.0, "fta": 5.0, "reb": 8.0, "ast": 4.0, "stl": 1.5, "blk": 1.0,
         "tov": 2.5}
FG3A_SHARE, FT_PCT, FG2_PCT, FG3_PCT = 0.4, 0.8, 0.5, 0.35

# One rookie per shape the design has to encode, all debuting in the target season.
#   100 lottery top-5, drafted the same year, a full preseason
#   101 second round, drafted three years earlier, a full preseason
#   102 undrafted, no preseason row at all
#   103 undrafted but carrying a stray draft year, a preseason row with zero minutes
ROOKIES = {100: (3.0, 2022.0), 101: (45.0, 2019.0), 102: (np.nan, np.nan),
           103: (np.nan, 2018.0)}
PRE_MINUTES = {100: 100.0, 101: 60.0, 103: 0.0}

# A veteran and one player per ladder rung, so the disjointness test is against a design
# that actually carries recovered rows rather than against an empty widening. Player 5 is
# the shape neither family reaches: his only prior season is five back, outside the
# ladder's `max_lag=3`, and he is not a rookie either because he has played.
VETERANS = {1: {s: 1500.0 for s in SEASONS},
            2: {"2021-22": 100.0, TARGET: 1500.0},
            3: {"2020-21": 1500.0, TARGET: 1500.0},
            4: {"2019-20": 1500.0, TARGET: 1500.0},
            5: {"2017-18": 1500.0, TARGET: 1500.0}}


def _season_rows(player: int, season: str, total_minutes: float, n_games: int = 10,
                 scale: float = 1.0) -> list[dict]:
    """One played season at exactly `scale x RATES` per 36 minutes."""
    minutes = total_minutes / n_games
    rows = []
    for _ in range(n_games):
        row = {"player_id": player, "season": season, "min": minutes, "played": 1}
        for c, r in RATES.items():
            row[c] = r * scale * minutes / 36.0
        row["fg3a"] = row["fga"] * FG3A_SHARE
        row["fg2a"] = row["fga"] - row["fg3a"]
        row["ftm"] = row["fta"] * FT_PCT
        row["fg2m"] = row["fg2a"] * FG2_PCT
        row["fg3m"] = row["fg3a"] * FG3_PCT
        rows.append(row)
    return rows


def _targets(careers: dict[int, dict[str, float]]) -> pd.DataFrame:
    rows = []
    for player, seasons in careers.items():
        for season, minutes in seasons.items():
            rows += _season_rows(player, season, minutes)
    return pd.DataFrame(rows)


def _bios(tmp_path, players):
    from src.data.fetch import _slug, nbastats_dir

    dest = nbastats_dir(tmp_path)
    dest.mkdir(parents=True, exist_ok=True)
    for season in SEASONS:
        pd.DataFrame({"PLAYER_ID": list(players),
                      "AGE": [22 for _ in players]}).to_csv(
            dest / f"player_bio_stats_{_slug(season)}.csv", index=False)
    return tmp_path


def _panel(scale: float = 1.0) -> pd.DataFrame:
    """A preseason row per `PRE_MINUTES` entry, at exactly `scale x RATES` per 36."""
    rows = []
    for player, minutes in PRE_MINUTES.items():
        row = {"season": TARGET, "player_id": player, "min_pre": minutes}
        for c in COUNT_HEADS:
            row[f"pre_per36_{c}"] = RATES[c] * scale if minutes else np.nan
        row["fga_pre"] = RATES["fga"] * scale * minutes / 36.0
        row["fg3a_pre"] = row["fga_pre"] * FG3A_SHARE
        row["fgm_pre"] = (row["fga_pre"] - row["fg3a_pre"]) * FG2_PCT + \
            row["fg3a_pre"] * FG3_PCT
        row["fg3m_pre"] = row["fg3a_pre"] * FG3_PCT
        row["fta_pre"] = RATES["fta"] * scale * minutes / 36.0
        row["ftm_pre"] = row["fta_pre"] * FT_PCT
        rows.append(row)
    return pd.DataFrame(rows)


def _draft() -> pd.DataFrame:
    rows = []
    for player, (number, year) in ROOKIES.items():
        for season in SEASONS:
            rows.append({"player_id": player, "season": season, "draft_number": number,
                         "draft_year": year})
    for player in VETERANS:
        for season in SEASONS:
            rows.append({"player_id": player, "season": season, "draft_number": 10.0,
                         "draft_year": 2014.0})
    return pd.DataFrame(rows)


def _careers() -> dict[int, dict[str, float]]:
    return {**VETERANS, **{p: {TARGET: 900.0} for p in ROOKIES}}


def _design(tmp_path, **kw) -> pd.DataFrame:
    careers = _careers()
    return RR.build_design(_targets(careers), SEASONS, _bios(tmp_path, careers),
                           _panel(**kw), _draft()).set_index("player_id")


def _ladder(k: float = 200.0, mu: float = 1.0) -> LagLadder:
    return LagLadder(rungs=tuple(LADDER_RUNGS),
                     shrinkage={c: k for c in rate_columns()},
                     means={c: mu for c in rate_columns()},
                     conversion={m: (10.0, 0.5) for m, _ in CONVERSION_HEADS})


# ── Disjointness ──────────────────────────────────────────────────────────────

def test_the_two_families_partition_the_population(tmp_path):
    """§5c's construction, and the guarantee Session 6's `units` union rests on.

    The veteran design is built at EVERY ladder rung, not the one rung the gate admitted:
    the claim is about where the boundary is, so a rung that ships later must not be able
    to reach a row this head already carries.
    """
    careers = _careers()
    targets, raw = _targets(careers), _bios(tmp_path, careers)
    rookie = RR.build_design(targets, SEASONS, raw, _panel(), _draft())
    veteran = veteran_design(targets, SEASONS, raw, ladder=_ladder())

    def keys(frame):
        return set(zip(frame["player_id"], frame["season"]))

    assert not keys(rookie) & keys(veteran)
    # And the widening is real — a ladder that recovered nothing would make the test vacuous.
    assert set(veteran["lag_rung"]) > {"veteran"}
    assert set(rookie.loc[rookie["season"] == TARGET, "player_id"]) == set(ROOKIES)


def test_the_population_is_first_played_season_and_not_absent_lags(tmp_path):
    """A player whose history falls outside the window has an NBA past the window cannot
    see, and calling him a rookie would hand this head a row it cannot serve. The test is
    `lag_recovery.classify`'s, reached through the builder rather than restated.

    Player 5 is that shape: his only prior season is five back, so at `max_lag=3` he has
    no lag column of any kind and looks exactly like a rookie to a test that reads lags.
    He belongs to NEITHER family — which is the long-tail half §7c left open — and the
    point here is that the rookie head does not quietly absorb him.
    """
    careers = {**VETERANS, 100: {TARGET: 900.0}}
    design = RR.build_design(_targets(careers), SEASONS, _bios(tmp_path, careers),
                             _panel(), _draft())
    assert set(design.loc[design["season"] == TARGET, "player_id"]) == {100}


def test_the_design_carries_no_lag_columns_at_all(tmp_path):
    """Structurally missing, so present-and-NaN would be a worse answer than absent: a
    caller who reached for `reb_p36_lag1` here should get a `KeyError`, not a silent NaN
    that propagates into a count prediction."""
    design = _design(tmp_path)
    assert not [c for c in design.columns if c.endswith(("_lag1", "_lag2", "_lag3"))]


# ── The slot block ────────────────────────────────────────────────────────────

def test_the_slot_block_encodes_the_bucket_and_its_staleness(tmp_path):
    design = _design(tmp_path)
    assert design.loc[100, "draft_lottery_top5"] == 1.0
    assert design.loc[100, RR.YEARS_SINCE_DRAFT] == 0.0
    assert design.loc[101, "draft_second_round"] == 1.0
    assert design.loc[101, RR.YEARS_SINCE_DRAFT] == 3.0
    assert design.loc[101, "draft_second_round__x__years_since_draft"] == 3.0
    # The interaction is a product and not a shared staleness column, which is what keeps
    # a same-year lottery pick distinguishable from an undrafted arrival.
    assert design.loc[100, "draft_lottery_top5__x__years_since_draft"] == 0.0


def test_an_undrafted_rookie_is_the_slot_blocks_exact_zero(tmp_path):
    """Zero-recovery for a block whose whole content is *where he was taken*. Player 103
    carries a stray draft year in the matrix and must still land on zero — an undrafted
    player has no draft to be years since."""
    design = _design(tmp_path)
    cols = RR.SLOT_COLS + [RR.YEARS_SINCE_DRAFT] + RR.SLOT_INTERACTION_COLS
    for player in (102, 103):
        assert design.loc[player, "draft_bucket"] == UNDRAFTED_BUCKET
        assert list(design.loc[player, cols]) == [0.0] * len(cols)


# ── The preseason block ───────────────────────────────────────────────────────

def test_the_shrunk_level_is_the_reliability_weighted_centred_level(tmp_path):
    """`w * (level - center)` at `w = m / (m + k)`, arithmetic and not a tolerance."""
    k, center = 100.0, 1.0
    design = _design(tmp_path)
    leveled = RR.with_levels(design, {m: (10.0, 0.5) for m, _ in CONVERSION_HEADS})
    out = RR.with_shrunk_level(leveled, "reb", k, center)
    level = np.log1p(RATES["reb"])
    for player, minutes in ((100, 100.0), (101, 60.0)):
        w = minutes / (minutes + k)
        assert np.isclose(out.loc[player, RR.shrunk_column("reb")],
                          w * (level - center))


def test_the_shrink_reaches_both_ends_of_its_own_range(tmp_path):
    """`k = 0` is the centred level unshrunk and a huge `k` is a column of zeros, so the
    grid brackets both endpoints and an interior optimum is a real one. The arm has to be
    able to say 'do not shrink': raw preseason per-36 loses to the anti-model on 6 of 8
    targets, and the fitted `k` is what separates the two."""
    center = 1.0
    leveled = RR.with_levels(_design(tmp_path),
                             {m: (10.0, 0.5) for m, _ in CONVERSION_HEADS})
    unshrunk = RR.with_shrunk_level(leveled, "reb", 0.0, center)
    swamped = RR.with_shrunk_level(leveled, "reb", 1e12, center)
    assert np.isclose(unshrunk.loc[100, RR.shrunk_column("reb")],
                      np.log1p(RATES["reb"]) - center)
    assert abs(swamped.loc[100, RR.shrunk_column("reb")]) < 1e-9


def test_a_missing_preseason_is_the_blocks_exact_zero_however_it_is_missing(tmp_path):
    """Two ways to have no reading — no panel row (102) and a panel row with no minutes
    (103) — and both must give exactly 0 on every one of the eleven columns, by the row's
    own volume rather than by a special case."""
    center = 3.0
    leveled = RR.with_levels(_design(tmp_path),
                             {m: (10.0, 0.5) for m, _ in CONVERSION_HEADS})
    for _, component, _ in RR.head_list():
        out = RR.with_shrunk_level(leveled, component, 160.0, center)
        for player in (102, 103):
            assert out.loc[player, RR.shrunk_column(component)] == 0.0
        assert out.loc[100, RR.shrunk_column(component)] != 0.0


def test_the_missing_indicators_say_who_is_missing_and_span_has_preseason(tmp_path):
    """The four age-split indicators partition the missing rows exactly, which is why
    `has_preseason` is not a twelfth feature: it is `1 - sum(indicators)` and adding it
    would make the block rank-deficient against the intercept."""
    design = _design(tmp_path)
    block = design[MISSING_AGE_COLS].to_numpy(dtype=float)
    assert np.allclose(block.sum(axis=1),
                       1.0 - design["has_preseason"].to_numpy(dtype=float))
    assert design.loc[102, "pre_missing__<24"] == 1.0
    assert design.loc[100, MISSING_AGE_COLS].sum() == 0.0
    assert not set(MISSING_AGE_COLS) & {"has_preseason"}
    assert "has_preseason" not in RR.head_features("reb")


def test_the_conversion_level_is_shrunk_on_attempts_not_on_minutes(tmp_path):
    """A percentage's reliability lives in its attempts — `carry_forward_conversion`'s
    device, pointed at the preseason. A rookie who went 0-for-2 from three has a raw
    preseason 3P% of exactly 0.000, and carrying that onto 200 regular-season attempts is
    not a weak estimator but a broken one."""
    k_att, league = 20.0, 0.5
    design = _design(tmp_path)
    leveled = RR.with_levels(design, {m: (k_att, league) for m, _ in CONVERSION_HEADS})
    attempts = design.loc[100, "fta_pre"]
    made = attempts * FT_PCT
    expected = (made + k_att * league) / (attempts + k_att)
    assert np.isclose(RR.conversion_pct(design, "ftm", k_att, league)[
        list(design.index).index(100)], expected)
    assert np.isclose(leveled.loc[100, RR.level_column("ftm")],
                      np.log(expected / (1 - expected)))
    # The device moves the reading toward the league mean and not away from it.
    assert league < expected < FT_PCT


# ── The floor ─────────────────────────────────────────────────────────────────

def _floor_frame() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = pd.DataFrame({
        "player_id": [1, 2], "season": [TARGET, TARGET],
        "draft_bucket": ["lottery", "lottery"], "min_pre": [60.0, 0.0],
        "pre_per36_reb": [10.0, np.nan], "total_minutes": [1000.0, 1000.0]})
    priors = pd.DataFrame({"season": [TARGET], "draft_bucket": ["lottery"],
                           "prior": [4.0], "prior_n": [50], "pooled_prior": [4.0]})
    return rows, priors


def test_the_floor_is_the_volume_blend_of_preseason_and_the_draft_bucket():
    """P4(b)'s selected estimator, reached through `rookie_priors.arm_predictions` rather
    than reimplemented — the head's benchmark and the measurement that chose the benchmark
    have to be the same arithmetic."""
    rows, priors = _floor_frame()
    k = 60.0
    arms = RR.floor_arms(rows, priors, "pre_per36_reb", k)
    w = 60.0 / (60.0 + k)
    assert np.isclose(arms[RR.FLOOR_ARM][0], w * 10.0 + (1 - w) * 4.0)
    assert arms[RR.INCUMBENT][0] == 4.0


def test_a_rookie_with_no_preseason_gets_his_bucket_prior_exactly():
    """By his own volume, not by a special case: `min_pre = 0` makes the weight 0 and the
    three arms agree on that row by construction."""
    rows, priors = _floor_frame()
    arms = RR.floor_arms(rows, priors, "pre_per36_reb", 160.0)
    assert arms[RR.FLOOR_ARM][1] == 4.0
    assert arms[RR.INCUMBENT][1] == 4.0
    assert arms["preseason"][1] == 4.0


def test_the_expanding_bucket_prior_never_reads_its_own_season():
    """Point-in-time by construction — season S's prior averages rookies from seasons
    strictly before S, so it is knowable in September. The first covered season gets no
    prior at all and is dropped rather than handed a constant that came from nowhere."""
    rows = pd.DataFrame({"season": ["2019-20"] * 4 + ["2020-21"] * 4,
                         "draft_bucket": ["lottery"] * 8,
                         "reb_p36": [2.0, 2.0, 2.0, 2.0, 9.0, 9.0, 9.0, 9.0]})
    covered = ["2019-20", "2020-21"]
    priors = RR.bucket_prior_table(rows, "reb_p36", covered)
    assert set(priors["season"]) == {"2020-21"}
    assert float(priors["prior"].iloc[0]) == 2.0
    assert list(RR.scorable_rows(rows, priors)) == [False] * 4 + [True] * 4


# ── The feature ladder ────────────────────────────────────────────────────────

def test_every_head_gets_the_same_four_blocks_and_its_own_level():
    """Eleven heads, one feature list shape. The level is per head because it is on that
    head's own link — a shared block would put `reb`'s preseason rebounding on `blk`'s
    linear predictor."""
    lists = {label: RR.head_features(component)
             for label, component, _ in RR.head_list()}
    assert len(lists) == 11
    shared = set.intersection(*(set(v) for v in lists.values()))
    assert shared == set(MISSING_AGE_COLS + RR.SLOT_COLS + RR.SLOT_INTERACTION_COLS
                         + [RR.YEARS_SINCE_DRAFT] + RR.BIO_COLS)
    for label, component, _ in RR.head_list():
        assert lists[label][0] == RR.shrunk_column(component)
        assert len(lists[label]) == len(shared) + 1
