"""The lag-recovery ladder — `component_rates`'s widened design and its §4 gate.

`docs/rookie-rates-plan.md` §5b widens the veteran design from *is the immediately-prior
season big enough* to *is any prior season constructible*, by imputing an unusable lag-1
block from the nearest usable one. The claim that lets it happen without refitting eleven
heads is narrow and mechanical — **rung 0 comes back bit-identical** — so that is what
these tests pin, alongside the rung boundaries and the 200-minute discontinuity §5b names
rather than solves.

Everything is deterministic: each synthetic player-season has an exact per-36 rate, so a
shrunk rate can be checked against `w * r + (1 - w) * mu` by arithmetic instead of by
tolerance.
"""

import numpy as np
import pandas as pd

from src.models.component_rates import (CONVERSION_HEADS, COUNT_HEADS, LADDER_RUNGS,
                                        LagLadder, MIN_PRIOR_MINUTES, build_design,
                                        fitting_rows, ladder_recovered, rate_columns)

SEASONS = ["2017-18", "2018-19", "2019-20", "2020-21", "2021-22", "2022-23"]

# One exact per-36 rate per column, so every expected value below is arithmetic.
RATES = {"fga": 20.0, "fta": 5.0, "reb": 8.0, "ast": 4.0, "stl": 1.5, "blk": 1.0,
         "tov": 2.5}
FG3A_SHARE, FT_PCT, FG2_PCT, FG3_PCT = 0.4, 0.8, 0.5, 0.35


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


def _targets(career: dict[int, dict[str, float]]) -> pd.DataFrame:
    """`{player_id: {season: total_minutes}}` -> the component-target frame."""
    rows = []
    for player, seasons in career.items():
        for season, minutes in seasons.items():
            rows += _season_rows(player, season, minutes)
    return pd.DataFrame(rows)


def _bios(tmp_path, players):
    from src.data.fetch import _slug, nbastats_dir
    dest = nbastats_dir(tmp_path)
    dest.mkdir(parents=True, exist_ok=True)
    for season in SEASONS:
        pd.DataFrame({"PLAYER_ID": list(players),
                      "AGE": [25 for _ in players]}).to_csv(
            dest / f"player_bio_stats_{_slug(season)}.csv", index=False)
    return tmp_path


def _ladder(rungs=LADDER_RUNGS, k: float = 200.0, mu: float = 1.0) -> LagLadder:
    """A ladder whose constants are chosen for arithmetic, not for realism."""
    return LagLadder(rungs=tuple(rungs),
                     shrinkage={c: k for c in rate_columns()},
                     means={c: mu for c in rate_columns()},
                     conversion={m: (10.0, 0.5) for m, _ in CONVERSION_HEADS})


# The four rungs, one player each, plus a veteran and a true rookie. The target season is
# the last one in every case.
CAREERS = {
    1: {"2017-18": 1500.0, "2018-19": 1500.0, "2019-20": 1500.0, "2020-21": 1500.0,
        "2021-22": 1500.0, "2022-23": 1500.0},                      # veteran
    2: {"2021-22": 100.0, "2022-23": 1500.0},                       # thin lag-1
    3: {"2020-21": 1500.0, "2022-23": 1500.0},                      # returnee, full lag-2
    4: {"2020-21": 100.0, "2022-23": 1500.0},                       # returnee, thin lag-2
    5: {"2019-20": 1500.0, "2022-23": 1500.0},                      # away two seasons
    6: {"2022-23": 1500.0},                                         # true rookie
}
RUNG_OF = {1: "veteran", 2: "thin_prior", 3: "returnee_lag2", 4: "returnee_thin",
           5: "no_usable_lag"}


def _designs(tmp_path, rungs=LADDER_RUNGS, **kw):
    targets = _targets(CAREERS)
    raw = _bios(tmp_path, CAREERS)
    plain = build_design(targets, SEASONS, raw)
    wide = build_design(targets, SEASONS, raw, ladder=_ladder(rungs, **kw))
    return plain, wide


# ── Rung 0 does not move ──────────────────────────────────────────────────────

def test_veteran_rows_come_back_bit_identical(tmp_path):
    """The one claim §3 constraint 4 rests on.

    The recovered rows are scored by coefficients fitted before the ladder existed, which
    is only legitimate if the rows those coefficients were fitted on are unchanged — not
    approximately, and not up to a tolerance. Every shared column is compared exactly,
    NaN for NaN.
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
    """`None` is the pre-ladder builder exactly — twelve modules import it and none of
    them asked for a wider frame or a `lag_rung` column they would have to ignore."""
    targets = _targets(CAREERS)
    raw = _bios(tmp_path, CAREERS)
    plain = build_design(targets, SEASONS, raw)
    assert not any(c.endswith(("_lag2", "_lag3")) for c in plain.columns)
    assert "lag_rung" not in plain.columns
    assert not ladder_recovered(plain).any()


def test_the_fitting_population_is_rung_zero_alone(tmp_path):
    """`fitting_rows` is §3 constraint 4 as code: the ladder widens SCORING only."""
    plain, wide = _designs(tmp_path)
    assert len(fitting_rows(wide)) == len(plain)
    assert len(wide) > len(plain)


# ── The rung boundaries ───────────────────────────────────────────────────────

def test_each_rung_admits_the_population_it_names(tmp_path):
    """One player per rung, labelled by why the shipped design has no row for him."""
    _, wide = _designs(tmp_path)
    last = wide[wide["season"] == SEASONS[-1]].set_index("player_id")
    assert {p: last.loc[p, "lag_rung"] for p in RUNG_OF} == RUNG_OF


def test_a_true_rookie_is_never_admitted_at_any_rung(tmp_path):
    """The boundary §3 constraint 2′ drew: no own-rate feature exists at any lag, so the
    ladder must leave him to the rookie heads. Disjointness is the property §5c's design
    depends on."""
    _, wide = _designs(tmp_path)
    assert 6 not in set(wide["player_id"])
    assert "true_rookie" not in set(wide["lag_rung"])


def test_only_the_configured_rungs_enter(tmp_path):
    """The gate came back per rung, so the config key is a list and it has to bind."""
    _, one = _designs(tmp_path, rungs=("returnee_lag2",))
    last = one[one["season"] == SEASONS[-1]]
    assert set(last["lag_rung"]) == {"veteran", "returnee_lag2"}
    assert set(last["player_id"]) == {1, 3}


# ── Provenance ────────────────────────────────────────────────────────────────

def test_every_row_says_where_its_block_came_from(tmp_path):
    """A scored unit whose provenance is not recorded cannot be audited on a board."""
    _, wide = _designs(tmp_path)
    last = wide[wide["season"] == SEASONS[-1]].set_index("player_id")
    assert last.loc[1, "lag_source"] == "lag1"        # veteran
    assert last.loc[2, "lag_source"] == "lag1"        # thin, but present
    assert last.loc[3, "lag_source"] == "lag2"        # returnee
    assert last.loc[5, "lag_source"] == "lag3"        # away two seasons
    # `lag_minutes` is the row-level half of the reliability weight; the per-head `k` is
    # the constant half and lives in `lag_recovery.csv`.
    assert last.loc[2, "lag_minutes"] == 100.0
    assert last.loc[3, "lag_minutes"] == 1500.0
    assert last.loc[1, "lag_minutes"] == 1500.0


def test_the_carried_block_is_the_source_season_not_the_missing_one(tmp_path):
    """A returnee's context columns are what his most recent season actually was — they
    are not estimates of a per-minute quantity, so they are carried raw."""
    _, wide = _designs(tmp_path)
    last = wide[wide["season"] == SEASONS[-1]].set_index("player_id")
    assert last.loc[3, "total_minutes_lag1"] == 1500.0
    assert last.loc[3, "gp_lag1"] == 10.0
    assert np.isclose(last.loc[3, "mpg_lag1"], 150.0)


# ── The shrink, and the discontinuity it creates ──────────────────────────────

def test_a_recovered_rate_is_the_reliability_weighted_blend(tmp_path):
    """`w * r + (1 - w) * mu` at `w = m / (m + k)`, arithmetic and not a tolerance."""
    k, mu = 200.0, 1.0
    _, wide = _designs(tmp_path, k=k, mu=mu)
    last = wide[wide["season"] == SEASONS[-1]].set_index("player_id")
    for player, minutes in ((2, 100.0), (3, 1500.0), (4, 100.0), (5, 1500.0)):
        w = minutes / (minutes + k)
        for c in COUNT_HEADS:
            assert np.isclose(last.loc[player, f"{c}_p36_lag1"],
                              w * RATES[c] + (1 - w) * mu), (player, c)


def test_the_threshold_discontinuity_is_real_and_sits_at_min_prior_minutes(tmp_path):
    """§5b names this rather than solving it, so it is pinned rather than smoothed.

    Two players identical but for one prior minute either side of the qualification
    threshold: the one above keeps a raw rate, the one below gets a shrunk one, and the
    jump is the whole weight gap `1 - m / (m + k)`. Removing it means shrinking everyone
    continuously, which moves the FITTING population and costs eleven refits — the
    escalation §3 constraint 4 holds in reserve.
    """
    over, under = float(MIN_PRIOR_MINUTES) + 1.0, float(MIN_PRIOR_MINUTES) - 1.0
    careers = {10: {"2021-22": over, "2022-23": 1500.0},
               11: {"2021-22": under, "2022-23": 1500.0}}
    targets = _targets(careers)
    k, mu = 200.0, 1.0
    wide = build_design(targets, SEASONS, _bios(tmp_path, careers),
                        ladder=_ladder(k=k, mu=mu))
    last = wide[wide["season"] == SEASONS[-1]].set_index("player_id")

    assert last.loc[10, "lag_rung"] == "veteran"
    assert last.loc[11, "lag_rung"] == "thin_prior"
    assert np.isclose(last.loc[10, "reb_p36_lag1"], RATES["reb"])       # raw
    w = under / (under + k)
    assert np.isclose(last.loc[11, "reb_p36_lag1"], w * RATES["reb"] + (1 - w) * mu)
    # One prior minute apart, and the rates differ by the whole weight gap.
    jump = abs(last.loc[10, "reb_p36_lag1"] - last.loc[11, "reb_p36_lag1"])
    assert np.isclose(jump, (1 - w) * abs(RATES["reb"] - mu))
    assert jump > 0.1 * RATES["reb"]


def test_the_shrink_reaches_both_ends_of_its_own_range(tmp_path):
    """`k = 0` is the raw carried rate and a huge `k` is the population mean. The arm has
    to be able to express 'do not shrink', because raw thin lag-1 is an anti-model and the
    fitted `k` is what separates the two."""
    careers = {20: {"2021-22": 100.0, "2022-23": 1500.0}}
    targets, raw = _targets(careers), _bios(tmp_path, careers)
    for k, expected in ((0.0, RATES["reb"]), (1e9, 1.0)):
        wide = build_design(targets, SEASONS, raw, ladder=_ladder(k=k, mu=1.0))
        last = wide[wide["season"] == SEASONS[-1]].set_index("player_id")
        assert np.isclose(last.loc[20, "reb_p36_lag1"], expected, atol=1e-5)


def test_the_conversion_block_is_shrunk_on_attempts_not_on_minutes(tmp_path):
    """A percentage's reliability lives in its attempts — the form
    `carry_forward_conversion` settled, pointed at the nearest usable season."""
    careers = {30: {"2020-21": 1500.0, "2022-23": 1500.0}}
    k_att, league = 10.0, 0.5
    wide = build_design(_targets(careers), SEASONS, _bios(tmp_path, careers),
                        ladder=LagLadder(rungs=LADDER_RUNGS,
                                         shrinkage={c: 200.0 for c in rate_columns()},
                                         means={c: 1.0 for c in rate_columns()},
                                         conversion={m: (k_att, league)
                                                     for m, _ in CONVERSION_HEADS}))
    last = wide[wide["season"] == SEASONS[-1]].set_index("player_id")
    attempts = RATES["fta"] * 1500.0 / 36.0
    made = attempts * FT_PCT
    assert np.isclose(last.loc[30, "ftm_pct_lag1"],
                      (made + k_att * league) / (attempts + k_att))
    # 208 attempts against 10 pseudo-attempts pulls 0.800 to 0.786 — the device working
    # rather than the device being absent, and it moves toward the league mean.
    assert league < last.loc[30, "ftm_pct_lag1"] < FT_PCT
    assert abs(last.loc[30, "ftm_pct_lag1"] - FT_PCT) < 0.02


# ── The gate's status quo ─────────────────────────────────────────────────────

def test_the_unserved_status_quo_is_a_point_mass_at_zero():
    """What happens to one of these players today: he is not in the design, so he is not
    in the tensor, so every draw scores him at zero. CRPS against a point mass at 0 is
    `|y - 0|`, and gate 1 is read against exactly that — which is why §4 pairs it with a
    band gate rather than reading it alone."""
    from src.models.lag_ladder import head_scores
    from src.models.stan_utils import crps_from_samples

    y = np.array([3.0, 17.0, 0.0])
    point_mass = np.zeros((64, len(y)))
    assert np.allclose(crps_from_samples(point_mass, y), np.abs(y))
    assert head_scores.__doc__ is not None


# ── Turning the ladder on — docs/rookie-rates-plan.md §5f's first act ─────────

def test_the_shipped_config_admits_the_one_rung_the_gate_cleared():
    """§7c's verdict, as the key the design actually reads.

    Pinned because the ladder is a config list and a silent edit widens the SCORING
    population of eleven heads at once — the four rungs were gated separately and three of
    them failed, so `[returnee_lag2]` is a result rather than a default.
    """
    import yaml

    cfg = yaml.safe_load(open("configs/default.yaml"))
    assert cfg["stan"]["components"]["lag_ladder"] == ["returnee_lag2"]


def test_every_fitting_path_takes_its_training_frame_through_fitting_rows():
    """§3 constraint 4, pinned by parsing rather than by discipline.

    `stan_components.head_design` carries the ladder, so any module that builds a training
    frame from it and hands that frame to a sampler is fitting on the recovered rows —
    which the imputation-only form exists to prevent, and which nothing would raise about.
    The three paths are named individually so a fourth one added later fails here.

    `season_terms` and `components_preseason` are deliberately absent: they call
    `component_rates.build_design` directly, which defaults to the pre-ladder design.
    """
    import pathlib

    for module, function in (("src/models/stan_components.py", "def run("),
                             ("src/models/posteriors.py", "def component_artifacts("),
                             ("src/models/model_cards.py", "def component_frames(")):
        source = pathlib.Path(module).read_text()
        start = source.index(function)
        end = source.index("\ndef ", start + 1)
        body = source[start:end]
        assert "fitting_rows(" in body, (
            f"{module}::{function.strip('def (')} builds a training frame from "
            f"`head_design` and does not restrict it to rung 0")

    import ast

    for module in ("src/models/season_terms.py", "src/models/components_preseason.py"):
        tree = ast.parse(pathlib.Path(module).read_text())
        reached = {alias.asname or alias.name
                   for node in ast.walk(tree)
                   if isinstance(node, ast.ImportFrom)
                   and node.module == "src.models.stan_components"
                   for alias in node.names}
        assert "head_design" not in reached, (
            f"{module} now reaches the ladder-carrying design; it has to route its "
            f"training frame through `component_rates.fitting_rows` too")


def test_no_head_fits_a_column_the_ladder_can_move():
    """The bit-identity claim rests on WHICH columns the heads read.

    `components_preseason.attach_preseason` writes a season-CENTRED twin of every
    preseason delta, and a season mean is a property of the frame — so those eleven
    columns do move when 189 recovered rows join the design. None of them is in any head's
    feature list: the shipped block is the volume-shrunk delta, which is a per-row product
    and cannot move. If a `_centered` arm is ever shipped, this fails, and the fix is to
    compute the centring on rung 0 rather than to delete the test.
    """
    from src.models.stan_components import head_preseason_cols

    for head in list(COUNT_HEADS) + [m for m, _ in CONVERSION_HEADS]:
        assert not [c for c in head_preseason_cols(head, True) if c.endswith("_centered")]
