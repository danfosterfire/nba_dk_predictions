import numpy as np
import pandas as pd
import pytest

from src.eda.preseason_value import (
    AVAIL_BLOCK,
    CENSUS_DIMENSIONS,
    attach_availability_block,
    attach_prior_shares,
    covered_seasons,
    log_rate,
    logit,
    missingness_census,
    shuffle_within_season,
    team_minutes_shares,
    training_frames,
)

SEASONS = [f"{y}-{str(y + 1)[-2:]}" for y in range(2000, 2010)]
HOME, AWAY = 1610612737, 1610612738


# ── Synthetic builders ────────────────────────────────────────────────────────

def _panel_rows(season: str, team: int, player: int, games: int,
                minutes: float, day0: int = 1) -> list[dict]:
    """Availability-panel rows: one per team game, `played` on all of them."""
    return [{"season": season, "player_id": player, "team_id": team,
             "game_id": team * 100 + g, "game_date": pd.Timestamp(f"2001-01-{day0 + g:02d}"),
             "min": minutes, "played": 1} for g in range(games)]


def _design(n: int, season: str = "2005-06") -> pd.DataFrame:
    """A design frame with the columns the block's deltas are taken against."""
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "season": season, "player_id": np.arange(n),
        "min_share_lag1": rng.uniform(0.02, 0.15, n),
        "minutes_per_game_lag1": rng.uniform(8.0, 34.0, n),
        "gp_share_lag1": rng.uniform(0.3, 1.0, n),
        "age": rng.uniform(20.0, 36.0, n),
        "gp_share": rng.uniform(0.2, 1.0, n),
        "minutes_per_game": rng.uniform(6.0, 36.0, n),
        "on_season_start_roster": 1.0,
    })


def _preseason(design: pd.DataFrame, players: list[int],
               jitter: bool = False) -> pd.DataFrame:
    """Panel rows for the players named.

    Without `jitter` they AGREE with the prior season exactly, which is what the nesting
    property is asserted on. With it, every column varies — needed wherever a full-rank
    block matters, because a degenerate design makes a least-squares comparison a statement
    about `lstsq`'s minimum-norm tie-break rather than about the columns.
    """
    sub = design[design["player_id"].isin(players)]
    n = len(sub)
    rng = np.random.default_rng(7)
    scale = rng.uniform(0.7, 1.4, n) if jitter else np.ones(n)
    tail = rng.integers(0, 3, n) if jitter else np.zeros(n, dtype=int)
    return pd.DataFrame({
        "season": sub["season"].to_numpy(), "player_id": sub["player_id"].to_numpy(),
        "min_pre": 60.0 + 80.0 * (scale - 0.7),
        "mpg_pre": sub["minutes_per_game_lag1"].to_numpy() * scale,
        "gp_share_pre": np.clip(scale - 0.2, 0.1, 1.0),
        "min_share_pre": sub["min_share_lag1"].to_numpy(),
        "min_share_pre_late": sub["min_share_lag1"].to_numpy() * scale,
        "min_rank_pre": 5.0, "missed_tail": tail,
        "played_final_game": (tail == 0).astype(int), "team_pre_games": 4,
        "pre_fg3a_share": 0.35, "fga_pre": 40.0, "fgm_pre": 18.0,
        "fg3a_pre": 14.0, "fg3m_pre": 5.0, "fta_pre": 10.0, "ftm_pre": 8.0,
        **{f"pre_per36_{c}": 5.0 for c in
           ("fga", "fta", "reb", "ast", "stl", "blk", "tov")},
    })


def _ols_r2(fit: pd.DataFrame, score: pd.DataFrame, cols: list[str],
            target: str) -> float:
    """Unpenalized least squares — the fit the invariance claim is exact for."""
    def design(f):
        return np.column_stack([np.ones(len(f)),
                                np.nan_to_num(f[cols].to_numpy(dtype=float))])
    beta, *_ = np.linalg.lstsq(design(fit), fit[target].to_numpy(dtype=float), rcond=None)
    y = score[target].to_numpy(dtype=float)
    resid = y - design(score) @ beta
    return float(1 - (resid ** 2).sum() / ((y - y.mean()) ** 2).sum())


# ── Link scales ───────────────────────────────────────────────────────────────

def test_a_zero_share_is_clipped_not_dropped():
    """Zero IS the observation — he played none of his team's late preseason games. A NaN
    would delete exactly the rows the participation signal lives on."""
    got = logit([0.0, 0.5, 1.0])
    assert np.isfinite(got).all()
    assert got[0] < got[1] < got[2]
    assert got[1] == pytest.approx(0.0)


def test_log_rate_is_defined_at_zero():
    assert log_rate([0.0])[0] == pytest.approx(0.0)
    assert log_rate([np.e - 1])[0] == pytest.approx(1.0)


# ── Difference coding ─────────────────────────────────────────────────────────

def test_a_preseason_that_agrees_with_the_prior_season_carries_a_zero_delta():
    """The nesting property: zero means "no new information", so a coefficient path
    through zero recovers the shipped head exactly."""
    design = _design(6)
    out = attach_availability_block(design, _preseason(design, list(range(6))))

    assert out["pre_d_min_share_late"].abs().max() == pytest.approx(0.0, abs=1e-12)
    assert out["pre_d_mpg"].abs().max() == pytest.approx(0.0, abs=1e-12)
    assert (out["has_preseason"] == 1.0).all()


def test_a_player_with_no_preseason_row_carries_zero_deltas_and_the_indicator():
    design = _design(6)
    out = attach_availability_block(design, _preseason(design, [0, 1, 2]))

    missing = out[out["player_id"] >= 3]
    assert (missing["has_preseason"] == 0.0).all()
    assert missing[AVAIL_BLOCK].drop(columns=["has_preseason"]).abs().to_numpy().max() == 0.0
    assert out[AVAIL_BLOCK].notna().all().all(), "the block must never reach a fit as NaN"


def test_the_fill_constant_does_not_change_an_unpenalized_fit():
    """The load-bearing half of the zero-fill convention. A column filled at constant `c`
    on the missing rows is `x*1{present} + c*(1 - has_preseason)`, so with the indicator in
    the block any two constants span the same column space. Exact for least squares; under
    an L2 penalty it holds only up to the penalty's basis, which is why the module says so
    rather than claiming invariance outright."""
    design = _design(120)
    # Both halves need present AND missing rows. With every present row in the fit half,
    # `has_preseason` is constant there, the fill never enters the fit at all, and the
    # comparison degenerates into "does changing the score rows change the score".
    covered = [p for p in range(120) if p % 3]
    zero_filled = attach_availability_block(
        design, _preseason(design, covered, jitter=True))

    fit, score = zero_filled.iloc[:90], zero_filled.iloc[90:]
    for half in (fit, score):
        assert 0.0 < half["has_preseason"].mean() < 1.0

    # The claim is about a column SPACE, so it says nothing when that space is degenerate:
    # `lstsq` would then be choosing between tied solutions and the two fills could break
    # the tie differently. Assert full rank on the FIT design rather than discover it as a
    # flaky test.
    block = np.column_stack([np.ones(len(fit)), fit[AVAIL_BLOCK].to_numpy(dtype=float)])
    assert np.linalg.matrix_rank(block) == block.shape[1]

    other = zero_filled.copy()
    missing = other["has_preseason"] == 0.0
    for col in ("pre_d_mpg", "pre_gp_share", "pre_log_min"):
        other.loc[missing, col] = -3.5

    fit2, score2 = other.iloc[:90], other.iloc[90:]
    assert _ols_r2(fit, score, AVAIL_BLOCK, "gp_share") == pytest.approx(
        _ols_r2(fit2, score2, AVAIL_BLOCK, "gp_share"), abs=1e-8)


# ── The prior-season within-team share ────────────────────────────────────────

def test_prior_minutes_shares_sum_to_one_within_a_team():
    """The same arithmetic check the preseason panel makes on its own shares — the two
    sides of the delta have to be the same quantity or the difference is a definition."""
    panel = pd.DataFrame(_panel_rows("2005-06", HOME, 1, 4, 30.0)
                         + _panel_rows("2005-06", HOME, 2, 4, 10.0)
                         + _panel_rows("2005-06", AWAY, 3, 4, 20.0))
    shares = team_minutes_shares(panel)
    assert shares["min_share"].sum() == pytest.approx(2.0), "one per team"
    assert shares.set_index("player_id").loc[1, "min_share"] == pytest.approx(0.75)
    assert shares.set_index("player_id").loc[1, "min_rank"] == 1.0


def test_a_traded_players_share_reads_his_last_team():
    """The project convention, and the one `preseason.py` uses — so a delta across a trade
    compares his new role with his old one rather than with an average of both."""
    panel = pd.DataFrame(_panel_rows("2005-06", HOME, 1, 2, 30.0, day0=1)
                         + _panel_rows("2005-06", HOME, 9, 2, 10.0, day0=1)
                         + _panel_rows("2005-06", AWAY, 1, 2, 20.0, day0=10)
                         + _panel_rows("2005-06", AWAY, 8, 2, 20.0, day0=10))
    shares = team_minutes_shares(panel).set_index("player_id")
    assert shares.loc[1, "min_share"] == pytest.approx(0.5), "his AWAY share, not HOME's"


def test_prior_shares_attach_as_lag_1():
    """The S-1 share lands on the S row. `with_lags` pairs on the season *index*, so the
    player needs a row in S as well — which every design row has, since the design and the
    shares are built from the same panel."""
    panel = pd.DataFrame(_panel_rows("2004-05", HOME, 1, 4, 30.0)
                         + _panel_rows("2004-05", HOME, 2, 4, 10.0)
                         + _panel_rows("2005-06", HOME, 1, 4, 12.0)
                         + _panel_rows("2005-06", HOME, 2, 4, 28.0))
    design = pd.DataFrame({"season": ["2005-06"], "player_id": [1]})
    out = attach_prior_shares(design, panel, SEASONS)
    assert out.loc[0, "min_share_lag1"] == pytest.approx(0.75)
    assert out.loc[0, "min_rank_lag1"] == 1.0


# ── The null ──────────────────────────────────────────────────────────────────

def test_the_shuffle_stays_inside_its_own_season():
    """Pooled, a shuffled 2011-12 lockout row could land on a 2019-20 player and the null
    would be measuring the calendar instead of the pairing."""
    frame = pd.DataFrame({"season": ["A"] * 4 + ["B"] * 4,
                          "x": [1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0]})
    out = shuffle_within_season(frame, ["x"], np.random.default_rng(1))

    assert sorted(out.loc[out["season"] == "A", "x"]) == [1.0, 2.0, 3.0, 4.0]
    assert sorted(out.loc[out["season"] == "B", "x"]) == [10.0, 20.0, 30.0, 40.0]


def test_the_shuffle_moves_the_block_and_nothing_else():
    frame = pd.DataFrame({"season": ["A"] * 6, "x": np.arange(6.0),
                          "y": np.arange(6.0)})
    out = shuffle_within_season(frame, ["x"], np.random.default_rng(3))
    assert list(out["y"]) == list(range(6)), "only the block is permuted"
    assert sorted(out["x"]) == list(range(6))


# ── Scope ─────────────────────────────────────────────────────────────────────

def test_only_seasons_with_an_intact_tail_are_in_scope():
    """2003-04's tail is the part `missed_tail` and the late-weighted shares are read
    over, so its block is a different feature rather than a noisier one."""
    coverage = pd.DataFrame({
        "season": ["2002-03", "2003-04", "2004-05"],
        "rows_kept": [0, 369, 1154],
        "coverage_class": ["absent", "tail_missing", "covered"]})
    assert covered_seasons(coverage) == ["2004-05"]


def test_training_frames_never_reach_validation_or_test():
    """P1 decides which heads get built. A screen that spends the selection split leaves
    P2 nothing to select on — so the guard is that neither inner frame contains a season
    `selection_split` would have handed to validation."""
    frame = pd.concat([_design(30, season=s).assign(team_games=82, gp=60)
                       for s in SEASONS], ignore_index=True)
    fit, score = training_frames(frame, SEASONS, test_seasons=2, inner=2)

    seen = set(fit["season"]) | set(score["season"])
    assert seen == set(SEASONS[:-4]), "the last 4 seasons are validation and test"
    assert set(score["season"]) == set(SEASONS[-6:-4])
    assert not set(fit["season"]) & set(score["season"])


def test_a_scope_shorter_than_the_inner_split_raises():
    """Two covered seasons would leave the fit frame empty — a silent zero-row fit is the
    failure mode a coverage restriction can produce and nothing else would object to."""
    frame = pd.concat([_design(10, season=s) for s in SEASONS], ignore_index=True)
    with pytest.raises(ValueError):
        training_frames(frame, SEASONS[:2], test_seasons=2, inner=2)


# ── The census ────────────────────────────────────────────────────────────────

def test_the_census_gap_is_absent_minus_present():
    """The sign has to point at the population the block exists for: a NEGATIVE gap means
    the players with no preseason row went on to play less."""
    design = _design(40)
    frame = attach_availability_block(design, _preseason(design, list(range(20))))
    frame.loc[frame["has_preseason"] == 0.0, "gp_share"] = 0.3
    frame.loc[frame["has_preseason"] == 1.0, "gp_share"] = 0.8

    rows = pd.DataFrame(missingness_census(frame, dimensions=("all",)))
    got = rows.set_index("metric")["value"]
    assert got["share_missing"] == pytest.approx(0.5)
    assert got["gap_gp_share"] == pytest.approx(-0.5)


def test_the_census_covers_every_declared_dimension():
    design = _design(60)
    frame = attach_availability_block(design, _preseason(design, list(range(40))))
    rows = pd.DataFrame(missingness_census(frame))
    assert set(rows["target"]) == set(CENSUS_DIMENSIONS)
    # Every cell of a cut has to account for every row of the frame.
    per_dim = rows[rows["metric"] == "n"].groupby("target")["value"].sum()
    assert (per_dim == len(frame)).all()
