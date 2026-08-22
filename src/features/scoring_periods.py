"""Scoring periods — the NBA week grid, and DK's tournament rounds on top of it.

A best-ball lineup is scored **weekly**: each scoring period the best 7 players by slot
count and the rest sit. So every downstream object in the simulation layer — the weekly
max, the round total, the advancement cut — is an aggregate over a *period*, and nothing
in the repo said what a period is. This module says it once.

**DK's scoring periods are NBA weeks.** `ScheduleLeagueV2` carries `weekNumber` and
`weekName` natively, the grid runs Monday-Sunday, and DK's published round windows are
ranges of it (`docs/simulations-plan.md`, "Scoring periods are NBA weeks"). The rules
define a period as "the date of the first game in the game set through the date of the
last game" (`docs/dk_best_ball_rules.md`), which is that grid restricted to dates that
have games.

### The week index is derived, and the derivation is validated against the NBA's own

`weekNumber` is populated from **2017-18** onward and is identically zero before it, so
the eight older seasons need a derivation. The rule is: anchor each game date to its
Monday, then **dense-rank the Mondays that actually carry games**. Checked against the
NBA's own numbering on all nine seasons that have it (2017-18 -> 2025-26), it agrees on
**every game, in every season** — 10,749 games, zero disagreements.

Dense-ranking rather than counting elapsed calendar weeks is the whole trick, and
2019-20 is why. The NBA suspended for four months and then resumed in the bubble, and
its own numbering treats the resumed games as weeks 22-24 — *consecutive with what came
before*, not 41-43. A `(date - first_date) // 7` index reproduces the other eight
seasons and breaks that one. Ranking occupied weeks reproduces all nine, because in a
normal season no week between the first and last game is empty, so the two definitions
coincide exactly where they can and differ only where the NBA itself skips.

### Three edge cases, owned here once rather than per use

- **A postponed or suspended game scores in the period it is PLAYED in**, not the one it
  was scheduled for (`dk_best_ball_rules.md`, "Rescheduled Games"). So the period is
  assigned from the realized `game_date` in the game logs, never from a scheduled date.
  This is currently a distinction without a difference and is implemented anyway:
  `ScheduleLeagueV2` serves the *realized* schedule, so scheduled and played dates agree
  on 7,380 of 7,380 checked games. It stops being free the moment the production board
  reads a *forward* schedule, where every date is still a plan — which is exactly when a
  silently wrong period would be most expensive. `PERIOD_DATE_COLUMN` names the column
  that decides, and `schedule_agreement` reports the disagreement rate per season.
- **The NBA Cup championship game does not score at all** (`dk_best_ball_rules.md`).
  It is identified from the schedule by its label, and it is *doubly* excluded: the NBA
  does not count it as a regular-season game either, so it carries game-id prefix `006`
  rather than `002` and never enters `game_logs.parquet` in the first place. Both facts
  are asserted rather than assumed, because the label is the only one of the two that
  will still be true if the NBA changes its mind about the box score.
- **The all-star gap breaks week adjacency.** In 2025-26 week 17's games stop on 2/12
  and week 18's start on 2/19. A grid derived from *game* spacing would fuse or shift
  those; a Monday-anchored calendar grid does not notice, because the break moves no
  Monday. The gap is reported (`break_gap_days`) rather than corrected.

### What is asserted and what is reported

Asserted, because a violation makes every downstream weekly aggregate wrong: every game
date lands in **exactly one** period, periods are contiguous Monday-anchored weeks, and
no game is assigned twice.

Reported, because it is a property of DK's published calendar rather than of the data:
how the derived round boundaries line up against the round windows in the rules copy.
Those windows (Round 1 closing 2/14, the Cup final on 12/11/2026) belong to a **single
season's** rules and cannot be anchored onto a backtest season without sliding — 2025-26's
week 17 closes 2/12, not 2/14. So the round map is specified as DK's *structure* in weeks
— `sim.scoring_periods.round_weeks`, 17 then three double weeks — which is the part that
is stable across seasons, and the calendar comparison is printed so drift stays visible.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.data.fetch import _slug, nbastats_dir

# The column the period is assigned from. Named rather than inlined because the whole
# "played, not scheduled" rule is this one choice, and a future forward-schedule consumer
# has to be able to see that it is the realized date it must supply.
PERIOD_DATE_COLUMN = "game_date"

# ScheduleLeagueV2 returns weekNumber == 0 for every game before this season.
SCHEDULE_API_FIRST_SEASON = "2017-18"

# NBA game-id prefixes. The Cup final is `006`; regular-season games are `002`.
REGULAR_SEASON_PREFIX = "002"
CUP_FINAL_PREFIX = "006"

# DK's published round-end boundaries as (month, day), from the rules copy. Used only for
# the printed calendar cross-check — never to assign a round. See the module docstring.
DK_PUBLISHED_ROUND_ENDS = ((2, 14), (3, 7), (3, 21), (4, 4))

OUTSIDE_CONTEST = 0


# ── The week grid ─────────────────────────────────────────────────────────────

def derive_week_index(dates: pd.Series) -> pd.Series:
    """Week index from realized game dates: dense rank over Mondays that carry games.

    Reproduces the NBA's own `weekNumber` on all nine seasons that publish one. See the
    module docstring for why this is a dense rank and not elapsed calendar weeks.
    """
    dates = pd.to_datetime(dates).dt.normalize()
    monday = dates - pd.to_timedelta(dates.dt.weekday, unit="D")
    return monday.rank(method="dense").astype(int)


def week_bounds(dates: pd.Series) -> tuple[pd.Series, pd.Series]:
    """The Monday and Sunday bracketing each date's calendar week."""
    dates = pd.to_datetime(dates).dt.normalize()
    monday = dates - pd.to_timedelta(dates.dt.weekday, unit="D")
    return monday, monday + pd.Timedelta(days=6)


# ── The schedule, cached ──────────────────────────────────────────────────────

def schedule_path(season: str, raw_dir: str | Path) -> Path:
    return nbastats_dir(raw_dir) / f"schedule_{_slug(season)}.csv"


def fetch_schedule(season: str, raw_dir: str | Path,
                   refresh: bool = False) -> pd.DataFrame:
    """ScheduleLeagueV2 for one season, flattened and cached to `data/raw`.

    Cached because this module is a cheap derivation that should not depend on a live
    endpoint to rebuild, and because the grid for a completed season never changes.
    """
    dest = schedule_path(season, raw_dir)
    if dest.exists() and not refresh:
        return pd.read_csv(dest, dtype={"game_id_raw": str})

    from nba_api.stats.endpoints import ScheduleLeagueV2

    payload = ScheduleLeagueV2(season=season, timeout=60).get_dict()
    game_dates = payload["leagueSchedule"]["gameDates"]
    rows = [
        {
            "game_id_raw": g["gameId"],
            "schedule_date": g["gameDateEst"],
            "week_number": g["weekNumber"],
            "week_name": g["weekName"],
            "game_label": g["gameLabel"],
            "game_sub_label": g["gameSubLabel"],
            "postponed_status": g["postponedStatus"],
        }
        for d in game_dates for g in d["games"]
    ]
    sched = pd.DataFrame(rows)
    sched["schedule_date"] = (pd.to_datetime(sched["schedule_date"], utc=True)
                              .dt.tz_localize(None).dt.normalize())
    dest.parent.mkdir(parents=True, exist_ok=True)
    sched.to_csv(dest, index=False)
    print(f"  fetched {len(sched):,} scheduled games → {dest}")
    return sched


def cup_final_mask(sched: pd.DataFrame) -> pd.Series:
    """The NBA Cup championship game, which DK does not score.

    Identified by label rather than by game id: the id prefix is the NBA's statement
    that the game is not a regular-season game, which is a separate claim that could
    change without DK's rule changing.
    """
    label = sched["game_label"].fillna("").str.contains("Cup", case=False)
    sub = sched["game_sub_label"].fillna("").str.strip().str.lower() == "championship"
    return label & sub


def has_week_numbers(sched: pd.DataFrame) -> bool:
    regular = sched[sched["game_id_raw"].str[:3] == REGULAR_SEASON_PREFIX]
    return bool(len(regular)) and bool((regular["week_number"] > 0).any())


# ── Rounds ────────────────────────────────────────────────────────────────────

def assign_rounds(week_index: np.ndarray, round_weeks: list[int]) -> np.ndarray:
    """Map each week index onto its DK tournament round, 0 if outside the contest.

    Rounds run consecutively from week 1: `round_weeks = [17, 2, 2, 2]` puts weeks 1-17
    in Round 1 and each following pair in Rounds 2, 3 and 4. Weeks past the last round
    are `OUTSIDE_CONTEST` — DK's Round 4 ends before the NBA regular season does, and
    those games are scored by nobody.
    """
    edges = np.cumsum([0] + list(round_weeks))
    out = np.full(len(week_index), OUTSIDE_CONTEST, dtype=int)
    for rnd, (lo, hi) in enumerate(zip(edges[:-1], edges[1:]), start=1):
        out[(week_index > lo) & (week_index <= hi)] = rnd
    return out


# ── Assertions ────────────────────────────────────────────────────────────────

def assert_partition(periods: pd.DataFrame) -> None:
    """Every game date lands in exactly one period, and every game exactly once.

    This is the property the whole module exists to guarantee. A date in two periods
    would double-count a night of basketball into two weekly maxima, which is silent:
    the totals stay plausible and only the lineup that produced them is wrong.
    """
    # Unassigned rows are checked first: `nunique` skips NaN, so a missing period would
    # otherwise surface as a date belonging to *zero* periods and be reported as the
    # opposite failure.
    missing = int(periods["period_index"].isna().sum())
    assert missing == 0, f"{missing} games carry no scoring period"

    dup = periods.duplicated(subset=["season", "game_id"]).sum()
    assert dup == 0, f"{dup} duplicated (season, game_id) rows"

    per_date = periods.groupby(["season", PERIOD_DATE_COLUMN])["period_index"].nunique()
    bad = per_date[per_date != 1]
    assert bad.empty, (
        f"{len(bad)} game dates land in more than one scoring period:\n"
        f"{bad.head(10).to_string()}"
    )


def assert_contiguous_weeks(periods: pd.DataFrame) -> None:
    """Period indices run 1..N with no gaps, and each is one Monday-Sunday week."""
    for season, grp in periods.groupby("season"):
        idx = np.sort(grp["period_index"].unique())
        expected = np.arange(1, len(idx) + 1)
        assert np.array_equal(idx, expected), (
            f"{season}: period indices are not contiguous from 1 — "
            f"got {idx.min()}..{idx.max()} with {len(idx)} distinct"
        )
        spans = grp.groupby("period_index").agg(start=("period_start", "nunique"),
                                                end=("period_end", "nunique"))
        assert (spans["start"] == 1).all() and (spans["end"] == 1).all(), (
            f"{season}: a period carries more than one week boundary"
        )
        bounds = grp.groupby("period_index")[["period_start", "period_end"]].first()
        width = (bounds["period_end"] - bounds["period_start"]).dt.days
        assert (width == 6).all(), f"{season}: a period is not a 7-day week"
        assert (bounds["period_start"].dt.weekday == 0).all(), (
            f"{season}: a period does not start on a Monday"
        )


def assert_no_cup_final(periods: pd.DataFrame, cup_final_ids: set[str]) -> None:
    """The Cup final scores nowhere, so it must not have reached the output."""
    present = periods["game_id_padded"].isin(cup_final_ids)
    assert not present.any(), (
        f"{int(present.sum())} NBA Cup championship game(s) reached the scoring "
        f"periods: {sorted(periods.loc[present, 'game_id_padded'].unique())}"
    )


# ── Build ─────────────────────────────────────────────────────────────────────

def build_periods(logs: pd.DataFrame, cfg: dict, raw_dir: str | Path,
                  refresh: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One row per (season, game_id): its scoring period and its tournament round."""
    sp_cfg = cfg["sim"]["scoring_periods"]
    round_weeks = list(sp_cfg["round_weeks"])

    games = (logs.groupby(["season", "game_id"], as_index=False)[PERIOD_DATE_COLUMN]
             .first())
    games[PERIOD_DATE_COLUMN] = (pd.to_datetime(games[PERIOD_DATE_COLUMN])
                                 .dt.normalize())
    games["game_id_padded"] = games["game_id"].astype(str).str.zfill(10)

    frames, audits, cup_ids = [], [], set()
    for season, grp in games.groupby("season", sort=True):
        # `reset_index` is load-bearing, not tidiness: `grp` carries `games`' original
        # labels, so every alignment below against a freshly merged frame would silently
        # mismatch for every season after the first.
        grp = grp.sort_values(PERIOD_DATE_COLUMN).reset_index(drop=True)

        sched, source = None, "derived"
        if season >= SCHEDULE_API_FIRST_SEASON:
            sched = fetch_schedule(season, raw_dir, refresh=refresh)
            cup_ids |= set(sched.loc[cup_final_mask(sched), "game_id_raw"])
            if has_week_numbers(sched):
                source = "schedule_api"

        grp["period_index"] = derive_week_index(grp[PERIOD_DATE_COLUMN])
        fallback_name = "Week " + grp["period_index"].astype(str)
        agreement, moved = np.nan, 0

        if source == "schedule_api":
            api = sched[sched["game_id_raw"].str[:3] == REGULAR_SEASON_PREFIX].copy()
            api["game_id_padded"] = api["game_id_raw"]
            merged = grp.merge(
                api[["game_id_padded", "week_number", "week_name", "schedule_date"]],
                on="game_id_padded", how="left")
            have = merged["week_number"].notna()
            agreement = float((merged.loc[have, "week_number"].astype(int)
                               == merged.loc[have, "period_index"]).mean())
            # "Played, not scheduled": the period is already indexed off the realized
            # date, so a game that moved is re-indexed by construction. Only the NBA's
            # cosmetic week *label* is borrowed, and only where the dates agree.
            sched_date = pd.to_datetime(merged["schedule_date"])
            same_date = have & (sched_date == merged[PERIOD_DATE_COLUMN])
            moved = int((have & ~same_date).sum())
            grp["period_name"] = merged["week_name"].where(same_date, fallback_name)
        else:
            grp["period_name"] = fallback_name

        start, end = week_bounds(grp[PERIOD_DATE_COLUMN])
        grp["period_start"], grp["period_end"] = start, end
        grp["tournament_round"] = assign_rounds(grp["period_index"].values, round_weeks)
        grp["period_source"] = source

        gaps = grp[PERIOD_DATE_COLUMN].drop_duplicates().sort_values().diff().dt.days
        frames.append(grp)
        audits.append({
            "season": season,
            "games": len(grp),
            "periods": int(grp["period_index"].nunique()),
            "period_source": source,
            "api_week_agreement": agreement,
            "games_moved_from_schedule": moved,
            "break_gap_days": float(gaps.max()) if gaps.notna().any() else np.nan,
            "first_game": grp[PERIOD_DATE_COLUMN].min(),
            "last_game": grp[PERIOD_DATE_COLUMN].max(),
            "weeks_outside_contest": int(
                grp.loc[grp["tournament_round"] == OUTSIDE_CONTEST,
                        "period_index"].nunique()),
        })

    periods = pd.concat(frames, ignore_index=True)
    assert_partition(periods)
    assert_contiguous_weeks(periods)
    assert_no_cup_final(periods, cup_ids)

    cols = ["season", "game_id", "game_id_padded", PERIOD_DATE_COLUMN, "period_index",
            "period_name", "period_start", "period_end", "tournament_round",
            "period_source"]
    return periods[cols], pd.DataFrame(audits)


def round_structure(periods: pd.DataFrame, round_weeks: list[int]) -> pd.DataFrame:
    """Weeks and games per (season, round), with whether the round is complete.

    A round is short only when the season ran out of weeks. Three seasons in the data do
    that — the 1998-99 and 2011-12 lockouts and the 2020-21 COVID season — and DK's rules
    already say what happens: a season shortened before the end of Round 1 refunds every
    contest, and one shortened later voids the rounds that did not complete. So a short
    round is recorded as incomplete rather than quietly scored as if it were a full one.
    """
    expected = {rnd: n for rnd, n in enumerate(round_weeks, start=1)}
    scored = periods[periods["tournament_round"] != OUTSIDE_CONTEST]
    out = (scored.groupby(["season", "tournament_round"])
           .agg(weeks=("period_index", "nunique"), games=("game_id", "size"),
                start=("period_start", "min"), end=("period_end", "max"))
           .reset_index())
    out["weeks_expected"] = out["tournament_round"].map(expected)
    out["complete"] = out["weeks"] == out["weeks_expected"]
    return out


def verify_round_structure(structure: pd.DataFrame, round_weeks: list[int],
                           season_weeks: pd.Series) -> pd.DataFrame:
    """Assert DK's published shape — 17 weeks then three double weeks — holds.

    The teeth are in the first two checks, which no season may fail: a round never runs
    *longer* than its published length, and an incomplete round is always the last one a
    season reaches. Together those say the map is anchored and the only way to lose weeks
    is to run off the end of the season. The third asserts the full 17/2/2/2 shape for
    every season long enough to carry it, which is the check the item asked for; short
    seasons are excluded by the count of weeks they have, not by being named.
    """
    full_length = sum(round_weeks)
    over, misordered, wrong_shape = [], [], []

    for season, grp in structure.groupby("season"):
        grp = grp.sort_values("tournament_round")
        for r in grp.itertuples():
            if r.weeks > r.weeks_expected:
                over.append(f"{season} round {r.tournament_round}: {r.weeks} weeks, "
                            f"published length is {r.weeks_expected}")
        incomplete = grp.loc[~grp["complete"], "tournament_round"]
        if len(incomplete):
            first_short = int(incomplete.min())
            after = grp[grp["tournament_round"] > first_short]
            if len(after):
                misordered.append(
                    f"{season}: round {first_short} is short but rounds "
                    f"{sorted(after['tournament_round'])} follow it")
        if season_weeks[season] >= full_length and not grp["complete"].all():
            got = ", ".join(f"R{int(r.tournament_round)}={int(r.weeks)}"
                            for r in grp.itertuples())
            wrong_shape.append(f"{season} ({season_weeks[season]} weeks): {got}")

    assert not over, "a round runs longer than DK publishes:\n" + "\n".join(over)
    assert not misordered, ("a round is short before the end of the season:\n"
                            + "\n".join(misordered))
    assert not wrong_shape, ("a full-length season does not reproduce DK's shape "
                             f"{round_weeks}:\n" + "\n".join(wrong_shape))
    return structure


def dk_calendar_check(structure: pd.DataFrame) -> pd.DataFrame:
    """How the derived round ends line up with the round windows in the rules copy.

    Reported, not asserted. The published windows belong to one season's rules — the
    same copy dates the Cup final to 12/11/2026 — so they slide against any other
    season's grid. A few days of drift is the all-star break moving; a few weeks would
    mean the round map has lost its anchor.
    """
    rows = []
    for (season, rnd), grp in structure.groupby(["season", "tournament_round"]):
        end = grp["end"].iloc[0]
        month, day = DK_PUBLISHED_ROUND_ENDS[rnd - 1]
        published = pd.Timestamp(end.year, month, day)
        rows.append({"season": season, "tournament_round": rnd,
                     "derived_end": end, "dk_published_end": published,
                     "drift_days": int((end - published).days)})
    return pd.DataFrame(rows)


def run(cfg: dict, refresh: bool = False, forward_seasons: list[str] = ()) -> Path:
    """Build the period grid from the realized game logs — plus, when a season is named
    in `forward_seasons`, from that season's synthetic game log.

    The forward triples come from `forward_design.synthetic_game_log` rather than from
    `ScheduleLeagueV2` directly, and the difference is the whole point: the published
    schedule for an unplayed season is 26 games short (the Cup knockout rides on TBD
    placeholders and the replacement games are not published at all), while the synthetic
    log fills every team to 82 with the SAME game ids the roster grid will carry. Periods
    keyed to the raw schedule would leave the filler games slotless, and a game with no
    slot scores for nobody. Every date in it is still a plan — `PERIOD_DATE_COLUMN`'s
    "played, not scheduled" rule has nothing realized to prefer until games happen, which
    is why a forward season's periods are rebuilt by the October runbook rather than
    trusted from August.

    Defaults to empty, so `make scoring-periods` is unchanged and a forward season enters
    this artifact only when someone names it.
    """
    processed_dir = Path(cfg["data"]["processed_dir"])
    features_dir = Path(cfg["data"]["features_dir"])
    raw_dir = Path(cfg["data"]["raw_dir"])
    round_weeks = list(cfg["sim"]["scoring_periods"]["round_weeks"])

    logs = pd.read_parquet(processed_dir / "game_logs.parquet",
                           columns=["season", "game_id", PERIOD_DATE_COLUMN])
    played = set(logs["season"].unique())
    for forward in forward_seasons:
        if forward in played:
            raise ValueError(
                f"{forward} is in the game logs, so it has been played and its periods "
                f"are built from realized dates — the forward path is for a season "
                f"nobody has played.")
        from src.features.forward_design import synthetic_game_log

        log, _ = synthetic_game_log(forward, raw_dir)
        triples = (log.rename(columns={"SEASON_YEAR": "season", "GAME_ID": "game_id",
                                       "GAME_DATE": PERIOD_DATE_COLUMN})
                   [["season", "game_id", PERIOD_DATE_COLUMN]]
                   .drop_duplicates(["season", "game_id"]))
        # The synthetic log carries string game ids and the stored logs carry int64 —
        # match the stored dtype or the concat mints an object column parquet refuses.
        # Leading zeros survive: `build_periods` re-pads with `zfill(10)`, the same
        # convention every played season already round-trips through.
        triples["game_id"] = triples["game_id"].astype(logs["game_id"].dtype)
        print(f"  {forward}: {len(triples):,} synthetic games appended to the grid — "
              f"every date is scheduled, none realized")
        logs = pd.concat([logs, triples], ignore_index=True)
    periods, audit = build_periods(logs, cfg, raw_dir, refresh=refresh)

    features_dir.mkdir(parents=True, exist_ok=True)
    dest = features_dir / "scoring_periods.parquet"
    periods.to_parquet(dest, index=False)

    season_weeks = audit.set_index("season")["periods"]
    structure = verify_round_structure(round_structure(periods, round_weeks),
                                       round_weeks, season_weeks)
    drift = dk_calendar_check(structure)

    eda_dir = Path(cfg["eda"]["output_dir"])
    eda_dir.mkdir(parents=True, exist_ok=True)
    audit_dest = eda_dir / "scoring_periods_audit.csv"
    audit.to_csv(audit_dest, index=False)
    rounds_dest = eda_dir / "scoring_periods_rounds.csv"
    structure.merge(drift, on=["season", "tournament_round"]).to_csv(rounds_dest,
                                                                    index=False)

    n_seasons = periods["season"].nunique()
    print(f"Scoring periods: {len(periods):,} games over {n_seasons} seasons → {dest}")

    api = audit[audit["period_source"] == "schedule_api"]
    if len(api):
        checked = int(api["games"].sum())
        print(f"  the validation — derived week index vs the NBA's own `weekNumber` "
              f"on {checked:,} games across {len(api)} seasons: "
              f"{api['api_week_agreement'].min():.4f} worst-season agreement")
        print(f"  games whose played date moved off the schedule: "
              f"{int(api['games_moved_from_schedule'].sum())} "
              "— scored in the period played, not scheduled")
    derived = audit[audit["period_source"] == "derived"]
    print(f"  week grid: {len(api)} seasons from ScheduleLeagueV2, "
          f"{len(derived)} derived from realized game dates")
    print(f"  Cup championship games reaching the output: 0 — asserted, not reported")

    latest = periods["season"].max()
    ls = structure[structure["season"] == latest]
    shape = ", ".join(f"R{int(r.tournament_round)}={int(r.weeks)}w/{int(r.games)}g"
                      for r in ls.itertuples())
    print(f"  round structure ({latest}): {shape}")
    full = season_weeks[season_weeks >= sum(round_weeks)]
    print(f"  {round_weeks[0]} weeks then three double weeks — asserted for all "
          f"{len(full)} full-length seasons")

    short = season_weeks[season_weeks < sum(round_weeks)]
    for season, weeks in short.items():
        got = structure[structure["season"] == season]
        done = int(got["complete"].sum())
        n_rounds = len(round_weeks)
        last = (f"round {n_rounds} never closes" if done + 1 == n_rounds
                else f"rounds {done + 1}-{n_rounds} never close")
        print(f"  {season} ran {weeks} weeks — {done} of {n_rounds} rounds close; "
              + ("Round 1 never closes, so DK refunds the contest"
                 if done == 0 else f"{last} and those stats do not count"))

    gap = audit.loc[audit["season"] == latest, "break_gap_days"].iloc[0]
    outside = audit.loc[audit["season"] == latest, "weeks_outside_contest"].iloc[0]
    print(f"  longest in-season gap ({latest}): {gap:.0f} days — the all-star break, "
          "which moves no Monday and so shifts no period")
    print(f"  weeks past Round 4 ({latest}): {int(outside)} — DK's contest ends before "
          "the NBA season does")

    clean = drift[drift["season"].isin(full.index)]
    latest_drift = ", ".join(f"R{int(r.tournament_round)}{int(r.drift_days):+d}d"
                             for r in clean[clean["season"] == latest].itertuples())
    print(f"  vs the round windows in the rules copy ({latest}): {latest_drift} — "
          "reported, not asserted; those dates belong to one season's rules")
    print(f"Audit: {len(audit):,} rows → {audit_dest}")
    print(f"Rounds: {len(structure):,} rows → {rounds_dest}")
    return dest


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--refresh", action="store_true",
                        help="re-pull cached schedules from ScheduleLeagueV2")
    parser.add_argument("--forward", nargs="*", default=[],
                        help="unplayed season(s) to append from the synthetic game log, "
                             "e.g. --forward 2026-27")
    args = parser.parse_args()

    cfg = yaml.safe_load(open("configs/default.yaml"))
    run(cfg, refresh=args.refresh, forward_seasons=list(args.forward))
