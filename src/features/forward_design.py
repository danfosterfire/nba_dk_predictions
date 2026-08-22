"""Design rows for a season that has not been played — and the rehearsal that prices them.

`docs/final-evaluation-plan.md` §6 measured why the production posteriors are not enough to
score 2026-27: **the design rows are harvested from what happened.** The component design
starts from `season_totals`, which aggregates the game log, so a season with no game log
yields no rows at all; and `season_availability` defines a player's team by his final
*appearance*, so `team_games` — the binomial denominator, "games he could have played" — is
undefined for a player who has not yet played.

Neither is a statement about the model. Every feature the heads consume is a function of
S-1 and static attributes, and the simulator's exposure is a **draw**
(`sim/season.py::draw_components` multiplies a per-minute rate by the drawn minutes), so
nothing downstream ever wants a realized target-season total. What is missing is a
membership rule that does not require the season to have started.

## The rule, and the inversion that makes it legal

`team_context.season_start_roster` is the project's point-in-time membership rule and is
**not** the answer: it reads game logs too — a player joins team T at the index of T's game
where he first appears. That reconstructs a roster after the opener, and the draft is
before it.

The source that works is `team_rosters_<season>.csv`, which `src/features/draft_pool.py`
explicitly **rejects** as a membership source — it is a current-status snapshot, so for a
played season it puts February signings in an October pool. That objection is exactly right
for a played season and **inverts for an unplayed one**: a snapshot taken before the opener
has no February signings in it yet, because February has not happened. The same file is
contaminated looking backwards and clean looking forwards.

## What this module holds

Two things, and they answer different questions.

**`synthetic_game_log` is the production path.** It crosses the published schedule with the
roster snapshot to produce a game-log-shaped frame for a season nobody has played. Because
every season-keyed frame in the project aggregates that one file — `build_season_panel`
reads it for membership and `team_schedule` reads it again for each team's game sequence —
synthesizing it makes the existing builders work unchanged, rather than teaching five
modules a second way to build themselves. Measured on 2026-27: 46,160 player-games, 82
games per team, and 489 availability design rows out of an unmodified `build_design`.

⚠️ It does not close the **component** design, and the reason inverted along the way. That
builder ends on `total_minutes > 0`; before the synthetic log existed nothing reached the
filter, because `season_totals` had no rows to filter. With the log in place the rows exist
and `MIN` is blank, so `total_minutes` sums to `0.0` and the filter drops every one. It is a
fitting-population rule and needs a forward exemption.

**`run` is the rehearsal**, below, which validates the membership rule the synthesis depends
on against a season where the answer is already known.

## What the rehearsal can and cannot establish

Rehearsing on a played season is what the October runbook asks for — *the real window is too
short to debug a join in* — and it can settle the mechanics exactly. It **cannot** settle
the population cleanly, and the reason is worth stating rather than discovering later.

A retrospective snapshot is contaminated in two directions and only one is detectable:

- it **includes** players who arrived mid-season, which `HOW_ACQUIRED` dates it, so the
  rehearsal can cut them;
- it **excludes** players who were on the October roster and were traded or waived away
  before the snapshot was taken, which nothing in the file records.

So the population comparison below is a **bound**, not an estimate, and it is a bound in a
known direction. In October 2026 neither contamination exists, because the snapshot is taken
before the opener. `part_b` reports it as a bound and says so.

## Two parts

**Part A — the mechanics.** Build the design forward with the population *held fixed* at the
game-log one, and compare every feature column against the retrospective design. Anything
that differs is a plumbing defect, because the two frames describe the same players in the
same season and every feature is a function of seasons before it.

**Part B — the population.** Build it forward from the roster snapshot instead, and price the
difference in the only unit that matters: what the added and dropped players actually
scored.

**Part D — the composition's per-player frame**, population held fixed like Part A: the
`w_share` / ordering / design columns `sim.season.composition_players` reads, built through
`forward_composition_players` and compared against the shipped `head_frame` path. The one
documented difference is the draft number, whose retrospective source (`bio_draft_number`)
does not exist before the opener — players with no matrix row before the season are
classified as expected rather than counted as defects.

(Part C, between them in the output, is not a comparison — it records the production
synthesis for the audit.)

Usage:
    python -m src.features.forward_design                    # 2023-24, a validation season
    python -m src.features.forward_design --season 2022-23
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.data.fetch import _season_start_year, _slug, nbastats_dir
from src.models.availability import FEATURE_COLS, build_design, season_start_dates

#: The target's own columns on a forward row. Set to a sentinel so the shared builder's
#: `dropna` keeps the row, then scrubbed on the way out — a forward row has no target, and
#: one carrying a plausible-looking zero is a row that could be fitted on by accident.
TARGET_COLS = ("gp", "gp_share", "window_games", "played_in_window", "trailing_missed",
               "end_play_rate", "start_play_rate", "minutes_per_game", "window_share",
               "available_rate", "total_minutes")

#: `HOW_ACQUIRED` free text carries the acquisition date for signings and trades —
#: "Signed on 12/22/23", "Traded from SAS on 02/10/22". Draft-pick rows carry a year only
#: and are always pre-opener for the season they appear in.
_ACQUIRED_DATE = re.compile(r"on (\d{2})/(\d{2})/(\d{2})")


def next_production_season(cfg: dict) -> str:
    """The season the production board is for — the pool's last label, as elsewhere."""
    pool = pd.read_parquet(Path(cfg["data"]["features_dir"]) / "draft_pool.parquet",
                           columns=["season"])
    return str(sorted(pool["season"].unique())[-1])


def acquired_on(text: object) -> pd.Timestamp | None:
    """The date in a `HOW_ACQUIRED` string, or `None` when it carries none."""
    if not isinstance(text, str):
        return None
    hit = _ACQUIRED_DATE.search(text)
    if not hit:
        return None
    month, day, year = (int(part) for part in hit.groups())
    return pd.Timestamp(year=2000 + year, month=month, day=day)


def roster_members(season: str, raw_dir: str | Path,
                   before: pd.Timestamp | None = None) -> pd.DataFrame:
    """`(player_id, team_id)` from the roster snapshot, optionally cut to pre-`before`.

    `before` is the rehearsal's honesty knob and is **not** used in production: for a
    forward season the snapshot is already taken before the opener, so there is nothing to
    cut. On a played season it removes the mid-season arrivals the snapshot swept up, which
    is one of the two contaminations — see the module docstring for the other, which cannot
    be removed at all.
    """
    path = nbastats_dir(raw_dir) / f"team_rosters_{_slug(season)}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"no {path} — the forward membership rule reads the roster snapshot, which "
            f"`make fetch` writes. For a season that has not started this is the only "
            f"admissible source; `season_start_roster` reads game logs.")
    roster = pd.read_csv(path)
    out = roster.rename(columns={"PLAYER_ID": "player_id", "TeamID": "team_id"})
    out = out[["player_id", "team_id", "HOW_ACQUIRED"]].copy()
    if before is not None:
        acquired = out["HOW_ACQUIRED"].map(acquired_on)
        out = out[~(acquired.notna() & (acquired >= before))]
    return out.drop_duplicates("player_id").reset_index(drop=True)


#: Regular-season games every NBA team plays. The published schedule does **not** say this
#: for a season that has not been played — see `schedule_team_games`.
SEASON_GAMES = 82


def schedule_team_games(season: str, raw_dir: str | Path,
                        refresh: bool = False) -> tuple[pd.Series, dict]:
    """Per-team regular-season games from `ScheduleLeagueV2`, and the Cup gap it hides.

    🔴 **The published schedule for an unplayed season is two games per team short, and
    nothing in it says so.** Measured 2026-08-21 against both a played and an unplayed
    season:

    | season | regular games | franchises | games each | TBD placeholder |
    |---|---|---|---|---|
    | 2025-26 (played) | 1,230 | 30 | **82** | none |
    | 2026-27 (published) | 1,206 | 30 | **80** | team id `0`, 12 games |

    The Emirates NBA Cup knockout rounds have no opponents until group play ends, so the
    league publishes 80 fixed games per team and carries the rest against a placeholder.
    Once the season is played the resolved games are backfilled and the same endpoint
    returns 82. Every team plays 82 either way — a team knocked out of the Cup has two
    ordinary games added.

    So a forward denominator taken as "rows in the schedule" is **80**, and `team_games` is
    the binomial denominator for the availability head: every simulated season would run
    each player's rate against 2 too few opportunities, a 2.4% shortfall straight into
    games played and from there into every season total and the draft board.

    This is the same class of defect `sim/season.py::roster_grid` already records from the
    other direction, where a traded player's denominator came out at 92.6 against the
    head's 82.0 and "every simulated season would have run the head's rate against ~13% too
    many opportunities". That one was found the hard way. This one is found in August.

    Returns the per-team count **corrected to `SEASON_GAMES`** plus a record of what the
    raw schedule said, so a caller can see the correction rather than inherit it.
    """
    from src.data.fetch import nbastats_dir as _dir

    dest = Path(_dir(raw_dir)) / f"schedule_teams_{_slug(season)}.csv"
    if dest.exists() and not refresh:
        games = pd.read_csv(dest, dtype={"game_id": str})
    else:
        from nba_api.stats.endpoints import ScheduleLeagueV2

        payload = ScheduleLeagueV2(season=season, timeout=60).get_dict()
        games = pd.DataFrame([
            {"game_id": g["gameId"], "date": g["gameDateEst"][:10],
             "home_team_id": g["homeTeam"]["teamId"],
             "away_team_id": g["awayTeam"]["teamId"],
             "game_label": g["gameLabel"]}
            for d in payload["leagueSchedule"]["gameDates"] for g in d["games"]])
        dest.parent.mkdir(parents=True, exist_ok=True)
        games.to_csv(dest, index=False)
        print(f"  fetched {len(games):,} scheduled games → {dest}")

    regular = games[games["game_id"].astype(str).str.startswith("002")]
    sides = pd.concat([regular["home_team_id"], regular["away_team_id"]])
    counts = sides.value_counts()
    # A real franchise id is ten digits; the Cup's TBD slot is `0`.
    franchises = counts[counts.index > 1_000_000]
    placeholder = int(counts[counts.index <= 1_000_000].sum())

    raw = sorted(franchises.unique().tolist())
    record = {"season": season, "n_regular_games": int(len(regular)),
              "n_franchises": int(franchises.size), "raw_games_each": raw,
              "n_placeholder_sides": placeholder,
              "corrected_to": SEASON_GAMES,
              "correction": SEASON_GAMES - int(max(raw)) if raw else np.nan}
    return pd.Series(SEASON_GAMES, index=franchises.index, dtype=int), record


def synthetic_game_log(season: str, raw_dir: str | Path,
                       dest: str | Path | None = None) -> tuple[pd.DataFrame, dict]:
    """A game-log-shaped frame for a season nobody has played — schedule x roster.

    **This is the whole forward path, and it is one artifact rather than five rewrites.**
    Every season-keyed frame in the project aggregates `game_logs_<season>.csv`:
    `build_season_panel` reads it for membership and calls `team_schedule`, which reads it
    again for each team's game sequence, and then builds the panel as roster x schedule —
    already the shape a forward season needs. Synthesizing that one file makes the existing
    builders work unchanged.

    `MIN` is left **empty**, so `played` comes out 0 everywhere and no target is
    fabricated. Nothing needs it: the availability design built with `played` faked to 1
    and with `played` never set gives identical feature columns to 0.00e+00, and the
    simulator draws availability per-sim onto the grid rather than reading it.

    ## The two games the published schedule does not have

    An unplayed season's schedule lists **80** games per team, not 82 — see
    `schedule_team_games`. Six of the missing games are the Cup knockout, published with
    the placeholder id `0` on both sides; the other twenty-four are the replacement games
    for teams that do not advance, and they are not published at all until the groups
    resolve in December.

    Leaving them out is not an option, because `team_games` is the availability head's
    binomial denominator **and** the grid the absences are laid out on. Eighty on one side
    and eighty-two on the other is not a 2.4% approximation, it is an inconsistency. So
    every team is given the games it is short, paired into real two-team games and dated
    inside the knockout window the real ones fall in, and the approximation is recorded in
    the returned record rather than hidden: **which** teams meet and **which** day inside
    that window is not knowable in October, and neither materially moves a season total.
    """
    counts, record = schedule_team_games(season, raw_dir)
    games = pd.read_csv(Path(nbastats_dir(raw_dir))
                        / f"schedule_teams_{_slug(season)}.csv", dtype={"game_id": str})
    regular = games[games["game_id"].str.startswith("002")]
    known = regular[(regular["home_team_id"] != 0) & (regular["away_team_id"] != 0)]
    tbd_dates = sorted(regular.loc[(regular["home_team_id"] == 0), "date"].unique())

    rows = pd.concat([
        known[["game_id", "date", "home_team_id"]].rename(
            columns={"home_team_id": "TEAM_ID"}),
        known[["game_id", "date", "away_team_id"]].rename(
            columns={"away_team_id": "TEAM_ID"})], ignore_index=True)

    teams = sorted(counts.index)
    short = {t: SEASON_GAMES - int((rows["TEAM_ID"] == t).sum()) for t in teams}
    if min(short.values()) < 0:
        raise ValueError(
            f"{season}: a team already has more than {SEASON_GAMES} scheduled games "
            f"({short}); the filler rule assumes the published schedule is short, never "
            f"long")
    filler = []
    half = len(teams) // 2
    for k in range(max(short.values())):
        date = tbd_dates[k % len(tbd_dates)] if tbd_dates else known["date"].max()
        for i in range(half):
            home, away = teams[i], teams[i + half]
            if short[home] <= k or short[away] <= k:
                continue
            gid = f"009{_season_start_year(season) % 100:02d}{k:02d}{i:03d}"
            filler += [{"game_id": gid, "date": date, "TEAM_ID": home},
                       {"game_id": gid, "date": date, "TEAM_ID": away}]
    rows = pd.concat([rows, pd.DataFrame(filler)], ignore_index=True) if filler else rows

    roster = pd.read_csv(Path(nbastats_dir(raw_dir))
                         / f"team_rosters_{_slug(season)}.csv")
    members = roster.rename(columns={"PLAYER_ID": "PLAYER_ID", "TeamID": "TEAM_ID",
                                     "PLAYER": "PLAYER_NAME"})
    log = members[["PLAYER_ID", "PLAYER_NAME", "TEAM_ID"]].drop_duplicates(
        "PLAYER_ID").merge(rows, on="TEAM_ID", how="inner")
    log = log.rename(columns={"game_id": "GAME_ID", "date": "GAME_DATE"})
    log["SEASON_YEAR"] = season
    log["MIN"] = np.nan               # nothing has been played, and nothing pretends it has
    log = log[["SEASON_YEAR", "PLAYER_ID", "PLAYER_NAME", "TEAM_ID", "GAME_ID",
               "GAME_DATE", "MIN"]]

    per_team = log.groupby("TEAM_ID")["GAME_ID"].nunique()
    record = record | {"n_filler_games": len(filler) // 2,
                       "filler_dates": ",".join(str(d) for d in tbd_dates),
                       "games_per_team": sorted(per_team.unique().tolist()),
                       "n_players": int(log["PLAYER_ID"].nunique()),
                       "n_rows": int(len(log))}
    if record["games_per_team"] != [SEASON_GAMES]:
        raise ValueError(
            f"{season}: after filling, teams carry {record['games_per_team']} games "
            f"rather than {SEASON_GAMES}. The grid and the binomial denominator must "
            f"agree or absences are laid out on a schedule of the wrong length.")
    if dest is not None:
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        log.to_csv(dest, index=False)
        print(f"  wrote {len(log):,} synthetic player-game rows → {dest}")
    return log, record


def scheduled_team_games(panel: pd.DataFrame, season: str) -> pd.Series:
    """`team_id -> games on the schedule`, which is what a forward denominator must use.

    The retrospective rule counts a player's *last team's* timeline, discovered from his
    final appearance. This counts the team's schedule, which is knowable in September and
    is the same number for every player on the roster.
    """
    rows = panel[panel["season"] == season]
    return (rows.groupby("team_id")["team_game_index"].max() + 1).astype(int)


def forward_component_design(cfg: dict, season: str,
                             targets: pd.DataFrame | None = None,
                             whole_frame: bool = False) -> pd.DataFrame:
    """The eleven rate heads' design for a season that has not been played.

    The whole chain in one call, because October is a two-day window and three separate
    invocations with a `forward_seasons` argument each is three chances to pass it twice
    and forget it once. The two population filters this clears are stacked and independent
    — `build_component_targets` drops blank-minute rows, and `build_design` drops
    zero-minute *seasons* — so clearing one and not the other yields an empty frame rather
    than a wrong one, which is the safe way for a half-applied change to fail.

    `targets` is the played seasons' component targets, read from disk when not supplied.
    They are what the lag columns come from; the forward season contributes rows and no
    values.
    """
    from src.features.targets import COMPONENTS, build_component_targets
    from src.models.component_rates import build_design

    if targets is None:
        targets = pd.read_parquet(
            Path(cfg["data"]["features_dir"]) / "component_targets.parquet")
    if season in set(targets["season"]):
        raise ValueError(
            f"{season} already has component targets on disk, so it has been played and "
            f"needs no forward design — build it the ordinary way.")

    log, _ = synthetic_game_log(season, cfg["data"]["raw_dir"])
    frame = log.rename(columns={"PLAYER_ID": "player_id", "TEAM_ID": "team_id",
                                "GAME_ID": "game_id", "GAME_DATE": "game_date",
                                "SEASON_YEAR": "season", "MIN": "min"})
    for component in COMPONENTS:
        frame[component] = np.nan
    frame["season_type"] = "regular"

    forward = build_component_targets(frame, forward_seasons=[season])
    stacked = pd.concat([targets.assign(is_forward=0), forward], ignore_index=True)
    seasons = list(cfg["data"]["seasons"])
    if season not in seasons:
        seasons = seasons + [season]
    design = build_design(stacked, seasons, cfg["data"]["raw_dir"],
                          forward_seasons=[season])
    out = design[design["season"] == season].reset_index(drop=True)
    if out.empty:
        raise ValueError(
            f"{season}: the forward component design is empty. Both population filters "
            f"have to be cleared — `build_component_targets(forward_seasons=...)` and "
            f"`build_design(forward_seasons=...)` — and clearing only one yields exactly "
            f"this.")
    # `whole_frame` hands back every season, because `sim.season.build_context` reads the
    # OTHER seasons' rows too — `allowed_seasons` and the no-design empirical rate. In a
    # rehearsal (a `targets` frame with the season cut out) the seasons AFTER the target
    # carry lags that can no longer see it; the callers only ever read the target's rows
    # and the labels of the rest, which is why this is not offered as a fitting frame.
    return design.reset_index(drop=True) if whole_frame else out


def forward_availability_design(cfg: dict, frame: pd.DataFrame, panel: pd.DataFrame,
                                season: str, members: pd.DataFrame) -> pd.DataFrame:
    """The availability design for `season`, built without that season's game log.

    The played seasons' rows are handed through untouched — they are what the lag columns
    are read from — and `season`'s row is **constructed** from membership plus the schedule
    rather than aggregated from appearances.

    The shared `build_design` is called rather than reimplemented, for the reason
    `posteriors.py` gives about recipes: duplicating the order of the steps is how a design
    drifts silently. The sentinel target is the price of that reuse, and it is scrubbed
    before the frame is returned.
    """
    seasons = list(cfg["data"]["seasons"])
    if season not in seasons:
        seasons = seasons + [season]
    team_games = scheduled_team_games(panel, season)
    start_year = int(pd.Series(panel.loc[panel["season"] == season,
                                         "season_start_year"]).iloc[0])

    forward = pd.DataFrame({
        "season": season,
        "season_start_year": start_year,
        "player_id": members["player_id"].to_numpy(),
        "team_id": members["team_id"].to_numpy(),
    })
    forward["team_games"] = forward["team_id"].map(team_games)
    forward = forward[forward["team_games"].notna()].copy()
    # The sentinel. `build_design` drops rows whose target is missing, which is correct for
    # every other caller and would drop every forward row.
    forward["gp"] = 0
    forward["gp_share"] = 0.0
    for column in frame.columns:
        if column not in forward.columns:
            forward[column] = np.nan

    played = frame[frame["season"] != season]
    stacked = pd.concat([played, forward[frame.columns.tolist()]], ignore_index=True)

    design = build_design(stacked, seasons, cfg["data"]["raw_dir"],
                          season_start_dates(panel))
    design["is_forward"] = (design["season"] == season).astype(int)
    # Scrub the sentinel: a forward row has no target, and one carrying a plausible zero is
    # a row something could fit on. Every fitting path in the project drops NaN targets.
    for column in TARGET_COLS:
        if column in design.columns:
            design.loc[design["is_forward"] == 1, column] = np.nan
    return design


#: `HOW_ACQUIRED` on a player still with his drafting team carries the slot itself —
#: "#27 Pick in 2024 Draft". The year is not captured: the slot is the feature, and a
#: rights-traded rookie ("Draft Rights Traded from DAL on 07/06/23") carries no slot at
#: all, so he lands in the undrafted bucket until his first bio row corrects him.
_DRAFT_PICK = re.compile(r"#(\d+) Pick in \d{4} Draft")


def forward_draft_numbers(cfg: dict, season: str) -> pd.DataFrame:
    """Draft slot per (player, `season`) from sources that exist before the opener.

    The retrospective source is `bio_draft_number` off the season matrix, which is built
    from `player_bio_stats_<season>.csv` — a season-statistics file that does not exist
    until games are played, the same class of gap as `load_ages`. Two preseason-legal
    sources replace it, in order:

    - **the player's own history** — the draft number is a static attribute, so his most
      recent matrix row from a season strictly *before* the target carries it. The cut to
      earlier seasons matters only in a rehearsal, where the matrix already has the
      target season's bio-derived rows and reading them would grade the rule against
      itself.
    - **the roster snapshot's `HOW_ACQUIRED` text** for players with no matrix row at all
      — drafted rookies still with their drafting team read "#N Pick in YYYY Draft".
      Measured on the 2026-27 roster (2026-08-21): 391 of 577 players resolve from the
      matrix, 38 more from the text, and where both exist they agree 107 of 107.

    What remains lands in the undrafted bucket, which is what `season_weights` does with
    a missing draft number everywhere else. Measured on the same roster that is 148
    players: 99 unsigned-drafted journeymen and undrafted rookies (correct), and 25
    rights-traded draftees whose slot the text does not carry (wrong bucket, priced by
    the Part D rehearsal rather than assumed away).
    """
    from src.models.stan_composition import _draft_bucket, draft_numbers

    known = draft_numbers(Path(cfg["data"]["features_dir"]))
    cutoff = _season_start_year(season)
    prior = known[known["season"].map(_season_start_year) < cutoff]
    latest = prior.sort_values("season").groupby("player_id")["draft_number"].last()

    roster = roster_members(season, cfg["data"]["raw_dir"])
    picks = roster["HOW_ACQUIRED"].map(
        lambda t: float(_DRAFT_PICK.search(t).group(1))
        if isinstance(t, str) and _DRAFT_PICK.search(t) else np.nan)
    out = pd.DataFrame({"player_id": roster["player_id"].to_numpy(),
                        "season": season,
                        "draft_number": roster["player_id"].map(latest)
                        .fillna(picks).to_numpy()})
    out["draft_bucket"] = out["draft_number"].map(_draft_bucket)
    return out


def forward_composition_players(cfg: dict, season: str, members: pd.DataFrame,
                                design: pd.DataFrame) -> pd.DataFrame:
    """One row per rostered player — what `sim.season.composition_players` reads.

    The composition head's fitting frame is one row per *played player-game*, so a season
    with no played games has no rows in it and `build_context` raises. But everything the
    simulator reads off that frame is constant within a player-season — `w_share`, the
    allocation order's keys (`no_prior`, `draft_number`), and the design columns behind
    `composition_eta` — and `composition_frame`'s own docstring says the ordering "is
    computable preseason". This builds exactly those columns, through the extracted
    `season_weights` and the shipped blend hook, so the values come out of the code that
    fits rather than a copy of it.

    `members` is the roster membership (the same frame `forward_availability_design`
    takes), and `design` is an availability design carrying `season`'s rows — the forward
    one in production, since the retrospective builder cannot produce them. Players the
    design does not qualify keep their row with NaN feature columns, which is what the
    persisted recipe's impute step expects (`design_missing` is a fitted feature, not an
    error path).
    """
    from src.models.stan_composition import (OWN, draft_numbers, played_frame,
                                             season_shares, season_weights,
                                             shipped_share_hook)

    features_dir = Path(cfg["data"]["features_dir"])
    panel = pd.read_parquet(features_dir / "availability_panel.parquet")
    lengths = pd.read_parquet(features_dir / "game_length.parquet")
    # The target season's own rows are cut BEFORE shares are built: in production the cut
    # is a no-op, and in a rehearsal it is what stops the forward path reading the very
    # season it claims not to need.
    played = played_frame(panel[panel["season"] != season], lengths)
    shares = season_shares(played)
    forward = pd.DataFrame({"player_id": members["player_id"].to_numpy(),
                            "season": season})
    if forward["player_id"].duplicated().any():
        raise ValueError(f"{season}: `members` repeats a player_id; one row per player")
    shares = pd.concat([shares, forward], ignore_index=True)

    seasons = list(cfg["data"]["seasons"])
    if season not in seasons:
        seasons = seasons + [season]
    drafts = draft_numbers(features_dir)
    drafts = pd.concat([drafts[drafts["season"] != season],
                        forward_draft_numbers(cfg, season)], ignore_index=True)

    lagged, carried = season_weights(shares, seasons, features_dir,
                                     shipped_share_hook(cfg), drafts=drafts)
    players = lagged.loc[lagged["season"] == season, carried].reset_index(drop=True)
    players[OWN] = np.log(players["w_share"] / (1 - players["w_share"]))
    players = players.merge(
        design.loc[design["season"] == season, ["player_id", "season"] + FEATURE_COLS],
        on=["player_id", "season"], how="left")
    return players


def retrospective_availability_design(cfg: dict, frame: pd.DataFrame,
                                      panel: pd.DataFrame) -> pd.DataFrame:
    """The design as it is built today, for the comparison to mean anything."""
    return build_design(frame, list(cfg["data"]["seasons"]), cfg["data"]["raw_dir"],
                        season_start_dates(panel))


# ── Part A: the mechanics ─────────────────────────────────────────────────────

def traded_denominator_rows(retrospective: pd.DataFrame, panel: pd.DataFrame,
                            season: str) -> set:
    """The rows where the retrospective denominator is the realized `gp`, not a schedule.

    `availability.build_design` ends on `team_games = max(team_games, gp)`, because a
    traded player whose two teams' schedules overlap can play more games than either team
    played, and `gp > n` makes `betabinom.logpmf` non-finite. Thirteen rows in thirty
    seasons, and they take the whole fit down silently without it.

    It is also the **one place a target-season quantity reaches the input side** of this
    head: `team_games` is the binomial denominator, which is read at prediction time. A
    forward design cannot reproduce it and should not try — in September nobody knows who
    will be traded — so these rows are expected to differ and are classified rather than
    counted as defects.
    """
    schedule = scheduled_team_games(panel, season)
    rows = retrospective[retrospective["season"] == season]
    team = (panel[panel["season"] == season].drop_duplicates("player_id")
            .set_index("player_id")["team_id"])
    expected = rows["player_id"].map(team).map(schedule)
    return set(rows.loc[rows["team_games"].to_numpy() > expected.to_numpy(),
                        "player_id"])


def part_a(forward: pd.DataFrame, retrospective: pd.DataFrame, season: str,
           columns: list[str], expected_differ: set | None = None) -> pd.DataFrame:
    """Every feature column, forward against retrospective, on the shared population.

    The population is held fixed on purpose. These two frames describe the same players in
    the same season, and every feature is a function of seasons *before* it — so anything
    that differs is the forward path computing a feature differently, which is the class of
    defect a two-day crunch cannot absorb.

    `expected_differ` names the players whose denominator the retrospective path sets from
    hindsight. They are reported in their own column rather than excluded, because "one
    mismatch, and it is the documented one" and "one mismatch" are different results.
    """
    expected_differ = expected_differ or set()
    f = forward[forward["season"] == season].set_index("player_id")
    r = retrospective[retrospective["season"] == season].set_index("player_id")
    shared = sorted(set(f.index) & set(r.index))
    rows = []
    for column in columns:
        if column not in f.columns or column not in r.columns:
            rows.append({"part": "A", "column": column, "n": len(shared),
                         "max_abs_diff": np.nan, "n_mismatched": np.nan,
                         "n_expected": np.nan, "n_unexplained": np.nan,
                         "note": "column absent on one side"})
            continue
        a = pd.to_numeric(f.loc[shared, column], errors="coerce")
        b = pd.to_numeric(r.loc[shared, column], errors="coerce")
        diff = (a - b).abs()
        both_nan = a.isna() & b.isna()
        off = (diff > 1e-9) | (a.isna() ^ b.isna())
        explained = int(off[[i in expected_differ for i in off.index]].sum())
        rows.append({"part": "A", "column": column, "n": len(shared),
                     "max_abs_diff": float(diff[~both_nan].max())
                     if (~both_nan).any() else 0.0,
                     "n_mismatched": int(off.sum()),
                     "n_expected": explained,
                     "n_unexplained": int(off.sum()) - explained, "note": ""})
    return pd.DataFrame(rows)


# ── Part B: the population ────────────────────────────────────────────────────

def realized_totals(cfg: dict, season: str) -> pd.DataFrame:
    """What each player actually scored that season — the unit a population error is felt in.

    A dropped player who scored 2,000 dk_pts is a different failure from a dropped player
    who scored 40, and a count of names cannot tell them apart.
    """
    targets = pd.read_parquet(
        Path(cfg["data"]["features_dir"]) / "component_targets.parquet",
        columns=["player_id", "season", "season_type", "dk_pts", "min", "played"])
    rows = targets[(targets["season"] == season)
                   & (targets["season_type"] == "regular")
                   & (targets["played"] == 1)]
    return (rows.groupby("player_id", as_index=False)
            .agg(dk_total=("dk_pts", "sum"), minutes=("min", "sum"),
                 games=("played", "sum")))


def part_b(cfg: dict, season: str, snapshot: set, gamelog: set) -> pd.DataFrame:
    """Who the two rules disagree about, priced by what those players actually scored.

    ⚠️ Both sides here are **design** populations, not roster lists. The design applies its
    own qualification — a prior season of real minutes, and a known age — so comparing a
    raw 532-name roster against a 450-row design would attribute the qualification rule's
    exclusions to the membership rule. The forward design is therefore built from the
    snapshot and *then* compared, so the only thing that varies is where membership came
    from.
    """
    realized = realized_totals(cfg, season).set_index("player_id")

    def priced(ids: set, label: str) -> dict:
        hit = realized.reindex(sorted(ids))
        return {"part": "B", "group": label, "n_players": len(ids),
                "dk_total": float(hit["dk_total"].fillna(0.0).sum()),
                "mean_dk_total": float(hit["dk_total"].fillna(0.0).mean())
                if len(ids) else 0.0,
                "max_dk_total": float(hit["dk_total"].fillna(0.0).max())
                if len(ids) else 0.0,
                "n_over_500_dk": int((hit["dk_total"].fillna(0.0) > 500).sum()),
                "minutes": float(hit["minutes"].fillna(0.0).sum())}

    return pd.DataFrame([priced(snapshot & gamelog, "in both"),
                         priced(snapshot - gamelog, "snapshot only (spurious)"),
                         priced(gamelog - snapshot, "game log only (missed)")])


def run(cfg: dict, season: str = "2023-24") -> Path:
    from src.features.availability import season_availability
    from src.models.held_out import assert_unlocked, selection_split

    out_dir = Path(cfg["evaluation"]["predictions_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    features_dir = Path(cfg["data"]["features_dir"])
    raw_dir = cfg["data"]["raw_dir"]

    panel = pd.read_parquet(features_dir / "availability_panel.parquet")
    # A rehearsal is not a measurement of the model, but it reads a season's rows and the
    # honest default is a season selection is allowed to see. A test season goes through
    # the same guard everything else does rather than being quietly permitted.
    design_seasons = sorted(panel["season"].unique())
    allowed = set(selection_split(pd.DataFrame({"season": design_seasons}),
                                  2)[0]["season"]) | set(
        selection_split(pd.DataFrame({"season": design_seasons}), 2)[1]["season"])
    if season not in allowed:
        assert_unlocked(f"rehearsing the forward design on {season}")

    print(f"Forward design rehearsal — {season}, a season that HAS been played")
    print(f"  The forward path may not read this season's game log. It reads the roster")
    print(f"  snapshot and the schedule, which is what September 2026 will have.")

    frame = season_availability(panel, "full")
    retrospective = retrospective_availability_design(cfg, frame, panel)
    ret_rows = retrospective[retrospective["season"] == season]
    gamelog_members = set(ret_rows["player_id"])
    print(f"\n  retrospective design: {len(ret_rows):,} rows for {season}")

    # ── Part A ────────────────────────────────────────────────────────────────
    held = pd.DataFrame({"player_id": sorted(gamelog_members)})
    held = held.merge(panel[panel["season"] == season]
                      .drop_duplicates("player_id")[["player_id", "team_id"]],
                      on="player_id", how="left")
    forward_fixed = forward_availability_design(cfg, frame, panel, season, held)
    fwd_rows = forward_fixed[forward_fixed["season"] == season]
    print(f"  forward design, population held fixed: {len(fwd_rows):,} rows")

    columns = list(FEATURE_COLS) + ["team_games", "n_prior_seasons", "career_year", "age"]
    traded = traded_denominator_rows(retrospective, panel, season)
    a = part_a(forward_fixed, retrospective, season, columns, expected_differ=traded)
    bad = a[(a["n_mismatched"].fillna(0) > 0) | a["max_abs_diff"].isna()]
    print(f"\n[Part A — the mechanics, population held fixed]")
    print(f"  {len(columns)} columns compared on {int(a['n'].max()):,} shared players")
    unexplained = a[a["n_unexplained"].fillna(1) > 0]
    if len(unexplained):
        print(f"  /!\\  {len(unexplained)} column(s) disagree for reasons the forward "
              f"path does not account for:")
        print(unexplained.to_string(index=False))
    else:
        print(f"  every column agrees to 1e-9 except where it is SUPPOSED to. The forward "
              f"path is\n  the same design, built from membership and the schedule "
              f"instead of from appearances.")
    if len(bad):
        print(f"\n  expected differences, classified rather than excluded:")
        print(bad[["column", "n", "max_abs_diff", "n_mismatched", "n_expected",
                   "n_unexplained"]].to_string(index=False))
        print(f"  `team_games` is the ONE place a target-season quantity reaches this "
              f"head's input\n  side: `build_design` ends on `max(team_games, gp)` so a "
              f"traded player whose two teams'\n  schedules overlap does not make "
              f"`betabinom.logpmf` non-finite. A forward design cannot\n  reproduce that "
              f"and should not try — in September nobody knows who will be traded.")

    # ── Part B ────────────────────────────────────────────────────────────────
    starts = season_start_dates(panel)
    opener = pd.Timestamp(starts[season])
    snapshot = roster_members(season, raw_dir, before=opener)
    uncut = roster_members(season, raw_dir)
    snapshot_members = set(snapshot["player_id"])
    print(f"\n[Part B — the population, and it is a BOUND rather than an estimate]")
    print(f"  roster snapshot: {len(uncut):,} players, {len(snapshot):,} after cutting "
          f"those acquired on or after {opener:%Y-%m-%d}")
    print(f"  ⚠️  the snapshot also EXCLUDES anyone traded or waived away before it was "
          f"taken,\n      and nothing in the file records them. In October 2026 neither "
          f"contamination\n      exists, because the snapshot precedes the opener.")

    forward_snapshot = forward_availability_design(cfg, frame, panel, season,
                                                   snapshot[["player_id", "team_id"]])
    snap_rows = forward_snapshot[forward_snapshot["season"] == season]
    print(f"  forward DESIGN from that snapshot: {len(snap_rows):,} rows against the "
          f"retrospective's {len(ret_rows):,}\n  (both after the design's own "
          f"prior-season and age qualification)")

    b = part_b(cfg, season, set(snap_rows["player_id"]), gamelog_members)
    print()
    print(b.round(1).to_string(index=False))

    both = b[b["group"] == "in both"].iloc[0]
    missed = b[b["group"] == "game log only (missed)"].iloc[0]
    spurious = b[b["group"] == "snapshot only (spurious)"].iloc[0]
    share = missed["dk_total"] / (both["dk_total"] + missed["dk_total"])
    print(f"\n  the forward design misses {missed['n_players']:,} players who scored "
          f"{missed['dk_total']:,.0f} dk_pts — {share:.2%} of the\n  realized production "
          f"the retrospective design covers, {int(missed['n_over_500_dk'])} of them over "
          f"500")
    print(f"  and carries {spurious['n_players']:,} it should not, worth "
          f"{spurious['dk_total']:,.0f} dk_pts")
    print(f"\n  ⚠️  BOTH numbers are inflated by the rehearsal rather than by the rule. "
          f"The misses are\n      mostly players traded away before the snapshot was "
          f"taken — invisible in the file —\n      and the spurious rows are mid-season "
          f"arrivals the `HOW_ACQUIRED` cut could not date.\n      Neither exists for a "
          f"snapshot taken before an opener, which is the production case.")

    # ── Part D: the composition's per-player frame ────────────────────────────
    from src.models.stan_composition import OWN, draft_numbers, head_frame
    from src.sim.season import composition_players

    print(f"\n[Part D — the composition's per-player frame, population held fixed]")
    retro_players = composition_players(head_frame(cfg), season)
    fwd_players = forward_composition_players(cfg, season,
                                              retro_players[["player_id"]],
                                              forward_fixed)
    comp_cols = (["w_share", "no_prior", "share_stale", "draft_number", OWN]
                 + (["order_share"] if "order_share" in retro_players.columns else [])
                 + list(FEATURE_COLS))
    # The documented difference: the retrospective draft number is bio-derived and the bio
    # file does not exist before the opener, so a player with no matrix row before
    # `season` reads his slot from the roster text or lands unbucketed. His `w_share` and
    # its logit can move with the bucket, which is why the set is per-player rather than
    # per-column. A second class is the rehearsal's own: the roster text is read off the
    # snapshot, and a player traded or waived away before the snapshot was taken has no
    # row to read — Part B's known contamination showing up one column over. In
    # production membership IS the snapshot, so every member has a roster row by
    # construction. Measured on 2023-24: 33 of the first class, 19 of the second, and
    # zero draft-number disagreements between two matrix rows.
    known = draft_numbers(features_dir)
    seen_before = set(known.loc[known["season"].map(_season_start_year)
                                < _season_start_year(season), "player_id"])
    no_history = set(retro_players["player_id"]) - seen_before
    off_snapshot = set(retro_players["player_id"]) - set(uncut["player_id"])
    d = part_a(fwd_players, retro_players, season, comp_cols,
               expected_differ=no_history | off_snapshot)
    d["part"] = "D"
    print(f"  {len(comp_cols)} columns compared on {int(d['n'].max()):,} shared players; "
          f"{len(no_history)} have no pre-{season} matrix row (draft number is "
          f"text-sourced there)\n  and {len(off_snapshot)} are no longer on the "
          f"snapshot at all — Part B's contamination, absent in production")
    d_unexplained = d[d["n_unexplained"].fillna(1) > 0]
    if len(d_unexplained):
        print(f"  /!\\  {len(d_unexplained)} column(s) disagree beyond the documented "
              f"draft-number source:")
        print(d_unexplained.to_string(index=False))
    else:
        print(f"  every column agrees to 1e-9 except on players whose draft number has "
              f"no preseason\n  source — the ordering and the offset are the shipped "
              f"head's, built without the season's games.")

    # The PRODUCTION synthesis, recorded beside the rehearsal so its figures are audited
    # rather than typed into a doc from a terminal. A doc quoting the pre-fill row count
    # next to the post-fill games-per-team is exactly the drift `make docs-audit` exists
    # for, and it happened once on 2026-08-21 before this block was added.
    forward = pd.DataFrame()
    target = next_production_season(cfg)
    try:
        _, record = synthetic_game_log(target, raw_dir)
        forward = pd.DataFrame([{"part": "C", "season": target, "measure": k,
                                 "value": v} for k, v in record.items()
                                if not isinstance(v, list)])
        print(f"\n[Part C — the production synthesis for {target}]")
        print(f"  {record['n_rows']:,} player-game rows, {record['n_players']} players, "
              f"{record['games_per_team'][0]} games per team "
              f"({record['raw_games_each'][0]} published + {record['n_filler_games']} "
              f"filler games at +{record['correction']} per team)")
    except (FileNotFoundError, KeyError) as problem:
        print(f"\n[Part C] skipped — {type(problem).__name__}: {problem}")

    out = pd.concat([a.assign(season=season), b.assign(season=season),
                     d.assign(season=season), forward], ignore_index=True)
    dest = out_dir / "forward_design_rehearsal.csv"
    out.to_csv(dest, index=False)
    print(f"\nSaved {len(out):,} rehearsal rows → {dest}")
    return dest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--season", default="2023-24",
                        help="the played season to rehearse on; must be one selection is "
                             "allowed to read")
    args = parser.parse_args()

    cfg = yaml.safe_load(open("configs/default.yaml"))
    run(cfg, season=args.season)
