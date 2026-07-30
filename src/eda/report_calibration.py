"""What an NBA injury-report designation is actually worth, measured against box scores.

The preseason injury snapshot is the **only genuinely new input** `docs/availability-plan.md`
identifies — everything else in the project is prior-season stats and season-start rosters.
But a snapshot says "Questionable — Left Ankle; Sprain", and the availability head needs
`P(play)`. That mapping is a *transfer function*, and until it is measured it is a guess.

The ablation that justifies the whole Tier-2 capture is blocked until the daily archive
covers an offseason→season boundary (practically, the 2026-27 opener). **This is not.** The
transfer function needs only an overlap between the PDF archive and the box-score backfill,
and one already exists inside 2025-26: ~13k player-report rows over ~98 game dates, every
one of them a game whose realized `played / dnp / inactive` outcome is on disk. So the
mapping can be fitted and validated now, and October becomes a measurement rather than an
invention.

Four measurements:

1. **`calibration`** — `P(outcome | designation)`, the transfer function itself. Reported
   per lead time, because a report published the day before a game and one published the
   morning of are different forecasts and the archive contains both.
2. **`revision`** — how a lead-1 designation is *revised* at lead 0. This is the closest
   thing the archive holds to "how does a stale designation decay", which is the question a
   preseason snapshot actually poses: it is read weeks, not hours, before game 1.
3. **`by_reason`** — `P(play | designation, reason_category)`. "Questionable — Rest" and
   "Questionable — Knee; Soreness" are not the same forecast, and the PDF states which.
4. **`coverage`** — match rate and what was dropped. A calibration table computed on the
   60% of rows that happened to join is a statement about the join, so the denominator is
   reported beside every probability.

**A player listed and then absent from the box score is an outcome, not a join failure**, and
the two are kept apart deliberately. A name that never resolves to a `player_id` anywhere in
the season is a join failure (`unmatched`); a name that resolves but has no row in *that*
game was not on that game's roster (`absent`) — which for an `Out` player is exactly the
"waived / not with team" case the plan wants separated from injury. Collapsing them would
silently convert a name-matching bug into a finding about roster mechanics.

Only games the backfill has actually reached are scored, mirroring `status_coverage` in
`src/eda/availability.py`. That excludes Summer League — the archive runs into July, and
those reports have no NBA box score behind them — and any game the backfill missed.
"""

import re
import unicodedata
from pathlib import Path

import pandas as pd
import yaml

from src.data.boxscore_status import load_status, pad_game_id
from src.data.fetch import _slug
from src.data.injury_reports import load_log

# The five participation designations, ordered from least to most likely to play. The
# ordering is the point: a calibration that does not come out monotone in this order is
# either a join bug or a genuinely surprising finding, and it should be obvious which.
DESIGNATIONS = ["Out", "Doubtful", "Questionable", "Probable", "Available"]

# Realized outcomes. `absent` and `unmatched` are distinct on purpose — see the module
# docstring; one is a fact about the roster and the other a fact about this code.
# `ambiguous` is a third kind: the name matched *two* players in one game, so the row is
# excluded rather than resolved. It is deliberately NOT in OUTCOMES — a fabricated match
# would otherwise be scored as a real one, and an unmatched rate would never show it.
OUTCOMES = ["played", "dnp", "inactive", "absent"]

# Name suffixes carry no identity and are written inconsistently across the two sources.
_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}
_PUNCT = re.compile(r"[^a-z ]")


def normalize_name(name: str) -> str:
    """A join key that survives both sources' spelling of the same player.

    The PDFs write `"Antetokounmpo, Alex"` and the box scores `"Alex Antetokounmpo"`, so the
    comma form is flipped first. Accents, punctuation (`O'Neale`, `Nurkic`) and generational
    suffixes are then stripped, because the two feeds disagree on all three and none of them
    identify anybody.
    """
    text = str(name or "").strip()
    if not text:
        return ""
    if "," in text:
        last, _, first = text.partition(",")
        text = f"{first.strip()} {last.strip()}"
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = _PUNCT.sub(" ", text.lower())
    parts = [p for p in text.split() if p and p not in _SUFFIXES]
    return " ".join(parts)


def parse_minutes(value) -> float:
    """Box-score `min` is `"MM:SS"`, not a number — `pd.to_numeric` returns all-NaN on it.

    Silent, because a column of NaN means "played" rows report a mean of nan rather than
    raising, so the mistake reads as missing data instead of a parse bug.
    """
    text = str(value or "").strip()
    if not text or text.lower() in {"nan", "none"}:
        return float("nan")
    if ":" in text:
        minutes, _, seconds = text.partition(":")
        try:
            return float(minutes) + float(seconds) / 60.0
        except ValueError:
            return float("nan")
    try:
        return float(text)
    except ValueError:
        return float("nan")


# ── The three sources, put on common keys ─────────────────────────────────────

def report_frame(log: pd.DataFrame) -> pd.DataFrame:
    """The archive, with a normalized name, a real date and a lead time.

    `NOT YET SUBMITTED` rows are dropped here: they carry a team and no player, and exist to
    record that a team missed the filing deadline. They are meaningful in the archive and
    meaningless in a per-player calibration.
    """
    if log.empty:
        return pd.DataFrame(columns=["report_date", "game_date", "team", "player_name",
                                     "name_key", "status", "reason_category", "lead"])
    frame = log[log["status"].isin(DESIGNATIONS)].copy()
    frame["game_date"] = pd.to_datetime(frame["game_date"], format="%m/%d/%Y",
                                        errors="coerce")
    frame["report_date"] = pd.to_datetime(frame["report_date"], errors="coerce")
    frame = frame[frame["game_date"].notna() & frame["report_date"].notna()]
    frame["lead"] = (frame["game_date"] - frame["report_date"]).dt.days
    frame["name_key"] = frame["player_name"].map(normalize_name)
    return frame[frame["name_key"] != ""].reset_index(drop=True)


def game_index(seasons: list[str], raw_dir: str | Path) -> pd.DataFrame:
    """(game_date, team) → game_id, from the game logs.

    Built from the logs rather than the box-score files because only the logs carry a date.
    A team that played is guaranteed at least one player row, so the mapping is complete even
    though the logs themselves hold only players who appeared.
    """
    raw_dir = Path(raw_dir)
    frames = []
    for season in seasons:
        for name in (f"game_logs_{_slug(season)}.csv",
                     f"game_logs_playoffs_{_slug(season)}.csv"):
            path = raw_dir / name
            if not path.exists():
                continue
            cols = ["GAME_ID", "GAME_DATE", "TEAM_NAME", "TEAM_ID"]
            part = pd.read_csv(path, usecols=cols, low_memory=False)
            part["season"] = season
            frames.append(part)
    if not frames:
        return pd.DataFrame(columns=["game_date", "team", "game_id", "team_id", "season"])
    index = pd.concat(frames, ignore_index=True).drop_duplicates(
        subset=["GAME_ID", "TEAM_ID"])
    index["game_date"] = pd.to_datetime(index["GAME_DATE"], errors="coerce").dt.normalize()
    index["game_id"] = index["GAME_ID"].map(pad_game_id)
    return index.rename(columns={"TEAM_NAME": "team", "TEAM_ID": "team_id"})[
        ["game_date", "team", "game_id", "team_id", "season"]]


def outcome_frame(seasons: list[str], raw_dir: str | Path) -> pd.DataFrame:
    """(game_id, name_key) → realized status, from the box-score backfill."""
    frames = []
    for season in seasons:
        status = load_status(season, raw_dir)
        if status.empty:
            continue
        status = status.copy()
        status["season"] = season
        frames.append(status)
    if not frames:
        return pd.DataFrame(columns=["game_id", "name_key", "status", "player_id",
                                     "reason", "season"])
    frame = pd.concat(frames, ignore_index=True)
    frame["game_id"] = frame["game_id"].map(pad_game_id)
    frame["name_key"] = frame["player_name"].map(normalize_name)
    return frame[frame["name_key"] != ""]


# ── The join ──────────────────────────────────────────────────────────────────

def join_outcomes(reports: pd.DataFrame, games: pd.DataFrame,
                  outcomes: pd.DataFrame) -> pd.DataFrame:
    """Attach the realized outcome to every report row that has one.

    Three-step, and each step's losses are counted rather than dropped quietly:
    report → game (does the archive name a game the backfill reached?), then
    (game, player) → status, then the `absent` / `unmatched` split described in the module
    docstring.

    **Name collisions are detected, not resolved.** `name_key` is not a unique key: two
    different players can normalize to the same string, and `drop_duplicates` would keep
    one arbitrarily and attach its realized outcome to the other — a *fabricated* row,
    which no unmatched rate can detect because it counts as a successful match. Measured
    over 20 backfilled seasons and 797,473 status rows there is exactly **one** such cell
    (game `0021200757`, two distinct `Chris Johnson`s), so this is rare rather than
    hypothetical. Those rows get `outcome = "ambiguous"` and are excluded from the
    calibration instead of being guessed at.

    Suffix stripping is what makes most cross-season collisions: `Gary Payton` and `Gary
    Payton II` share a key. The join is scoped to a single `game_id`, so those two cannot
    collide in practice — but `known` below is season-wide, so a name that only ever
    belonged to the *other* Payton would read as `absent` rather than `unmatched`.
    """
    if reports.empty or games.empty or outcomes.empty:
        return pd.DataFrame(columns=list(reports.columns) + ["game_id", "outcome"])

    joined = reports.merge(games, on=["game_date", "team"], how="left")
    joined["covered"] = joined["game_id"].isin(set(outcomes["game_id"]))

    cols = ["game_id", "name_key", "status", "player_id", "min"]
    per_cell = outcomes.groupby(["game_id", "name_key"])["player_id"].nunique()
    ambiguous = set(per_cell[per_cell > 1].index)

    status = outcomes[cols].drop_duplicates(subset=["game_id", "name_key"])
    joined = joined.merge(status, on=["game_id", "name_key"], how="left")

    # A name that appears somewhere in the season is a real player; a name that appears
    # nowhere is this module's problem, not the roster's.
    known = set(outcomes["name_key"])
    joined["outcome"] = joined["status_y"].where(
        joined["status_y"].notna(),
        joined["name_key"].map(lambda k: "absent" if k in known else "unmatched"))
    if ambiguous:
        collided = pd.Series(list(zip(joined["game_id"], joined["name_key"])),
                             index=joined.index).isin(ambiguous)
        joined.loc[collided, "outcome"] = "ambiguous"
    joined.loc[~joined["covered"], "outcome"] = "uncovered"
    return joined.rename(columns={"status_x": "designation"}).drop(columns=["status_y"])


# ── Measurements ──────────────────────────────────────────────────────────────

def _row(measurement: str, key: str, metric: str, value: float,
         n: int | None = None) -> dict:
    return {"measurement": measurement, "key": key, "metric": metric,
            "n": n, "value": value}


def coverage(joined: pd.DataFrame) -> list[dict]:
    """What the calibration is computed on, and what fell out of it."""
    n = len(joined)
    rows = [_row("coverage", "all", "report_rows", float(n), n)]
    if not n:
        return rows
    counts = joined["outcome"].value_counts()
    for name in ["played", "dnp", "inactive", "absent", "unmatched", "uncovered",
                 "ambiguous"]:
        rows.append(_row("coverage", "all", f"share_{name}",
                         float(counts.get(name, 0)) / n, int(counts.get(name, 0))))
    scored = joined[joined["outcome"].isin(OUTCOMES)]
    rows.append(_row("coverage", "all", "scored_rows", float(len(scored)), len(scored)))
    rows.append(_row("coverage", "all", "match_rate",
                     float(len(scored)) / n if n else float("nan"), n))
    rows.append(_row("coverage", "all", "game_dates",
                     float(joined["game_date"].nunique()), n))
    rows.append(_row("coverage", "all", "players",
                     float(joined["name_key"].nunique()), n))
    return rows


def calibration(joined: pd.DataFrame) -> list[dict]:
    """`P(outcome | designation)` overall and per lead time — the transfer function."""
    scored = joined[joined["outcome"].isin(OUTCOMES)]
    rows: list[dict] = []
    if scored.empty:
        return rows
    for lead_key, part in [("all", scored)] + [
            (f"lead_{int(lead)}", grp) for lead, grp in scored.groupby("lead")]:
        for designation in DESIGNATIONS:
            sub = part[part["designation"] == designation]
            n = len(sub)
            key = f"{designation}|{lead_key}"
            rows.append(_row("calibration", key, "n", float(n), n))
            if not n:
                continue
            counts = sub["outcome"].value_counts()
            for outcome in OUTCOMES:
                rows.append(_row("calibration", key, f"p_{outcome}",
                                 float(counts.get(outcome, 0)) / n, n))
            played = sub[sub["outcome"] == "played"]
            rows.append(_row("calibration", key, "mean_min_given_played",
                             float(played["min"].map(parse_minutes).mean())
                             if len(played) else float("nan"), len(played)))
    return rows


def revision(joined: pd.DataFrame) -> list[dict]:
    """How a lead-1 designation is revised at lead 0, and what each pair realizes.

    The archive's only handle on designation *staleness*: a preseason snapshot is read weeks
    before game 1, and this is the one place the same forecast is observable at two horizons.
    """
    scored = joined[joined["outcome"].isin(OUTCOMES)]
    if scored.empty or "lead" not in scored.columns:
        return []
    keys = ["game_date", "team", "name_key"]
    early = scored[scored["lead"] == 1][keys + ["designation"]]
    late = scored[scored["lead"] == 0][keys + ["designation", "outcome"]]
    if early.empty or late.empty:
        return []
    pair = early.merge(late, on=keys, how="inner", suffixes=("_early", "_late"))
    rows = [_row("revision", "all", "paired_rows", float(len(pair)), len(pair))]
    for designation in DESIGNATIONS:
        sub = pair[pair["designation_early"] == designation]
        n = len(sub)
        rows.append(_row("revision", designation, "n", float(n), n))
        if not n:
            continue
        rows.append(_row("revision", designation, "p_unchanged",
                         float((sub["designation_late"] == designation).mean()), n))
        rows.append(_row("revision", designation, "p_played",
                         float((sub["outcome"] == "played").mean()), n))
    return rows


def by_reason(joined: pd.DataFrame, min_n: int = 30) -> list[dict]:
    """`P(outcome | designation, reason_category)` — the stated reason as a modifier.

    This is where the usable structure is, not in the designation alone: the five-level
    scale is **not monotone** at the top (`Available` plays less often than `Probable`)
    because the designations carry different reason mixes, and G-League assignment is the
    bulk of the difference. Conditioning here is what restores the ordering.

    Cells thinner than `min_n` are dropped rather than reported at whatever precision a
    handful of rows gives; the surviving cell count is itself reported.
    """
    scored = joined[joined["outcome"].isin(OUTCOMES)]
    rows: list[dict] = []
    if scored.empty or "reason_category" not in scored.columns:
        return rows
    kept = 0
    for (designation, reason), sub in scored.groupby(["designation", "reason_category"]):
        if len(sub) < min_n or not str(reason).strip():
            continue
        kept += 1
        key = f"{designation}|{reason}"
        rows.append(_row("by_reason", key, "n", float(len(sub)), len(sub)))
        counts = sub["outcome"].value_counts()
        for outcome in OUTCOMES:
            rows.append(_row("by_reason", key, f"p_{outcome}",
                             float(counts.get(outcome, 0)) / len(sub), len(sub)))
    rows.insert(0, _row("by_reason", "all", "cells_reported", float(kept), len(scored)))

    # The monotonicity check, reported as a number so a future re-run cannot quietly lose
    # it. G League is the contaminating reason; excluding it is the like-for-like scale.
    health = scored[scored["reason_category"] != "G League"]
    for designation in DESIGNATIONS:
        sub = health[health["designation"] == designation]
        if len(sub) < min_n:
            continue
        rows.append(_row("by_reason", f"{designation}|ex_gleague", "p_played",
                         float((sub["outcome"] == "played").mean()), len(sub)))
    return rows


# ── Entry point ───────────────────────────────────────────────────────────────

def measure(joined: pd.DataFrame, min_reason_n: int = 30) -> pd.DataFrame:
    rows = (coverage(joined) + calibration(joined) + revision(joined)
            + by_reason(joined, min_reason_n))
    return pd.DataFrame(rows)[["measurement", "key", "metric", "n", "value"]]


def transfer_table(joined: pd.DataFrame, min_n: int = 30) -> pd.DataFrame:
    """The artifact the model consumes: `P(outcome | designation, reason)` with its n.

    Keyed on (designation, reason_category), with `reason_category = ""` for the marginal
    over reasons, because the marginal is what a caller falls back to when the snapshot
    states a reason too rare to have its own cell. **Prefer the conditioned row where one
    exists** — the marginal is not monotone in the designation scale and the conditioned
    rows are.

    Kept separate from the long report on purpose: `outputs/eda/` is for reading,
    `data/features/` is for downstream code, per `CLAUDE.md`.
    """
    scored = joined[joined["outcome"].isin(OUTCOMES)]

    def _cell(sub: pd.DataFrame, designation: str, reason: str) -> dict:
        n = len(sub)
        counts = sub["outcome"].value_counts()
        row = {"designation": designation, "reason_category": reason, "n": n}
        for outcome in OUTCOMES:
            row[f"p_{outcome}"] = float(counts.get(outcome, 0)) / n if n else float("nan")
        played = sub[sub["outcome"] == "played"]
        row["mean_min_given_played"] = (float(played["min"].map(parse_minutes).mean())
                                        if len(played) else float("nan"))
        return row

    rows = [_cell(scored[scored["designation"] == d], d, "") for d in DESIGNATIONS]
    for (designation, reason), sub in scored.groupby(["designation", "reason_category"]):
        if len(sub) >= min_n and str(reason).strip():
            rows.append(_cell(sub, designation, reason))
    return pd.DataFrame(rows).sort_values(["designation", "reason_category"],
                                          ignore_index=True)


def run(cfg: dict) -> Path:
    raw_dir = Path(cfg["data"]["raw_dir"])
    out_dir = Path(cfg["eda"]["output_dir"])
    features_dir = Path(cfg["data"]["features_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    features_dir.mkdir(parents=True, exist_ok=True)
    seasons = cfg["data"]["seasons"]
    cfg_rc = cfg["eda"].get("report_calibration", {})
    min_reason_n = cfg_rc.get("min_reason_cell", 30)

    reports = report_frame(load_log(raw_dir / "injury_reports"))
    print(f"Injury-report archive: {len(reports):,} player-report rows with a designation, "
          f"{reports['report_date'].nunique() if len(reports) else 0} report days")

    games = game_index(seasons, raw_dir)
    outcomes = outcome_frame(seasons, raw_dir)
    print(f"Box-score outcomes: {len(outcomes):,} player-game statuses over "
          f"{outcomes['game_id'].nunique() if len(outcomes) else 0:,} backfilled games")

    joined = join_outcomes(reports, games, outcomes)
    table = measure(joined, min_reason_n)

    def pick(measurement: str, key: str, metric: str) -> float:
        m = table[(table.measurement == measurement) & (table.key == key)
                  & (table.metric == metric)]
        return float(m["value"].iloc[0]) if len(m) else float("nan")

    scored = int(pick("coverage", "all", "scored_rows"))
    print(f"\nJoined {scored:,} of {len(joined):,} report rows "
          f"({pick('coverage', 'all', 'match_rate'):.1%}) over "
          f"{pick('coverage', 'all', 'game_dates'):.0f} game dates, "
          f"{pick('coverage', 'all', 'players'):.0f} players")
    print(f"  dropped: {pick('coverage', 'all', 'share_uncovered'):.1%} uncovered "
          f"(no backfilled box score — Summer League and any gap), "
          f"{pick('coverage', 'all', 'share_unmatched'):.1%} unmatched name")

    if scored:
        print("\nP(outcome | designation) — the transfer function:")
        wide = (table[(table.measurement == "calibration")
                      & (table.key.str.endswith("|all"))]
                .assign(designation=lambda d: d.key.str.split("|").str[0])
                .pivot_table(index="designation", columns="metric", values="value")
                .reindex(DESIGNATIONS))
        cols = ["n", "p_played", "p_dnp", "p_inactive", "p_absent",
                "mean_min_given_played"]
        print(wide[[c for c in cols if c in wide.columns]].round(3).to_string())
        print("  `absent` is a real outcome — listed on the report, not on that game's "
              "roster.\n  `unmatched` names never reach this table; they are in coverage "
              "above.")

        rev = table[table.measurement == "revision"]
        if not rev.empty and pick("revision", "all", "paired_rows") > 0:
            print(f"\nHow a day-before designation is revised "
                  f"({pick('revision', 'all', 'paired_rows'):,.0f} paired rows):")
            rw = (rev[rev.key != "all"]
                  .pivot_table(index="key", columns="metric", values="value")
                  .reindex([d for d in DESIGNATIONS if d in set(rev.key)]))
            print(rw.round(3).to_string())
            print("  The closest handle the archive has on how a stale designation decays — "
                  "which is\n  the question a preseason snapshot poses, read weeks rather "
                  "than hours ahead.")

        reasons = table[(table.measurement == "by_reason") & (table.key != "all")
                        & (~table.key.str.endswith("|ex_gleague"))]
        if not reasons.empty:
            print("\nP(outcome | designation, stated reason) — cells with n >= "
                  f"{min_reason_n}:")
            rr = (reasons.pivot_table(index="key", columns="metric", values="value")
                  .sort_values("p_played"))
            cols = ["n", "p_played", "p_dnp", "p_inactive", "p_absent"]
            print(rr[[c for c in cols if c in rr.columns]].round(3).to_string())

        ex = table[(table.measurement == "by_reason")
                   & (table.key.str.endswith("|ex_gleague"))]
        if not ex.empty:
            print("\nThe designation scale is NOT monotone; conditioning on reason fixes "
                  "it:")
            marg = {d: pick("calibration", f"{d}|all", "p_played") for d in DESIGNATIONS}
            exg = {r.key.split("|")[0]: r.value for r in ex.itertuples()}
            for d in DESIGNATIONS:
                if d in exg:
                    print(f"  {d:<13} p_play {marg[d]:.3f} marginal  "
                          f"{exg[d]:.3f} excluding G League")
            print("  `Available` plays LESS often than `Probable` on the raw scale. Most "
                  "of that is\n  reason mix — G-League assignment — and what survives is "
                  "inside noise, so treat\n  Probable and Available as one designation "
                  "rather than as an ordered pair.")

    dest = out_dir / "report_calibration.csv"
    table.to_csv(dest, index=False)
    print(f"\nReport calibration: {len(table):,} measurement rows → {dest}")

    transfer = transfer_table(joined, min_reason_n)
    tdest = features_dir / "report_transfer.parquet"
    transfer.to_parquet(tdest, index=False)
    print(f"Transfer table: {len(transfer):,} designations → {tdest}")
    return dest


if __name__ == "__main__":
    cfg = yaml.safe_load(open("configs/default.yaml"))
    from src.eda.report_calibration import run as _run

    _run(cfg)
