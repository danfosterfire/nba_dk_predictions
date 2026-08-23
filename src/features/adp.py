"""Build the ADP panel: DraftKings and consensus boards, joined and point-in-time safe.

One row per (snapshot, player, source_detail). Consumes what
`src/data/adp_draftkings.py` and `src/data/adp_fantasypros.py` write, joins both to
`nba_api` `player_id`, and attaches the dating a backtest needs to use them honestly.

## Point-in-time discipline — the reason this module exists rather than a `concat`

ADP is a market forecast of the same target the model predicts, so it is the most
leakage-prone input in the project. Three distinct dates have to be kept apart, and the
FantasyPros source carries all three **in one table**:

| column | meaning |
|---|---|
| `season` | the season the ADP is drafting for (flip-detected upstream) |
| `season_start_date` | when that season's first game was played |
| `as_of_date` | when the board was **observed** |

A mid-season capture of a frozen board — e.g. 2025-01-19 serving 2024-25 ADP — carries a
*preseason* value observed *after* the season began. The value is legitimately preseason
information; the observation is not. Rather than guess, both facts are recorded
(`snapshot_lag_days`, `captured_before_season_start`) and `training_rows` is the one
supported read for anything that feeds a model or a backtest.

`status_at_capture` and `team` on the FantasyPros side describe `as_of_date`, **not** the
draft date, and must never be joined to a preseason injury feature. See
`src/data/adp_fantasypros.py`.

Usage:
    python -m src.features.adp
"""

import argparse
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.data.adp_draftkings import (
    NAME_ALIASES,
    RECENT_SEASONS,
    _prefix_match,
    _reversed_key,
)
from src.data.fetch import nbastats_dir
from src.data.preprocess import REGULAR_SEASON, _parse_log_filename

# Season start dates come from the game logs rather than a hardcoded calendar, because the
# calendar is wrong twice in the sample: 2019-20 and 2020-21 were both COVID-shifted, and
# 2020-21 tipped off on 2020-12-22 rather than in October.
_GAMELOG_GLOB = "game_logs_*.csv"

PANEL_COLS = ["season", "season_start_date", "as_of_date", "snapshot_lag_days",
              "captured_before_season_start", "snapshot_source", "source_detail",
              "player_id", "dk_player_id", "player_name", "player_key", "team",
              "position", "adp", "adp_rank", "adp_censored", "pool_size",
              "status_at_capture", "match_method"]


def normalize_name(name: str) -> str:
    """Fold a player name to a join key: ASCII, no punctuation, no suffix, lowercase."""
    n = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    n = re.sub(r"\b(Jr|Sr|II|III|IV|V)\.?\b", "", n)
    n = re.sub(r"[^A-Za-z ]", "", n)
    return re.sub(r"\s+", " ", n).strip().lower()


def season_start_dates(raw_dir: str | Path) -> dict[str, str]:
    """First **regular-season** game date per season, from the raw game logs.

    The season type is resolved by `preprocess._parse_log_filename` rather than by a
    substring test on the filename, because this date is the freeze line every ADP
    snapshot is judged legal or illegal against. A preseason file carries the same
    `SEASON_YEAR` as its regular season and starts ~3 weeks earlier, so classifying it
    by anything less exact would move the opener backwards and silently re-decide which
    captures are point-in-time safe.
    """
    starts: dict[str, str] = {}
    for path in sorted(nbastats_dir(raw_dir).glob(_GAMELOG_GLOB)):
        if _parse_log_filename(path.stem)[0] != REGULAR_SEASON:
            continue
        try:
            df = pd.read_csv(path, usecols=["SEASON_YEAR", "GAME_DATE"])
        except (ValueError, pd.errors.EmptyDataError):
            continue
        for season, g in df.groupby("SEASON_YEAR"):
            day = pd.to_datetime(g["GAME_DATE"]).min().date().isoformat()
            starts[season] = min(starts.get(season, day), day)
    return starts


# ── Joining consensus names to player_id ──────────────────────────────────────

def match_players(frame: pd.DataFrame, roster: pd.DataFrame) -> pd.DataFrame:
    """Attach `player_id` to a consensus board.

    The same cascade `src/data/adp_draftkings.py::build_id_map` uses, with its primitives
    imported rather than reimplemented, plus a season-aware first step: a name is matched
    against the roster *for that season* before widening, which disambiguates recurring
    names with no fuzzy logic at all.

    The fuzzy steps carry the same era guard, keyed on the **board's** season rather than
    the newest season on file, since a 2014-15 board should resolve to players active
    around 2014-15. Without it, prefix matching invents people — see the comment on
    `RECENT_SEASONS`.

    Unmatched rows keep a null `player_id` and say so; they are never dropped, because a
    silent drop would bias the panel toward long-tenured players.
    """
    ref = roster.copy()
    ref["player_key"] = ref["player_name"].map(normalize_name)
    ref["_yr"] = ref["season"].str.slice(0, 4).astype(int)

    out = frame.copy()
    out["player_key"] = out["player_name"].map(normalize_name)
    out["_yr"] = out["season"].str.slice(0, 4).astype(int)

    by_season = ref.drop_duplicates(["season", "player_key"])[
        ["season", "player_key", "player_id"]]
    out = out.merge(by_season, on=["season", "player_key"], how="left")
    out["match_method"] = out["player_id"].notna().map({True: "name+season", False: ""})

    any_season = (ref.sort_values("season").drop_duplicates("player_key", keep="last")
                  [["player_key", "player_id"]].rename(columns={"player_id": "_pid"}))
    out = out.merge(any_season, on="player_key", how="left")
    fill = out["player_id"].isna() & out["_pid"].notna()
    out.loc[fill, "player_id"] = out.loc[fill, "_pid"]
    out.loc[fill, "match_method"] = "name"
    out = out.drop(columns=["_pid"])

    # Candidates carry their **whole career span**, not one season. Keying the era guard on
    # a single season silently narrows it to that year: `Enes Kanter` → `Enes Freedom`
    # (first seen 2011-12) then resolves on a 2014-15 board and fails on a 2018-19 one,
    # even though he played both.
    span = ref.groupby(["player_key", "player_id"])["_yr"].agg(["min", "max"])
    by_key: dict[str, list[tuple[int, int, int]]] = {}
    by_surname: dict[str, list[tuple[str, int, int, int]]] = {}
    for (key, pid), (lo, hi) in span.iterrows():
        by_key.setdefault(key, []).append((pid, lo, hi))
        parts = key.split()
        if len(parts) >= 2:
            by_surname.setdefault(" ".join(parts[1:]), []).append((parts[0], pid, lo, hi))

    aliases = {normalize_name(k): normalize_name(v) for k, v in NAME_ALIASES.items()}

    def _near(hits, yr):
        """The single candidate whose career comes within RECENT_SEASONS of the board."""
        near = {pid for pid, lo, hi in hits
                if lo - RECENT_SEASONS <= yr <= hi + RECENT_SEASONS}
        return next(iter(near)) if len(near) == 1 else None

    def _resolve(key: str, yr: int) -> tuple[int | None, str]:
        alias = aliases.get(key)
        if alias and alias in by_key:
            hit = _near(by_key[alias], yr)
            if hit is not None:
                return hit, "alias"
        rev = _reversed_key(key)
        if rev in by_key:
            hit = _near(by_key[rev], yr)
            if hit is not None:
                return hit, "reversed"
        parts = key.split()
        if len(parts) >= 2:
            first, last = parts[0], " ".join(parts[1:])
            hits = [(pid, lo, hi) for f, pid, lo, hi in by_surname.get(last, [])
                    if _prefix_match(first, f)]
            hit = _near(hits, yr)
            if hit is not None:
                return hit, "prefix"
        return None, "unmatched"

    missing = out["player_id"].isna()
    if missing.any():
        idx = out.index[missing]
        resolved = [_resolve(k, y) for k, y in
                    zip(out.loc[missing, "player_key"], out.loc[missing, "_yr"])]
        # Built as a float Series rather than assigned from a list: a list holding None
        # against a float64 column raises on modern pandas instead of coercing to NaN.
        out.loc[idx, "player_id"] = pd.Series([r[0] for r in resolved],
                                              index=idx, dtype="float64")
        out.loc[idx, "match_method"] = pd.Series([r[1] for r in resolved], index=idx)
    return out.drop(columns=["_yr"])


# ── Point-in-time ─────────────────────────────────────────────────────────────

def attach_dating(frame: pd.DataFrame, starts: dict[str, str]) -> pd.DataFrame:
    """Add `season_start_date`, `snapshot_lag_days` and the before-start flag."""
    out = frame.copy()
    out["season_start_date"] = out["season"].map(starts)
    lag = (pd.to_datetime(out["as_of_date"]) -
           pd.to_datetime(out["season_start_date"])).dt.days
    out["snapshot_lag_days"] = lag
    # Unknown season start (a season with no game logs yet — e.g. an upcoming one) is
    # treated as "before", since a board for a season that has not been played cannot
    # postdate it.
    out["captured_before_season_start"] = lag.isna() | (lag <= 0)
    return out


def training_rows(panel: pd.DataFrame) -> pd.DataFrame:
    """The only supported read for anything that feeds a model or a backtest.

    Keeps observations made at or before the season's first game. A frozen board observed
    mid-season still carries a preseason *value*, but this project's rule is that a
    historical row is filled only from a source dated at or before the prediction date —
    so the qualifying observation, not the qualifying value, is what counts.
    """
    return panel[panel["captured_before_season_start"]].copy()


def assert_point_in_time(panel: pd.DataFrame) -> None:
    """Raise if any row would hand a season information observed after it began."""
    bad = panel[~panel["captured_before_season_start"]]
    if len(bad):
        worst = bad.nlargest(1, "snapshot_lag_days").iloc[0]
        raise AssertionError(
            f"{len(bad):,} ADP rows were observed after their season began — e.g. "
            f"{worst.player_name} ({worst.season}) captured {worst.as_of_date}, "
            f"{int(worst.snapshot_lag_days)} days after {worst.season_start_date}. "
            f"Filter with `training_rows` before using this for training.")


# ── Name-match audit ──────────────────────────────────────────────────────────
#
# `CLAUDE.md` records this as the most dangerous measurement in the repo: **an unmatched rate
# is monotonically increasing in the error it is supposed to detect.** Every fabricated match
# improves it. The rejected "same surname + same first initial" rule scored 0.0% unmatched
# while inventing `Cameron Boozer` → Carlos Boozer, `RJ Davis` → Ricky Davis and nine more.
#
# The standing rule that follows — "list every non-exact match and read them" — had no
# artifact. This section is that artifact, and it does two things a coverage number cannot:
#
# 1. **Emits every surviving fuzzy match with both names**, so the tier stays small enough to
#    eyeball. A fuzzy tier too large to read is a fuzzy tier too large to trust.
# 2. **Keeps the rejected rule running forever, as an ablation.** It sits next to the metric
#    it discredits, so if someone later "simplifies" the cascade back toward it, the audit
#    says so on the next run rather than after the next modelling decision.

# Methods that require the normalized name to match exactly. Everything else is fuzzy and
# gets listed by name. `roster_snapshot` is exact — it is a normalized-name equality inside
# the board's own season's roster file, with an ambiguous key yielding nothing — so it is
# not a tier that needs reading, and counting it as fuzzy would grow the read-it-all list
# by 71 rows that no rule guessed at.
EXACT_METHODS = ("name+team", "name+season", "name", "roster_snapshot")
FUZZY_METHODS = ("reversed", "prefix", "alias")

# Not defects: a board row for a player who has never played an NBA game has no `player_id`
# to find. Kept apart from `unmatched` per the standing rule.
#
# **Revisited for `docs/rookie-rates-plan.md` §5g and left as it is.** The roster-snapshot
# tier gives the never-played *rostered* player an id, which is what a rookie design row
# needs to join to — so the class shrinks from 162 to 91 on the two real boards. The
# residual is still not a defect: it is DK's deep pool below the 577-player snapshot,
# carrying no ADP, and no preseason-legal source holds an id for a player nobody has
# rostered. Promoting it to `unmatched` would put a permanent 91-row failure on a join that
# has nothing left to find.
NON_DEFECT_METHODS = ("no_nba_history",)


def _roster_index(roster: pd.DataFrame) -> pd.DataFrame:
    """Per player_id: normalized name, display name, and first/last season seen."""
    ref = roster.copy()
    ref["player_key"] = ref["player_name"].map(normalize_name)
    ref["_yr"] = ref["season"].str.slice(0, 4).astype(int)
    return (ref.groupby("player_id")
            .agg(roster_name=("player_name", "last"),
                 roster_key=("player_key", "last"),
                 first_year=("_yr", "min"), last_year=("_yr", "max"))
            .reset_index())


def surname_initial_matches(keys: pd.Series, roster: pd.DataFrame) -> pd.Series:
    """The **rejected** rule, kept runnable: same surname, same first initial.

    No prefix requirement and no era guard — the two independent guards the real cascade
    uses. Ties are broken toward the most recently seen candidate, which is the charitable
    reading of the rule; it still fabricates, which is the point.

    Returns `player_id` per input key, NaN where even this rule finds nothing.
    """
    idx = _roster_index(roster)
    buckets: dict[tuple[str, str], list[tuple[int, int]]] = {}
    for r in idx.itertuples():
        parts = str(r.roster_key).split()
        if len(parts) >= 2:
            buckets.setdefault((" ".join(parts[1:]), parts[0][:1]), []).append(
                (r.last_year, r.player_id))

    def _hit(key: str):
        parts = str(key).split()
        if len(parts) < 2:
            return np.nan
        candidates = buckets.get((" ".join(parts[1:]), parts[0][:1]))
        return max(candidates)[1] if candidates else np.nan

    return keys.map(_hit)


def match_audit(panel: pd.DataFrame, roster: pd.DataFrame) -> pd.DataFrame:
    """Every surviving fuzzy match, the rejected rule's extra matches, and the counts.

    One row per (source, player_key, season) — the panel carries a row per snapshot, and the
    same name resolved the same way in forty snapshots is one match to read, not forty.
    """
    rows = panel.dropna(subset=["player_key"]).copy()
    rows["match_method"] = rows["match_method"].fillna("unmatched").replace("", "unmatched")
    # A panel whose every row is unmatched carries `player_id` as `object`, which will not
    # merge against the roster's int64. Coerce once here rather than at each join.
    rows["player_id"] = pd.to_numeric(rows["player_id"], errors="coerce")
    unique = (rows.sort_values("as_of_date")
              .drop_duplicates(subset=["snapshot_source", "player_key", "season"],
                              keep="last"))
    idx = _roster_index(roster)
    unique = unique.merge(idx, on="player_id", how="left")
    board_year = unique["season"].str.slice(0, 4).astype(int)
    # Distance from the board's season to the nearest season the matched player was on a
    # roster. 0 means he was active that year; a large gap is what invents people.
    unique["seasons_apart"] = np.where(
        unique["first_year"].isna(), np.nan,
        np.maximum(0, np.maximum(unique["first_year"] - board_year,
                                 board_year - unique["last_year"])))

    def _row(section: str, **kw) -> dict:
        base = {"section": section, "source": "", "rule": "", "board_name": "",
                "matched_name": "", "board_season": "", "seasons_apart": np.nan,
                "player_id": np.nan, "metric": "", "value": np.nan}
        return base | kw

    out = []
    fuzzy = unique[unique["match_method"].isin(FUZZY_METHODS)]
    for r in fuzzy.sort_values(["match_method", "player_name"]).itertuples():
        out.append(_row("fuzzy_match", source=r.snapshot_source, rule=r.match_method,
                        board_name=r.player_name, matched_name=r.roster_name,
                        board_season=r.season, seasons_apart=r.seasons_apart,
                        player_id=r.player_id))

    # ── the rejected rule, re-run on the rows exact matching could not resolve ──
    needs_fuzzy = unique[~unique["match_method"].isin(EXACT_METHODS)]
    ablation = needs_fuzzy.assign(
        ablation_id=surname_initial_matches(needs_fuzzy["player_key"], roster))
    hit = ablation[ablation["ablation_id"].notna()].merge(
        idx.rename(columns={"player_id": "ablation_id",
                            "roster_name": "ablation_name",
                            "last_year": "ablation_last_year"})[
            ["ablation_id", "ablation_name", "ablation_last_year"]],
        on="ablation_id", how="left")
    for r in hit.itertuples():
        agrees = (not pd.isna(r.player_id)) and r.player_id == r.ablation_id
        out.append(_row("ablation_match", source=r.snapshot_source,
                        rule="surname_initial", board_name=r.player_name,
                        matched_name=r.ablation_name, board_season=r.season,
                        seasons_apart=float(abs(int(str(r.season)[:4])
                                                - r.ablation_last_year))
                        if not pd.isna(r.ablation_last_year) else np.nan,
                        player_id=r.ablation_id,
                        metric="agrees_with_cascade", value=float(agrees)))

    # ── the counts, including the perverse one ────────────────────────────────
    n = len(unique)
    matchable = unique[~unique["match_method"].isin(NON_DEFECT_METHODS)]
    cascade_unmatched = int((matchable["match_method"] == "unmatched").sum())
    false_matches = int((~hit.apply(
        lambda r: (not pd.isna(r["player_id"])) and r["player_id"] == r["ablation_id"],
        axis=1)).sum()) if len(hit) else 0
    ablation_unmatched = max(cascade_unmatched - len(hit), 0)
    counts = {
        "rows_audited": float(n),
        "exact_matches": float(unique["match_method"].isin(EXACT_METHODS).sum()),
        "fuzzy_matches": float(len(fuzzy)),
        "unmatched": float(cascade_unmatched),
        "no_nba_history": float((unique["match_method"] == "no_nba_history").sum()),
        "matchable_rows": float(len(matchable)),
        "unmatched_rate_cascade": (cascade_unmatched / len(matchable)
                                   if len(matchable) else np.nan),
        "ablation_matches": float(len(hit)),
        "ablation_false_matches": float(false_matches),
        "unmatched_rate_surname_initial": (ablation_unmatched / len(matchable)
                                           if len(matchable) else np.nan),
    }
    for metric, value in counts.items():
        out.append(_row("summary", rule="surname_initial" if "surname" in metric
                        or "ablation" in metric else "cascade",
                        metric=metric, value=value))
    return pd.DataFrame(out)


# ── Build ─────────────────────────────────────────────────────────────────────

def build(features_dir: str | Path = "data/features",
          raw_dir: str | Path = "data/raw") -> pd.DataFrame:
    features_dir, raw_dir = Path(features_dir), Path(raw_dir)
    roster_path = features_dir / "season_matrix_roster_tierA.parquet"
    roster = pd.read_parquet(roster_path,
                             columns=["player_id", "player_name", "season"])
    starts = season_start_dates(raw_dir)

    frames = []

    dk_path = features_dir / "adp_draftkings.parquet"
    if dk_path.exists():
        dk = pd.read_parquet(dk_path).rename(columns={"capture_date": "as_of_date"})
        id_map = pd.read_parquet(features_dir / "adp_dk_id_map.parquet")
        dk = dk.merge(id_map[["dk_player_id", "player_id", "match_method", "key"]],
                      on="dk_player_id", how="left").rename(columns={"key": "player_key"})
        dk["status_at_capture"] = ""
        frames.append(dk)

    fp_path = features_dir / "adp_fantasypros.parquet"
    if fp_path.exists():
        fp = pd.read_parquet(fp_path).rename(columns={"capture_date": "as_of_date"})
        fp = fp[fp["season"].astype(bool)]
        fp = match_players(fp, roster)
        fp["dk_player_id"] = pd.NA
        fp["position"] = fp["positions"]
        fp["adp_censored"] = False
        fp["pool_size"] = fp.groupby("as_of_date")["player_name"].transform("size")
        frames.append(fp)

    if not frames:
        raise FileNotFoundError(
            "No ADP sources found. Run `make adp-draftkings` and/or `make adp-fantasypros`.")

    panel = pd.concat(frames, ignore_index=True)
    panel = attach_dating(panel, starts)
    for col in PANEL_COLS:
        if col not in panel.columns:
            panel[col] = pd.NA
    return panel[PANEL_COLS].sort_values(
        ["season", "as_of_date", "source_detail", "adp"], na_position="last"
    ).reset_index(drop=True)


def run(features_dir: str | Path = "data/features",
        raw_dir: str | Path = "data/raw",
        eda_dir: str | Path = "outputs/eda") -> pd.DataFrame:
    panel = build(features_dir, raw_dir)
    dest = Path(features_dir)
    dest.mkdir(parents=True, exist_ok=True)
    out = dest / "adp_panel.parquet"
    panel.to_parquet(out, index=False)

    train = training_rows(panel)
    assert_point_in_time(train)

    seasons = sorted(panel["season"].dropna().unique())
    with_adp = panel["adp"].notna()
    matchable = panel[with_adp & (panel["match_method"] != "no_nba_history")]
    unmatched = int((matchable["match_method"] == "unmatched").sum())
    print(f"  Built {len(panel):,} ADP rows over {len(seasons)} seasons "
          f"({seasons[0]} → {seasons[-1]}) → {out}")
    print(f"  {int(with_adp.sum()):,} carry an ADP; {unmatched:,} of "
          f"{len(matchable):,} matchable rows unmatched "
          f"({unmatched / max(len(matchable), 1):.2%})")
    print(f"  {len(train):,} rows observed before their season began "
          f"({len(train) / max(len(panel), 1):.1%}) — the point-in-time-safe subset")
    by_source = panel.groupby(["snapshot_source", "source_detail"]).size()
    for (src, detail), n in by_source.items():
        print(f"    {src:12s} {detail:8s} {n:7,}")

    # ── the name-match audit ─────────────────────────────────────────────────
    roster = pd.read_parquet(Path(features_dir) / "season_matrix_roster_tierA.parquet",
                             columns=["player_id", "player_name", "season"])
    audit = match_audit(panel, roster)
    eda_dest = Path(eda_dir)
    eda_dest.mkdir(parents=True, exist_ok=True)
    audit_path = eda_dest / "adp_match_audit.csv"
    audit.to_csv(audit_path, index=False)

    counts = audit[audit["section"] == "summary"].set_index("metric")["value"]
    fuzzy = audit[audit["section"] == "fuzzy_match"]
    print(f"\n  Name-match audit — {int(counts['rows_audited']):,} unique "
          f"(source, name, season) rows:")
    print(f"    exact {int(counts['exact_matches']):,}   "
          f"fuzzy {int(counts['fuzzy_matches']):,}   "
          f"unmatched {int(counts['unmatched']):,}   "
          f"no_nba_history {int(counts['no_nba_history']):,} (not a defect)")
    if len(fuzzy):
        print(f"    every non-exact match, to be read rather than trusted "
              f"({len(fuzzy)} of them):")
        for r in fuzzy.itertuples():
            gap = "?" if pd.isna(r.seasons_apart) else f"{r.seasons_apart:+.0f}"
            print(f"      [{r.rule:8s}] {r.board_name:<24} → {r.matched_name:<22}"
                  f" {r.board_season}  {gap} seasons  {r.source}")
    print(f"\n    THE ABLATION — the rejected 'same surname + same first initial' rule, "
          "re-run:")
    print(f"      it makes {int(counts['ablation_matches']):,} matches, of which "
          f"{int(counts['ablation_false_matches']):,} disagree with the cascade")
    print(f"      and it scores {counts['unmatched_rate_surname_initial']:.2%} unmatched "
          f"against the cascade's {counts['unmatched_rate_cascade']:.2%}")
    print("      A LOWER unmatched rate on a rule that fabricates matches is the whole "
          "warning:\n      the metric is monotonically increasing in the error it is "
          "supposed to detect.")
    bad = audit[(audit["section"] == "ablation_match") & (audit["value"] == 0.0)]
    for r in bad.head(12).itertuples():
        print(f"        invents: {r.board_name:<24} → {r.matched_name}")
    print(f"→ {audit_path}  ({len(audit):,} rows)")
    return panel


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the ADP panel.")
    parser.add_argument("--features-dir", default=None)
    args = parser.parse_args()

    cfg = yaml.safe_load(open("configs/default.yaml"))
    data_cfg = cfg.get("data", {})
    run(features_dir=args.features_dir or data_cfg.get("features_dir", "data/features"),
        raw_dir=data_cfg.get("raw_dir", "data/raw"),
        eda_dir=cfg.get("eda", {}).get("output_dir", "outputs/eda"))
