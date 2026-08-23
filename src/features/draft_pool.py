"""The draft pool — who is draftable, where he can be slotted, and what the field thinks.

One row per `(season, player_id)`: his team, his **DK position eligibility**, his ADP, and
the prior-season key every model head scores him from. This is the board
`src/sim/draft.py` picks from and `dashboard/draft_room.py` renders, and it is the one
artifact in the simulation layer that decides *which* players exist at all.

Position eligibility is the load-bearing half. A best-ball week starts 2 G / 2 F / 1 C and
2 UTIL out of 16 (`docs/dk_best_ball_rules.md`), so eligibility decides which of the seven
slots a player can occupy — and with it every weekly maximum, every round total and every
advancement cut. Get it wrong and nothing downstream complains; the lineups simply fill
differently and every simulated score is quietly off.

## 🔴 DraftKings is SINGLE-position, and `docs/simulations-plan.md` assumed otherwise

The plan's "Data the layer needs" says `team_rosters_*.csv` carries `POSITION` "with
DK-shaped dual eligibility (`G-F`, `F-C`, `C-F`, `F-G`)". The first half is true and the
second half is not. **Both DK boards print exactly one of `G` / `F` / `C` for every player
— 1,640 board rows across two seasons, zero duals, zero slashes.** NBA.com hands out a dual
to 18.2% of rostered players; DK hands out none.

That is not a parsing artifact of one file. It reproduces on both boards independently, and
DK's own label is **99.85% stable** across them (1 change in 667 shared ids), so the label is
a settled per-player attribute rather than something the board recomputes.

The two sources are therefore two *opinions*, not a coarse view and a fine view, and the
mapping has to be measured rather than assumed. Joined on the persistent DK id through
`adp_dk_id_map.parquet` — never on a name, per `docs/adp-plan.md` — against the
contemporaneous 2025-26 roster:

| | agreement |
|---|---|
| DK's letter lies inside NBA.com's position set | **92.61%** |
| DK's letter equals NBA.com's **primary** letter | **86.63%** |
| …restricted to NBA singles (`G`, `F`, `C`) | 90.98% |
| …restricted to NBA duals (`G-F`, `F-C`, …) | **100.00%** inside the pair, 67.03% on the primary |

Read the last two rows together, because they say something specific. Where NBA.com commits
to one letter, DK contradicts it **9.0%** of the time. Where NBA.com says "tweener", DK
always picks one of the two it named — but which one is close to a coin toss, and on `G-F`
it is 16 G against 19 F on 35 players. **Every disagreement is between adjacent classes;
there is not one G↔C swap in either board.**

### What ships, and the error it accepts

`position` is a single letter for every row, because that is the shape DK uses. For a season
with a DK board the letter **is** DK's. For every other season it is NBA.com's primary
letter, which the table above prices at 86.6% correct.

The alternative — granting both letters of a dual — was measured and rejected. It is better
on one error and much worse on the other: it never *misses* DK's letter (100% containment)
but it hands a second slot to the 18.2% of players NBA.com calls tweeners, and a spurious
eligibility inflates every lineup it touches. Primary-only misassigns ~13% of players
symmetrically, which is noise in *which* slot a player fills; dual grants systematic extra
flexibility, which is an upward bias in every simulated score and in exactly the direction
that would make a strategy look profitable when it is not. Between symmetric noise and
optimistic bias, take the noise.

`dual_g` / `dual_f` / `dual_c` carry the rejected convention anyway, so the sensitivity run
is a column swap rather than a rebuild.

**A fitted majority map was also rejected**, and deliberately: it scores 87.2% against the
primary rule's 86.6%, and the entire difference is flipping `G-F` to `F` on a 19-vs-16 split.
Fitting a coin toss on 35 observations to buy 0.6 points is not a map, it is noise with a
lookup table.

### The caveat this leaves, stated rather than buried

Backtest seasons get mapped positions (86.6% right) and the production season gets DK's own
(right by definition). So the backtest understates lineup fit relative to the live board.
That is the safe direction — it makes the measured edge conservative — but it means a
strategy tuned on how *awkward* rosters are to fill is tuned on an artifact.

### Is validating against the boards a split read?

The only DK boards that exist are Oct-2025 (a **test** season) and Jul-2026 (the production
season). No train or validation season has one, so the plan's instruction to validate against
them cannot be followed without touching 2025-26.

It is not a split violation, for a reason worth stating rather than waving at: a position
label is not a target-season outcome. It carries nothing about 2025-26 scoring, availability
or minutes, and the guarded frames in `src/models/held_out.py` exist to stop a *score* being
read, not a roster attribute. The audit nevertheless reports the same measurement against
train-only roster rows, which is what makes the argument checkable rather than asserted:
restricted to rosters through 2021-22 the 2026-27 board reads **95.42%** containment and
**88.55%** primary agreement against the contemporaneous **92.61%** / **86.63%**. The finding
does not come from the held-out rows and does not change when they are removed.

## The rest of the row

- **Who is in the pool.** `team_context.season_start_roster` — a player is on team T's
  season-start roster if his first appearance for T falls inside T's first 10 games. That is
  this project's existing definition and it is point-in-time legal;
  `team_rosters_<season>.csv` is **not** an acceptable substitute for membership, because it
  is a *current-status* snapshot (the 2025-26 file carries `HOW_ACQUIRED = "Signed on
  03/04/26"`) and would put February signings in an October draft pool. The roster CSV is
  read for `POSITION` only, which is a static physical attribute rather than a
  season-outcome.
- **2026-27 has no game log**, so its pool is the DK board itself — 942 players with DK's own
  team, position and ADP. That is the production board, and it is the right answer rather
  than a fallback: DK's player pool *is* the draftable set. When `commonteamroster` is
  fetched for 2026-27 its `POSITION` nulls (unsigned and two-way players) resolve through the
  same board, and `position_source` records how many needed it.
- **ADP** comes from `adp_panel.parquet` through `adp.training_rows`, so a row is filled only
  from a board observed at or before the season's first game — the qualifying *observation*,
  not the qualifying value. `adp_dk_scale` additionally puts consensus ADP on DK's scale
  through the isotonic map in `adp_transfer.parquet`, which `docs/adp-plan.md` binds the
  opponent model to use in place of the raw consensus. Both are emitted, because that map is
  fitted on a single anchor and the plan requires backtests to report with and without it.
- **The prior-season key** is `(player_id, prior_start_year)`, which is what
  `component_rates.build_design` joins the season matrix on. Both the qualified matrix
  (`GP >= 20 & MIN >= 10`) and its inclusive twin are checked, because the composition head
  covers players the other heads filter out.

## 🔴 Point-in-time costs four of the nine ADP seasons, and the plan's coverage list is
panel presence rather than legality

`docs/simulations-plan.md` lists ADP coverage as nine seasons. Under `training_rows` only
**five** survive — 2014-15, 2022-23, 2023-24, 2025-26 and 2026-27. In 2017-18, 2018-19,
2019-20 and 2024-25 *every* archived snapshot postdates the season's first game, so the board
that season drafted on was never captured and the frozen value that remains is not admissible.

Two consequences. **Both validation seasons survive**, which is the coverage the realized
backtest needs and the one the task called load-bearing. But the plan's "cheap widening" to
2014-15 / 2017-18 / 2018-19 / 2019-20 loses three of its four extra seasons, so that fallback
is a two-season widening, not a four-season one. And 2024-25 — one of the two test seasons —
carries no legal ADP at all, which item 10's risk readout has to account for.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.data.fetch import _season_start_year, _slug, nbastats_dir
from src.features.adp import assert_point_in_time, training_rows
from src.features.team_context import season_start_roster

# DK's three position classes. The lineup is 2 G / 2 F / 1 C / 2 UTIL, so these are the
# slots eligibility resolves to and there is no fourth.
DK_CLASSES = ("G", "F", "C")

# Where a row's `position` came from, in resolution order. `dk_board` is DK's own letter
# for that season; `nba_roster` is that season's `POSITION` mapped to a single letter;
# the two `_carry` tiers borrow the same player's label from another season, which is
# legitimate precisely because both sources treat position as a static attribute — DK's
# own label moves on 1 of 667 players between boards.
POSITION_SOURCES = ("dk_board", "nba_roster", "nba_roster_carry", "dk_board_carry", "none")

# Roster CSV columns actually used. POSITION is the point of the file; the rest key it.
ROSTER_COLS = ["SEASON", "PLAYER_ID", "PLAYER", "POSITION"]


def _season_label(start_year: int) -> str:
    """The inverse of `fetch._season_start_year` (2013 → '2013-14', 1999 → '1999-00')."""
    return f"{start_year}-{(start_year + 1) % 100:02d}"


# ── Positions ─────────────────────────────────────────────────────────────────

def primary_letter(position_nba: pd.Series) -> pd.Series:
    """NBA.com's `POSITION` mapped to the single DK class, by taking the primary.

    `G-F` means "primarily a guard who also plays forward", so the first token is
    NBA.com's own statement of which class the player mainly occupies. The module
    docstring prices this at 86.63% against DK's board and records why a fitted map was
    not preferred.
    """
    return (position_nba.astype("string").str.strip().str.upper()
            .str.split("-").str[0].where(lambda s: s.isin(DK_CLASSES)))


def position_set(position_nba: pd.Series) -> pd.Series:
    """Every class NBA.com names, as a frozenset — `G-F` → {G, F}, `C` → {C}."""
    def _split(value):
        if not isinstance(value, str):
            return frozenset()
        return frozenset(t for t in value.strip().upper().split("-") if t in DK_CLASSES)

    return position_nba.map(_split)


def eligibility_flags(letters: pd.Series, prefix: str = "pos") -> pd.DataFrame:
    """One boolean column per DK class from a single-letter position series."""
    return pd.DataFrame(
        {f"{prefix}_{c.lower()}": letters.eq(c).fillna(False).to_numpy(dtype=bool)
         for c in DK_CLASSES},
        index=letters.index)


def dual_flags(position_nba: pd.Series, shipped: pd.Series,
               prefix: str = "dual") -> pd.DataFrame:
    """The rejected convention: every class NBA.com names, **union** the shipped letter.

    Kept so the sensitivity run is a column swap. See the module docstring for why it
    does not ship: it never misses DK's letter and grants a spurious second slot to 18%
    of players, and a spurious eligibility biases every simulated lineup upward.

    The union is what makes it a *swap* rather than a different board. NBA.com's set
    alone is not a superset of what ships: it names nothing at all for the 2026 draft
    class (205 rows of the production board), and where DK contradicts a single NBA.com
    letter outright it names the other class. Either case would leave a player eligible
    somewhere he cannot play and, worse, not eligible where he can — so a sensitivity run
    would be measuring two changes at once. Unioning keeps the difference to exactly the
    one thing the convention is about: whether a tweener gets his second slot.
    """
    sets = position_set(position_nba)
    return pd.DataFrame(
        {f"{prefix}_{c.lower()}": (sets.map(lambda s, c=c: c in s)
                                   | shipped.eq(c).fillna(False)).to_numpy(dtype=bool)
         for c in DK_CLASSES},
        index=position_nba.index)


# ── Sources ───────────────────────────────────────────────────────────────────

def load_rosters(seasons: list[str], raw_dir: str | Path) -> pd.DataFrame:
    """`POSITION` per `(season, player_id)` from the roster CSVs.

    Read for the position label only. Membership comes from
    `team_context.season_start_roster`, because these files are current-status snapshots
    and carry mid-season signings — see the module docstring.
    """
    frames = []
    for season in seasons:
        path = nbastats_dir(raw_dir) / f"team_rosters_{_slug(season)}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path, usecols=lambda c: c in ROSTER_COLS)
        df = df.rename(columns={"SEASON": "season", "PLAYER_ID": "player_id",
                                "PLAYER": "roster_name", "POSITION": "position_nba"})
        df["season"] = season
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"no team_rosters_*.csv under {raw_dir}")
    out = pd.concat(frames, ignore_index=True)
    out["player_id"] = out["player_id"].astype("int64")
    # One roster row per player-season. The raw files carry a single duplicate across 30
    # seasons; keeping the first is arbitrary and affects one row.
    return out.drop_duplicates(subset=["season", "player_id"], keep="first")


def load_dk_boards(features_dir: str | Path) -> pd.DataFrame:
    """The DK boards with `player_id` attached through the persistent-id map.

    `docs/adp-plan.md`: the DK `ID` is a stable player key, so this is an id join and
    never a name join.

    **A board row with no `player_id` is kept, not dropped.** 162 of the 942 rows on the
    2026-27 board are players who have never played an NBA game — almost all of them the
    2026 draft class — so `adp_dk_id_map` reported them as `no_nba_history` rather than as
    a join failure. They are nonetheless *draftable*, and several go early: AJ Dybantsa at
    ADP 41.8 is a fourth-round pick. Dropping them would break the draft simulator in a
    way that has nothing to do with whether the model can score them, because the field
    takes them at their ADP regardless, and who is still on the board at pick k is the
    quantity a snake draft turns on.

    Those without an id get a **surrogate, `-dk_player_id`**, which is negative so it can
    never collide with an `nba_api` id and never silently matches a season-matrix row — a
    player with no NBA history has no prior season, and `has_nba_id` says so out loud.

    **71 of the 162 stopped needing the surrogate on 2026-08-22**
    (`docs/rookie-rates-plan.md` §5g): `build_id_map`'s roster-snapshot tier gives a
    never-played *rostered* player his real `nba_api` id, which is what a rookie rate design
    row has to join to before the board can price him. `has_nba_id` is therefore no longer a
    proxy for "has a prior season" — it never promised to be, and the rookie design's own
    population rule (`lag_recovery.classify`) is the thing that says so. The 91 that remain
    are DK's deep pool below the roster snapshot and keep the surrogate.
    """
    features_dir = Path(features_dir)
    board = pd.read_parquet(features_dir / "adp_draftkings.parquet")
    id_map = pd.read_parquet(features_dir / "adp_dk_id_map.parquet")
    id_map = id_map[["dk_player_id", "player_id"]].dropna(subset=["player_id"])
    id_map = id_map.drop_duplicates("dk_player_id")

    out = board.merge(id_map, on="dk_player_id", how="left")
    out["has_nba_id"] = out["player_id"].notna()
    out["player_id"] = (out["player_id"]
                        .fillna(-out["dk_player_id"]).astype("int64"))
    assert (out.loc[out["has_nba_id"], "player_id"] > 0).all(), (
        "an nba_api player_id is not positive, so the negative surrogate can collide")
    out = out.rename(columns={"position": "position_dk", "team": "team_dk"})
    keep = ["season", "player_id", "dk_player_id", "player_name", "team_dk",
            "position_dk", "adp", "capture_date", "has_nba_id"]
    return out[keep].drop_duplicates(subset=["season", "player_id"], keep="last")


def pool_membership(seasons: list[str], raw_dir: str | Path,
                    boards: pd.DataFrame, window_games: int) -> pd.DataFrame:
    """Who is draftable in each season, and for which team.

    Seasons with a game log use `season_start_roster`, the project's existing
    point-in-time definition. A season with no game log has not been played, so its pool
    is the DK board — which for 2026-27 is not a fallback but the authoritative answer,
    since DK's player pool is the draftable set by definition.
    """
    frames = []
    for season in seasons:
        roster = season_start_roster(season, raw_dir, window_games)
        if len(roster):
            frames.append(pd.DataFrame({
                "season": season,
                "player_id": roster["player_id"].astype("int64").to_numpy(),
                "team": roster["team_abbreviation"].to_numpy(),
                "pool_source": "nba_roster",
                # Membership came from a game log, so the id is an nba_api id by
                # construction — the surrogate only ever enters through a board.
                "has_nba_id": True,
            }))
            continue
        board = boards[boards["season"] == season]
        if not len(board):
            continue
        frames.append(pd.DataFrame({
            "season": season,
            "player_id": board["player_id"].to_numpy(),
            "team": board["team_dk"].to_numpy(),
            "pool_source": "dk_board",
            "has_nba_id": board["has_nba_id"].to_numpy(),
        }))
    if not frames:
        raise ValueError("no season produced a draft pool")
    return pd.concat(frames, ignore_index=True)


def resolve_positions(pool: pd.DataFrame, rosters: pd.DataFrame,
                      boards: pd.DataFrame) -> pd.DataFrame:
    """Attach `position`, its source, and both eligibility conventions.

    The cascade, in order: DK's own letter for that season; that season's NBA.com
    `POSITION`; the same player's nearest NBA.com label from another season; DK's letter
    from any board. The two carry tiers exist because both sources treat position as a
    static per-player attribute, and they cover the ~5% of season-start players the
    current-status roster CSV has already dropped by the time it is fetched.
    """
    out = pool.copy()
    ros = rosters[["season", "player_id", "position_nba", "roster_name"]]
    out = out.merge(ros, on=["season", "player_id"], how="left")

    dk_season = boards[["season", "player_id", "position_dk", "player_name"]]
    out = out.merge(dk_season, on=["season", "player_id"], how="left")
    # The DK id is persistent across boards, so it is a property of the player rather
    # than of the season — and it is what the draft room needs to write a pre-draft
    # rankings CSV back to DK, which keys on `ID` and refuses names.
    dk_ids = (boards.sort_values("season").drop_duplicates("player_id", keep="last")
              .set_index("player_id")["dk_player_id"])
    out["dk_player_id"] = out["player_id"].map(dk_ids).astype("Int64")

    # Carry tiers, keyed on the player alone: the nearest roster label in either
    # direction, so a player whose only roster row postdates the pool season still
    # resolves. Later is admissible here because position is a static attribute, not a
    # season outcome — `position_source` records every row that took this path.
    out["_start_year"] = out["season"].map(_season_start_year)
    ros_carry = ros.dropna(subset=["position_nba"]).copy()
    ros_carry["_start_year"] = ros_carry["season"].map(_season_start_year)
    nearest = _nearest_label(out, ros_carry, "position_nba")

    board_carry = (boards.dropna(subset=["position_dk"])
                   .sort_values("season")
                   .drop_duplicates("player_id", keep="last")
                   .set_index("player_id")["position_dk"])

    from_nba = primary_letter(out["position_nba"])
    from_nba_carry = primary_letter(nearest)
    from_dk_carry = out["player_id"].map(board_carry)

    letter = out["position_dk"].astype("string")
    source = pd.Series(np.where(letter.notna(), "dk_board", None),
                       index=out.index, dtype="object")
    for candidate, name in ((from_nba, "nba_roster"),
                            (from_nba_carry, "nba_roster_carry"),
                            (from_dk_carry, "dk_board_carry")):
        fill = letter.isna() & pd.Series(candidate, index=out.index).notna()
        letter = letter.where(~fill, pd.Series(candidate, index=out.index))
        source = source.where(~fill, name)
    source = source.fillna("none")

    # The dual convention reads NBA.com wherever a label exists — that season's or the
    # carried one — because it only means anything for the source that emits duals.
    nba_any = out["position_nba"].where(out["position_nba"].notna(), nearest)

    out["position"] = letter
    out["position_source"] = source
    out["position_nba"] = nba_any
    out["dual_eligible"] = position_set(nba_any).map(len).gt(1).to_numpy(dtype=bool)
    # A row resolved through a carry tier has no name from its own season either, so the
    # name follows the same per-player lookup the position did.
    names = pd.concat([
        rosters.dropna(subset=["roster_name"]).rename(
            columns={"roster_name": "name"})[["player_id", "name"]],
        boards.dropna(subset=["player_name"]).rename(
            columns={"player_name": "name"})[["player_id", "name"]],
    ]).drop_duplicates("player_id").set_index("player_id")["name"]
    out["player_name"] = (out["player_name"].fillna(out["roster_name"])
                          .fillna(out["player_id"].map(names)))
    out = pd.concat([out.drop(columns=["roster_name", "_start_year"]),
                     eligibility_flags(letter),
                     dual_flags(nba_any, letter)], axis=1)
    return out


def _nearest_label(pool: pd.DataFrame, labelled: pd.DataFrame,
                   column: str) -> pd.Series:
    """Each pool row's nearest same-player label from another season.

    Nearest by absolute season distance, breaking ties toward the *earlier* season so a
    backtest row prefers a label it could have known. Position is static enough that this
    is a labelling convenience rather than a point-in-time claim, and `position_source`
    records every row that needed it.
    """
    if not len(labelled):
        return pd.Series(pd.NA, index=pool.index, dtype="object")
    merged = pool[["player_id", "_start_year"]].reset_index().merge(
        labelled[["player_id", "_start_year", column]].rename(
            columns={"_start_year": "_label_year"}),
        on="player_id", how="left")
    merged["_distance"] = (merged["_label_year"] - merged["_start_year"]).abs()
    merged = merged.sort_values(["index", "_distance", "_label_year"])
    best = merged.dropna(subset=[column]).drop_duplicates("index").set_index("index")
    return best[column].reindex(pool.index)


# ── The mapping audit — the measurement this module exists to make ────────────

def position_audit(rosters: pd.DataFrame, boards: pd.DataFrame,
                   roster_windows: dict[str, tuple[str, str]]) -> pd.DataFrame:
    """Agreement between NBA.com's `POSITION` and DK's own, per board and window.

    One row per `(board season, roster window, NBA position)` plus an `ALL` roll-up.
    `roster_windows` names the season ranges compared — the contemporaneous join is the
    headline and the train-only one is what shows the finding does not rest on held-out
    rows.
    """
    rows = []
    for board_season, board in boards.groupby("season"):
        board = board.dropna(subset=["position_dk"])
        for window, (lo, hi) in roster_windows.items():
            pool = rosters[(rosters["season"] >= lo) & (rosters["season"] <= hi)]
            pool = pool.dropna(subset=["position_nba"]).sort_values("season")
            pool = pool.drop_duplicates("player_id", keep="last")
            joined = board.merge(pool[["player_id", "position_nba"]], on="player_id")
            if not len(joined):
                continue
            contemporaneous = lo == hi == board_season
            sets = position_set(joined["position_nba"])
            inside = [d in s for d, s in zip(joined["position_dk"], sets)]
            joined = joined.assign(
                inside=inside,
                on_primary=primary_letter(joined["position_nba"]).eq(
                    joined["position_dk"]).to_numpy())
            for label, grp in list(joined.groupby("position_nba")) + [("ALL", joined)]:
                rows.append({
                    "board_season": board_season,
                    "roster_window": window,
                    "contemporaneous": contemporaneous,
                    "position_nba": label,
                    "n": len(grp),
                    "dk_inside_nba_set": float(grp["inside"].mean()),
                    "dk_equals_nba_primary": float(grp["on_primary"].mean()),
                    "dual": "-" in str(label),
                    **{f"dk_{c.lower()}": int((grp["position_dk"] == c).sum())
                       for c in DK_CLASSES},
                })
    return pd.DataFrame(rows)


def assert_dk_is_single_position(boards: pd.DataFrame) -> None:
    """DK prints one class per player. The whole eligibility design rests on it.

    Asserted rather than reported because it is the claim that reverses the plan: if a
    future board ever carries `G-F`, the single-letter convention below is wrong and this
    must fail loudly rather than silently dropping the second class.
    """
    values = set(boards["position_dk"].dropna().unique())
    bad = values - set(DK_CLASSES)
    assert not bad, (
        f"a DK board carries a position outside {DK_CLASSES}: {sorted(bad)} — the "
        "single-position convention in this module no longer holds")


# ── ADP ───────────────────────────────────────────────────────────────────────

def select_adp(features_dir: str | Path) -> pd.DataFrame:
    """One ADP observation per `(season, player_id)` — the board that season drafted on.

    `training_rows` first, so only boards observed at or before the season's first game
    qualify; then DraftKings ahead of the consensus, and the latest qualifying snapshot
    ahead of an earlier one. A frozen mid-season snapshot still carries a preseason
    *value* and is still excluded, which is `src/features/adp.py`'s rule and the reason
    four of the nine panel seasons contribute nothing here.
    """
    features_dir = Path(features_dir)
    panel = pd.read_parquet(features_dir / "adp_panel.parquet")
    pit = training_rows(panel).dropna(subset=["adp"]).copy()
    assert_point_in_time(pit)
    # The same surrogate `load_dk_boards` assigns, for the same reason: a DK row for a
    # player with no NBA history still carries the ADP the field will draft him at.
    surrogate = pit["player_id"].isna() & pit["dk_player_id"].notna()
    pit.loc[surrogate, "player_id"] = -pit.loc[surrogate, "dk_player_id"]
    pit = pit.dropna(subset=["player_id"])
    pit["player_id"] = pit["player_id"].astype("int64")

    pit["_source_rank"] = np.where(pit["snapshot_source"] == "draftkings", 0, 1)
    pit = pit.sort_values(["season", "player_id", "_source_rank", "as_of_date"],
                          ascending=[True, True, True, False])
    best = pit.drop_duplicates(subset=["season", "player_id"], keep="first")

    out = best.rename(columns={"snapshot_source": "adp_source",
                               "as_of_date": "adp_as_of"})
    cols = ["season", "player_id", "adp", "adp_rank", "adp_censored", "adp_source",
            "adp_as_of", "snapshot_lag_days"]
    return out[cols].rename(columns={"snapshot_lag_days": "adp_lag_days"})


def to_dk_scale(adp: pd.Series, source: pd.Series,
                features_dir: str | Path) -> pd.Series:
    """Consensus ADP put on DK's scale through the fitted isotonic map.

    `docs/adp-plan.md` binds the opponent model to the **recalibrated** consensus rather
    than the raw one: the map is worth 7 picks of cross-validated error and removes a
    category-league bias that would otherwise read as model edge. DK-sourced rows pass
    through unchanged. The map is fitted on a single anchor, which is why the raw `adp`
    ships beside this rather than being replaced by it.
    """
    transfer = pd.read_parquet(Path(features_dir) / "adp_transfer.parquet")
    grid = transfer["adp_consensus"].to_numpy(dtype=float)
    fitted = transfer["adp_dk_fitted"].to_numpy(dtype=float)
    values = adp.to_numpy(dtype=float)
    mapped = np.interp(values, grid, fitted, left=fitted[0], right=fitted[-1])
    out = np.where(source.to_numpy() == "draftkings", values, mapped)
    return pd.Series(np.where(np.isnan(values), np.nan, out), index=adp.index)


# ── The prior-season key ──────────────────────────────────────────────────────

def attach_prior_season(pool: pd.DataFrame, features_dir: str | Path) -> pd.DataFrame:
    """The `(player_id, prior_start_year)` key the model heads score a player from.

    `component_rates.build_design` joins the season matrix on exactly this pair. Both the
    qualified matrix and its inclusive twin are reported: the count and conversion heads
    filter at `GP >= 20 & MIN >= 10`, while the composition head cannot drop anyone
    because a team's minutes must sum, so "has a prior row" has two different answers and
    a consumer needs to know which one it is asking about.
    """
    features_dir = Path(features_dir)
    out = pool.copy()
    out["season_start_year"] = out["season"].map(_season_start_year)
    out["prior_start_year"] = out["season_start_year"] - 1
    out["prior_season"] = out["prior_start_year"].map(_season_label)

    cols = ["player_id", "season_start_year", "gp", "min_total"]
    qualified = pd.read_parquet(features_dir / "season_matrix_tierA.parquet",
                                columns=cols)
    inclusive = pd.read_parquet(features_dir / "season_matrix_roster_tierA.parquet",
                                columns=cols)

    key = ["player_id", "prior_start_year"]
    q_key = set(map(tuple, qualified[["player_id", "season_start_year"]]
                    .astype("int64").to_numpy()))
    pairs = list(map(tuple, out[key].astype("int64").to_numpy()))
    out["prior_in_matrix"] = [p in q_key for p in pairs]

    inc = inclusive.rename(columns={"season_start_year": "prior_start_year",
                                    "gp": "prior_gp", "min_total": "prior_min_total"})
    inc = inc.drop_duplicates(subset=key)
    out = out.merge(inc, on=key, how="left")
    out["prior_in_roster_matrix"] = out["prior_min_total"].notna()
    return out


# ── Build ─────────────────────────────────────────────────────────────────────

def build(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """The pool, the position-mapping audit, and per-season coverage."""
    raw_dir = Path(cfg["data"]["raw_dir"])
    features_dir = Path(cfg["data"]["features_dir"])
    window_games = int(cfg["features"]["team_context"]["roster_window_games"])

    boards = load_dk_boards(features_dir)
    assert_dk_is_single_position(boards)

    seasons = sorted(set(cfg["data"]["seasons"]) | set(boards["season"].unique()))
    rosters = load_rosters(seasons, raw_dir)

    pool = pool_membership(seasons, raw_dir, boards, window_games)
    pool = resolve_positions(pool, rosters, boards)
    pool = attach_prior_season(pool, features_dir)

    adp = select_adp(features_dir)
    pool = pool.merge(adp, on=["season", "player_id"], how="left")
    pool["adp_dk_scale"] = to_dk_scale(pool["adp"], pool["adp_source"], features_dir)

    # Contemporaneous first (the headline), then train-only (the split-clean check), then
    # the widest join for reference. The windows are season ranges over roster files.
    train_end = _season_label(_season_start_year(max(cfg["data"]["seasons"]))
                              - 2 * int(cfg["features"]["availability"]["test_seasons"]))
    windows = {f"rosters_through_{train_end}": ("1996-97", train_end),
               "rosters_all_seasons": ("1996-97", "9999")}
    for board_season in sorted(boards["season"].unique()):
        if board_season in set(rosters["season"]):
            windows[f"rosters_{board_season}"] = (board_season, board_season)
    audit = position_audit(rosters, boards, windows)

    pool["position"] = pool["position"].astype(str)
    pool["adp_censored"] = pool["adp_censored"].astype("boolean")
    pool = pool.sort_values(["season", "adp", "player_id"],
                            na_position="last").reset_index(drop=True)
    pool, dropped = drop_unslottable(pool)
    assert_pool(pool)
    assert_drop_rate(dropped, pool)
    return pool[_ORDERED_COLUMNS], audit, coverage(pool, dropped)


_ORDERED_COLUMNS = [
    "season", "season_start_year", "player_id", "has_nba_id", "player_name", "team",
    "pool_source",
    "position", "pos_g", "pos_f", "pos_c", "position_source",
    "position_nba", "position_dk", "dual_eligible", "dual_g", "dual_f", "dual_c",
    "adp", "adp_rank", "adp_dk_scale", "adp_source", "adp_as_of", "adp_lag_days",
    "adp_censored", "prior_season", "prior_start_year", "prior_in_matrix",
    "prior_in_roster_matrix", "prior_gp", "prior_min_total", "dk_player_id",
]


def drop_unslottable(pool: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Remove players no source gives a position for, and hand back what went.

    Keeping them would be worse than dropping them: a player eligible at no class can be
    drafted, consumes a roster spot, and can never be started — so he would silently
    shrink a 16-man roster rather than announce a gap. They are players `commonteamroster`
    has no row for in *any* season who also never reached a DK board, which is a coverage
    hole in the 1996-2007 files rather than a property of the player.
    """
    unslottable = pool["position_source"] == "none"
    return pool[~unslottable].reset_index(drop=True), pool[unslottable].copy()


def assert_pool(pool: pd.DataFrame) -> None:
    """What must hold for a lineup to be fillable at all.

    Every check here is silent if it fails: a duplicate player would let one man occupy
    two slots, a missing team would break the two-team roster rule, and a player with no
    eligibility would simply never be selected while still consuming a draft pick.
    """
    dup = pool.duplicated(subset=["season", "player_id"]).sum()
    assert dup == 0, f"{dup} duplicated (season, player_id) rows"

    missing_team = int(pool["team"].isna().sum())
    assert missing_team == 0, f"{missing_team} pool rows carry no team"

    flags = pool[[f"pos_{c.lower()}" for c in DK_CLASSES]]
    assert (flags.sum(axis=1) == 1).all(), (
        f"{int((flags.sum(axis=1) != 1).sum())} rows are not eligible at exactly one "
        "DK class — the board is single-position, see the module docstring")

    unresolved = pool[pool["position_source"] == "none"]
    assert unresolved.empty, (
        f"{len(unresolved)} players have no position from any source and could never be "
        f"slotted:\n{unresolved[['season', 'player_id', 'player_name']].head().to_string()}")

    bad_source = set(pool["position_source"]) - set(POSITION_SOURCES)
    assert not bad_source, f"unknown position sources: {sorted(bad_source)}"

    assert pool["player_name"].notna().all(), (
        f"{int(pool['player_name'].isna().sum())} pool rows carry no player name")

    # The sensitivity convention has to be a strict superset of what ships, or swapping
    # to it would change two things at once and stop measuring the dual question.
    shipped = pool[[f"pos_{c.lower()}" for c in DK_CLASSES]].to_numpy()
    duals = pool[[f"dual_{c.lower()}" for c in DK_CLASSES]].to_numpy()
    narrower = int((shipped & ~duals).any(axis=1).sum())
    assert narrower == 0, (
        f"{narrower} rows are eligible somewhere under `pos_*` and not under `dual_*` — "
        "the sensitivity convention must widen eligibility, never move it")


def assert_drop_rate(dropped: pd.DataFrame, pool: pd.DataFrame,
                     max_share: float = 0.02, max_per_season: int = 30) -> None:
    """The coverage hole stays a hole and does not become a hollowed-out pool.

    A bar rather than a report, because `drop_unslottable` removes rows silently and the
    seasons it removes them from are the oldest ones — exactly where nobody would look.
    """
    total = len(dropped) + len(pool)
    share = len(dropped) / total if total else 0.0
    assert share <= max_share, (
        f"{len(dropped):,} of {total:,} pool rows ({share:.2%}) have no position from "
        f"any source, over the {max_share:.0%} bar")
    if len(dropped):
        worst = dropped.groupby("season").size()
        assert worst.max() <= max_per_season, (
            f"{worst.idxmax()} loses {worst.max()} players to missing positions, over "
            f"the per-season bar of {max_per_season}")


def coverage(pool: pd.DataFrame, dropped: pd.DataFrame) -> pd.DataFrame:
    """Per-season: pool size, where positions came from, ADP and prior-row coverage."""
    lost = dropped.groupby("season").size() if len(dropped) else pd.Series(dtype=int)
    rows = []
    for season, grp in pool.groupby("season"):
        row = {
            "season": season,
            "players": len(grp),
            "teams": int(grp["team"].nunique()),
            "pool_source": "|".join(sorted(grp["pool_source"].unique())),
            "dropped_no_position": int(lost.get(season, 0)),
            "adp_players": int(grp["adp"].notna().sum()),
            "adp_source": "|".join(sorted(grp["adp_source"].dropna().unique())) or "none",
            "prior_in_matrix": float(grp["prior_in_matrix"].mean()),
            "prior_in_roster_matrix": float(grp["prior_in_roster_matrix"].mean()),
            "dual_eligible_share": float(grp["dual_eligible"].mean()),
        }
        for src in POSITION_SOURCES:
            row[f"pos_{src}"] = int((grp["position_source"] == src).sum())
        for c in DK_CLASSES:
            row[f"n_{c.lower()}"] = int(grp[f"pos_{c.lower()}"].sum())
        rows.append(row)
    return pd.DataFrame(rows)


def run(cfg: dict) -> Path:
    features_dir = Path(cfg["data"]["features_dir"])
    eda_dir = Path(cfg["eda"]["output_dir"])

    pool, audit, cover = build(cfg)

    features_dir.mkdir(parents=True, exist_ok=True)
    dest = features_dir / "draft_pool.parquet"
    pool.to_parquet(dest, index=False)

    eda_dir.mkdir(parents=True, exist_ok=True)
    audit_dest = eda_dir / "draft_pool_position_audit.csv"
    audit.to_csv(audit_dest, index=False)
    cover_dest = eda_dir / "draft_pool_coverage.csv"
    cover.to_csv(cover_dest, index=False)

    n_seasons = pool["season"].nunique()
    print(f"Draft pool: {len(pool):,} player-seasons over {n_seasons} seasons → {dest}")

    latest = pool["season"].max()
    recent = pool[pool["season"] == latest]
    print(f"  {latest}: {len(recent):,} draftable, "
          f"{int(recent['pos_g'].sum())} G / {int(recent['pos_f'].sum())} F / "
          f"{int(recent['pos_c'].sum())} C, {int(recent['adp'].notna().sum())} with ADP")

    boards = load_dk_boards(features_dir)
    print(f"  🔴 DK is SINGLE-position — {len(boards):,} board rows across "
          f"{boards['season'].nunique()} boards, zero duals, and DK's own label moves on "
          "1 of 667 shared ids. `docs/simulations-plan.md` assumed NBA.com's duals were "
          "DK-shaped; they are not")
    roll = audit[audit["position_nba"] == "ALL"]
    for row in roll.sort_values(["board_season", "roster_window"]).itertuples():
        mark = " ← contemporaneous" if row.contemporaneous else ""
        print(f"    board {row.board_season} vs {row.roster_window} (n={row.n:,}): "
              f"DK's letter is inside NBA.com's set {row.dk_inside_nba_set * 100:.2f}%, "
              f"equals its primary {row.dk_equals_nba_primary * 100:.2f}%{mark}")
    duals = audit[audit["dual"] & audit["contemporaneous"]]
    if len(duals):
        n = int(duals["n"].sum())
        inside = float((duals["dk_inside_nba_set"] * duals["n"]).sum() / n)
        primary = float((duals["dk_equals_nba_primary"] * duals["n"]).sum() / n)
        print(f"    on NBA duals (n={n}): DK picks one of the named pair "
              f"{inside * 100:.2f}% of the time, the primary {primary * 100:.2f}% — "
              "which one is close to a coin toss")
    print("    → `position` ships a single letter: DK's own where a board exists, "
          "NBA.com's primary otherwise. `dual_*` carries the rejected convention")

    mix = cover[[c for c in cover.columns if c.startswith("pos_")]].sum()
    print("  position sources: "
          + ", ".join(f"{k[4:]} {int(v):,}" for k, v in mix.items() if v))
    fallback = int(mix.get("pos_dk_board_carry", 0) + mix.get("pos_dk_board", 0))
    have_2026 = (nbastats_dir(cfg["data"]["raw_dir"]) / "team_rosters_2026_27.csv").exists()
    print(f"    {fallback:,} rows resolved from a DK board. `commonteamroster` has not "
          "been fetched for 2026-27" if not have_2026 else
          f"    {fallback:,} rows resolved from a DK board")
    if not have_2026:
        print("      — so 0 rows needed the board as a *fallback* for a null `POSITION`; "
              "the board is the whole 2026-27 pool, and the cascade already prefers it")
    lost = int(cover["dropped_no_position"].sum())
    worst = cover.nlargest(3, "dropped_no_position")
    print(f"  dropped as unslottable: {lost:,} players with no `POSITION` in any roster "
          "file and no DK board row — a coverage hole in the 1996-2007 files. Worst "
          + ", ".join(f"{r.season} {r.dropped_no_position}" for r in worst.itertuples()))

    with_adp = cover[cover["adp_players"] > 0]
    print(f"  ADP: {int(cover['adp_players'].sum()):,} rows across "
          f"{len(with_adp)} seasons ({', '.join(with_adp['season'])}) — "
          "point-in-time legal, i.e. observed at or before the season's first game")
    print("    🔴 four panel seasons (2017-18, 2018-19, 2019-20, 2024-25) contribute "
          "nothing: every archived snapshot postdates their first game. The plan's "
          "nine-season coverage list is panel presence, not legality")

    val = pool[pool["season"].isin(["2022-23", "2023-24"])]
    print(f"  validation seasons carry {int(val['adp'].notna().sum()):,} ADP'd players "
          f"of {len(val):,} — the coverage the realized backtest needs")
    print(f"  prior-season rows: {pool['prior_in_matrix'].mean() * 100:.1f}% in the "
          f"qualified matrix, {pool['prior_in_roster_matrix'].mean() * 100:.1f}% in the "
          "inclusive twin")
    print(f"Position audit: {len(audit):,} rows → {audit_dest}")
    print(f"Coverage: {len(cover):,} rows → {cover_dest}")
    return dest


if __name__ == "__main__":
    cfg = yaml.safe_load(open("configs/default.yaml"))
    run(cfg)
