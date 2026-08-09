"""Tournament economics, derived from the two captured DraftKings CSVs.

Every figure the drafting tab shows about rake, advance rates and payout convexity
is computed here from `data/raw/dk_best_ball_tournament_{metadata,prize_structure}
.csv` rather than typed in. **No Streamlit import** — this is the arithmetic, and the
tests exercise it directly.

Two things about the source data need deliberate handling:

- The metadata file carries **18 trailing unnamed columns** from the spreadsheet it
  was exported out of, one of which holds a stray value. They are dropped by name
  rather than by position.
- The prize file records the **advance cutoff** for round 1 (`low_place = 2`) but not
  the pod size, because round-1 pods are a contest-structure fact rather than a
  payout one. `ROUND_ONE_POD = 12` comes from `docs/dk_best_ball_rules.md`, and
  `check_round_one_pod` verifies it against the entry counts: only a 12-man round-1
  pod makes every subsequent round's field size come out an exact integer for all
  five tournaments. Two independent routes to the same number.
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

METADATA = "data/raw/dk_best_ball_tournament_metadata.csv"
PRIZES = "data/raw/dk_best_ball_tournament_prize_structure.csv"

# From docs/dk_best_ball_rules.md — "12-man round 1 contests". Not in the prize CSV.
ROUND_ONE_POD = 12


# ── Loading ───────────────────────────────────────────────────────────────────

def load_metadata(root: Path = ROOT) -> pd.DataFrame:
    """One row per tournament, with the spreadsheet's trailing columns dropped."""
    frame = pd.read_csv(root / METADATA)
    keep = [c for c in frame.columns if not str(c).startswith("Unnamed:")]
    return frame[keep].copy()


def load_prizes(root: Path = ROOT) -> pd.DataFrame:
    return pd.read_csv(root / PRIZES)


# ── Rake and the break-even edge hurdle ───────────────────────────────────────

def rake(entries: float, fee: float, prizes: float) -> float:
    """Share of the buy-in pool the house keeps."""
    pool = entries * fee
    if pool <= 0:
        raise ValueError("buy-in pool must be positive")
    return 1.0 - prizes / pool


def break_even_hurdle(rake_share: float) -> float:
    """The edge needed just to return the entry fee: `1/(1−rake) − 1`.

    The right units for comparing against a *measured* edge. A raw rake percentage
    is not denominated the same way — losing 15% of the pool means you have to beat
    the field by more than 15% to break even, not by exactly 15%.
    """
    if rake_share >= 1.0:
        raise ValueError("rake must be below 1")
    return 1.0 / (1.0 - rake_share) - 1.0


def economics(root: Path = ROOT) -> pd.DataFrame:
    """Per-tournament rake, break-even hurdle and top-prize convexity."""
    meta = load_metadata(root)
    prizes = load_prizes(root)

    meta = meta.rename(columns={"type": "tournament"})
    meta["buy_in_pool"] = meta["total_entries"] * meta["entry_fee_per_team"]
    meta["rake"] = [rake(r.total_entries, r.entry_fee_per_team, r.total_prizes)
                    for r in meta.itertuples()]
    meta["break_even_hurdle"] = meta["rake"].map(break_even_hurdle)

    final = prizes[prizes["prize_type"] == "cash"]
    first = (final.sort_values(["tournament", "round", "high_place"])
             .groupby("tournament").apply(_first_prize, include_groups=False)
             .rename("first_prize").reset_index())
    meta = meta.merge(first, on="tournament", how="left")
    meta["first_prize_multiple"] = meta["first_prize"] / meta["entry_fee_per_team"]
    return meta.sort_values("entry_fee_per_team").reset_index(drop=True)


def _first_prize(group: pd.DataFrame) -> float:
    last_round = group["round"].max()
    top = group[(group["round"] == last_round) & (group["high_place"] == 1)]
    return float(top["cash_amount"].iloc[0]) if len(top) else float("nan")


# ── Round structure ───────────────────────────────────────────────────────────

def advance_table(root: Path = ROOT,
                  round_one_pod: int = ROUND_ONE_POD) -> pd.DataFrame:
    """Per tournament × round: pod size, how many advance, and the field left.

    Pod sizes for rounds 2+ are the largest place the round pays or advances. Round
    1 pays nothing at all — it is a **zero-consolation knockout**, which is why its
    row has no cash line and why its pod size has to come from the rules doc.
    """
    meta = load_metadata(root).rename(columns={"type": "tournament"})
    prizes = load_prizes(root)
    entries = dict(zip(meta["tournament"], meta["total_entries"]))

    rows = []
    for tournament in meta["tournament"]:
        sub = prizes[prizes["tournament"] == tournament]
        field = float(entries[tournament])
        for rnd in sorted(sub["round"].unique()):
            r = sub[sub["round"] == rnd]
            adv = r[r["prize_type"] == "advance"]
            cash = r[r["prize_type"] == "cash"]
            n_advance = int((adv["low_place"] - adv["high_place"] + 1).sum())
            cash_places = int((cash["low_place"] - cash["high_place"] + 1).sum())
            if rnd == 1:
                pod = round_one_pod
            elif n_advance:
                pod = int(r["low_place"].max())
            else:
                # The final round is one contest of everyone who got there. All five
                # captured tournaments pay every finalist, but the pod is still taken
                # from the chained field rather than from the paid places, because
                # nothing in the rules says a final table has to pay out in full.
                pod = int(round(field))
            rows.append({
                "tournament": tournament,
                "round": int(rnd),
                "field_entries": field,
                "pod_size": pod,
                "n_advance": n_advance,
                "advance_rate": (n_advance / pod) if pod else float("nan"),
                "cash_places": cash_places,
                "paid_share": (cash_places / field) if field else float("nan"),
                "min_cash": float(cash["cash_amount"].min()) if len(cash) else 0.0,
                "zero_consolation": bool(n_advance and cash_places == 0),
            })
            if n_advance:
                field = field / pod * n_advance
    return pd.DataFrame(rows)


def round_one_pod_evidence(root: Path = ROOT,
                           round_one_pod: int = ROUND_ONE_POD) -> dict:
    """Two independent consistency tests of a candidate round-1 pod size.

    The prize CSV records round 1's *advance cutoff* but not its pod size, so the
    12 in `ROUND_ONE_POD` comes from `docs/dk_best_ball_rules.md`. Chaining each
    tournament's entry count forward through the per-round pod sizes and advance
    counts checks it against the data:

    - `integral` — every round's field size is a whole number of entries. Necessary
      but not sufficient: any divisor of 12 also passes, since halving the pod just
      doubles the next field.
    - `final_field_equals_paid` — how many of the five have a final round that pays
      *exactly* its field. At 12 this is **5 of 5**; every other pod size scores 0
      or 1. That is what pins it.

    It read 4 of 5 until 2026-08-09, `15k_and_one` appearing to pay 24 of 42 — a
    transcription slip in the prize CSV, corrected at source. It was found by
    `src/sim/bracket.py`'s symmetric-field null, which prices an exchangeable entry
    and must return exactly `-rake`: that tournament's bands paid $13,200 of its
    stated $15,000 pool while the other four reconciled to the cent. The tolerance
    here (`>= n - 1`) is what let it through, and it is deliberately kept — a real
    contest need not pay its whole final table — so the tight data check lives in
    the null instead, where a test pins all five.
    """
    table = advance_table(root, round_one_pod)
    fields = table["field_entries"]
    final = table[table["n_advance"] == 0]
    return {
        "round_one_pod": round_one_pod,
        "integral": bool(((fields - fields.round()).abs() < 1e-9).all()
                         and (fields > 0).all()),
        "final_field_equals_paid": int(
            (final["field_entries"].round() == final["cash_places"]).sum()),
        "pays_more_places_than_entrants": int(
            (final["cash_places"] > final["field_entries"] + 1e-9).sum()),
        "n_tournaments": int(len(final)),
    }


def check_round_one_pod(root: Path = ROOT,
                        round_one_pod: int = ROUND_ONE_POD) -> bool:
    """Whether this round-1 pod size is consistent with the captured entry counts.

    Both arms of `round_one_pod_evidence` must hold: an integral chain, and the
    final-round-pays-its-field coincidence on at least four of the five.
    """
    ev = round_one_pod_evidence(root, round_one_pod)
    return bool(ev["integral"]
                and ev["pays_more_places_than_entrants"] == 0
                and ev["final_field_equals_paid"] >= ev["n_tournaments"] - 1)


def payout_curve(root: Path = ROOT, tournament: str = "") -> pd.DataFrame:
    """Final-round cash by finishing place, expanded from the banded rows."""
    prizes = load_prizes(root)
    sub = prizes[(prizes["tournament"] == tournament)
                 & (prizes["prize_type"] == "cash")]
    if sub.empty:
        return pd.DataFrame(columns=["place", "cash"])
    sub = sub[sub["round"] == sub["round"].max()]
    rows = []
    for band in sub.itertuples():
        for place in range(int(band.high_place), int(band.low_place) + 1):
            rows.append({"place": place, "cash": float(band.cash_amount)})
    return pd.DataFrame(rows).sort_values("place").reset_index(drop=True)
