import gzip

import pandas as pd

from src.data.adp_draftkings import (
    NAME_ALIASES,
    TEAM_ALIASES,
    _prefix_match,
    _reversed_key,
    build_id_map,
    capture_date_from_name,
    id_stability,
    read_board,
    season_of_board,
)
from src.data.adp_fantasypros import (
    FROZEN_THRESHOLD,
    assign_seasons,
    calendar_season,
    identical_share,
    parse_board,
    parse_player_cell,
    sources_present,
)
from src.features.adp import (
    assert_point_in_time,
    attach_dating,
    match_audit,
    match_players,
    normalize_name,
    surname_initial_matches,
    training_rows,
)


# ── Synthetic builders ────────────────────────────────────────────────────────

def _board(rows, header=("Rank", "Player", "Yahoo", "ESPN", "AVG")):
    """A minimal FantasyPros ADP page with the real table id."""
    ths = "".join(f"<th>{h}</th>" for h in header)
    trs = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f"<html><body><table id='data'><thead><tr>{ths}</tr></thead>" \
           f"<tbody>{trs}</tbody></table></body></html>"


def _adp_frame(names, avgs, season="2025-26"):
    return pd.DataFrame({"player_name": names, "AVG": avgs, "season": season})


def _roster(rows, team="XXX"):
    """rows: (player_id, player_name, season). `team` never matches a board on purpose,
    so the fuzzy steps are what is under test rather than the exact name+team join."""
    df = pd.DataFrame(rows, columns=["player_id", "player_name", "season"])
    df["team_abbreviation"] = team
    return df


# ── Player cell: three formats across twelve years ────────────────────────────

def test_player_cell_handles_every_observed_format():
    assert parse_player_cell("Kevin Durant ( OKC )") == {
        "player_name": "Kevin Durant", "team": "OKC",
        "positions": "", "status_at_capture": ""}
    assert parse_player_cell("LeBron James CLE")["player_name"] == "LeBron James"
    modern = parse_player_cell("Russell Westbrook (OKC - PG)")
    assert modern["player_name"] == "Russell Westbrook"
    assert modern["team"] == "OKC" and modern["positions"] == "PG"


def test_status_token_is_stripped_not_enumerated():
    """The vocabulary is open: DTD/OUT in 2022, a bare O in 2015, G-League in 2025."""
    assert parse_player_cell("Kevin Durant OKC O")["player_name"] == "Kevin Durant"
    assert parse_player_cell("Kevin Durant OKC O")["status_at_capture"] == "O"
    gl = parse_player_cell("Kristaps Porzingis (BOS - PF,C) G-League")
    assert gl["player_name"] == "Kristaps Porzingis"
    assert gl["status_at_capture"] == "G-League"
    assert parse_player_cell("Larry Nance Jr. (LAL - PF) OUT")["player_name"] \
        == "Larry Nance Jr."


def test_player_cell_does_not_eat_uppercase_initials():
    """CJ McCollum and JJ Redick are names, not team codes."""
    for name in ("CJ McCollum (POR - PG,SG)", "JJ Redick (LAC - SG)"):
        parsed = parse_player_cell(name)
        assert parsed["player_name"] in ("CJ McCollum", "JJ Redick")


# ── Parsing a page ────────────────────────────────────────────────────────────

def test_parse_board_reads_table_data_and_reports_sources():
    html = _board([("1", "Nikola Jokic (DEN - C)", "1", "2", "1.5")])
    board = parse_board(html)
    assert len(board) == 1
    assert board.iloc[0]["player_name"] == "Nikola Jokic"
    assert board.iloc[0]["AVG"] == 1.5
    assert sources_present(board) == ["Yahoo", "ESPN"]


def test_parse_board_inflates_gzip():
    """Wayback's id_ endpoint returns the origin's gzip; decoding it as UTF-8 yields a
    page with no table, which looks exactly like a JS-rendered page and is not one."""
    html = _board([("1", "Nikola Jokic (DEN - C)", "1", "2", "1.5")])
    assert len(parse_board(gzip.compress(html.encode()))) == 1


def test_parse_board_returns_empty_for_the_placeholder_page():
    """FantasyPros serves this on some early-September dates. A page state, not a bug."""
    html = _board([("Sorry, this report is not available at the moment",)],
                  header=("Rank",))
    assert parse_board(html).empty


def test_cbs_column_is_read_when_present():
    html = _board([("1", "Anthony Davis (LAL - PF,C)", "2", "2", "1", "1.7")],
                  header=("Rank", "Player", "Yahoo", "ESPN", "CBS", "AVG"))
    assert sources_present(parse_board(html)) == ["Yahoo", "ESPN", "CBS"]


# ── The freeze rule ───────────────────────────────────────────────────────────

def test_identical_share_separates_a_frozen_board_from_a_changed_one():
    a = _adp_frame(["a", "b", "c"], [1.0, 2.0, 3.0])
    assert identical_share(a, a.copy())[0] == 1.0
    moved = _adp_frame(["a", "b", "c"], [9.0, 8.0, 7.0])
    assert identical_share(a, moved)[0] == 0.0


def test_calendar_season_splits_on_october():
    assert calendar_season("2025-10-05") == "2025-26"
    assert calendar_season("2025-09-06") == "2024-25"
    assert calendar_season("2026-01-19") == "2025-26"


def test_frozen_board_inherits_across_the_covid_boundary():
    """2020-21 tipped off 2020-12-22, so its drafts ran in December.

    The 2020-10-23 snapshot is byte-identical to 2020-09-16 — still the 2019-20 board —
    and a calendar rule alone would call it 2020-21. Inheritance carries 2019-20 through.
    This is the trap the module exists to avoid; pinning it stops a "simplification"
    from reintroducing it.
    """
    names = [f"p{i}" for i in range(60)]
    avgs = [float(i) for i in range(60)]
    frozen = _adp_frame(names, avgs)
    boards = [("s1", "2020-09-16", frozen), ("s2", "2020-10-23", frozen.copy())]
    labels = assign_seasons(boards)
    assert labels[0]["season"] == "2019-20"
    assert labels[1]["season"] == "2019-20", "frozen board must not jump a season"
    assert labels[1]["frozen"] and labels[1]["inherited"]


def test_a_changed_board_in_october_takes_the_new_season():
    names = [f"p{i}" for i in range(60)]
    old = _adp_frame(names, [float(i) for i in range(60)])
    new = _adp_frame(names, [float(i + 100) for i in range(60)])
    labels = assign_seasons([("s1", "2025-09-06", old), ("s2", "2025-10-05", new)])
    assert labels[0]["season"] == "2024-25"
    assert labels[1]["season"] == "2025-26"
    assert not labels[1]["frozen"]


def test_small_overlap_cannot_flip_a_season():
    """Below the overlap floor there is no evidence either way; do not act on noise."""
    a = _adp_frame(["a", "b"], [1.0, 2.0])
    b = _adp_frame(["a", "b"], [5.0, 6.0])
    labels = assign_seasons([("s1", "2020-09-16", a), ("s2", "2020-10-23", b)])
    assert labels[1]["frozen"] is False
    assert FROZEN_THRESHOLD > 0.3360, "must sit above the highest non-frozen pair measured"


# ── DraftKings boards ─────────────────────────────────────────────────────────

def test_capture_date_accepts_abbreviated_and_full_month_names():
    """Real captures use both. A narrow pattern skips the board silently, which is the
    worst failure for a source that cannot be re-captured."""
    assert capture_date_from_name("DkPreDraftRankings_Oct17_2025.csv") == "2025-10-17"
    assert capture_date_from_name("DkPreDraftRankings_July28_2026.csv") == "2026-07-28"
    assert capture_date_from_name("nonsense.csv") is None


def test_season_of_board_reads_the_upcoming_season():
    assert season_of_board("2025-10-17") == "2025-26"
    assert season_of_board("2026-07-28") == "2026-27"


def test_new_orleans_alias_is_not_a_thirty_first_team():
    assert TEAM_ALIASES["NO"] == "NOP"


def test_read_board_applies_team_aliases_and_flags_censoring(tmp_path):
    csv = tmp_path / "DkPreDraftRankings_Oct17_2025.csv"
    csv.write_text("ID,Name,Position,ADP,Team\n"
                   "1,Alpha Player,C,1.5,NO\n"
                   "2,Beta Player,G,180.0,DEN\n"
                   "3,Gamma Player,F,,LAL\n")
    board = read_board(csv)
    assert board.loc[board.player_name == "Alpha Player", "team"].iloc[0] == "NOP"
    assert not board.loc[board.player_name == "Alpha Player", "adp_censored"].iloc[0]
    assert board.loc[board.player_name == "Beta Player", "adp_censored"].iloc[0]
    assert board["adp"].isna().sum() == 1, "pool rows without an ADP are kept"


def test_id_stability_detects_a_recycled_id():
    stable = pd.DataFrame({"dk_player_id": [1, 1], "player_name": ["A", "A"]})
    assert id_stability(stable)["name_agreement"] == 1.0
    recycled = pd.DataFrame({"dk_player_id": [1, 1], "player_name": ["A", "B"]})
    assert id_stability(recycled)["name_agreement"] == 0.0


# ── Name matching: the cascade must not invent people ─────────────────────────

def test_prefix_match_accepts_short_forms_and_rejects_different_names():
    assert _prefix_match("alexandre", "alex")
    assert not _prefix_match("cameron", "carlos")
    assert not _prefix_match("darryn", "drew")
    assert not _prefix_match("rj", "ricky")


def test_reversed_key_handles_surname_first_romanization():
    assert _reversed_key("hansen yang") == "yang hansen"
    assert _reversed_key("nene") == "nene"


def test_same_initial_is_not_a_match():
    """The cheap 'same surname + same first initial' rule produced 11 false matches out
    of 12 on real boards — Cameron Boozer -> Carlos Boozer among them — while scoring a
    perfect unmatched rate. A join that fabricates rows must fail this test."""
    boards = pd.DataFrame({
        "dk_player_id": [1], "player_name": ["Cameron Boozer"], "team": ["MEM"],
        "capture_date": ["2026-07-28"], "season": ["2026-27"]})
    roster = _roster([(2430, "Carlos Boozer", "2013-14")])
    out = build_id_map(boards, roster)
    assert out.iloc[0]["match_method"] != "prefix"
    assert pd.isna(out.iloc[0]["player_id"])


def test_era_guard_rejects_a_plausible_string_from_the_wrong_decade():
    """`Mikel Brown Jr.` really does start with `Mike`, so the prefix rule alone matches
    Mike Brown (last seen 1996-97). Only era separates them."""
    boards = pd.DataFrame({
        "dk_player_id": [1], "player_name": ["Mikel Brown Jr."], "team": ["BKN"],
        "capture_date": ["2026-07-28"], "season": ["2026-27"]})
    roster = _roster([(1479, "Mike Brown", "1996-97"),
                      (1642905, "Yang Hansen", "2025-26")])
    out = build_id_map(boards, roster)
    assert pd.isna(out.iloc[0]["player_id"])


def test_rookies_are_no_nba_history_not_unmatched():
    """Collapsing the two would turn 'the 2026 draft class exists' into a join failure."""
    boards = pd.DataFrame({
        "dk_player_id": [1], "player_name": ["AJ Dybantsa"], "team": ["WAS"],
        "capture_date": ["2026-07-28"], "season": ["2026-27"]})
    roster = _roster([(1642905, "Yang Hansen", "2025-26")])
    assert build_id_map(boards, roster).iloc[0]["match_method"] == "no_nba_history"


def test_consensus_cascade_resolves_real_name_variants():
    frame = _adp_frame(["Hansen Yang", "Louis Williams", "Enes Kanter"],
                       [10.0, 20.0, 30.0], season="2021-22")
    roster = _roster([(1642905, "Yang Hansen", "2021-22"),
                      (101150, "Lou Williams", "2021-22"),
                      (202683, "Enes Freedom", "2021-22")])
    out = match_players(frame, roster)
    assert set(out["match_method"]) <= {"reversed", "prefix", "alias"}
    assert out["player_id"].notna().all()
    assert "Enes Kanter" in NAME_ALIASES


def test_normalize_name_folds_diacritics_and_suffixes():
    assert normalize_name("Kristaps Porziņģis") == "kristaps porzingis"
    assert normalize_name("Larry Nance Jr.") == "larry nance"


# ── Point-in-time ─────────────────────────────────────────────────────────────

def _dated(as_of, season="2025-26"):
    return pd.DataFrame({"season": [season], "as_of_date": [as_of],
                         "player_name": ["A"], "adp": [1.0]})


def test_capture_after_the_season_starts_is_flagged_not_silently_used():
    starts = {"2025-26": "2025-10-21"}
    before = attach_dating(_dated("2025-10-17"), starts)
    after = attach_dating(_dated("2026-01-19"), starts)
    assert bool(before["captured_before_season_start"].iloc[0])
    assert not bool(after["captured_before_season_start"].iloc[0])
    assert int(after["snapshot_lag_days"].iloc[0]) == 90


def test_training_rows_drops_post_start_captures_and_the_assert_agrees():
    starts = {"2025-26": "2025-10-21"}
    panel = attach_dating(
        pd.concat([_dated("2025-10-17"), _dated("2026-01-19")], ignore_index=True), starts)
    train = training_rows(panel)
    assert len(train) == 1
    assert_point_in_time(train)          # must not raise
    try:
        assert_point_in_time(panel)
    except AssertionError:
        pass
    else:                                 # pragma: no cover
        raise AssertionError("assert_point_in_time must reject a post-start capture")


def test_unknown_season_start_is_treated_as_before():
    """A board for a season with no games played yet cannot postdate that season."""
    out = attach_dating(_dated("2026-07-28", season="2026-27"), {})
    assert bool(out["captured_before_season_start"].iloc[0])


# ── Fabrication guards (see CLAUDE.md: an unmatched rate does not validate a join) ─────

def test_ambiguous_prefix_candidates_resolve_to_nothing():
    """Two plausible short-form candidates cannot disambiguate each other. Picking one is
    a coin flip that scores as a successful match."""
    frame = _adp_frame(["Chris Johnson"], [10.0], season="2012-13")
    roster = _roster([(202419, "Christopher Johnson", "2012-13"),
                      (203187, "Christian Johnson", "2012-13")])
    out = match_players(frame, roster)
    assert out["match_method"].iloc[0] == "unmatched"
    assert pd.isna(out["player_id"].iloc[0])


def test_match_players_never_multiplies_rows():
    """Row inflation is silent fabrication: every duplicate copy counts as matched."""
    frame = _adp_frame(["Alpha Beta", "Gamma Delta"], [1.0, 2.0], season="2021-22")
    roster = _roster([(1, "Alpha Beta", "2021-22"),
                      (1, "Alpha Beta", "2020-21"),      # same player, two seasons
                      (2, "Gamma Delta", "2021-22")])
    out = match_players(frame, roster)
    assert len(out) == 2, "one board row must stay one row"
    assert out["player_id"].tolist() == [1, 2]


def test_a_match_never_crosses_eras():
    """The era guard, stated as a property rather than through one example."""
    frame = _adp_frame(["Mikel Brown"], [50.0], season="2026-27")
    roster = _roster([(1479, "Mike Brown", "1996-97")])
    out = match_players(frame, roster)
    assert out["match_method"].iloc[0] == "unmatched"


def test_every_non_exact_match_is_individually_checkable():
    """The fuzzy tier must stay small enough to read. A cascade that resolves a large
    fraction by fuzzy rules is not auditable, and an unmatched rate will not say so."""
    frame = _adp_frame(["Alpha Beta", "Gamma Delta", "Hansen Yang"],
                       [1.0, 2.0, 3.0], season="2021-22")
    roster = _roster([(1, "Alpha Beta", "2021-22"), (2, "Gamma Delta", "2021-22"),
                      (1642905, "Yang Hansen", "2021-22")])
    out = match_players(frame, roster)
    fuzzy = out[out["match_method"].isin(["reversed", "prefix", "alias"])]
    assert len(fuzzy) / len(out) <= 0.5
    assert fuzzy["player_id"].notna().all()


# ── The name-match audit, and the rejected rule kept as a permanent ablation ───

def _panel_rows(rows: list[tuple[str, str, str, object]],
                source: str = "fantasypros") -> pd.DataFrame:
    """rows: (player_name, season, match_method, player_id)."""
    df = pd.DataFrame(rows, columns=["player_name", "season", "match_method", "player_id"])
    df["player_key"] = df["player_name"].map(normalize_name)
    df["snapshot_source"] = source
    df["as_of_date"] = "2025-10-01"
    return df


def test_every_fuzzy_match_is_listed_with_both_names_and_the_season_gap():
    """"List every non-exact match and read them" — as an artifact, not an instruction."""
    roster = _roster([(1, "Alex Sarr", "2024-25"), (2, "Lou Williams", "2011-12")])
    panel = _panel_rows([("Alexandre Sarr", "2024-25", "prefix", 1),
                         ("Louis Williams", "2011-12", "prefix", 2),
                         ("Alex Sarr", "2024-25", "name+season", 1)])
    audit = match_audit(panel, roster)
    fuzzy = audit[audit["section"] == "fuzzy_match"]
    assert len(fuzzy) == 2
    assert set(fuzzy["board_name"]) == {"Alexandre Sarr", "Louis Williams"}
    assert set(fuzzy["matched_name"]) == {"Alex Sarr", "Lou Williams"}
    assert (fuzzy["seasons_apart"] == 0).all()


def test_an_exact_match_is_not_listed_in_the_fuzzy_tier():
    roster = _roster([(1, "Alex Sarr", "2024-25")])
    panel = _panel_rows([("Alex Sarr", "2024-25", "name+season", 1)])
    audit = match_audit(panel, roster)
    assert audit[audit["section"] == "fuzzy_match"].empty
    counts = audit[audit["section"] == "summary"].set_index("metric")["value"]
    assert counts["exact_matches"] == 1.0


def test_a_stale_fuzzy_match_reports_the_seasons_apart_gap():
    """The gap is the second guard: a candidate decades away is not the same person."""
    roster = _roster([(1, "Mike Brown", "1996-97")])
    panel = _panel_rows([("Mikel Brown Jr.", "2026-27", "prefix", 1)])
    audit = match_audit(panel, roster)
    row = audit[audit["section"] == "fuzzy_match"].iloc[0]
    assert row["seasons_apart"] == 30.0


def test_no_nba_history_is_counted_apart_from_unmatched():
    """Collapsing them turns "the 2026 draft class exists" into a fake defect."""
    roster = _roster([(1, "Alex Sarr", "2024-25")])
    panel = _panel_rows([("Alex Sarr", "2024-25", "name+season", 1),
                         ("Cameron Boozer", "2026-27", "no_nba_history", None),
                         ("Nobody Here", "2024-25", "unmatched", None)])
    counts = (match_audit(panel, roster)
              .pipe(lambda d: d[d["section"] == "summary"])
              .set_index("metric")["value"])
    assert counts["no_nba_history"] == 1.0
    assert counts["unmatched"] == 1.0
    # The rate's denominator excludes the players who simply have no NBA history.
    assert counts["matchable_rows"] == 2.0
    assert abs(counts["unmatched_rate_cascade"] - 0.5) < 1e-12


def test_the_rejected_rule_fabricates_a_match_the_cascade_refuses():
    """`Cameron Boozer` -> Carlos Boozer: same surname, same initial, different person."""
    roster = _roster([(1, "Carlos Boozer", "2013-14")])
    panel = _panel_rows([("Cameron Boozer", "2026-27", "no_nba_history", None)])
    audit = match_audit(panel, roster)
    ablation = audit[audit["section"] == "ablation_match"]
    assert len(ablation) == 1
    assert ablation.iloc[0]["matched_name"] == "Carlos Boozer"
    assert ablation.iloc[0]["rule"] == "surname_initial"
    # `agrees_with_cascade == 0` is what marks it as invented.
    assert ablation.iloc[0]["value"] == 0.0


def test_the_rejected_rule_scores_a_better_unmatched_rate_while_being_wrong():
    """The whole warning: the metric is increasing in the error it should detect."""
    roster = _roster([(1, "Carlos Boozer", "2013-14"), (2, "Ricky Davis", "2007-08")])
    panel = _panel_rows([("Cameron Boozer", "2026-27", "unmatched", None),
                         ("RJ Davis", "2026-27", "unmatched", None)])
    counts = (match_audit(panel, roster)
              .pipe(lambda d: d[d["section"] == "summary"])
              .set_index("metric")["value"])
    assert counts["unmatched_rate_surname_initial"] < counts["unmatched_rate_cascade"]
    assert counts["ablation_false_matches"] == 2.0


def test_the_rejected_rule_agrees_where_the_cascade_already_matched():
    """It is not wrong about everything — which is why the false-match count is the metric."""
    roster = _roster([(1, "Alex Sarr", "2024-25")])
    panel = _panel_rows([("Alexandre Sarr", "2024-25", "prefix", 1)])
    audit = match_audit(panel, roster)
    ablation = audit[audit["section"] == "ablation_match"]
    assert len(ablation) == 1
    assert ablation.iloc[0]["value"] == 1.0
    counts = audit[audit["section"] == "summary"].set_index("metric")["value"]
    assert counts["ablation_false_matches"] == 0.0


def test_surname_initial_needs_a_surname_and_an_initial():
    roster = _roster([(1, "Carlos Boozer", "2013-14")])
    got = surname_initial_matches(pd.Series(["boozer", "cameron boozer", "carlos"]), roster)
    assert pd.isna(got.iloc[0])          # no first name at all
    assert got.iloc[1] == 1
    assert pd.isna(got.iloc[2])          # no surname


def test_the_audit_collapses_repeated_snapshots_of_the_same_name():
    """Forty snapshots resolving the same way are one match to read, not forty."""
    roster = _roster([(1, "Alex Sarr", "2024-25")])
    rows = [("Alexandre Sarr", "2024-25", "prefix", 1)] * 40
    panel = _panel_rows(rows)
    panel["as_of_date"] = [f"2024-10-{1 + i % 28:02d}" for i in range(40)]
    audit = match_audit(panel, roster)
    assert len(audit[audit["section"] == "fuzzy_match"]) == 1
