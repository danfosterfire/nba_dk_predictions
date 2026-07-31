"""The dashboard's pure layer: palette rules, figure builders, registry, audit.

After the package split the palette tests import `dashboard.theme` and
`dashboard.charts` directly rather than loading `app.py` by file path through
`importlib` — simpler, and it stops a test from executing the whole app module to
check a colour constant.

`decisions.py`, `economics.py` and `audit.py` import no Streamlit, which is what lets
the registry and the audit checks be exercised here as plain functions.
"""

import ast
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from dashboard import audit, charts, decisions, economics, theme
from dashboard.tabs import TAB_NAMES, TABS

ROOT = Path(__file__).resolve().parent.parent


# ── Synthetic builders ────────────────────────────────────────────────────────

def _bars_frame(n: int = 6) -> pd.DataFrame:
    return pd.DataFrame({"feature": [f"f{i}" for i in range(n)],
                         "a": np.linspace(0.1, 0.9, n),
                         "b": np.linspace(0.9, 0.1, n)})


def _scatter_frame(n: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "pc1": rng.normal(size=n), "pc2": rng.normal(size=n),
        "age": rng.integers(20, 38, n).astype(float),
        "archetype_name": rng.choice(list("abcdefghi"), n),
        "player_name": [f"P{i}" for i in range(n)],
    })


def _decision(**kw) -> decisions.Decision:
    base = dict(id="synthetic", topic="problem", claim="A claim.",
                because="A reason.", status="settled", source="CLAUDE.md",
                reviewed="2026-07-30", date="2026-07-30",
                reproduce="make thing → outputs/eda/thing.csv")
    return decisions.Decision(**{**base, **kw})


# ── Palette rules ─────────────────────────────────────────────────────────────

def test_both_modes_supply_the_same_eight_slots_in_a_fixed_order():
    light, dark = theme.theme("light"), theme.theme("dark")
    assert len(light["series"]) == len(dark["series"]) == 8
    # slot order is the colour-blind-safety mechanism, so it must not be re-sorted
    assert light["series"][0] == "#2a78d6" and light["series"][1] == "#eb6834"
    assert dark["series"][0] == "#3987e5" and dark["series"][1] == "#d95926"


def test_the_all_pairs_cap_is_three():
    """Only the first three slots clear the CVD floors on every pair."""
    assert theme.ALL_PAIRS_CAP == 3


def test_sequential_ramp_is_one_hue_and_reverses_for_the_dark_surface():
    light = [c for _, c in theme.theme("light")["sequential"]]
    dark = [c for _, c in theme.theme("dark")["sequential"]]
    assert light == theme.BLUE_RAMP
    assert dark == list(reversed(theme.BLUE_RAMP))
    # near-zero recedes toward the surface in both modes
    assert light[0] == "#cde2fb" and dark[0] == "#0d366b"


def test_diverging_scale_is_two_hues_with_a_neutral_midpoint():
    for mode in ("light", "dark"):
        th = theme.theme(mode)
        stops = th["diverging"]
        assert [s[0] for s in stops] == [0.0, 0.5, 1.0]      # equal arms
        assert stops[1][1] == th["neutral"]                   # gray, not a hue
        assert stops[0][1] in th["series"] and stops[2][1] in th["series"]


def test_ordinal_ramp_stays_clear_of_the_surface():
    """An ordinal step must not sink into the background at either end."""
    assert "#cde2fb" not in theme.theme("light")["ordinal"]   # lighter than step 250
    assert "#0d366b" not in theme.theme("dark")["ordinal"]    # darker than step 600


def test_ordinal_colors_are_distinct_and_count_matched():
    for n in (2, 5, 10):
        colors = theme.ordinal_colors(theme.theme("light"), n)
        assert len(colors) == n
        assert len(set(colors)) == n


# ── Chrome ────────────────────────────────────────────────────────────────────

def test_apply_theme_pins_the_validated_surface_and_solid_hairlines():
    th = theme.theme("dark")
    fig = theme.apply_theme(go.Figure(), th)
    assert fig.layout.paper_bgcolor == th["surface"] == "#1a1a19"
    assert fig.layout.plot_bgcolor == th["surface"]
    # dashed grid reads as "threshold" when it is only a grid
    assert fig.layout.xaxis.griddash == "solid"
    assert fig.layout.yaxis.gridcolor == th["grid"]


def test_light_mode_pins_the_surface_the_palette_was_validated_against():
    assert theme.theme("light")["surface"] == "#fcfcfb"


# ── Figure builders ───────────────────────────────────────────────────────────

def test_bars_assign_slots_in_order_and_separate_fills_with_the_surface():
    th = theme.theme("light")
    fig = charts.fig_bars(_bars_frame(), "feature", ["a", "b"], th, "t")
    assert len(fig.data) == 2
    assert fig.data[0].marker.color == th["series"][0]
    assert fig.data[1].marker.color == th["series"][1]
    # a 2px surface gap, not a border drawn around the marks
    assert all(t.marker.line.color == th["surface"] and t.marker.line.width == 2
               for t in fig.data)
    assert fig.layout.showlegend is True


def test_a_single_bar_series_carries_no_legend_box():
    fig = charts.fig_bars(_bars_frame(), "feature", ["a"], theme.theme("light"), "t")
    assert fig.layout.showlegend is False


def test_emphasis_paints_one_category_and_mutes_the_rest():
    th = theme.theme("light")
    df = _bars_frame(4)
    fig = charts.fig_bars(df, "feature", ["a"], th, "t", emphasis="f2")
    colors = list(fig.data[0].marker.color)
    assert colors == [th["muted"], th["muted"], th["series"][0], th["muted"]]


def test_scatter_emphasis_never_exceeds_the_all_pairs_cap():
    th = theme.theme("light")
    df = _scatter_frame()
    fig = charts.fig_scatter(df, "pc1", "pc2", th, "t", color_by="archetype_name",
                             highlight=list("abcdefghi"))
    highlighted = [t for t in fig.data if t.name != "everything else"]
    assert len(highlighted) == theme.ALL_PAIRS_CAP
    assert [t.marker.color for t in highlighted] == th["series"][:theme.ALL_PAIRS_CAP]


def test_scatter_mutes_the_unhighlighted_population():
    th = theme.theme("light")
    fig = charts.fig_scatter(_scatter_frame(), "pc1", "pc2", th, "t",
                             color_by="archetype_name", highlight=["a"])
    rest = next(t for t in fig.data if t.name == "everything else")
    assert rest.marker.color == th["muted"]
    assert rest.marker.opacity < 0.5


def test_a_continuous_key_uses_the_sequential_ramp_and_has_no_pair_limit():
    th = theme.theme("light")
    fig = charts.fig_scatter(_scatter_frame(), "pc1", "pc2", th, "t", color_by="age",
                             continuous=True)
    assert len(fig.data) == 1
    assert list(fig.data[0].marker.colorscale) == [
        (s[0], s[1]) for s in th["sequential"]]
    assert fig.data[0].marker.showscale is True


def test_a_single_series_scatter_uses_slot_one_and_no_legend():
    th = theme.theme("light")
    fig = charts.fig_scatter(_scatter_frame(), "pc1", "pc2", th, "t")
    assert fig.data[0].marker.color == th["series"][0]
    assert fig.layout.showlegend is False


def test_heatmap_picks_sequential_for_magnitude_and_diverging_for_polarity():
    th = theme.theme("light")
    z = pd.DataFrame(np.random.default_rng(0).normal(size=(4, 5)),
                     index=list("abcd"), columns=list("vwxyz"))
    mag = charts.fig_heatmap(z, th, "t", "c")
    pol = charts.fig_heatmap(z, th, "t", "c", diverging=True, zmid=0.0)
    assert list(mag.data[0].colorscale) == [(s[0], s[1]) for s in th["sequential"]]
    assert list(pol.data[0].colorscale) == [(s[0], s[1]) for s in th["diverging"]]
    assert pol.data[0].zmid == 0.0
    assert mag.data[0].zmid is None


def test_lines_are_two_px_with_markers_ringed_in_the_surface():
    th = theme.theme("dark")
    df = pd.DataFrame({"age": range(20, 30), "a": range(10), "b": range(10, 20)})
    fig = charts.fig_lines(df, "age", {"a": "A", "b": "B"}, th, "t")
    assert len(fig.data) == 2
    for i, trace in enumerate(fig.data):
        assert trace.line.width == 2
        assert trace.marker.size >= 8
        assert trace.marker.line.color == th["surface"]
        assert trace.line.color == th["series"][i]


def test_lines_direct_label_the_endpoint_rather_than_every_point():
    th = theme.theme("light")
    df = pd.DataFrame({"age": range(20, 30), "a": range(10)})
    fig = charts.fig_lines(df, "age", {"a": "A"}, th, "t", label_last=True)
    assert len(fig.layout.annotations) == 1
    assert fig.layout.annotations[0].text == "A"


def test_lines_skip_a_series_the_frame_does_not_carry():
    th = theme.theme("light")
    df = pd.DataFrame({"age": range(5), "a": range(5)})
    fig = charts.fig_lines(df, "age", {"a": "A", "missing": "M"}, th, "t")
    assert len(fig.data) == 1


# ── Wiring ────────────────────────────────────────────────────────────────────

def test_the_module_declares_nine_tabs():
    """Nine *topics* now, not nine `src/eda/` modules — the point of the revamp.

    The count coinciding with the pre-split app's nine is worth noticing and not
    relying on, so this asserts the declaration rather than a remembered number
    twice.
    """
    assert len(TABS) == 9
    assert len(TAB_NAMES) == len(TABS)
    assert TAB_NAMES[0] == "Problem" and TAB_NAMES[-1] == "Decision log"
    assert len(set(TAB_NAMES)) == 9


def test_every_tab_name_has_a_renderer():
    from dashboard.tabs import (availability, components, data, decision_log,
                                drafting, eda, minutes, problem, simulations)

    modules = [problem, data, eda, availability, minutes, components, simulations,
               drafting, decision_log]
    assert len(modules) == len(TABS)
    for module in modules:
        assert callable(module.render)
    # and the declaration dispatches to exactly those callables
    assert [render for _, render in TABS] == [m.render for m in modules]


def test_the_repo_root_is_on_the_path_for_package_imports():
    from dashboard import artifacts
    assert str(ROOT) == str(artifacts.ROOT)


def test_the_dashboard_imports_nothing_from_src():
    """The invariant the revamp bought: it reads artifacts and nothing else.

    `season_pairs()` in the pre-split app was the single `src/` import, so dropping
    tab 3's figures made this checkable rather than merely conventional.
    """
    offenders = []
    for path in sorted((ROOT / "dashboard").rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            for name in names:
                if name == "src" or name.startswith("src."):
                    offenders.append(f"{path.relative_to(ROOT)}: {name}")
    assert offenders == []


def test_pure_modules_do_not_import_streamlit():
    """decisions / economics / audit hold the only new logic worth testing."""
    for name in ("decisions.py", "economics.py", "audit.py"):
        path = ROOT / "dashboard" / name
        if not path.exists():
            continue
        assert "streamlit" not in path.read_text(), f"{name} must stay Streamlit-free"


# ── The registry ──────────────────────────────────────────────────────────────

def test_every_entry_uses_the_closed_status_vocabulary():
    for d in decisions.REGISTRY:
        assert d.status in decisions.STATUSES, (d.id, d.status)
        assert d.topic in decisions.TOPICS, (d.id, d.topic)


def test_every_entry_has_a_claim_a_reason_and_a_source_that_exists():
    for d in decisions.REGISTRY:
        assert d.claim.strip(), d.id
        assert d.because.strip(), d.id
        assert (ROOT / d.source).exists(), f"{d.id} cites missing {d.source}"


def test_entry_ids_are_unique():
    ids = [d.id for d in decisions.REGISTRY]
    assert len(ids) == len(set(ids))


def test_reviewed_and_date_are_iso_dates():
    for d in decisions.REGISTRY:
        date.fromisoformat(d.reviewed)
        date.fromisoformat(d.date)


def test_every_reproduce_artifact_exists_unless_the_status_exempts_it():
    """The anti-drift guard — the one audit check that is a hard test.

    An entry citing a target that was renamed or never built is exactly the drift
    `docs/dashboard-plan.md` names as its main risk, so this one fails the suite
    while the other three checks stay report-only.
    """
    assert audit.missing_artifacts() == []


def test_a_withdrawn_entry_keeps_its_replacement_and_what_caught_it():
    """Reversals are content — an entry that loses those fields loses the point."""
    for d in decisions.by_status("withdrawn"):
        assert d.replaced_by.strip(), d.id
        assert d.caught_by.strip(), d.id


def test_a_deadline_entry_names_its_date():
    for d in decisions.by_status("deadline"):
        assert d.due.strip(), d.id


def test_a_blocked_entry_names_the_unblocking_condition():
    for d in decisions.by_status("blocked"):
        assert d.unblocks.strip(), d.id


def test_incident_entries_carry_no_live_figure():
    """The deliberate exception to the provenance rule, and its boundary."""
    for d in decisions.by_status("incident"):
        assert not decisions.artifact_paths(d), (
            f"{d.id} is an incident but names an artifact — if it has a figure it is "
            f"not an incident")


def test_reproduce_parses_into_a_target_and_its_artifacts():
    d = _decision(reproduce="make stan-availability → outputs/predictions/a.csv, "
                            "outputs/predictions/b.csv")
    assert decisions.make_target(d) == "make stan-availability"
    assert decisions.artifact_paths(d) == ("outputs/predictions/a.csv",
                                           "outputs/predictions/b.csv")


def test_an_entry_with_no_reproduce_yields_no_artifacts():
    d = _decision(status="open", reproduce="")
    assert decisions.artifact_paths(d) == ()
    assert decisions.needs_artifact(d) is False


def test_status_mix_counts_every_status_in_the_vocabulary():
    mix = decisions.status_mix()
    assert set(mix) == set(decisions.STATUSES)
    assert sum(mix.values()) == len(decisions.REGISTRY)


def test_every_topic_has_at_least_one_entry():
    """A tab whose registry slice is empty renders a heading and nothing else."""
    for topic in decisions.TOPICS:
        assert decisions.by_topic(topic), topic


# ── The audit ─────────────────────────────────────────────────────────────────

def test_the_staleness_check_flags_an_entry_older_than_its_source_doc():
    stale = _decision(id="stale", reviewed="2020-01-01")
    fresh = _decision(id="fresh", reviewed="2026-12-31")
    moved = lambda path, root: date(2026, 6, 1)          # noqa: E731 — injected stub

    flagged = audit.stale_entries((stale, fresh), ROOT, commit_date=moved)
    assert [f.subject for f in flagged] == ["stale"]
    assert "2026-06-01" in flagged[0].detail


def test_the_staleness_check_is_silent_when_git_knows_nothing_about_the_doc():
    d = _decision(reviewed="2020-01-01")
    assert audit.stale_entries((d,), ROOT, commit_date=lambda p, r: None) == []


def test_the_staleness_check_uses_commit_dates_not_mtime(tmp_path):
    """A fresh checkout resets mtime, so mtime would flag the whole registry."""
    doc = tmp_path / "CLAUDE.md"
    doc.write_text("x")
    assert audit.last_commit_date("CLAUDE.md", tmp_path) is None


def test_the_orphan_check_flags_an_artifact_no_entry_and_no_tab_references(tmp_path):
    art = tmp_path / "outputs" / "eda"
    art.mkdir(parents=True)
    (art / "referenced_by_registry.csv").write_text("a\n1\n")
    (art / "referenced_by_a_tab.csv").write_text("a\n1\n")
    (art / "nobody_reads_this.csv").write_text("a\n1\n")

    pkg = tmp_path / "dashboard"
    pkg.mkdir()
    (pkg / "some_tab.py").write_text(
        'path = ctx.eda("referenced_by_a_tab.csv")\n')

    registry = (_decision(reproduce="make thing → outputs/eda/referenced_by_registry.csv"),)
    flagged = audit.orphaned_artifacts(root=tmp_path, registry=registry, package=pkg,
                                       dirs=["outputs/eda"])
    assert [f.subject for f in flagged] == ["outputs/eda/nobody_reads_this.csv"]


def test_the_orphan_check_credits_a_glob_for_a_whole_artifact_family(tmp_path):
    """A target that writes twenty files gets one registry line, not twenty."""
    art = tmp_path / "data" / "features"
    art.mkdir(parents=True)
    for name in ("pca_tierA_pooled.pkl", "pca_tierB_within_season_scores.parquet"):
        (art / name).write_text("x")
    (art / "unrelated.parquet").write_text("x")

    pkg = tmp_path / "dashboard"
    pkg.mkdir()
    (pkg / "empty.py").write_text("x = 1\n")

    registry = (_decision(reproduce="make pca → data/features/pca_*"),)
    flagged = audit.orphaned_artifacts(root=tmp_path, registry=registry, package=pkg,
                                       dirs=["data/features"])
    assert [f.subject for f in flagged] == ["data/features/unrelated.parquet"]


def test_a_tier_parameterized_read_credits_the_whole_family(tmp_path):
    """`f"team_context_tier{tier}.parquet"` never contains the tier literally."""
    art = tmp_path / "data" / "features"
    art.mkdir(parents=True)
    (art / "team_context_tierA.parquet").write_text("x")
    (art / "team_context_tierB.parquet").write_text("x")

    pkg = tmp_path / "dashboard"
    pkg.mkdir()
    (pkg / "tab.py").write_text(
        'p = ctx.features(f"team_context_tier{ctx.tier}.parquet")\n')

    assert audit.orphaned_artifacts(root=tmp_path, registry=(), package=pkg,
                                    dirs=["data/features"]) == []


def test_the_missing_artifact_check_reports_a_renamed_target(tmp_path):
    registry = (_decision(id="renamed",
                          reproduce="make thing → outputs/eda/never_written.csv"),)
    flagged = audit.missing_artifacts(registry, tmp_path)
    assert [f.subject for f in flagged] == ["renamed"]
    assert "make thing" in flagged[0].detail


def test_the_missing_artifact_check_skips_the_unbacked_statuses(tmp_path):
    for status in decisions.UNBACKED_STATUSES:
        d = _decision(status=status, reproduce="")
        assert audit.missing_artifacts((d,), tmp_path) == []


def test_an_entry_that_owes_a_figure_and_names_none_is_flagged(tmp_path):
    d = _decision(status="measured", reproduce="")
    flagged = audit.missing_artifacts((d,), tmp_path)
    assert len(flagged) == 1
    assert "obliges" in flagged[0].detail


def test_the_pending_marker_count_is_the_provenance_gap(tmp_path):
    pkg = tmp_path / "dashboard"
    pkg.mkdir()
    (pkg / "gated.py").write_text(
        "from dashboard.layout import pending_marker\n"
        "def render(ctx):\n"
        "    pending_marker('make variance-budget', 'the variance budget')\n")
    (pkg / "clean.py").write_text("def render(ctx):\n    pass\n")

    flagged = audit.pending_markers(package=pkg)
    assert len(flagged) == 1
    assert flagged[0].subject.endswith("gated.py:3")


def test_the_pending_marker_count_ignores_the_definition_itself(tmp_path):
    pkg = tmp_path / "dashboard"
    pkg.mkdir()
    (pkg / "layout.py").write_text("def pending_marker(target, figure):\n    pass\n")
    assert audit.pending_markers(package=pkg) == []


def test_source_literals_read_fstring_parts_but_not_comments(tmp_path):
    pkg = tmp_path / "dashboard"
    pkg.mkdir()
    (pkg / "m.py").write_text(
        '# mentions ghost_artifact.csv in a comment only\n'
        'a = "real_artifact.csv"\n'
        'b = f"prefixed_tier{tier}.parquet"\n'
        'c = "short.csv"\n')
    literals = audit.source_literals(pkg)
    assert "real_artifact.csv" in literals
    assert "prefixed_tier" in literals
    assert not any("ghost_artifact" in lit for lit in literals)
    assert "short.csv" not in literals                    # below MIN_LITERAL


def test_the_audit_report_renders_every_check_and_the_two_headline_counts():
    results = audit.run()
    text = audit.report(results)
    for heading in audit.HEADINGS.values():
        assert heading in text
    assert "typed constants pending:" in text
    assert "orphaned artifacts:" in text


# ── Tournament economics ──────────────────────────────────────────────────────

def test_rake_and_hurdle_match_hand_computed_values_on_a_synthetic_table():
    # 100 entries at $10 is a $1,000 pool; paying $800 keeps 20%.
    r = economics.rake(entries=100, fee=10, prizes=800)
    assert abs(r - 0.20) < 1e-12
    # Keeping 20% of the pool means beating the field by 25% to return the fee —
    # 1/0.8 = 1.25 — not by 20%. That gap is the whole reason for the hurdle unit.
    assert abs(economics.break_even_hurdle(r) - 0.25) < 1e-12


def test_a_zero_rake_contest_has_a_zero_hurdle():
    assert economics.break_even_hurdle(economics.rake(10, 5, 50)) == 0.0


def test_rake_rejects_an_empty_pool():
    try:
        economics.rake(entries=0, fee=10, prizes=100)
    except ValueError:
        return
    raise AssertionError("an empty buy-in pool must raise")


def test_the_metadata_loader_drops_the_spreadsheets_trailing_columns():
    meta = economics.load_metadata()
    assert not any(str(c).startswith("Unnamed:") for c in meta.columns)
    assert {"type", "total_entries", "entry_fee_per_team", "total_prizes"} <= set(
        meta.columns)


def test_the_five_real_tournaments_derive_their_own_economics():
    econ = economics.economics()
    assert len(econ) == 5
    # every rake is a plausible house cut, and the hurdle always exceeds it
    assert (econ["rake"] > 0).all() and (econ["rake"] < 0.30).all()
    assert (econ["break_even_hurdle"] > econ["rake"]).all()
    # the pool is the identity it is defined by
    assert (econ["buy_in_pool"]
            == econ["total_entries"] * econ["entry_fee_per_team"]).all()


def test_the_derived_advance_counts_reproduce_the_five_known_tournaments():
    adv = economics.advance_table()
    advancing = adv[adv["n_advance"] > 0]
    assert set(advancing["tournament"]) == set(economics.load_metadata()["type"])
    # the plan's stated range: 1-of-12 through 2-of-6
    pairs = set(zip(advancing["n_advance"], advancing["pod_size"]))
    assert (1, 12) in pairs and (2, 6) in pairs and (2, 12) in pairs
    assert advancing["advance_rate"].between(0, 1).all()


def test_round_one_is_a_zero_consolation_knockout_in_every_tournament():
    adv = economics.advance_table()
    r1 = adv[adv["round"] == 1]
    assert len(r1) == 5
    assert (r1["cash_places"] == 0).all()      # ranks 3-12 get nothing
    assert (r1["n_advance"] == 2).all()
    assert bool(r1["zero_consolation"].all())


def test_the_round_one_pod_size_is_confirmed_by_the_entry_counts():
    """The prize CSV does not carry it, so it needs a second, independent route."""
    assert economics.check_round_one_pod(round_one_pod=12) is True
    for wrong in (2, 3, 4, 6, 8, 10, 11, 13, 14, 20, 24):
        assert economics.check_round_one_pod(round_one_pod=wrong) is False


def test_integrality_alone_does_not_pin_the_round_one_pod():
    """Worth pinning, because it is the trap in this derivation.

    Any divisor of 12 gives a whole-number chain — halving the pod just doubles the
    next round's field. What actually identifies 12 is that four of the five
    tournaments then have a final round paying *exactly* its own field.
    """
    six = economics.round_one_pod_evidence(round_one_pod=6)
    twelve = economics.round_one_pod_evidence(round_one_pod=12)
    assert six["integral"] is True                    # necessary, not sufficient
    assert six["final_field_equals_paid"] == 0
    assert twelve["final_field_equals_paid"] == 4     # only 15k_and_one differs


def test_the_payout_curve_expands_the_banded_prize_rows():
    curve = economics.payout_curve(tournament="88k_alley_oop")
    assert list(curve["place"]) == [1, 2, 3, 4]
    assert curve["cash"].is_monotonic_decreasing
    # and an unknown tournament yields an empty frame rather than raising
    assert economics.payout_curve(tournament="no_such_contest").empty


def test_payout_convexity_differs_by_an_order_of_magnitude_across_the_five():
    econ = economics.economics()
    multiples = econ["first_prize_multiple"]
    assert multiples.max() / multiples.min() > 10


# ── Artifact loading ──────────────────────────────────────────────────────────

def test_optional_returns_none_for_a_missing_path_without_raising(tmp_path):
    from dashboard.artifacts import optional
    assert optional(tmp_path / "nope.csv") is None


def test_optional_reads_a_real_table(tmp_path):
    from dashboard.artifacts import optional
    path = tmp_path / "small.csv"
    pd.DataFrame({"a": [1, 2]}).to_csv(path, index=False)
    frame = optional(path)
    assert frame is not None and list(frame["a"]) == [1, 2]


def test_the_inventory_counts_parquet_rows_from_metadata(tmp_path):
    """A 37 MB parquet must cost a footer seek, not a load."""
    from dashboard.artifacts import _row_count
    path = tmp_path / "f.parquet"
    pd.DataFrame({"a": range(37)}).to_parquet(path)
    assert _row_count(path) == 37


def test_the_inventory_counts_csv_rows_without_the_header(tmp_path):
    from dashboard.artifacts import _row_count
    path = tmp_path / "f.csv"
    pd.DataFrame({"a": range(5)}).to_csv(path, index=False)
    assert _row_count(path) == 5


def test_pipeline_health_expects_exactly_what_the_registry_names():
    from dashboard.artifacts import pipeline_health
    health = pipeline_health()
    assert set(health["artifact"]) == set(decisions.all_artifacts())
