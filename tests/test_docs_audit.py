"""The docs-audit machinery: precision inference, the three checks, and the registry.

The audit exists because prose drifted away from its artifact twice without anyone
noticing, so these tests are mostly about the *guards* — that a wrong figure is caught,
that a claim which stops describing the doc is caught, and that a missing artifact is
skipped rather than failed.
"""

from pathlib import Path

import pandas as pd

from src import docs_audit as A

ROOT = Path(__file__).resolve().parent.parent


# ── Synthetic builders ────────────────────────────────────────────────────────

def _claim(quoted: str, value: float, doc: str = "docs/fake.md",
           artifact: str = "outputs/eda/fake.csv") -> A.Claim:
    return A.Claim(doc=doc, quoted=quoted, artifact=artifact,
                   actual=lambda: value, label=f"synthetic {quoted}")


def _doc(tmp_path: Path, name: str, text: str) -> Path:
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


# ── Precision inference ───────────────────────────────────────────────────────

def test_tolerance_is_half_a_unit_in_the_last_quoted_place():
    """A figure quoted to two decimals must not be held to four."""
    assert A.implied_tolerance("22.7") == 0.05
    assert A.implied_tolerance("0.0635") == 0.00005
    assert A.implied_tolerance("0.283") == 0.0005
    assert A.implied_tolerance("12,406") == 0.5
    assert A.implied_tolerance("254") == 0.5


def test_quoted_values_parse_through_separators_and_units():
    assert A.parse_quoted("12,406") == 12406.0
    assert A.parse_quoted("22.7×") == 22.7
    assert A.parse_quoted("26.7%") == 26.7
    assert A.parse_quoted("−0.084") == -0.084          # unicode minus, not a hyphen


def test_a_percentage_is_compared_against_a_fraction(tmp_path):
    """`26.7%` in prose against `0.267325` in the artifact is agreement, not drift."""
    _doc(tmp_path, "docs/fake.md", "the share is 26.7% of players")
    (tmp_path / "outputs/eda").mkdir(parents=True)
    (tmp_path / "outputs/eda/fake.csv").write_text("a\n1\n")
    bad, skipped = A.check_values((_claim("26.7%", 0.267325),), tmp_path)
    assert bad == [] and skipped == []


# ── The value check ───────────────────────────────────────────────────────────

def test_a_correctly_rounded_figure_passes(tmp_path):
    _doc(tmp_path, "docs/fake.md", "0.317")
    (tmp_path / "outputs/eda").mkdir(parents=True)
    (tmp_path / "outputs/eda/fake.csv").write_text("a\n1\n")
    bad, _ = A.check_values((_claim("0.317", 0.317265),), tmp_path)
    assert bad == []


def test_a_figure_beyond_its_own_rounding_is_caught(tmp_path):
    """The season-total R² failure, in miniature: 0.10 quoted against 0.1414."""
    _doc(tmp_path, "docs/fake.md", "0.10")
    (tmp_path / "outputs/eda").mkdir(parents=True)
    (tmp_path / "outputs/eda/fake.csv").write_text("a\n1\n")
    bad, _ = A.check_values((_claim("0.10", 0.141406),), tmp_path)
    assert len(bad) == 1
    assert bad[0].check == "value-mismatch"
    assert "0.10" in bad[0].detail and "0.141406" in bad[0].detail


def test_a_missing_artifact_is_skipped_rather_than_failed(tmp_path):
    """A fresh checkout without `make eda` must not report a wall of red."""
    _doc(tmp_path, "docs/fake.md", "0.317")
    bad, skipped = A.check_values((_claim("0.317", 0.317265),), tmp_path)
    assert bad == []
    assert len(skipped) == 1 and skipped[0].check == "missing-artifact"


def test_a_lookup_that_finds_no_row_is_skipped_not_failed(tmp_path):
    """A renamed column should not masquerade as a figure being wrong."""
    _doc(tmp_path, "docs/fake.md", "0.317")
    (tmp_path / "outputs/eda").mkdir(parents=True)
    (tmp_path / "outputs/eda/fake.csv").write_text("a\n1\n")
    bad, skipped = A.check_values((_claim("0.317", float("nan")),), tmp_path)
    assert bad == []
    assert len(skipped) == 1 and skipped[0].check == "no-such-row"


# ── The presence check ────────────────────────────────────────────────────────

def test_a_claim_whose_text_left_the_doc_is_flagged(tmp_path):
    """The anti-rot guard: a claim that stops describing the doc is itself drift.

    Without this, correcting a figure in prose and forgetting the registry leaves a
    claim that passes forever while describing nothing — the same failure one level up.
    """
    _doc(tmp_path, "docs/fake.md", "the share is 0.469 now")
    stale = A.check_presence((_claim("0.470", 0.4691),), tmp_path)
    assert len(stale) == 1 and stale[0].check == "stale-claim"
    assert "0.470" in stale[0].detail


def test_a_claim_still_present_in_the_doc_is_not_flagged(tmp_path):
    _doc(tmp_path, "docs/fake.md", "| scratch | **0.469** |")
    assert A.check_presence((_claim("0.469", 0.4691),), tmp_path) == []


def test_presence_matches_inside_markdown_emphasis(tmp_path):
    """Figures are routinely bolded in these tables, which must not defeat the check."""
    _doc(tmp_path, "docs/fake.md", "CRPS **10.795** against")
    assert A.check_presence((_claim("10.795", 10.795207),), tmp_path) == []


# ── Coverage ──────────────────────────────────────────────────────────────────

def test_season_labels_and_dates_are_not_counted_as_measurements():
    text = "Held out on 2024-25 and 2025-26, settled 2026-07-29, since 1996."
    assert A.numeric_literals(text) == []


def test_fenced_and_inline_code_is_excluded():
    assert A.numeric_literals("```\nmean 0.803\n```") == []
    assert A.numeric_literals("`gp_share_lag1 = 0.317`") == []


def test_a_range_is_not_read_as_a_negative_number():
    """`12–24` is a bucket, not minus twenty-four."""
    assert "-24" not in A.numeric_literals("bucket 12–24 mpg")


def test_measurements_are_decimals_percentages_and_separated_counts():
    assert A.is_measurement("0.483") and A.is_measurement("26.7%")
    assert A.is_measurement("12,406")
    assert not A.is_measurement("30")          # seasons
    assert not A.is_measurement("82")          # games


def test_coverage_reports_both_denominators():
    cov = A.coverage(A.AVAIL)
    assert cov["measurements"] <= cov["numbers"]
    assert cov["covered"] <= cov["measurements"]
    assert 0.0 <= cov["share"] <= 1.0


# ── The registry itself ───────────────────────────────────────────────────────

def test_every_claim_names_a_doc_that_exists():
    for claim in A.CLAIMS:
        assert (ROOT / claim.doc).exists(), claim.label


def test_every_claim_names_an_artifact_under_a_tracked_directory():
    for claim in A.CLAIMS:
        assert claim.artifact.startswith(("outputs/", "data/")), claim.label


def test_every_claim_has_a_label_and_a_parseable_quote():
    for claim in A.CLAIMS:
        assert claim.label.strip(), claim.quoted
        A.parse_quoted(claim.quoted)           # raises if it is not a number
        assert claim.tolerance() > 0


def test_the_registry_covers_every_head_the_plan_documents():
    """A doc section losing its claims entirely would otherwise be invisible."""
    artifacts = {c.artifact for c in A.CLAIMS}
    for required in (A.PROFILE, A.METRICS, A.ABLATION, A.MIN_NONLIN,
                     A.STAN_MIN_M, A.STAN_MIN_D, A.STAN_AV_M, A.STAN_AV_D,
                     A.SEASON_TOTAL, A.REPORT_CAL):
        assert required in artifacts, required


def test_no_claim_text_has_drifted_out_of_its_doc():
    """Hard test — the registry must keep describing the docs it audits."""
    assert A.check_presence() == []


def test_every_quoted_figure_agrees_with_its_artifact():
    """The gate. A doc contradicting its artifact is a defect, not a preference.

    Skips cleanly when the artifacts have not been built, so this does not fail on a
    fresh checkout.
    """
    bad, skipped = A.check_values()
    if len(skipped) == len(A.CLAIMS):
        return                                  # nothing built; nothing to check
    assert bad == [], "\n".join(f"{f.label}: {f.detail}" for f in bad)
