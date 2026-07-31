"""`make dashboard-audit` — a report on registry drift, not a gate.

Drift between `dashboard/decisions.py` and `CLAUDE.md` is the single largest risk in
`docs/dashboard-plan.md`, and it gets four checks rather than one:

| check | why it matters |
|---|---|
| every `reproduce` artifact exists | catches an entry citing a target that was renamed or never built |
| no source doc has a commit newer than the entry's `reviewed` date | the docs-folder sweep the drift risk actually needs |
| artifacts on disk that no tab and no entry references | the inverse check — nine orphaned families accumulated because nothing was looking |
| pending provenance markers remaining | should trend to zero as `docs/provenance-plan.md` lands; a rising count is a regression |

Only the first is also a `pytest` test. The other three are **report-only** on
purpose: failing the suite because somebody edited a doc would train people to
ignore the suite.

Pure logic plus a `__main__` block — **no Streamlit import**, so the tests exercise
each check directly. Run as `python -m dashboard.audit`, a deliberate minor
departure from this repo's `python -m src.<module>` convention: the audit is about
the dashboard, not the data pipeline, and filing it under `src/eda/` would misfile
it.
"""

import ast
import fnmatch
import subprocess
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import yaml

from dashboard import decisions as D

ROOT = Path(__file__).resolve().parent.parent
PACKAGE = ROOT / "dashboard"

# A string literal in the dashboard source counts as an artifact reference when it
# is long enough and shaped like a filename fragment. f-strings contribute their
# constant parts, so `f"team_context_tier{tier}.parquet"` yields
# "team_context_tier" — which is how a tier-parameterized read is credited without
# the tier ever appearing literally.
MIN_LITERAL = 10


@dataclass(frozen=True)
class Finding:
    check: str
    subject: str
    detail: str


# ── Check 1 · every `reproduce` artifact exists ────────────────────────────────

def _matches(root: Path, pattern: str) -> list[Path]:
    """Resolve one `reproduce` path, which may be a glob for a whole family."""
    if "*" in pattern:
        return sorted(root.glob(pattern))
    candidate = root / pattern
    return [candidate] if candidate.exists() else []


def missing_artifacts(registry: tuple[D.Decision, ...] = D.REGISTRY,
                      root: Path = ROOT) -> list[Finding]:
    """Entries obliged to name a live artifact whose artifact is not on disk.

    `open`/`blocked`/`deadline`/`incident` are skipped: those statuses exist
    precisely to describe something that has no artifact yet, or by design never
    will.
    """
    out = []
    for d in registry:
        if not D.needs_artifact(d):
            continue
        paths = D.artifact_paths(d)
        if not paths:
            out.append(Finding("missing-artifact", d.id,
                               f"status `{d.status}` obliges a `reproduce` link and "
                               f"the entry has none"))
            continue
        for pattern in paths:
            if not _matches(root, pattern):
                out.append(Finding("missing-artifact", d.id,
                                   f"`{pattern}` not on disk — "
                                   f"`{D.make_target(d)}` should write it"))
    return out


# ── Check 2 · source docs newer than the `reviewed` date ───────────────────────

def last_commit_date(path: str, root: Path = ROOT) -> date | None:
    """Git commit date, not filesystem mtime — a fresh checkout resets mtime."""
    try:
        out = subprocess.run(
            ["git", "log", "-1", "--format=%cI", "--", path],
            cwd=root, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    stamp = out.stdout.strip()
    if not stamp:
        return None
    return date.fromisoformat(stamp[:10])


def stale_entries(registry: tuple[D.Decision, ...] = D.REGISTRY,
                  root: Path = ROOT,
                  commit_date=last_commit_date) -> list[Finding]:
    """Entries whose source doc changed after the entry was last checked.

    Grouped by doc in the report, because the useful sentence is
    "`docs/availability-plan.md` changed since these 9 entries were last checked",
    not nine separate lines.
    """
    out = []
    cache: dict[str, date | None] = {}
    for d in registry:
        if d.source not in cache:
            cache[d.source] = commit_date(d.source, root)
        moved = cache[d.source]
        if moved is None:
            continue
        try:
            reviewed = date.fromisoformat(d.reviewed)
        except ValueError:
            out.append(Finding("stale-entry", d.id,
                               f"`reviewed` is not an ISO date: {d.reviewed!r}"))
            continue
        if moved > reviewed:
            out.append(Finding("stale-entry", d.id,
                               f"`{d.source}` last changed {moved}, "
                               f"reviewed {reviewed}"))
    return out


# ── Check 3 · artifacts nothing references ────────────────────────────────────

def source_literals(package: Path = PACKAGE) -> set[str]:
    """Filename-shaped string literals across the dashboard package.

    Read with `ast` rather than a regex so an f-string's constant parts are
    recovered as literals and a comment mentioning an artifact is not.
    """
    found: set[str] = set()

    def keep(value: str) -> None:
        if len(value) >= MIN_LITERAL and ("_" in value or "." in value):
            found.add(value)

    for path in sorted(package.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                keep(node.value)
            elif isinstance(node, ast.JoinedStr):
                for part in node.values:
                    if isinstance(part, ast.Constant) and isinstance(part.value, str):
                        keep(part.value)
    return found


def _referenced_by_source(rel: str, literals: set[str]) -> bool:
    name = Path(rel).name
    return any(lit in name or name in lit or lit in rel for lit in literals)


def _referenced_by_registry(rel: str,
                            registry: tuple[D.Decision, ...]) -> bool:
    for d in registry:
        for pattern in D.artifact_paths(d):
            if pattern == rel or fnmatch.fnmatch(rel, pattern):
                return True
    return False


def artifact_dirs(root: Path = ROOT) -> list[str]:
    cfg = yaml.safe_load((root / "configs" / "default.yaml").read_text())
    return [cfg["data"]["features_dir"], cfg["eda"]["output_dir"],
            cfg["evaluation"]["predictions_dir"]]


def on_disk(root: Path = ROOT, dirs: list[str] | None = None) -> list[str]:
    out = []
    for rel_dir in (dirs if dirs is not None else artifact_dirs(root)):
        d = root / rel_dir
        if not d.exists():
            continue
        for path in sorted(d.iterdir()):
            if path.is_file() and not path.name.startswith("."):
                out.append(f"{rel_dir}/{path.name}")
    return out


def orphaned_artifacts(root: Path = ROOT,
                       registry: tuple[D.Decision, ...] = D.REGISTRY,
                       package: Path = PACKAGE,
                       dirs: list[str] | None = None) -> list[Finding]:
    """Artifacts on disk that no tab reads and no registry entry accounts for.

    The check that would have caught the problem this whole revamp exists to fix:
    `serial_correlation.csv`, `availability_profile.csv`, every `stan_*` output and
    both tournament CSVs were unreachable from the dashboard, and nothing said so
    because nothing was asking.

    An artifact is accounted for either way round — a tab that renders it, or a
    registry entry that names its make target. Both count, because a deliberately
    deferred family (the PCA line) is *recorded* rather than rendered, and that is
    the honest state to report.
    """
    literals = source_literals(package)
    out = []
    for rel in on_disk(root, dirs):
        if _referenced_by_registry(rel, registry):
            continue
        if _referenced_by_source(rel, literals):
            continue
        out.append(Finding("orphaned-artifact", rel,
                           "no tab reads it and no registry entry names it"))
    return out


# ── Check 4 · pending provenance markers ──────────────────────────────────────

def pending_markers(package: Path = PACKAGE) -> list[Finding]:
    """Calls to `layout.pending_marker`, each standing in for an unbacked figure.

    This is the plan's "count of provenance-marked typed constants remaining": a
    marker is what a panel renders *instead of* typing a number it cannot read from
    an artifact, so the marker count is the count of figures still owed a `make`
    target. It should trend to zero, and a rise is a regression.
    """
    out = []
    for path in sorted(package.rglob("*.py")):
        if path.name == "layout.py":
            continue                       # its own definition, not a use
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = (func.attr if isinstance(func, ast.Attribute)
                    else getattr(func, "id", None))
            if name == "pending_marker":
                out.append(Finding(
                    "pending-marker", f"{path.relative_to(package.parent)}:{node.lineno}",
                    "a figure with no artifact behind it"))
    return out


# ── Report ────────────────────────────────────────────────────────────────────

def run(root: Path = ROOT, package: Path = PACKAGE) -> dict[str, list[Finding]]:
    return {
        "missing-artifact": missing_artifacts(root=root),
        "stale-entry": stale_entries(root=root),
        "orphaned-artifact": orphaned_artifacts(root=root, package=package),
        "pending-marker": pending_markers(package=package),
    }


HEADINGS = {
    "missing-artifact": "Registry entries whose artifact is not on disk",
    "stale-entry": "Entries whose source doc changed after they were reviewed",
    "orphaned-artifact": "Artifacts no tab reads and no entry names",
    "pending-marker": "Pending provenance markers (typed constants owed a target)",
}


def report(results: dict[str, list[Finding]]) -> str:
    lines = ["Dashboard audit", "=" * 60, ""]
    mix = {k: v for k, v in D.status_mix().items() if v}
    lines.append(f"registry: {len(D.REGISTRY)} entries across "
                 f"{len({d.topic for d in D.REGISTRY})} topics")
    lines.append("  " + " · ".join(f"{k} {v}" for k, v in mix.items()))
    lines.append("")

    for check, findings in results.items():
        lines.append(f"{HEADINGS[check]}: {len(findings)}")
        for f in findings:
            lines.append(f"  - {f.subject}: {f.detail}")
        lines.append("")

    total = sum(len(v) for v in results.values())
    lines.append("-" * 60)
    lines.append(f"typed constants pending: {len(results['pending-marker'])}")
    lines.append(f"orphaned artifacts:      {len(results['orphaned-artifact'])}")
    lines.append(f"total findings:          {total}")
    return "\n".join(lines)


if __name__ == "__main__":
    results = run()
    text = report(results)
    print(text)
    # A report, not a gate: exit 0 even with findings, so the weekly job's log is a
    # two-minute read rather than a failure to triage.
    sys.exit(0)
