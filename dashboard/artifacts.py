"""Artifact resolution and cached reads — the dashboard's only I/O layer.

**The dashboard reads artifacts and nothing else.** There is no import from `src/`
anywhere in this package: dropping tab 3's `season_pairs()` helper, which was the
single such import, turned that from a convention into an invariant, and a test
pins it. Nothing here fits, clusters, refits or fetches; the tier and appearance
toggles switch which artifact is read, never recompute one.
"""

import pickle
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st
import yaml

from dashboard import decisions as D

ROOT = Path(__file__).resolve().parent.parent

# Directories the inventory walks, in the order the sidebar reports them.
ARTIFACT_DIRS = ("data/features", "outputs/eda", "outputs/predictions")


@st.cache_data(show_spinner=False)
def load_cfg() -> dict:
    return yaml.safe_load((ROOT / "configs" / "default.yaml").read_text())


def features_dir() -> Path:
    return ROOT / load_cfg()["data"]["features_dir"]


def eda_dir() -> Path:
    return ROOT / load_cfg()["eda"]["output_dir"]


def predictions_dir() -> Path:
    return ROOT / load_cfg()["evaluation"]["predictions_dir"]


@st.cache_data(show_spinner=False)
def read_table(path_str: str, columns: tuple[str, ...] | None = None) -> pd.DataFrame:
    path = Path(path_str)
    if path.suffix == ".parquet":
        return pd.read_parquet(path, columns=list(columns) if columns else None)
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def read_pickle(path_str: str):
    with open(path_str, "rb") as f:
        return pickle.load(f)


def optional(path: Path, columns: tuple[str, ...] | None = None,
             target: str = "") -> pd.DataFrame | None:
    """Read an artifact, or name the make target that produces it and return None."""
    if not path.exists():
        hint = f"run `{target}`" if target else "run `make eda`"
        st.warning(f"`{_rel(path)}` not found — {hint} to build it.")
        return None
    return read_table(str(path), columns)


def _rel(path: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


# ── The context object every tab renders against ──────────────────────────────

@dataclass(frozen=True)
class Ctx:
    """What a tab needs, in one object.

    The pre-split renderers took `(th, tier, mode)` positionally and
    inconsistently — some took two, some three, one took only the theme. One
    context stops that from growing nine different ways.
    """

    th: dict                     # the resolved theme dict, per appearance mode
    appearance: str              # "light" | "dark"
    tier: str                    # "A" | "B" — scopes the coverage heatmap only
    mode: str                    # "within_season" | "pooled" — likewise

    def features(self, name: str) -> Path:
        return features_dir() / name

    def eda(self, name: str) -> Path:
        return eda_dir() / name

    def predictions(self, name: str) -> Path:
        return predictions_dir() / name


# ── Inventory and pipeline health ─────────────────────────────────────────────

def _row_count(path: Path) -> int | None:
    """Rows without reading the frame, where the format allows it.

    Parquet carries `num_rows` in its footer, so a 37 MB file costs a seek rather
    than a load — which is what keeps `component_targets.parquet` off the read path.
    """
    try:
        if path.suffix == ".parquet":
            import pyarrow.parquet as pq
            return pq.ParquetFile(path).metadata.num_rows
        if path.suffix == ".csv":
            with open(path, "rb") as f:
                return max(sum(1 for _ in f) - 1, 0)
    except Exception:
        return None
    return None


@st.cache_data(show_spinner=False)
def inventory() -> pd.DataFrame:
    """Every artifact on disk with its size, row count and mtime.

    Also the input to the audit's orphan check, which is the inverse question:
    nine artifact families accumulated unreachable from the dashboard precisely
    because nothing was asking which ones nothing reads.
    """
    rows = []
    for rel_dir in ARTIFACT_DIRS:
        d = ROOT / rel_dir
        if not d.exists():
            continue
        for path in sorted(d.iterdir()):
            if not path.is_file() or path.name.startswith("."):
                continue
            stat = path.stat()
            rows.append({
                "directory": rel_dir,
                "artifact": path.name,
                "path": f"{rel_dir}/{path.name}",
                "rows": _row_count(path),
                "mb": round(stat.st_size / 1e6, 3),
                "modified": datetime.fromtimestamp(
                    stat.st_mtime, tz=timezone.utc).strftime("%Y-%m-%d %H:%M"),
            })
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def pipeline_health() -> pd.DataFrame:
    """One row per artifact the registry expects, present or not.

    Expectations come from the registry's `reproduce` fields rather than a second
    hand-maintained list, so "what should exist" has exactly one definition and
    `make dashboard-audit` checks the same set the sidebar reports.
    """
    rows = []
    for d in D.REGISTRY:
        if not D.needs_artifact(d):
            continue
        for rel in D.artifact_paths(d):
            rows.append({
                "artifact": rel,
                "target": D.make_target(d),
                "decision": d.id,
                "present": (ROOT / rel).exists(),
            })
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.drop_duplicates("artifact").sort_values("artifact")
