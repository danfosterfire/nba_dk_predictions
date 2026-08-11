"""Artifact resolution and cached reads — the dashboard's only I/O layer.

**The dashboard reads artifacts and nothing else.** There is no import from `src/`
anywhere in this package, and a test pins it. Nothing here fits, projects or refits: the
PCA was fitted by `make pca`, and this module opens the three files it wrote.
"""

from pathlib import Path

import pandas as pd
import streamlit as st
import yaml

ROOT = Path(__file__).resolve().parent.parent


@st.cache_data(show_spinner=False)
def load_cfg() -> dict:
    return yaml.safe_load((ROOT / "configs" / "default.yaml").read_text())


def features_dir() -> Path:
    return ROOT / load_cfg()["data"]["features_dir"]


def predictions_dir() -> Path:
    """Where the model and simulation layers write their measured artifacts."""
    return ROOT / load_cfg()["evaluation"]["predictions_dir"]


def eda_dir() -> Path:
    """Where the EDA and capture-status reports land."""
    return ROOT / load_cfg()["eda"]["output_dir"]


@st.cache_data(show_spinner=False)
def read_table(path_str: str, columns: tuple[str, ...] | None = None) -> pd.DataFrame:
    path = Path(path_str)
    if path.suffix == ".parquet":
        return pd.read_parquet(path, columns=list(columns) if columns else None)
    return pd.read_csv(path)


def optional(path: Path, columns: tuple[str, ...] | None = None,
             target: str = "make pca") -> pd.DataFrame | None:
    """Read an artifact, or name the make target that produces it and return None."""
    if not path.exists():
        st.warning(f"`{rel(path)}` not found — run `{target}` to build it.")
        return None
    return read_table(str(path), columns)


def rel(path: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return str(path)
