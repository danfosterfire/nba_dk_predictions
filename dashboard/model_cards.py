"""The model pages' pure layer — one head's card, cut into the seven blocks a page draws.

`src/models/model_cards.py` (`make model-cards`) writes eight flat artifacts describing
every fitted head: what it is, what it was fed, what it learned, and how well it predicts.
This module reshapes those into the frames `dashboard/views/model_page.py` hands to
`charts.py`, and **nothing else**. It fits nothing, refits nothing and draws no posterior —
that is the whole reason the emitter exists, and `docs/model-cards-plan.md` is the contract
between them.

**No Streamlit import**, so every rule below is exercised as a plain function in
`tests/test_dashboard.py` rather than through a rendered page, and no function here opens a
file: the view reads the artifacts and hands frames in.

## The three things this module knows that the artifacts do not

**Which heads make a page.** `CLASSES` is the sidebar's four model pages and the head order
inside each — declared here rather than read off `model_card_index.csv`, because the
*order* is a reading order (entry, then onset, then duration, then exit is how a tenure
runs) and no column carries it. `test_every_carded_head_belongs_to_exactly_one_model_page`
holds the coverage, so a twenty-first head fails a test instead of vanishing from the
navigation.

**Where each head's sampler diagnostics live.** `make stan` writes one diagnostics table per
model class with a label per *arm* of that class's variant ladder, so finding a head's row is
a per-class rule rather than a lookup — `DIAGNOSTICS` carries it as a template filled from
the head's own index row, so a head that ships a different variant follows its own row
instead of needing an edit here.

**That the two sampler runs on a page are different fits.** The index's R̂ and divergences
come from `make posteriors`, which refits each head once at its shipped variant and keeps
the draws; the ESS, treedepth and wall clock come from `make stan`, which fitted the whole
ladder and threw the draws away. Same specification, different chains, and a page that
stacked them in one row would be claiming they were one run. `sampler_runs` returns two
rows and names both.

## The one reading this module insists on

**Block 5 is a distance, not a verdict.** At n ≈ 10⁴ a posterior-predictive ribbon is one to
two ECDF points wide and every head in this project falls outside it somewhere — the
observed curve is inside the 95% band at 22% of grid points for `availability` and 7% for
the composition. `band_distance` therefore reports the largest vertical gap from `q50`
*and* the coverage share, in that order, because a page that renders in-or-out as a pass/fail
will report that every head fails. See `docs/model-cards-plan.md`, "the predictive half".
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

# ── The artifacts, and the targets that write them ────────────────────────────

INDEX_FILE = "model_card_index.csv"
COEFFICIENTS_FILE = "model_card_coefficients.csv"
FEATURES_FILE = "model_card_features.csv"
CORRELATION_FILE = "model_card_feature_corr.csv"
DENSITY_FILE = "model_card_feature_density.parquet"
ECDF_FILE = "model_card_ecdf.csv"
CALIBRATION_FILE = "model_card_calibration.csv"
SAMPLE_FILE = "model_card_sample.parquet"

#: `make posteriors` writes this beside the pickles the dashboard may not open. It is a flat
#: CSV of what each persisted fit cost and what it converged to, which is the half of block 7
#: no `stan_*_diagnostics.csv` carries.
MANIFEST_FILE = "manifest.csv"
POSTERIOR_WINDOW = "train"

MAKE_CARDS = "make model-cards"
MAKE_POSTERIORS = "make posteriors"
MAKE_STAN = "make stan"

#: The closed split vocabulary, mirroring the emitter's. There is no test column: the cards
#: are carved by `held_out.selection_split`, which never materializes a held-out row.
SPLITS = ("train", "validation")
SPLIT_LABELS = {"train": "Train", "validation": "Validation"}


# ── The four model pages ──────────────────────────────────────────────────────

@dataclass(frozen=True)
class ModelClass:
    """One model page: which heads it carries, in which order, and what they have in common.

    `intro` is typed prose and is the one place a model page is allowed any — it describes
    the *specification* and never a result, the same rule `HeadSpec.description` follows in
    the emitter. Every number on these pages is read from an artifact.
    """

    key: str
    title: str
    icon: str
    url_path: str
    heads: tuple[str, ...]
    intro: str


CLASSES: dict[str, ModelClass] = {
    "availability": ModelClass(
        key="availability",
        title="Availability",
        icon=":material/event_available:",
        url_path="availability",
        # The season-level head first, then the tenure decomposition in the order the
        # process runs: a tenure begins, spells start inside it, each spell lasts, the
        # tenure ends.
        heads=("availability", "gp_entry", "gp_onset", "gp_duration", "gp_exit"),
        intro=(
            "Two ways of predicting the same quantity — how many of his team's games a "
            "player is available for. **`availability`** models it directly, as a "
            "beta-binomial over games played out of team games. The **games-played tenure "
            "decomposition** models the process that generates it instead: an entry index "
            "and an exit index bound the stretch of the schedule a player is with the "
            "team, and inside that tenure a two-state chain starts absence spells at a "
            "fitted hazard and a beta-geometric decides how long each one lasts."),
    ),
    "minutes": ModelClass(
        key="minutes",
        title="Minutes",
        icon=":material/timer:",
        url_path="minutes",
        heads=("minutes", "composition"),
        intro=(
            "Two heads over minutes at two different units. The marginal head models "
            "`min | available` per player-season; the composition allocates each "
            "team-game's minutes among the players who played, as sequential beta-binomial "
            "trials ordered by prior-season share."),
    ),
    "components": ModelClass(
        key="components",
        title="Box-score components",
        icon=":material/sports_basketball:",
        url_path="components",
        heads=("fga", "fg3a_given_fga", "fg2m_given_fg2a", "fg3m_given_fg3a",
               "fta", "ftm_given_fta", "reb", "ast", "stl", "blk", "tov"),
        intro=(
            "The eleven heads `dk_pts` is reassembled from — seven negative-binomial counts "
            "on a minutes exposure, and four beta-binomial conversions on their own "
            "attempts. Total attempts are the count and the three-point mix is a share of "
            "them, so `fg2a` is derived rather than fitted."),
    ),
    "game_length": ModelClass(
        key="game_length",
        title="Game length",
        icon=":material/av_timer:",
        url_path="game-length",
        heads=("game_length_ot", "game_length_depth"),
        intro=(
            "The one input a forward simulation cannot look up: how long a game is. A "
            "beta-binomial on whether a game reaches overtime, and a beta-geometric on how "
            "many extra periods it then plays."),
    ),
}


def model_class(key: str) -> ModelClass:
    if key not in CLASSES:
        raise KeyError(f"no model page for {key!r}; declared pages are {sorted(CLASSES)}")
    return CLASSES[key]


def heads_of(index: pd.DataFrame, key: str) -> list[str]:
    """The class's heads in declared order, restricted to those actually on disk.

    Restricted rather than assumed, so a partially-built `outputs/predictions/` draws the
    heads it has instead of raising on the first one it does not.
    """
    carded = set(index["head"])
    return [head for head in model_class(key).heads if head in carded]


def head_row(index: pd.DataFrame, head: str) -> pd.Series:
    row = index[index["head"] == head]
    if row.empty:
        raise KeyError(f"{head!r} is not in {INDEX_FILE}; run `{MAKE_CARDS}`")
    return row.iloc[0]


# ── Block 1 · what this head is ───────────────────────────────────────────────

def specification(row: pd.Series) -> pd.DataFrame:
    """The head's declared specification as a two-column table, in reading order.

    Every value is read from `model_card_index.csv` — including `unit`, which is the one
    thing every model page must state prominently and the one thing most tempting to type
    into a view, where it goes stale on the next refit at a different grain.
    """
    seasons = f"{row['first_season']} → {row['last_season']}"
    fields = [
        ("Unit", str(row["unit"]), "One row of this head's fit"),
        ("Likelihood", str(row["likelihood"]), "The observation model"),
        ("Response", str(row["response_label"]),
         "What its predictive is a distribution over"),
        ("Selected variant", str(row["variant"]),
         "The arm its own ladder selected, on validation"),
        ("Dispersion", str(row["dispersion"]) or "—",
         "The second parameter of a two-parameter likelihood"),
        ("Fitting rows", f"{int(row['n_fit']):,}", "Rows the head actually fitted"),
        ("Validation rows", f"{int(row['n_validation']):,}",
         "Held back from the fit, and the only split selection may read"),
        ("Fitted seasons", seasons if str(row["first_season"]) else "—",
         "Target seasons in the fit. The test split begins after the last of them"),
        ("Posterior draws", f"{int(row['n_draws']):,}",
         "Thinned draws persisted by `make posteriors`"),
        ("Coefficient scale", str(row["coefficient_scale"]),
         "Every design column is standardized before fitting"),
    ]
    if str(row.get("row_filter", "")) not in ("", "nan"):
        fields.append(("Row filter", str(row["row_filter"]),
                       "Rows the head drops internally, so `n_fit` is below the frame"))
    return pd.DataFrame(fields, columns=["Field", "Value", "Note"])


# ── Block 2 · the features it was fed ─────────────────────────────────────────

def feature_order(features: pd.DataFrame, head: str) -> list[str]:
    """Features grouped by `term_family`, bases in order inside a family.

    A spline basis is six columns over one underlying quantity; scattering them through an
    alphabetical grid makes six unrelated panels out of one feature.
    """
    part = features[features["head"] == head]
    if part.empty:
        return []
    keys = (part[["feature", "term_family", "basis_index"]].drop_duplicates()
            .sort_values(["term_family", "basis_index", "feature"]))
    return keys["feature"].tolist()


def feature_summary(features: pd.DataFrame, head: str) -> pd.DataFrame:
    """One row per feature: n, mean and sd per split, and the share that was imputed.

    `missing_share` is carried per split because it differs by split and the difference is
    a finding — the composition imputes 17.0% of its training rows and 12.6% of its
    validation ones, and one number for both would hide that.
    """
    part = features[features["head"] == head]
    if part.empty:
        return pd.DataFrame(columns=["Feature", "Family", "Basis"])
    stats = (part.groupby(["feature", "split"])
             .first()[["term_family", "basis_index", "n", "mean", "sd", "min", "max",
                       "missing_share"]])
    order = feature_order(features, head)
    rows = []
    for feature in order:
        keyed = {split: stats.loc[(feature, split)] for split in SPLITS
                 if (feature, split) in stats.index}
        first = next(iter(keyed.values()))
        row = {"Feature": feature, "Family": first["term_family"],
               "Basis": (int(first["basis_index"])
                         if first["basis_index"] >= 0 else pd.NA)}
        for split in SPLITS:
            label = SPLIT_LABELS[split]
            stat = keyed.get(split)
            row[f"n ({label})"] = int(stat["n"]) if stat is not None else pd.NA
            row[f"Mean ({label})"] = float(stat["mean"]) if stat is not None else np.nan
            row[f"SD ({label})"] = float(stat["sd"]) if stat is not None else np.nan
        row["Min"] = float(first["min"])
        row["Max"] = float(first["max"])
        row["Imputed share"] = float(first["missing_share"])
        rows.append(row)
    return pd.DataFrame(rows)


def histogram_panel(features: pd.DataFrame, head: str,
                    order: list[str] | None = None) -> pd.DataFrame:
    """The binned counts both splits share, long, in the small-multiple's own order.

    `density` rather than `count` is what a panel plots: a 751-row validation histogram and
    an 8,232-row training one are drawn on one pair of axes, and only shares are comparable.
    """
    part = features[features["head"] == head]
    if part.empty:
        return part
    order = order or feature_order(features, head)
    rank = {feature: i for i, feature in enumerate(order)}
    out = part[part["feature"].isin(rank)].copy()
    out["panel_index"] = out["feature"].map(rank)
    # A discrete feature's edges are its own values, so left == right and the bar centres on
    # the value. Its width then has to be *derived* — plotly will not take a zero-width bar
    # and a NaN one raises — so a discrete bar takes most of the gap to its nearest
    # neighbouring value, which is what makes an imputation flag read as two bars at 0 and 1
    # rather than as two hairlines.
    out["bin_center"] = (out["bin_left"] + out["bin_right"]) / 2.0
    out["bin_width"] = out["bin_right"] - out["bin_left"]
    for feature, part_frame in out.groupby("feature"):
        if (part_frame["bin_width"] > 0).any():
            continue
        centers = np.sort(part_frame["bin_center"].unique())
        gap = float(np.diff(centers).min()) if len(centers) > 1 else 1.0
        out.loc[out["feature"] == feature, "bin_width"] = gap * DISCRETE_BAR_SHARE
    return out.sort_values(["panel_index", "split", "bin_index"]).reset_index(drop=True)


#: How much of the gap to the next value a discrete feature's bar takes. Under 1 so two
#: adjacent values stay two bars.
DISCRETE_BAR_SHARE = 0.8


def imputed_features(summary: pd.DataFrame) -> pd.DataFrame:
    """The features carrying any imputation, which is what block 2's flag rows are for."""
    if summary.empty or "Imputed share" not in summary:
        return summary
    return summary[summary["Imputed share"] > 0].sort_values(
        "Imputed share", ascending=False)


# ── Block 3 · feature relationships ───────────────────────────────────────────

def correlation_square(correlations: pd.DataFrame, head: str,
                       split: str = "train") -> pd.DataFrame:
    """The whole matrix as a square frame, ordered by the emitter's own feature index.

    A reshape rather than a reconstruction, which is why the emitter ships the diagonal:
    ordering by `i`/`j` keeps the heatmap's axes in design-matrix order, and a column that
    is constant on this split stays on the axis as an empty row instead of vanishing.
    """
    part = correlations[(correlations["head"] == head)
                        & (correlations["split"] == split)]
    if part.empty:
        return pd.DataFrame()
    order = (part[["feature_x", "i"]].drop_duplicates().sort_values("i")["feature_x"]
             .tolist())
    square = part.pivot(index="feature_x", columns="feature_y", values="r")
    return square.reindex(index=order, columns=order)


def constant_features(correlations: pd.DataFrame, head: str, split: str) -> list[str]:
    """Features whose correlations are all undefined on this split — i.e. constant here.

    `DataFrame.corr` leaves a constant column as NaN rather than as a spurious zero, so an
    all-NaN off-diagonal row *is* the finding: no validation row sits in that spline basis's
    knot span, or nothing was imputed there.
    """
    square = correlation_square(correlations, head, split)
    if square.empty:
        return []
    off_diagonal = square.where(~np.eye(len(square), dtype=bool))
    return [str(name) for name, row in off_diagonal.iterrows() if row.isna().all()]


def pair_menu(correlations: pd.DataFrame, head: str) -> pd.DataFrame:
    """The pairs that earned a precomputed 2-D density, strongest |r| first.

    Read off the training split's `top_pair` flags, which is the ranking the emitter binned
    over — so every row here has a panel behind it and the menu cannot offer one it does
    not have. **Not a pair-plot matrix**: 12–20 features is 150–400 panels, unreadable at
    any size that fits a page (`feature-correlation-not-pair-plots`).
    """
    part = correlations[(correlations["head"] == head)
                        & (correlations["split"] == "train")
                        & correlations["top_pair"] & (correlations["i"]
                                                      < correlations["j"])]
    if part.empty:
        return pd.DataFrame(columns=["feature_x", "feature_y", "r", "pair_rank"])
    return (part[["feature_x", "feature_y", "r", "abs_r", "pair_rank"]]
            .sort_values("pair_rank").reset_index(drop=True))


def pair_label(row: pd.Series) -> str:
    return f"{row['feature_x']} × {row['feature_y']}  (r = {float(row['r']):+.2f})"


def density_panel(density: pd.DataFrame, head: str, feature_x: str, feature_y: str,
                  split: str) -> pd.DataFrame:
    """One pair's binned joint on one split, with the cell centres a heatmap needs."""
    part = density[(density["head"] == head) & (density["feature_x"] == feature_x)
                   & (density["feature_y"] == feature_y) & (density["split"] == split)]
    if part.empty:
        return part
    out = part.copy()
    out["x_center"] = (out["x_left"] + out["x_right"]) / 2.0
    out["y_center"] = (out["y_left"] + out["y_right"]) / 2.0
    return out.sort_values(["x_index", "y_index"]).reset_index(drop=True)


# ── Block 4 · coefficients ────────────────────────────────────────────────────

#: Terms that are not comparable to the others on a sorted bar panel. The intercept and the
#: dispersion are on their own scales — every other term is a slope on a standardized design
#: column — so they are tiled beside the panel rather than drawn inside it, where the
#: intercept would set the axis and the comparison the panel exists for would be lost.
SCALAR_ROLES = ("intercept", "dispersion")


def coefficient_panel(coefficients: pd.DataFrame, head: str,
                      collapse: bool = False) -> pd.DataFrame:
    """The slope terms, sorted, with spline bases kept together under their family.

    Families are ordered by their **strongest** member and members run in basis order
    inside the family, so a six-column basis reads as one feature's shape rather than as
    six unrelated bars adjacent by accident. `collapse` keeps one row per family — the
    basis with the largest |mean|, labelled as such — for a head whose ladder puts a
    12-knot basis on every feature.

    Every coefficient is on the standardized design scale (`coefficient_scale` in the
    index says so), which is what makes sorting them against each other a legitimate
    comparison rather than a plot of measurement units.
    """
    part = coefficients[(coefficients["head"] == head)
                        & (~coefficients["term_role"].isin(SCALAR_ROLES))].copy()
    if part.empty:
        return pd.DataFrame(columns=["term", "term_family", "mean", "q2.5", "q97.5"])

    part["abs_mean"] = part["mean"].abs()
    strongest = part.groupby("term_family")["abs_mean"].transform("max")
    part["family_rank"] = strongest
    part["n_bases"] = part.groupby("term_family")["term"].transform("size")

    if collapse:
        part = (part.sort_values("abs_mean", ascending=False)
                .groupby("term_family", as_index=False).first())
        part["label"] = np.where(
            part["n_bases"] > 1,
            part["term_family"] + " · widest of " + part["n_bases"].astype(str) + " bases",
            part["term"])
    else:
        part["label"] = part["term"]

    # Strongest family first, bases in order inside it. Descending, because the figure puts
    # row 0 at the top and a panel that opens on its weakest term buries the answer — caught
    # by rendering it rather than by reading this line.
    part = part.sort_values(["family_rank", "term_family", "basis_index", "term"],
                            ascending=[False, True, True, True])
    return part.reset_index(drop=True)


def scalar_terms(coefficients: pd.DataFrame, head: str) -> pd.DataFrame:
    """The intercept and the dispersion terms, which the panel above deliberately drops."""
    part = coefficients[(coefficients["head"] == head)
                        & coefficients["term_role"].isin(SCALAR_ROLES)]
    return part.sort_values(["term_role", "term"]).reset_index(drop=True)


def coefficient_table(panel: pd.DataFrame) -> pd.DataFrame:
    """The panel's table twin — every drawn bar, with the interval it was drawn from."""
    if panel.empty:
        return panel
    return pd.DataFrame({
        "Term": panel["label"] if "label" in panel else panel["term"],
        "Family": panel["term_family"],
        "Mean": panel["mean"], "SD": panel["sd"],
        "2.5%": panel["q2.5"], "97.5%": panel["q97.5"],
        "P(> 0)": panel["p_positive"],
    }).round({"Mean": 4, "SD": 4, "2.5%": 4, "97.5%": 4, "P(> 0)": 3})


# ── Block 5 · predictive calibration ──────────────────────────────────────────

def ecdf_panel(ecdf: pd.DataFrame, head: str, split: str) -> pd.DataFrame:
    part = ecdf[(ecdf["head"] == head) & (ecdf["split"] == split)]
    return part.sort_values("grid_index").reset_index(drop=True)


def band_distance(ecdf: pd.DataFrame, head: str) -> pd.DataFrame:
    """How far the observed ECDF sits from the predictive median, per split.

    **The distance is the reading, and the coverage share is the footnote.** At n ≈ 10⁴ a
    posterior-predictive ribbon is one to two ECDF points wide and every head in this
    project falls outside it somewhere, so a page rendering in-or-out as a verdict reports
    that every head fails. `max_gap` is in ECDF units: 0.037 for `availability` means the
    observed curve is never more than 3.7 percentage points of probability from the median
    replicate.
    """
    rows = []
    for split in SPLITS:
        part = ecdf_panel(ecdf, head, split)
        if part.empty:
            continue
        gap = (part["observed"] - part["q50"]).abs()
        inside = ((part["observed"] >= part["q2.5"])
                  & (part["observed"] <= part["q97.5"]))
        rows.append({
            "split": split, "label": SPLIT_LABELS[split],
            "max_gap": float(gap.max()), "mean_gap": float(gap.mean()),
            "inside_95": float(inside.mean()), "n_grid": int(len(part)),
            "n_rows": int(part["n_rows"].iloc[0]),
            "n_draws": int(part["n_draws"].iloc[0]),
            "value_at_max": float(part.loc[gap.idxmax(), "value"]),
        })
    return pd.DataFrame(rows)


# ── Block 6 · predicted against observed, and residuals ───────────────────────

#: The four panels, as (artifact panel, split) — two panels each on two splits. Order is
#: row-major: fitted-vs-observed on both splits, then residual-vs-fitted on both.
PANELS = ("fitted_observed", "residual_fitted")
PANEL_LABELS = {"fitted_observed": "Predicted against observed",
                "residual_fitted": "Residual against predicted"}
PANEL_AXES = {"fitted_observed": ("Predicted", "Observed"),
              "residual_fitted": ("Predicted", "Observed − predicted")}


def calibration_panel(calibration: pd.DataFrame, head: str, panel: str,
                      split: str) -> pd.DataFrame:
    part = calibration[(calibration["head"] == head) & (calibration["panel"] == panel)
                       & (calibration["split"] == split)]
    if part.empty:
        return part
    out = part.copy()
    out["x_center"] = (out["x_left"] + out["x_right"]) / 2.0
    out["y_center"] = (out["y_left"] + out["y_right"]) / 2.0
    return out.sort_values(["x_index", "y_index"]).reset_index(drop=True)


def sample_points(sample: pd.DataFrame, head: str, split: str) -> pd.DataFrame:
    """The bounded subsample that goes over the density, for texture."""
    part = sample[(sample["head"] == head) & (sample["split"] == split)]
    return part.reset_index(drop=True)


def calibration_summary(calibration: pd.DataFrame, head: str) -> pd.DataFrame:
    """The four panels' means and spreads, **weighted off the same cells they draw**.

    Derived from the binned density rather than from the sample overlay on purpose: the
    figure and its table twin are then two readings of one object, and cannot disagree.
    The cost is that every statistic is a binned approximation, which the caption says.
    """
    rows = []
    for panel in PANELS:
        for split in SPLITS:
            cells = calibration_panel(calibration, head, panel, split)
            if cells.empty:
                continue
            weight = cells["count"].to_numpy(dtype=float)
            x, y = cells["x_center"].to_numpy(), cells["y_center"].to_numpy()
            mean_x = float(np.average(x, weights=weight))
            mean_y = float(np.average(y, weights=weight))
            rows.append({
                "Panel": PANEL_LABELS[panel], "Split": SPLIT_LABELS[split],
                "n": int(cells["n"].iloc[0]), "Cells": int(len(cells)),
                f"Mean {PANEL_AXES[panel][0].lower()}": mean_x,
                f"Mean {PANEL_AXES[panel][1].lower()}": mean_y,
                "Spread (y)": float(np.sqrt(np.average((y - mean_y) ** 2,
                                                       weights=weight))),
            })
    return pd.DataFrame(rows)


# ── Block 7 · diagnostics ─────────────────────────────────────────────────────

AVAILABILITY_DIAGNOSTICS = "stan_availability_diagnostics.csv"
GAMES_PLAYED_DIAGNOSTICS = "stan_games_played_diagnostics.csv"
MINUTES_DIAGNOSTICS = "stan_minutes_diagnostics.csv"
COMPOSITION_DIAGNOSTICS = "stan_composition_diagnostics.csv"
COMPONENT_DIAGNOSTICS = "stan_component_diagnostics.csv"
GAME_LENGTH_DIAGNOSTICS = "stan_game_length_diagnostics.csv"

#: Which `make stan` table carries each head's selection fit, and under which label. The
#: label is a template filled from the head's own index row, so a head that ships a
#: different variant follows its own row rather than needing an edit here. The three heads
#: with an explicit label are the ones whose ladder does not name itself after its variant.
DIAGNOSTICS: dict[str, tuple[str, str]] = {
    "availability": (AVAILABILITY_DIAGNOSTICS, "stan_posterior"),
    "gp_entry": (GAMES_PLAYED_DIAGNOSTICS, "{variant}/val/entry"),
    "gp_exit": (GAMES_PLAYED_DIAGNOSTICS, "{variant}/val/exit"),
    "gp_onset": (GAMES_PLAYED_DIAGNOSTICS, "{variant}/val/onset"),
    "gp_duration": (GAMES_PLAYED_DIAGNOSTICS, "{variant}/val/duration"),
    "minutes": (MINUTES_DIAGNOSTICS, "{variant}/val"),
    "composition": (COMPOSITION_DIAGNOSTICS, "{variant}/val"),
    "game_length_ot": (GAME_LENGTH_DIAGNOSTICS, "game_length/{variant}"),
    "game_length_depth": (GAME_LENGTH_DIAGNOSTICS, "game_length/depth"),
}
#: The eleven component heads share one table and one label rule, keyed on the head's own
#: label because that is what `stan_components.py` writes (`fg3a|fga/logit_own_spline/val`).
DIAGNOSTICS_BY_CLASS: dict[str, tuple[str, str]] = {
    "components": (COMPONENT_DIAGNOSTICS, "{label}/{variant}/val"),
}


def diagnostics_source(row: pd.Series) -> tuple[str, str]:
    """`(filename, label)` for one head's selection fit, or a `KeyError` naming the head."""
    spec = DIAGNOSTICS.get(str(row["head"])) or DIAGNOSTICS_BY_CLASS.get(
        str(row["model_class"]))
    if spec is None:
        raise KeyError(
            f"no `make stan` diagnostics source for {row['head']!r}. Every head's sampler "
            f"row lives in its class's table under a label its own ladder writes — add it "
            f"to `DIAGNOSTICS` rather than leaving block 7 silently empty.")
    filename, template = spec
    return filename, template.format(**{k: str(v) for k, v in row.items()})


def _diagnostics_row(diagnostics: pd.DataFrame | None, label: str) -> pd.Series | None:
    if diagnostics is None or diagnostics.empty:
        return None
    match = diagnostics[diagnostics["label"] == label]
    return match.iloc[0] if len(match) else None


#: What a cell reads when its source does not carry that quantity. A literal `None` on a
#: diagnostics table reads as a measurement, which is exactly what it is not: `make
#: posteriors` records no ESS and `make stan` kept no draws.
ABSENT = "—"


def _fmt(value, spec: str = "") -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return ABSENT
    return format(value, spec) if spec else str(value)


def sampler_runs(row: pd.Series, manifest: pd.DataFrame | None,
                 diagnostics: pd.DataFrame | None) -> pd.DataFrame:
    """The two fits behind one card, as two rows that say which is which.

    **They are different chains of the same specification.** `make posteriors` refits each
    head once at its shipped variant and keeps the draws every other block on this page is
    cut from; `make stan` fitted the whole variant ladder, recorded the full diagnostic
    block and threw the draws away. So ESS and treedepth are known for the selection fit
    and not for the persisted one, and R̂ is known for both and need not agree — stacking
    them in one row would claim a single run.

    Formatted here rather than by a `column_config`, because half these cells are *absent*
    rather than zero and a column with one missing value renders the whole column as text
    anyway. `ABSENT` says which.
    """
    _, label = diagnostics_source(row)
    manifest_row = None
    if manifest is not None and not manifest.empty:
        match = manifest[manifest["head"] == row["head"]]
        manifest_row = match.iloc[0] if len(match) else None
    selection = _diagnostics_row(diagnostics, label)

    runs = [{
        "Fit": f"Persisted · {MAKE_POSTERIORS}",
        "Label": str(row["head"]),
        "Max R̂": _fmt(float(row["max_rhat"]), ".5f"),
        "ESS bulk": ABSENT,
        "ESS tail": ABSENT,
        "Divergences": f"{int(row['divergences']):,}",
        "Treedepth hits": ABSENT,
        "Draws": (f"{int(manifest_row['n_draws_before_thinning']):,} → "
                  f"{int(row['n_draws']):,} kept" if manifest_row is not None
                  else f"{int(row['n_draws']):,} kept"),
        "Wall clock": (f"{float(manifest_row['fit_seconds']):,.0f} s"
                       if manifest_row is not None else ABSENT),
        "CmdStan": str(manifest_row["cmdstan"]) if manifest_row is not None else ABSENT,
        "Git SHA": str(row["git_sha"])[:8],
    }]
    if selection is not None:
        runs.append({
            "Fit": f"Selection · {MAKE_STAN}",
            "Label": label,
            "Max R̂": _fmt(float(selection["max_rhat"]), ".5f"),
            "ESS bulk": f"{float(selection['min_ess_bulk']):,.0f}",
            "ESS tail": f"{float(selection['min_ess_tail']):,.0f}",
            "Divergences": f"{int(selection['divergences']):,}",
            "Treedepth hits": f"{int(selection['treedepth_saturated']):,}",
            "Draws": f"{int(selection['n_draws']):,} → none kept",
            "Wall clock": f"{float(selection['wall_clock_s']):,.0f} s",
            "CmdStan": str(selection["cmdstan"]),
            "Git SHA": ABSENT,
        })
    return pd.DataFrame(runs)


def build_checks(row: pd.Series) -> pd.DataFrame:
    """What `make model-cards` verified before it was allowed to write this card.

    The emitter fails the build rather than writing a wrong artifact, so these are green by
    construction — which is exactly why `design_check` ships beside them. It reads `vacuous`
    for the nine heads whose recipe carries no design steps, where check 2 compares a frame
    against itself, and a green tick that cannot fail is worth less than no tick.
    """
    checks = [
        ("Recipe against the head's own ladder",
         f"{float(row['recipe_design_error']):.1e}", str(row["design_check"]),
         "a moved imputation mean, spline knot or bin edge"),
        ("Artifact round-trip", f"{float(row['roundtrip_prediction_error']):.1e}", "ladder",
         "a wrong link, or a mis-thinned draw block"),
        ("Drawn mean against reported mean",
         f"{float(row['predictive_bias']):+.2%}"
         if np.isfinite(row["predictive_bias"]) else "not checkable",
         str(row["predictive_check"]),
         "a missing exposure, or a link applied twice"),
        ("Ribbon stability at the draw budget", f"{float(row['ecdf_band_mc']):.4f}",
         "gated" if row["ecdf_band_gated"] else "reported only",
         "a band read off too few draws to be stable"),
    ]
    return pd.DataFrame(checks, columns=["Check", "Reading", "Kind", "What it catches"])


def predictive_provenance(row: pd.Series) -> str:
    """One sentence a page prints under blocks 5 and 6, saying what was actually drawn."""
    train = int(row["n_predictive_train"])
    validation = int(row["n_predictive_validation"])
    capped = " (a subsample — the head fitted more)" if row["predictive_rows_capped"] else ""
    weighted = (" Cell frames are expanded by their multiplicity first, so the ribbon is "
                "over games rather than over cells." if row["predictive_weighted"] else "")
    return (f"{int(row['predictive_draws'])} posterior draws over {train:,} training rows "
            f"and {validation:,} validation rows{capped}.{weighted}")
