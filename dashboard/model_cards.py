"""The model pages' pure layer — one head's card, cut into the seven blocks a page draws.

`src/models/model_cards.py` (`make model-cards`) writes nine flat artifacts describing
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

**What each head is a rate of.** `COMPONENT_BASIS` declares, per box-score head, whether it
is a count, an attempt *share* or a conversion, and anchors that claim to the design column
carrying its own prior-season rate. The share head's column is `logit_fg3a_pct_lag1`, whose
`_pct_` is `fg3a / fga` and not `fg3m / fg3a` — the confusion
`stan_components.conversion_variants` takes an explicit `own=` parameter to prevent in the
fitting code, and one a page that reprinted the column name without saying which ratio it is
would reintroduce on the way out. A test holds each anchor against the shipped artifact.

**Which arm of which artifact is which head.** The minutes page compares two heads at two
units, and neither unit's artifact names them the way the cards do: at the season unit they
are `composition_sum` and `minutes_head` in `minutes_unification.csv`, and at the
composition's own fitted unit the marginal head appears as `independent_comparator`, the
control `make stan-composition` refits inside its own run. `unit_board` carries that map, so
the page never has to know it.

## The one reading this module insists on

**Blocks 5 and 6 report distances, not verdicts.** At n ≈ 10⁴ a posterior-predictive ribbon
is one to two ECDF points wide and every head in this project falls outside it somewhere —
the observed curve is inside the 95% band at 22% of grid points for `availability` and 7% for
the composition. `band_distance` therefore reports the largest vertical gap from `q50`
*and* the coverage share, in that order, because a page that renders in-or-out as a pass/fail
will report that every head fails. `quantile_distance` is the same rule one block down: a KS
distance of the scaled residual from uniform is a *size*, and the same sample sizes make any
uniformity test reject everything. See `docs/model-cards-plan.md`, "the predictive half".
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
QUANTILE_FILE = "model_card_quantile.csv"
SAMPLE_FILE = "model_card_sample.parquet"
#: The head's own Stan program, verbatim — emitted beside the cards rather than read live
#: from `src/stan/`, so the code a page shows is a snapshot taken with the cards it ships
#: with and the dashboard still reads artifacts only.
STAN_FILE = "model_card_stan.csv"

#: `make posteriors` writes this beside the pickles the dashboard may not open. It is a flat
#: CSV of what each persisted fit cost and what it converged to, which is the half of block 7
#: no `stan_*_diagnostics.csv` carries.
MANIFEST_FILE = "manifest.csv"
POSTERIOR_WINDOW = "train"

MAKE_CARDS = "make model-cards"
MAKE_POSTERIORS = "make posteriors"
MAKE_STAN = "make stan"

#: The two variant ladders a model page reads *beside* its cards, because the cards
#: describe the head that shipped and say nothing about what it was chosen over. Both are
#: written by `make stan` and both carry a mandatory no-fit floor as a row of the ladder.
COMPONENT_METRICS_FILE = "stan_component_metrics.csv"
GAME_LENGTH_METRICS_FILE = "stan_game_length_metrics.csv"
GAME_LENGTH_PPC_FILE = "stan_game_length_ppc.csv"
GAME_LENGTH_DEPTH_FILE = "stan_game_length_depth.csv"
COMPOSITION_METRICS_FILE = "stan_composition_metrics.csv"

#: The minutes page reads one more artifact than the other three, and it is not a ladder:
#: `make minutes-unification` scores **both** minutes heads at the season unit off their
#: persisted posteriors, so it is the only file in the project where the two are read
#: against each other at a unit neither was fitted at.
MINUTES_UNIFICATION_FILE = "minutes_unification.csv"

MAKE_STAN_COMPONENTS = "make stan-components"
MAKE_STAN_GAME_LENGTH = "make stan-game-length"
MAKE_STAN_COMPOSITION = "make stan-composition"
MAKE_MINUTES_UNIFICATION = "make minutes-unification"

#: What the no-fit floor is called in each ladder. Different words, one idea: the arm that
#: does no fitting at all, which every head in this project is quoted against.
COMPONENT_FLOOR = "carry_forward"
GAME_LENGTH_FLOOR = "floor"

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
    #: Every carded head of the class, in page order — the full membership, which is what
    #: `test_every_carded_head_belongs_to_exactly_one_model_page` holds. The page itself
    #: renders the subset `heads_of` admits: the heads in the simulator's draw path.
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
        # tenure ends. Only the draw-path subset renders — see `heads_of`.
        heads=("availability", "gp_entry", "gp_onset", "gp_duration", "gp_exit"),
        intro=(
            "**The two heads a simulated season's absences are assembled from.** The "
            "season draw asks how many of his team's games a player misses and then which "
            "ones: **`availability`** supplies the count, as a beta-binomial over games "
            "played out of team games, and **`gp_duration`** supplies the shape the "
            "misses are laid out in, as a beta-geometric spell length. This page carries "
            "only the heads `make simulate-season` reads at draw time."),
    ),
    "minutes": ModelClass(
        key="minutes",
        title="Minutes",
        icon=":material/timer:",
        url_path="minutes",
        # The marginal head is class membership only; `heads_of` keeps it off the selector
        # because the simulator never reads it — see the module docstring of
        # `views/minutes.py` for where its season-level spread enters instead.
        heads=("minutes", "composition"),
        intro=(
            "**The head every simulated minute comes from.** The composition allocates "
            "each team-game's `5 × game_length` minutes among the players who played, as "
            "sequential beta-binomial trials ordered by prior-season share — so the team "
            "total is exact by construction, and a teammate's absence redistributes "
            "minutes as a fitted quantity rather than an assumption. The season-level "
            "spread its iid per-game draws cannot produce is injected at draw time as a "
            "per-(player, season) effect, calibrated against a marginal season-level "
            "baseline that appears below only as the reference the composition is scored "
            "against."),
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
    """The class's heads in declared order, restricted to those actually on disk — and,
    since 2026-08-23, to those in the simulator's **draw path**.

    Restricted rather than assumed, so a partially-built `outputs/predictions/` draws the
    heads it has instead of raising on the first one it does not.

    The draw-path restriction is the model pages' charter: a page discusses the heads a
    simulated season is assembled from, and a head that is fitted, converged and carded but
    never read at draw time — the games-played tenure decomposition, the marginal minutes
    head — stays carded and off the page. The filter reads the artifact's own
    `in_draw_path` rather than a list typed here, so a refactor of `src/sim/season.py`
    moves the page through `make model-cards` instead of going stale against it. An index
    built before the column existed degrades to the old behaviour, the same way
    `chain_role_phrase` loses a caption rather than printing `nan`.
    """
    carded = set(index["head"])
    if "in_draw_path" in index.columns:
        carded &= set(index.loc[index["in_draw_path"].astype(bool), "head"])
    return [head for head in model_class(key).heads if head in carded]


def head_row(index: pd.DataFrame, head: str) -> pd.Series:
    row = index[index["head"] == head]
    if row.empty:
        raise KeyError(f"{head!r} is not in {INDEX_FILE}; run `{MAKE_CARDS}`")
    return row.iloc[0]


# ── Block 1 · what this head is ───────────────────────────────────────────────

def text(value, fallback: str = "—") -> str:
    """A cell's own text, or a dash — never the string `nan`.

    A missing CSV cell reads back as a float `nan` whose `str()` is `"nan"`, which is
    truthy, prints as four letters on the page and looks like a value. The game-length
    depth head is the first head in the project with no season span at all — it is fitted
    on depth cells rather than on seasons — so it is the first row where an absent field
    reached a caption, and `nan` on a page is the same class of defect as `undefined`.
    """
    if value is None:
        return fallback
    if isinstance(value, float) and not np.isfinite(value):
        return fallback
    rendered = str(value).strip()
    return fallback if rendered in ("", "nan", "None", "NaT", "<NA>") else rendered


def season_span(row: pd.Series, fallback: str = "—") -> str:
    """`first → last`, or a dash for a head that is not fitted over seasons at all."""
    first, last = text(row["first_season"], ""), text(row["last_season"], "")
    if not first and not last:
        return fallback
    return f"{first or fallback} → {last or fallback}"


def specification(row: pd.Series) -> pd.DataFrame:
    """The head's declared specification as a two-column table, in reading order.

    Every value is read from `model_card_index.csv` — including `unit`, which is the one
    thing every model page must state prominently and the one thing most tempting to type
    into a view, where it goes stale on the next refit at a different grain.
    """
    fields = [
        ("Unit", text(row["unit"]), "One row of this head's fit"),
        # No backticks: an `st.dataframe` cell is canvas text with no markdown, so every
        # other Note in this table is plain prose and a stray pair would render as itself.
        ("In the shipped chain", text(row.get("chain_role_label")),
         "What a simulated season does with this head, if anything"),
        ("Likelihood", text(row["likelihood"]), "The observation model"),
        ("Response", text(row["response_label"]),
         "What its predictive is a distribution over"),
        ("Selected variant", text(row["variant"]),
         "The arm its own ladder selected, on validation"),
        ("Dispersion", text(row["dispersion"]),
         "The second parameter of a two-parameter likelihood"),
        ("Fitting rows", f"{int(row['n_fit']):,}", "Rows the head actually fitted"),
        ("Validation rows", f"{int(row['n_validation']):,}",
         "Held back from the fit, and the only split selection may read"),
        ("Fitted seasons", season_span(row),
         "Target seasons in the fit. The test split begins after the last of them"),
        ("Posterior draws", f"{int(row['n_draws']):,}",
         f"Thinned draws persisted by {MAKE_POSTERIORS}"),
        ("Coefficient scale", text(row["coefficient_scale"]),
         "Every design column is standardized before fitting"),
    ]
    if text(row.get("row_filter"), ""):
        fields.append(("Row filter", text(row["row_filter"]),
                       "Rows the head drops internally, so n_fit is below the frame"))
    return pd.DataFrame(fields, columns=["Field", "Value", "Note"])


def chain_role_phrase(row: pd.Series) -> str:
    """What this head does when a season is drawn, as one sentence plus its mechanism.

    Read from `model_card_index.csv`, not typed here — a head being fitted, converged and
    carded says nothing about whether `src/sim/season.py` calls it, and a view that decided
    that for itself would go stale on the next refactor of the simulator. `in_draw_path`
    picks the auxiliary, which is why the boolean ships beside the label rather than being
    inferred from it: *"in a simulated season it **draws the games-played count**"* against
    *"in a simulated season it is **not called at draw time**"*.

    Empty for an index built before the column existed, so the page loses a caption rather
    than printing `nan`.
    """
    label = text(row.get("chain_role_label"), "")
    if not label:
        return ""
    verb = "it" if bool(row.get("in_draw_path", False)) else "it is"
    note = text(row.get("chain_role_note"), "")
    return f"**In a simulated season {verb} {label}.** {note}".strip()


# ── The Stan program, between blocks 3 and 4 ──────────────────────────────────

def stan_row(stan: pd.DataFrame, head: str) -> pd.Series | None:
    """The head's own Stan program — file name, line count and verbatim source.

    From `model_card_stan.csv`, where the emitter snapshots the program beside the cards:
    the page shows the code the fit was compiled from rather than describing it, and reads
    it from an artifact rather than from `src/stan/`, which the dashboard may not open.
    `None` for a head the artifact does not carry, so an older file loses the block rather
    than raising.
    """
    part = stan[stan["head"] == head]
    if part.empty:
        return None
    return part.iloc[0]


def stan_siblings(stan: pd.DataFrame, index: pd.DataFrame, head: str) -> list[str]:
    """The labels of the other carded heads compiled from the same program.

    Four `.stan` sources serve every carded head, so sharing is the norm rather than the
    exception, and the page says who else is on the file — data flags select the blocks
    (`S = 0` disables the year effect exactly, and so on), which is why one program can be
    ten heads without any of them fitting another's structure.
    """
    row = stan_row(stan, head)
    if row is None:
        return []
    shared = stan[(stan["stan_file"] == row["stan_file"]) & (stan["head"] != head)]
    labels = index.set_index("head")["label"] if "label" in index.columns else None
    return [str(labels.get(h, h)) if labels is not None else str(h)
            for h in shared["head"]]


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


#: How many imputed features block 2 names inline before the table takes over.
IMPUTED_NAMED = 6


def imputed_shares(summary: pd.DataFrame, limit: int = IMPUTED_NAMED) -> str:
    """The imputed features as `` `name` 4.11% ``, strongest first — or an empty string.

    Formatted here rather than in the view because `DataFrame.itertuples` **renames any
    column whose name is not an identifier**: `Imputed share` arrives in the namedtuple as
    `_10`, so reading it back by its own name is a `KeyError` — one raised only on a head
    that actually imputed something, which is why nine heads and the whole availability
    page rendered it fine and `ftm|fta` did not.
    """
    imputed = imputed_features(summary)
    if imputed.empty:
        return ""
    return ", ".join(f"`{row['Feature']}` {row['Imputed share']:.2%}"
                     for _, row in imputed.head(limit).iterrows())


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

    **Below two columns there is no finding to have.** A lone design column has an empty
    off-diagonal by arithmetic rather than by measurement, and reporting it as "correlates
    with nothing" would put a claim about the data on the page that is really a claim about
    the column count — which is what the game-length onset head, whose whole design is one
    season term, did on the first cut of page 6.
    """
    square = correlation_square(correlations, head, split)
    if len(square) < 2:
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


# ── Block 6 · predicted against observed, and the scaled quantile residual ────

#: The calibration artifact's panels, as (artifact panel, split). **One panel since
#: 2026-08-10**: `residual_fitted` — the raw residual against the fitted value — was replaced
#: by the scaled quantile residual below, because a negative binomial's residual on a season
#: rebound total and a beta-binomial's on a conversion count are not on one scale, so the
#: same-looking panel meant different things on four pages that share a renderer.
PANELS = ("fitted_observed",)
PANEL_LABELS = {"fitted_observed": "Predicted against observed"}
PANEL_AXES = {"fitted_observed": ("Predicted", "Observed")}


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
    """The bounded subsample that goes over either density, for texture.

    Four columns and two panels: `fitted`/`observed` for the calibration density, and
    `u`/`predicted_rank` for the quantile residual. One subsample rather than two, taken at
    the same thinned rows, so a point in one panel is the same row as the point in the other.
    """
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


# ── Block 6 · the scaled quantile residual ────────────────────────────────────
#
# DHARMa's residual, cut by `make model-cards` from the same draws block 5's ribbon is:
# each observation's randomized quantile inside its own replicate distribution, which is
# uniform on [0, 1] iff the head is calibrated *whatever its likelihood is*. That property is
# the whole reason it replaced the raw residual panel — one renderer serves four model pages,
# and a negative binomial on a season total, a beta-binomial on a rate and a beta-geometric on
# a spell length do not share a residual scale. They share this one.
#
# One difference from R's DHARMa, which the page states in a line: DHARMa simulates at the
# fitted model's point estimate and these draws integrate over the posterior, so this is a
# Bayesian PIT residual — the same reading, carrying parameter uncertainty rather than
# conditioning it away.

QQ_PANEL = "qq"
RESIDUAL_PANEL = "residual"
LINE_PANEL = "quantile"

#: The levels `make model-cards` bins, and what a calibrated head puts them at. A page draws
#: the empirical lines against these, so "flat at 0.25 / 0.5 / 0.75" is read off the axis
#: rather than asserted in a caption.
QUANTILE_LEVELS = (0.25, 0.5, 0.75)

#: The build gate the page cites when it says the tiled distance is a reading of the head
#: rather than of the draw budget. Mirrors `src/models/model_cards.KS_MC_TOL`, the way
#: `POSTERIOR_WINDOW` mirrors that module's `WINDOW`; a test holds the two together.
KS_MC_TOL = 0.02


def _quantile_part(quantile: pd.DataFrame, head: str, panel: str,
                   split: str) -> pd.DataFrame:
    part = quantile[(quantile["head"] == head) & (quantile["panel"] == panel)
                    & (quantile["split"] == split)]
    return part.sort_values(["x_index", "y_index"]).reset_index(drop=True)


def qq_panel(quantile: pd.DataFrame, head: str, split: str) -> pd.DataFrame:
    """The QQ-uniform curve: expected quantile, observed order statistic, envelope.

    Renamed off the artifact's generic `(x, y)` grammar rather than passed through, because
    a figure builder taking `x`/`y`/`lo`/`hi` cannot say which is which and the axis titles
    are the entire content of a QQ plot.
    """
    part = _quantile_part(quantile, head, QQ_PANEL, split)
    if part.empty:
        return part
    return pd.DataFrame({
        "expected": part["x"].to_numpy(dtype=float),
        "observed": part["y"].to_numpy(dtype=float),
        "lo": part["lo"].to_numpy(dtype=float),
        "hi": part["hi"].to_numpy(dtype=float),
        "n": part["n"].to_numpy(dtype=int),
    })


def residual_cells(quantile: pd.DataFrame, head: str, split: str) -> pd.DataFrame:
    """The binned density of the scaled residual against rank-transformed predicted.

    `x_center` / `y_center` are aliases of the artifact's own `x` / `y`, so this frame is the
    same shape `calibration_panel` returns and one heatmap builder draws both.
    """
    part = _quantile_part(quantile, head, RESIDUAL_PANEL, split)
    if part.empty:
        return part
    out = part.copy()
    out["x_center"] = out["x"]
    out["y_center"] = out["y"]
    return out


def quantile_lines(quantile: pd.DataFrame, head: str, split: str) -> pd.DataFrame:
    """The 0.25 / 0.5 / 0.75 lines through that panel, one row per (bin, level).

    Binned on the density's own x edges by the emitter, and a bin with too few rows for a
    quartile is absent rather than drawn — so a line has a gap where the evidence does.
    """
    part = _quantile_part(quantile, head, LINE_PANEL, split)
    return part.sort_values(["level", "x_index"]).reset_index(drop=True)


def quantile_distance(quantile: pd.DataFrame, head: str) -> pd.DataFrame:
    """The KS distance per split, and how far the three lines sit from their own levels.

    **A distance, never a verdict**, which is the same rule `band_distance` carries one block
    up and for the same arithmetic reason: at n ≈ 10⁴ a strict uniformity test rejects every
    head in this project, so a page rendering in-or-out would report twenty failures. `ks` is
    in the units of the residual's own CDF — 0.03 means the residual's distribution is never
    more than 3 percentage points of probability from uniform.

    `line_gap` is the second reading and the one that says *where* a miss is: the largest
    absolute deviation of a binned quantile line from its nominal level. A head can sit close
    to uniform overall and still drift across the predicted range, which is exactly what the
    rank-transformed panel exists to show and what a single KS cannot.
    """
    rows = []
    for split in SPLITS:
        part = _quantile_part(quantile, head, QQ_PANEL, split)
        if part.empty:
            continue
        lines = quantile_lines(quantile, head, split)
        gap = float((lines["y"] - lines["level"]).abs().max()) if len(lines) else float("nan")
        rows.append({
            "split": split, "label": SPLIT_LABELS[split],
            "ks": float(part["ks"].iloc[0]), "n": int(part["n"].iloc[0]),
            "line_gap": gap, "n_bins": int(lines["x_index"].nunique()),
        })
    return pd.DataFrame(rows)


def quantile_note(row: pd.Series) -> str:
    """Why a head has no residual panel, or an empty string when it has one."""
    if str(row.get("quantile_scope", "drawn")) == "drawn":
        return ""
    return text(row.get("quantile_reason"), "no reason recorded")


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


# ── Page 5's own block · the eleven heads against their no-fit floor ──────────
#
# **A page's own block is named rather than numbered.** The seven numbered blocks are the
# contract every model page keeps; this is the one thing the box-score page owes that the
# other three do not, and it sits under block 1 because "did this head need to exist" is the
# first question about a component head rather than the seventh.
#
# It reads `stan_component_metrics.csv` — the head's own variant ladder — and not the model
# cards, because the cards describe the arm that shipped and carry no record of what it was
# chosen over. The floor is a row of that ladder.

ROLE_COUNT = "count"
ROLE_SHARE = "attempt share"
ROLE_CONVERSION = "conversion"


@dataclass(frozen=True)
class ComponentBasis:
    """What one box-score head is a rate *of* — declared, and anchored to a design column.

    Eleven heads, three roles, and one of them is routinely misread. `fg3a | fga` is a
    **share of attempts** — how much of a player's shot diet is threes — and not a shooting
    percentage; the two are different quantities on different heads. The column names make
    that easy to get wrong rather than easy to get right, because the share head's own
    prior-season term is `logit_fg3a_pct_lag1`, where `_pct_` means `fg3a / fga`. Three-point
    *shooting* percentage is `fg3m / fg3a` and lives on `fg3m | fg3a` as
    `logit_fg3m_pct_lag1`.

    That is the confusion `stan_components.conversion_variants` takes an explicit `own=`
    parameter to prevent in the fitting code, and a page that reprints the design column
    without saying which ratio it is would reintroduce it on the way out.

    So `own_family` is an **anchor** in the sense `pca.COMPONENTS` uses the word: the claim
    the page makes about a head is tied to a column the artifact has to carry, and
    `test_every_component_basis_is_anchored_to_a_real_design_family` fails if a refit renames
    it. Interpretation is allowed here; unpinned interpretation is not.
    """

    role: str
    #: What the head is a rate of, as arithmetic — the unambiguous form of the sentence.
    ratio: str
    #: The `term_family` its own prior-season rate arrives as, on the head's design frame.
    own_family: str


COMPONENT_BASIS: dict[str, ComponentBasis] = {
    "fga": ComponentBasis(ROLE_COUNT, "fga per minute", "log_fga_p36_lag1"),
    "fg3a_given_fga": ComponentBasis(ROLE_SHARE, "fg3a / fga", "logit_fg3a_pct_lag1"),
    "fg2m_given_fg2a": ComponentBasis(ROLE_CONVERSION, "fg2m / fg2a",
                                      "logit_fg2m_pct_lag1"),
    "fg3m_given_fg3a": ComponentBasis(ROLE_CONVERSION, "fg3m / fg3a",
                                      "logit_fg3m_pct_lag1"),
    "fta": ComponentBasis(ROLE_COUNT, "fta per minute", "log_fta_p36_lag1"),
    "ftm_given_fta": ComponentBasis(ROLE_CONVERSION, "ftm / fta", "logit_ftm_pct_lag1"),
    "reb": ComponentBasis(ROLE_COUNT, "reb per minute", "log_reb_p36_lag1"),
    "ast": ComponentBasis(ROLE_COUNT, "ast per minute", "log_ast_p36_lag1"),
    "stl": ComponentBasis(ROLE_COUNT, "stl per minute", "log_stl_p36_lag1"),
    "blk": ComponentBasis(ROLE_COUNT, "blk per minute", "log_blk_p36_lag1"),
    "tov": ComponentBasis(ROLE_COUNT, "tov per minute", "log_tov_p36_lag1"),
}

#: The share head, named once so the page and its test agree on which head is the odd one.
SHARE_HEAD = "fg3a_given_fga"
#: The head carrying the quantity the share head is mistaken for.
SHOOTING_PCT_HEAD = "fg3m_given_fg3a"


def component_basis(head: str) -> ComponentBasis:
    if head not in COMPONENT_BASIS:
        raise KeyError(f"{head!r} has no declared basis; every head on the box-score page "
                       f"states what it is a rate of. Declared: "
                       f"{sorted(COMPONENT_BASIS)}")
    return COMPONENT_BASIS[head]


def basis_note(head: str) -> str:
    """One paragraph saying what this head's rate is a rate of, and what it is not.

    Specification prose only, which is what a model page allows — no result appears here.
    The share head gets the long form because it is the one a reader mis-reads, and because
    the mis-reading is invisible: `logit_fg3a_pct_lag1` looks exactly like a shooting
    percentage and is not one.
    """
    basis = component_basis(head)
    if basis.role == ROLE_COUNT:
        return (f"**A count on a minutes exposure.** The head fits `{basis.ratio}` and "
                f"multiplies by the minutes it is given, so the season total is a "
                f"consequence of the rate and the availability rather than a quantity of "
                f"its own. Its own prior-season term is `{basis.own_family}` — the same "
                f"rate a season earlier, per 36 — put in on the **log** scale, so the head "
                f"fits a scale on the player's own rate rather than a curvature over it.")
    if basis.role == ROLE_SHARE:
        return (
            f"**This head is a share of attempts, not a shooting percentage.** It models "
            f"`{basis.ratio}` — how much of a player's shot diet is threes — as successes "
            f"out of his total field-goal attempts. Its own prior-season term is "
            f"`{basis.own_family}`, and the `_pct_` in that name is **`{basis.ratio}`**. "
            f"Three-point *shooting* percentage is "
            f"`{component_basis(SHOOTING_PCT_HEAD).ratio}`, a different quantity that lives "
            f"on the `{SHOOTING_PCT_HEAD.replace('_given_', '|')}` head as "
            f"`{component_basis(SHOOTING_PCT_HEAD).own_family}`. Reading one for the other "
            f"is the failure `conversion_variants` takes an explicit `own=` parameter to "
            f"prevent in the fitting code, and it is why `fg2a` is **derived** "
            f"(`fga − fg3a`) rather than fitted: a three substitutes for a two by "
            f"construction, so the mix is a share of the count and not a second count.")
    return (f"**A conversion** — `{basis.ratio}`, makes out of that player's own attempts, "
            f"so the trials are a quantity another head on this page produces. Its own "
            f"prior-season term is `{basis.own_family}`, a percentage on the logit scale.")


def metrics_key(row: pd.Series) -> str:
    """The name a head goes by in its `make stan` ladder — `fg3a|fga`, not the head id."""
    return str(row["label"])


def floor_ladder(metrics: pd.DataFrame, label: str) -> pd.DataFrame:
    """One head's whole variant ladder, floor first, with the selected arm marked.

    The floor's own `beats_floor` is `True` in the artifact and means nothing — it is the
    floor — so it is blanked here rather than rendered as a tick the reader would compare
    against the real ones.
    """
    part = metrics[metrics["head"] == label]
    if part.empty:
        return pd.DataFrame(columns=["Arm", "Features", "R²", "MAE", "CRPS", "PIT KS",
                                     "Shipped", "Clears the floor"])
    floor_first = part.assign(
        _order=(part["variant"] != COMPONENT_FLOOR).astype(int)).sort_values(
        ["_order", "val_r2"], ascending=[True, False])
    return pd.DataFrame({
        "Arm": floor_first["variant"].where(floor_first["variant"] != COMPONENT_FLOOR,
                                            f"{COMPONENT_FLOOR} · the no-fit floor"),
        "Features": floor_first["n_features"].astype(int),
        "R²": floor_first["val_r2"].round(4),
        "MAE": floor_first["val_mae"].round(4),
        "CRPS": floor_first["val_crps"].round(4),
        "PIT KS": floor_first["val_pit_ks"].round(4),
        "Shipped": floor_first["selected"],
        "Clears the floor": floor_first["beats_floor"].where(
            floor_first["variant"] != COMPONENT_FLOOR, pd.NA),
    }).reset_index(drop=True)


def floor_board(metrics: pd.DataFrame, index: pd.DataFrame,
                class_key: str = "components") -> pd.DataFrame:
    """Every head on the page against its own no-fit floor, in page order.

    One row per head: the floor's validation R², the shipped arm's, and the margin between
    them. **The margin is the point of the frame**, because a component head's whole claim
    is that fitting bought something over a prior per-36 rate carried forward — and one head
    on this page has a negative one, which is a finding rather than a defect to hide.

    R² is each head's own, on its own response, so a *height* here reads as "how much the
    fit added" and never as "this head is better than that one": a count head's R² is over a
    season total and a conversion head's is over a rate.
    """
    rows = []
    for head in heads_of(index, class_key):
        row = head_row(index, head)
        label = metrics_key(row)
        part = metrics[metrics["head"] == label]
        floor = part[part["variant"] == COMPONENT_FLOOR]
        shipped = part[part["selected"]]
        if floor.empty or shipped.empty:
            continue
        floor_r2 = float(floor["val_r2"].iloc[0])
        shipped_r2 = float(shipped["val_r2"].iloc[0])
        rows.append({
            "head": head, "label": label,
            "role": component_basis(head).role, "ratio": component_basis(head).ratio,
            "variant": str(shipped["variant"].iloc[0]),
            "floor_r2": floor_r2, "shipped_r2": shipped_r2,
            "margin": shipped_r2 - floor_r2,
            "clears": bool(shipped["beats_floor"].iloc[0]),
        })
    return pd.DataFrame(rows)


def floor_table(board: pd.DataFrame) -> pd.DataFrame:
    """The board figure's table twin — every drawn bar, with the numbers behind it."""
    if board.empty:
        return board
    return pd.DataFrame({
        "Head": board["label"], "Role": board["role"], "Models": board["ratio"],
        "Shipped arm": board["variant"],
        "No-fit floor R²": board["floor_r2"].round(4),
        "Shipped R²": board["shipped_r2"].round(4),
        "Margin": board["margin"].round(4),
        "Clears the floor": board["clears"],
    })


# ── Page 6's own block · game length against its no-fit floor ─────────────────
#
# Same idea, different ladder, and here it is repairing a real gap rather than adding a
# headline: the onset head is fitted on 26 season cells and its **validation ECDF is two
# grid points**, so block 5's ribbon cannot say whether the head is right. What can is the
# unit the head is actually consumed at — games — which `make stan-game-length` already
# writes as a posterior-predictive count per game class.

#: The onset ladder's four game classes, in order. `regulation` is `n_games` minus the other
#: three by construction, for the floor and for every fitted arm alike, so it is carried in
#: the table and left off the figure: drawn, it is a 2,322-long bar that flattens the three
#: classes the arms actually differ on.
ONSET_CLASSES = ("regulation", "1OT", "2OT", "3OT+")

#: How the two game-length heads name their own floor and their own fit. The depth head's
#: floor is a plain geometric — a constant continuation hazard — and its arm is the
#: beta-geometric that integrates a Beta frailty out of that hazard.
SERIES_OBSERVED = "observed"


def onset_counts(ppc: pd.DataFrame, variant: str) -> pd.DataFrame:
    """Observed against predicted games per class, for the floor and one fitted arm."""
    part = ppc[ppc["variant"].isin([GAME_LENGTH_FLOOR, variant])]
    if part.empty:
        return pd.DataFrame(columns=["class_label", "series", "count", "drawn"])
    rows = []
    for class_label in ONSET_CLASSES:
        cell = part[part["class"] == class_label]
        if cell.empty:
            continue
        drawn = class_label != "regulation"
        rows.append({"class_label": class_label, "series": SERIES_OBSERVED,
                     "count": float(cell["observed"].iloc[0]), "drawn": drawn})
        for arm, name in ((GAME_LENGTH_FLOOR, "no-fit floor"), (variant, variant)):
            arm_cell = cell[cell["variant"] == arm]
            if arm_cell.empty:
                continue
            rows.append({"class_label": class_label, "series": name,
                         "count": float(arm_cell["predicted"].iloc[0]), "drawn": drawn})
    return pd.DataFrame(rows)


def depth_counts(depth: pd.DataFrame, split: str = "val") -> pd.DataFrame:
    """Observed against the geometric and the beta-geometric, per overtime depth.

    `depth == 0` is not a depth: it is the artifact's per-OT-game log-likelihood row, which
    carries no `observed` count and would be drawn as an empty class. It is dropped here and
    read separately by `depth_likelihood`.
    """
    part = depth[(depth["split"] == split) & (depth["depth"] > 0)]
    if part.empty:
        return pd.DataFrame(columns=["class_label", "series", "count", "drawn"])
    rows = []
    for _, cell in part.sort_values("depth").iterrows():
        for series, column in ((SERIES_OBSERVED, "observed"),
                               ("geometric · the no-fit floor", "geometric"),
                               ("beta-geometric", "beta_geometric")):
            rows.append({"class_label": str(cell["label"]), "series": series,
                         "count": float(cell[column]), "drawn": True})
    return pd.DataFrame(rows)


def depth_likelihood(depth: pd.DataFrame, split: str = "val") -> pd.Series | None:
    """The `depth == 0` row — nats per overtime game for both arms, or `None`."""
    part = depth[(depth["split"] == split) & (depth["depth"] == 0)]
    return part.iloc[0] if len(part) else None


def class_counts(row: pd.Series, ppc: pd.DataFrame | None,
                 depth: pd.DataFrame | None, split: str = "val") -> pd.DataFrame:
    """Whichever of the two game-length ladders belongs to this head."""
    if str(row["head"]) == "game_length_depth":
        return (depth_counts(depth, split) if depth is not None
                else pd.DataFrame(columns=["class_label", "series", "count", "drawn"]))
    return (onset_counts(ppc, str(row["variant"])) if ppc is not None
            else pd.DataFrame(columns=["class_label", "series", "count", "drawn"]))


def class_table(panel: pd.DataFrame) -> pd.DataFrame:
    """The class figure's table twin — every class including the ones left off the figure."""
    if panel.empty:
        return panel
    wide = panel.pivot(index="class_label", columns="series", values="count")
    # `pivot` sorts both axes alphabetically, which would put the fitted arm before the
    # floor in the table and after it in the figure's legend. Both axes are restored to the
    # panel's own order, so the twin reads in the same order as the thing it is a twin of.
    rows = list(dict.fromkeys(panel["class_label"]))
    columns = list(dict.fromkeys(panel["series"]))
    return (wide.reindex(index=rows, columns=columns).round(2)
            .rename_axis(index=None, columns=None)
            .reset_index().rename(columns={"index": "Class"}))


# ── Page 4's own blocks · one posterior, two units, and the constraint ────────
#
# Three named blocks rather than one, each keyed on the numbered block whose question it
# extends. The two-unit verdict follows block 1, because "which unit is this a model at" is
# the first thing to know about either minutes head. The injected player-season effect
# follows block 5, because what fails at the season unit is *calibration* and sigma is what
# moves the PIT KS. And the teammate coupling follows block 6, because it is precisely the
# thing four panels of marginal residuals cannot show: no marginal metric can see whether a
# head carries the zero-sum team constraint.
#
# None of it comes from the model cards. A card describes one head at its own fitted unit;
# every claim here is a comparison *between* two heads or *across* two units, which is what
# `minutes_unification.csv` and the composition's own `make stan-composition` ladder carry.

HEAD_COMPOSITION = "composition"
HEAD_MARGINAL = "minutes"

#: `minutes_unification.csv` is several analyses in one long table, keyed by `unit`. The
#: name is the analysis rather than a measurement grain — `paired_bootstrap` and
#: `ps_effect_sweep` are both readings at the season unit — so a lookup that filters on the
#: arm alone is ambiguous. `src/docs_audit.py` learned that the hard way: the injected arm
#: carries the same name on the validation grid and the train grid.
UNIT_SEASON = "season_total"
UNIT_PAIRED = "paired_bootstrap"
UNIT_SWEEP = "ps_effect_sweep"
UNIT_TRAIN_SWEEP = "ps_sigma_on_train"
UNIT_COUPLING = "teammate_coupling"
UNIT_HEADROOM = "variance_decomposition"

ARM_COMPOSITION = "composition_sum"
ARM_MARGINAL = "minutes_head"
ARM_ALL_ROWS = "composition_sum_all_rows"
ARM_PAIRED = "composition_minus_minutes"
ARM_INJECTED = "composition_sum_plus_player_season_effect"
ARM_HEADROOM = "season_effect_headroom"

#: What the marginal head is called inside the composition's own ladder. `make
#: stan-composition` refits it as the control its per-game headline is measured against, so
#: the two heads meet at *both* units in an artifact where nothing was fitted twice.
COMPARATOR_ARM = "independent_comparator"

#: Both minutes ladders and the unification artifact name their no-fit arm the same thing
#: the component ladder does — a prior-season quantity carried forward with nothing fitted.
MINUTES_FLOOR = COMPONENT_FLOOR

#: The per-unit slot each head takes in every figure on the minutes page. Fixed rather than
#: following the head selector: three of this page's four figures are comparisons *between*
#: the two heads, so a colour that moved with the selector would mean two different things
#: on one screen. The reader learns it once and the tiles say which head is open.
MINUTES_SLOTS = {HEAD_COMPOSITION: 0, HEAD_MARGINAL: 1}


def head_label(index: pd.DataFrame | None, head: str, fallback: str = "") -> str:
    """A head's own `label` from the index, or the fallback — never a `KeyError` here.

    The two-unit block draws whether or not both cards are on disk, because its point is a
    comparison and a half-built `outputs/predictions/` should still show the half it has.
    """
    if index is None or index.empty:
        return fallback
    row = index[index["head"] == head]
    return str(row["label"].iloc[0]) if len(row) else fallback


def unit_board(unification: pd.DataFrame | None, ladder: pd.DataFrame | None,
               index: pd.DataFrame | None) -> pd.DataFrame:
    """Both minutes heads at both units, each against **that unit's own** no-fit floor.

    The frame this page exists for. One posterior scored at two units gives opposite
    verdicts, and the only way to draw that as one picture is to make the two units
    commensurable — so `improvement` is the CRPS gain over the floor *of that unit*, a
    ratio, rather than the CRPS itself, which is 4.5 minutes at one unit and 170 at the
    other and cannot share an axis.

    The floor is not a normalization of convenience: every head in this project is quoted
    against one, and here it is what makes the reversal a *verdict* rather than a change of
    scale. The composition clears its per-game floor and fails the season one; the marginal
    head does the reverse.
    """
    columns = ["unit", "unit_label", "head", "label", "arm", "crps", "floor_crps",
               "improvement", "clears", "n"]
    rows: list[dict] = []

    if ladder is not None and not ladder.empty:
        floor = ladder[ladder["variant"] == MINUTES_FLOOR]
        shipped = ladder[ladder["selected"].astype(bool)]
        comparator = ladder[ladder["variant"] == COMPARATOR_ARM]
        unit_label = _unit_of(index, HEAD_COMPOSITION, "player-game")
        if not (floor.empty or shipped.empty or comparator.empty):
            for head, part in ((HEAD_COMPOSITION, shipped), (HEAD_MARGINAL, comparator)):
                rows.append({
                    "unit": "fitted", "unit_label": unit_label, "head": head,
                    "arm": str(part["variant"].iloc[0]),
                    "crps": float(part["val_crps"].iloc[0]),
                    "floor_crps": float(floor["val_crps"].iloc[0]), "n": pd.NA})

    if unification is not None and not unification.empty:
        season = unification[unification["unit"] == UNIT_SEASON]
        floor = season[season["arm"] == MINUTES_FLOOR]
        unit_label = _unit_of(index, HEAD_MARGINAL, "player-season")
        if not floor.empty:
            for head, arm in ((HEAD_COMPOSITION, ARM_COMPOSITION),
                              (HEAD_MARGINAL, ARM_MARGINAL)):
                part = season[season["arm"] == arm]
                if part.empty:
                    continue
                rows.append({
                    "unit": "season", "unit_label": f"{unit_label}, summed", "head": head,
                    "arm": arm, "crps": float(part["crps_minutes"].iloc[0]),
                    "floor_crps": float(floor["crps_minutes"].iloc[0]),
                    "n": int(part["n"].iloc[0])})

    if not rows:
        return pd.DataFrame(columns=columns)
    board = pd.DataFrame(rows)
    board["label"] = [head_label(index, head, head) for head in board["head"]]
    board["improvement"] = (board["floor_crps"] - board["crps"]) / board["floor_crps"]
    board["clears"] = board["crps"] < board["floor_crps"]
    return board[columns]


def _unit_of(index: pd.DataFrame | None, head: str, fallback: str) -> str:
    """A head's declared unit, read from the index rather than typed into the view."""
    if index is None or index.empty:
        return fallback
    row = index[index["head"] == head]
    return text(row["unit"].iloc[0], fallback) if len(row) else fallback


def unit_table(board: pd.DataFrame) -> pd.DataFrame:
    """The verdict figure's table twin — the CRPS the ratio was computed from."""
    if board.empty:
        return board
    return pd.DataFrame({
        "Scored per": board["unit_label"], "Head": board["label"],
        "Arm": board["arm"], "CRPS": board["crps"].round(4),
        "No-fit floor": board["floor_crps"].round(4),
        "Against the floor": board["improvement"].round(4),
        "Clears it": board["clears"],
        "Rows": board["n"],
    })


def season_reading(unification: pd.DataFrame | None, head: str,
                   column: str) -> float | None:
    """One head's own value of one season-unit column, or `None` if it is not there.

    The single accessor for the two places a *rival's* number is needed rather than the open
    head's: the reference line under the sigma grid, and the calibration figure the injected
    arm is tiled against. Going through one function means the arm→head map is applied once.
    """
    if unification is None or unification.empty:
        return None
    arm = {HEAD_COMPOSITION: ARM_COMPOSITION, HEAD_MARGINAL: ARM_MARGINAL}.get(head)
    part = unification[(unification["unit"] == UNIT_SEASON) & (unification["arm"] == arm)]
    if part.empty or column not in part:
        return None
    value = float(part[column].iloc[0])
    return value if np.isfinite(value) else None


def season_gap(unification: pd.DataFrame | None,
               head: str = HEAD_COMPOSITION) -> pd.Series | None:
    """The paired bootstrap at the season unit, oriented for whichever head is open.

    The artifact stores one direction — composition minus marginal — and a page with a head
    selector has to be able to say "your head is 25.70 CRPS minutes *worse*" and "your head
    is 25.70 better" off the same row. Flipping the sign means flipping the interval's ends
    too, which is the kind of thing that is wrong in exactly one of the two branches.
    """
    if unification is None or unification.empty:
        return None
    part = unification[(unification["arm"] == ARM_PAIRED)
                       & (unification["unit"] == UNIT_PAIRED)]
    if part.empty:
        return None
    row = part.iloc[0].copy()
    if head == HEAD_MARGINAL:
        row["crps_delta"] = -float(row["crps_delta"])
        row["crps_delta_shared_target"] = -float(row["crps_delta_shared_target"])
        row["ci_lo"], row["ci_hi"] = -float(row["ci_hi"]), -float(row["ci_lo"])
    return row


#: What each metric of the season-unit comparison is a statement *about*. Declared rather
#: than derived: that MAE and bias describe where a predictive sits and that its sd and PIT
#: KS describe how wide it is are definitions, not readings. Splitting them is the block's
#: whole argument — the two heads tie on the first pair and separate 4.68x on the second.
GROUP_MEAN = "Where the predictive sits"
GROUP_SPREAD = "How wide it is"

SPREAD_METRICS: tuple[tuple[str, str, str, str], ...] = (
    ("mae_minutes", "MAE (minutes)", GROUP_MEAN, ",.2f"),
    ("bias_minutes", "Bias (minutes)", GROUP_MEAN, "+,.2f"),
    ("predictive_sd", "Predictive sd (minutes)", GROUP_SPREAD, ",.2f"),
    ("pit_ks", "PIT KS", GROUP_SPREAD, ".4f"),
)


def spread_panel(unification: pd.DataFrame | None,
                 index: pd.DataFrame | None = None) -> pd.DataFrame:
    """Four readings of the same 742 season totals, two per head, in one long frame.

    **The means are the control and the spread is the finding.** A page that drew only MAE
    would say the two heads are the same model; a page that drew only the predictive sd
    would leave a reader wondering whether the composition is simply worse. Both, side by
    side, are the actual result: it fits as well and it is 4.68x too narrow.

    `reference` carries the head's own residual sd onto the sd panel — the spread a
    calibrated season-total predictive has to cover — and is empty on the other three,
    where there is no target value to draw.
    """
    columns = ["metric", "metric_label", "group", "head", "label", "value", "text",
               "reference"]
    if unification is None or unification.empty:
        return pd.DataFrame(columns=columns)
    season = unification[unification["unit"] == UNIT_SEASON]
    headroom = unification[(unification["unit"] == UNIT_HEADROOM)
                           & (unification["arm"] == ARM_HEADROOM)]
    residual = float(headroom["resid_sd"].iloc[0]) if len(headroom) else np.nan

    rows = []
    for metric, label, group, spec in SPREAD_METRICS:
        for head, arm in ((HEAD_COMPOSITION, ARM_COMPOSITION),
                          (HEAD_MARGINAL, ARM_MARGINAL)):
            part = season[season["arm"] == arm]
            if part.empty or metric not in part:
                continue
            value = float(part[metric].iloc[0])
            rows.append({
                "metric": metric, "metric_label": label, "group": group, "head": head,
                "label": head_label(index, head, arm), "value": value,
                "text": format(value, spec),
                "reference": residual if metric == "predictive_sd" else np.nan})
    return pd.DataFrame(rows, columns=columns)


def coverage_row(unification: pd.DataFrame | None) -> pd.Series | None:
    """The composition's wider validation set — a coverage advantage, not a win.

    Reported as its own row in the artifact and as its own line on the page, because the
    369 extra player-seasons are rookies and low-minute players who are *easier* to predict:
    folding them into the comparison would flatter the composition on rows the other head
    never sees.
    """
    if unification is None or unification.empty:
        return None
    part = unification[(unification["arm"] == ARM_ALL_ROWS)
                       & (unification["unit"] == UNIT_SEASON)]
    return part.iloc[0] if len(part) else None


def sigma_sweep(unification: pd.DataFrame | None) -> pd.DataFrame:
    """The injected player-season effect's two grids, merged on sigma.

    Same arithmetic, same code path, **disjoint rows**: the validation grid scores the 742
    gate player-seasons and the train grid the last two *training* seasons. Merged rather
    than stacked because the reading is a per-sigma comparison — two optima one grid step
    apart on rows that share nothing is what says the effect size was not moved by the
    evaluation data.

    The two CRPS columns are levels on different row sets and are **not** comparable to each
    other; only the location of each minimum is. The figure gives them separate panels for
    that reason.
    """
    columns = ["sigma", "val_crps", "val_delta", "val_ci_lo", "val_ci_hi", "val_pit_ks",
               "val_sd", "val_n", "verdict", "train_crps", "train_pit_ks", "train_sd",
               "train_n", "train_seasons"]
    if unification is None or unification.empty:
        return pd.DataFrame(columns=columns)
    injected = unification[unification["arm"] == ARM_INJECTED]
    validation = injected[injected["unit"] == UNIT_SWEEP]
    train = injected[injected["unit"] == UNIT_TRAIN_SWEEP]
    if validation.empty and train.empty:
        return pd.DataFrame(columns=columns)

    left = validation[["sigma", "crps_minutes", "crps_delta", "ci_lo", "ci_hi", "pit_ks",
                       "predictive_sd", "n", "verdict"]].rename(columns={
        "crps_minutes": "val_crps", "crps_delta": "val_delta", "ci_lo": "val_ci_lo",
        "ci_hi": "val_ci_hi", "pit_ks": "val_pit_ks", "predictive_sd": "val_sd",
        "n": "val_n"})
    right = train[["sigma", "crps_minutes", "pit_ks", "predictive_sd", "n",
                   "seasons"]].rename(columns={
        "crps_minutes": "train_crps", "pit_ks": "train_pit_ks",
        "predictive_sd": "train_sd", "n": "train_n", "seasons": "train_seasons"})
    out = left.merge(right, on="sigma", how="outer").sort_values("sigma")
    return out.reindex(columns=columns).reset_index(drop=True)


def shipped_sigma(index: pd.DataFrame | None) -> float | None:
    """The sigma the composition's persisted posterior already carries, from its own card.

    Read from the artifact rather than typed, and it is not decoration: `make posteriors`
    records `player_season_sigma` per head and `minutes_unification.rehydrate_composition`
    applies it, so a consumer gets the effect by loading the head. A page that typed 0.450
    would keep printing it after the shipped value moved.
    """
    if index is None or index.empty or "player_season_sigma" not in index:
        return None
    row = index[index["head"] == HEAD_COMPOSITION]
    if row.empty:
        return None
    value = float(row["player_season_sigma"].iloc[0])
    return value if np.isfinite(value) else None


def shipped_sigma_by_role(index: pd.DataFrame | None) -> list[float] | None:
    """The per-role injection the draw actually uses, or `None` when it is shared.

    Since 2026-08-16 `sim.minutes.player_season_sigma_by_role` grades the injection over the
    composition's own `rho_bin` (`docs/draw-time-calibration-plan.md`), while
    `shipped_sigma` above stays the SHARED rung the grading was selected against — which is
    what the sweep block on this page is about. Both are read from the card rather than
    typed, and a page showing only the first would describe a draw nobody makes.
    """
    if index is None or index.empty or "player_season_sigma_by_role" not in index:
        return None
    row = index[index["head"] == HEAD_COMPOSITION]
    if row.empty:
        return None
    cell = str(row["player_season_sigma_by_role"].iloc[0] or "").strip()
    if not cell or cell.lower() == "nan":
        return None
    return [float(part) for part in cell.split("|")]


#: How close a grid sigma has to be to the shipped one to be labelled as it. The grid is
#: written at three decimals and the index at two, so an equality test on floats is the one
#: way this label can silently stop appearing.
SIGMA_TOLERANCE = 1e-6


def sigma_label(sigma: float, shipped: float | None = None) -> str:
    """`sigma = 0.450 · shipped`, with the un-injected arm named rather than left bare."""
    label = f"σ = {float(sigma):.3f}"
    if abs(float(sigma)) < SIGMA_TOLERANCE:
        return f"{label} · the un-injected head"
    if shipped is not None and abs(float(sigma) - float(shipped)) < SIGMA_TOLERANCE:
        return f"{label} · shipped"
    return label


def sigma_gaps(sweep: pd.DataFrame, shipped: float | None = None) -> pd.DataFrame:
    """The validation grid as paired gaps against the marginal head, shaped for `fig_paired`.

    Deliberately the same frame shape the tournament page's paired block uses, and drawn by
    the same builder: **an interval that straddles zero is a tie**, and this dashboard
    already has one triply-redundant encoding for that — a hollow marker, a heavier rule and
    a visible crossing of the reference line. Two encodings for one idea would be worse than
    either.
    """
    columns = ["sigma", "strategy", "gap", "gap_lo", "gap_hi", "crosses_zero", "verdict"]
    if sweep.empty or "val_delta" not in sweep:
        return pd.DataFrame(columns=columns)
    part = sweep[sweep["val_delta"].notna()]
    if part.empty:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame({
        "sigma": part["sigma"].astype(float),
        "strategy": [sigma_label(s, shipped) for s in part["sigma"]],
        "gap": part["val_delta"].astype(float),
        "gap_lo": part["val_ci_lo"].astype(float),
        "gap_hi": part["val_ci_hi"].astype(float),
        "crosses_zero": (part["val_ci_lo"] < 0) & (part["val_ci_hi"] > 0),
        "verdict": part["verdict"],
    }).reset_index(drop=True)


def sigma_row(sweep: pd.DataFrame, sigma: float | None) -> pd.Series | None:
    """One grid row by sigma, within the same tolerance the label uses."""
    if sweep.empty or sigma is None:
        return None
    match = sweep[(sweep["sigma"] - float(sigma)).abs() < SIGMA_TOLERANCE]
    return match.iloc[0] if len(match) else None


def sigma_table(sweep: pd.DataFrame, shipped: float | None = None) -> pd.DataFrame:
    """Both grids side by side — the twin for a figure that draws them in two panels."""
    if sweep.empty:
        return sweep
    return pd.DataFrame({
        "Effect size": [sigma_label(s, shipped) for s in sweep["sigma"]],
        "Validation CRPS": sweep["val_crps"].round(2),
        "Gap vs the marginal head": sweep["val_delta"].round(2),
        "95% interval": [
            "—" if not np.isfinite(lo) else f"[{lo:+.2f}, {hi:+.2f}]"
            for lo, hi in zip(sweep["val_ci_lo"], sweep["val_ci_hi"])],
        "Verdict": sweep["verdict"],
        "Validation PIT KS": sweep["val_pit_ks"].round(4),
        "Predictive sd": sweep["val_sd"].round(2),
        "Train CRPS": sweep["train_crps"].round(2),
        "Train PIT KS": sweep["train_pit_ks"].round(4),
    })


def teammate_coupling(unification: pd.DataFrame | None,
                      index: pd.DataFrame | None = None) -> pd.DataFrame:
    """What each head says about two teammates' season minutes, against what physics forces.

    A team's season minutes are a fixed pot, so a fixed sum over K players forces a mean
    pairwise correlation of **−1/(K−1)**. That is arithmetic, not a fit — and the artifact
    carries it per head at that head's *own* measured roster size, which differs because the
    marginal head's `>= 200 prior minutes` filter drops real teammates. Drawing one shared
    reference line would therefore be wrong for one of the two rows.

    `forced` is the reference and `measured` is the reading, which is why the figure draws
    them as one row rather than as two series: the quantity is the **gap** between them.
    """
    columns = ["head", "label", "measured", "forced", "team_sd", "roster", "n"]
    if unification is None or unification.empty:
        return pd.DataFrame(columns=columns)
    part = unification[unification["unit"] == UNIT_COUPLING]
    rows = []
    for head, arm in ((HEAD_COMPOSITION, ARM_COMPOSITION), (HEAD_MARGINAL, ARM_MARGINAL)):
        cell = part[part["arm"] == arm]
        if cell.empty:
            continue
        cell = cell.iloc[0]
        rows.append({"head": head, "label": head_label(index, head, arm),
                     "measured": float(cell["r_teammates"]),
                     "forced": float(cell["r_implied_by_fixed_sum"]),
                     "team_sd": float(cell["team_season_sum_sd"]),
                     "roster": float(cell["roster_size"]),
                     "n": int(cell["n"])})
    return pd.DataFrame(rows, columns=columns)


def coupling_table(panel: pd.DataFrame) -> pd.DataFrame:
    """The coupling figure's table twin, including the two columns it does not draw."""
    if panel.empty:
        return panel
    return pd.DataFrame({
        "Head": panel["label"],
        "Mean pairwise r between teammates": panel["measured"].round(4),
        "Forced by a fixed team total": panel["forced"].round(4),
        "Roster size it was measured at": panel["roster"].round(2),
        "Predictive sd of the team's season total (minutes)": panel["team_sd"].round(1),
        "Player-seasons": panel["n"],
    })


def predictive_provenance(row: pd.Series) -> str:
    """One sentence a page prints under blocks 5 and 6, saying what was actually drawn."""
    train = int(row["n_predictive_train"])
    validation = int(row["n_predictive_validation"])
    capped = " (a subsample — the head fitted more)" if row["predictive_rows_capped"] else ""
    weighted = (" Cell frames are expanded by their multiplicity first, so the ribbon is "
                "over games rather than over cells." if row["predictive_weighted"] else "")
    return (f"{int(row['predictive_draws'])} posterior draws over {train:,} training rows "
            f"and {validation:,} validation rows{capped}.{weighted}")
