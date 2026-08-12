"""Persist every fitted Stan head's posterior — the simulation layer's step zero.

`make stan` fits eleven-plus heads and writes metrics, diagnostics and per-row predictions.
**It writes no coefficient draws.** The single exception is `stan_composition._checkpoint`,
which pickles `alpha_draws` / `beta_draws` / `rho_draws` / `scaler` / `features` per arm —
incidentally, to survive a 9.9-hour loop, not as a consumable artifact.

So "simulate from the joint posterior" has meant *refit*: ~137 minutes for the component
heads and ~2.7 h for one composition arm, every run. That is not a thing a draft room can do
once and not a thing a strategy sweep can do a hundred times. This module generalizes the
checkpoint from a crash artifact into a contract: one pickle per head under
`data/features/posteriors/`, carrying

- **thinned coefficient draws** — `alpha_draws`, `beta_draws`, and whichever dispersion the
  family carries (`rho_draws`, `phi_draws`, `kappa_draws`), 1,000 by default and thinned
  across the **whole** posterior via `stan_utils.thin`. Never sliced off the front:
  `season_terms._draw_components` already records why that silently takes different parts of
  the posterior from different heads;
- the **design recipe** needed to score an arbitrary frame — the ordered, *fitted* feature
  steps (imputation means, log / logit transforms, spline knots), the feature list, the
  fitted `StandardScaler`, and the selected variant name;
- **provenance** — fit window, CmdStan version, git SHA, timestamp, and the sampler budget.

Once this exists, `make posteriors` is the only thing in the simulation layer that needs a
CmdStan toolchain. Everything downstream reads a pickle.

## The recipe is captured from the real ladder, not re-derived beside it

Each head's variant ladder lives in its own module (`stan_components.count_variants`,
`stan_minutes.variants`, `stan_composition.variants`). This module **calls those functions**
to get the transformed frames and the feature list, and only expresses the *fitted state*
they leave implicit — the imputation means, the `SplineTransformer` knots, the dispersion bin
edges. Every spline is refitted from the ladder's own returned training frame, which carries
the pre-spline column, so nothing is reconstructed from scratch.

That still duplicates the *order* of the steps, which is how a recipe drifts silently. So
every artifact is verified at build time: the recipe applied to the raw probe rows must
reproduce the head's own standardized design matrix and the head's own predictions, exactly.
A ladder that changes shape fails the build rather than writing a wrong artifact.

## Three fit windows, three consumers, three directories

Artifacts are namespaced by window (`posteriors/<window>/<head>.pkl`) because the windows are
not versions of one thing — they are three artifacts wanted at once:

- **`train` is the default**, and it is what anything scored against *realized* 2022-23 or
  2023-24 must read. `docs/simulations-plan.md`'s honest readout replays simulated portfolios
  against those seasons' real box scores, and at `train_val` every one of those rows was in
  the fit — 883 of 10,361 availability rows and 773 of 9,403 component rows. Small per row,
  and exactly the shape of the four sub-1% reversals this repo has already logged.
- **`train_val`** is for the one-shot test readout on 2024-25 / 2025-26, for the same reason
  `src/final_evaluation.py` refits on train **plus** validation before scoring test: a
  held-out figure should describe the model that would actually deploy.
- **`full`** is the 2026-27 production board. It reads the held-out seasons, so it goes
  through `held_out.assert_unlocked`.

The doc's config block originally specified `train_val` as the default on the grounds that it
matches the four simulator inputs already calibrated that way — the residual copula, the
game-level minutes dispersion, the block variance inflation and the bonus overdispersion.
That analogy does not transfer, and the default was changed on 2026-08-08. Those four are
*given* to the simulator and never scored against realized data; they set the shape of the
noise. The coefficients generate the board that the realized backtest then scores.

The window is stamped into every artifact regardless, and `require_window` is there so a
consumer can refuse the wrong one instead of discovering it in a result — the leak it prevents
is quiet, because a backtest reading heads fitted at a wider window has read those seasons
*through the coefficients*, and no frame-level split guard can see that.

## The fit window is not the season truncation, and they are named apart

A second season axis arrived with the windowed availability head (2026-08-11) and the
composition's pilot window before it, and the two are easy to confuse because both are "which
seasons are in the fit":

| axis | what it decides | where it lives | what enforces it |
|---|---|---|---|
| **fit window** | which *split* may be fitted — `train` / `train_val` / `full` | `provenance["fit_window"]`, and the directory | `require_window`, `assert_unlocked` |
| **season truncation** | which recent *suffix* of that split is actually fitted | `extras["fit_first_season"]` | the head's own `fitting_rows`, re-applied by `model_cards` |

They are orthogonal: a `train` head truncated at 2012-13 and a `train` head over the full
history sit in the same directory, pass the same `require_window`, and are different models.
`fit_first_season` therefore reaches the manifest as its own column rather than being left to
`provenance["first_season"]`, which is a *readout* of the fitted frame's span and would say
`2012-13` for a truncated head and a full-window head fitted on a frame that happens to start
there.
"""

from __future__ import annotations

import pickle
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.preprocessing import SplineTransformer

from src.models.availability import split_seasons
from src.models.held_out import assert_unlocked, selection_split
from src.models.stan_utils import cmdstan_version, thin

# Draws kept per head. Thinned across the whole posterior, so this is a statement about
# how much of the posterior a consumer carries, not about which part of it.
POSTERIOR_DRAWS = 1000
# `train`, not `train_val` — see "Three fit windows, three consumers" above. The realized
# backtest scores 2022-23 and 2023-24, which `train_val` fits on.
FIT_WINDOW = "train"
WINDOWS = ("train", "train_val", "full")

# Rows kept as the round-trip probe. Evenly spaced through the validation frame via the
# same `thin` used on the draws, so the probe spans the design rather than sampling one
# corner of it. 400 x ~30 float columns is a few hundred KB per head.
PROBE_ROWS = 400

# The round-trip bar. Both paths do the same arithmetic on the same doubles, so the honest
# tolerance is floating-point noise on a matrix product, not a modelling tolerance.
DESIGN_TOL = 1e-9
PREDICTION_TOL = 1e-8

POSTERIOR_DIR = "posteriors"

# Head groups, in the order `make posteriors` runs them: cheapest first, so a failure in
# the plumbing surfaces in three minutes rather than three hours.
GROUPS = ("game-length", "availability", "games-played", "minutes", "components",
          "composition")


# ── The design recipe ─────────────────────────────────────────────────────────

def _impute_step(train: pd.DataFrame, columns: list[str],
                 transformed: pd.DataFrame) -> dict:
    """`component_rates.impute`'s fitted state: which columns were filled, and the mean.

    Which columns were filled is read off the **transformed frame's columns**, not off the
    head's feature list. `impute` skips a column that is complete on both frames, so the
    flag set is a property of the data — and `stan_composition.variants` then *discards*
    the per-column flags in favour of one `design_missing` indicator, because a rookie
    loses all 18 design columns at once and 18 identical flags are a degenerate subspace
    the sampler pays for in treedepth. Deriving the imputed set from the features would
    therefore find nothing there and silently write a recipe that does not impute at all.
    That is exactly what the first version of this did, and the build-time round-trip is
    what caught it.
    """
    means = {col: float(train[col].mean()) for col in columns
             if f"{col}__miss" in transformed.columns}
    return {"kind": "impute", "means": means}


def _spline_step(train: pd.DataFrame, column: str, n_knots: int,
                 names: list[str]) -> dict:
    """A `SplineTransformer` refitted on the ladder's own training column.

    `add_spline` and `stan_minutes._spline` both fit on `train[[col]]` and discard the
    transformer. Refitting on the same values is deterministic and gives back the identical
    basis, which the build-time round-trip then confirms against the ladder's own output.
    """
    st = SplineTransformer(n_knots=n_knots, degree=3, extrapolation="linear",
                           include_bias=False)
    st.fit(train[[column]].to_numpy(dtype=float))
    bsplines = getattr(st, "bsplines_", None)
    knots = (np.asarray(bsplines[0].t, dtype=float) if bsplines is not None
             else np.zeros(0))
    return {"kind": "spline", "column": column, "names": list(names),
            "transformer": st, "knots": knots, "n_knots": int(n_knots)}


def _apply_step(step: dict, frame: pd.DataFrame) -> pd.DataFrame:
    """One fitted design step, applied in place on a copy.

    The vocabulary is closed and small on purpose: every head's ladder is some ordering of
    these eight, and a step kind that is not here is a ladder this module has not been
    taught rather than something to guess at.
    """
    kind = step["kind"]
    out = frame
    if kind == "impute":
        out = frame.copy()
        for col, mean in step["means"].items():
            out[f"{col}__miss"] = frame[col].isna().astype(float)
            out[col] = frame[col].fillna(mean)
    elif kind == "isna_flag":
        out = frame.copy()
        out[step["name"]] = frame[step["column"]].isna().astype(float)
    elif kind == "log1p":
        out = frame.copy()
        for col, name in step["columns"].items():
            out[name] = np.log1p(np.clip(frame[col].to_numpy(dtype=float), 0.0, None))
    elif kind == "logit":
        out = frame.copy()
        lo = float(step["clip"])
        for col, name in step["columns"].items():
            p = np.clip(frame[col].to_numpy(dtype=float), lo, 1 - lo)
            out[name] = np.log(p / (1 - p))
    elif kind == "square":
        out = frame.copy()
        for col, name in step["columns"].items():
            out[name] = frame[col].to_numpy(dtype=float) ** 2
    elif kind == "product":
        out = frame.copy()
        out[step["name"]] = (frame[step["left"]].to_numpy(dtype=float)
                             * frame[step["right"]].to_numpy(dtype=float))
    elif kind == "spline":
        out = frame.copy()
        basis = step["transformer"].transform(
            frame[[step["column"]]].to_numpy(dtype=float))
        for j, name in enumerate(step["names"]):
            out[name] = basis[:, j]
    elif kind == "bins":
        out = frame.copy()
        values = frame[step["column"]].to_numpy(dtype=float)
        out[step["name"]] = np.searchsorted(np.asarray(step["edges"], dtype=float),
                                            values, side="right") + 1
    elif kind == "cut":
        # `stan_availability.role_bins`, as a recipe step: explicit interval edges that
        # close BOTH ends, 1-based, with anything outside them — a NaN prior season, or a
        # value above the top edge — falling into the LOWEST bucket.
        #
        # A second binning kind rather than a flag on `bins` above, because the two are
        # different conventions and collapsing them would make the difference invisible at
        # the call site: `bins` holds INTERIOR quantile cuts that `searchsorted` extends to
        # +/- infinity, so it has no outside, where these edges are constants with an
        # outside that has to be given a rule. Both write a 1-based `rho_bin`; only this
        # one can be handed a value the fit never saw.
        out = frame.copy()
        idx = pd.cut(frame[step["column"]].to_numpy(dtype=float),
                     np.asarray(step["edges"], dtype=float), labels=False)
        out[step["name"]] = (np.nan_to_num(np.asarray(idx, dtype=float), nan=0.0)
                             .astype(int) + 1)
    elif kind == "join":
        # A per-unit feature block joined on keys, with ONE missingness indicator for the
        # whole block and train means for the holes — `stan_composition.attach_team_context`
        # expressed as a recipe step. The block travels INSIDE the artifact rather than
        # being re-read from parquet, so a rebuilt `team_context_tierA.parquet` cannot
        # silently change what a persisted posterior scores; the flip side is that a season
        # the block does not cover scores as missing, which is what `flag` is for and why a
        # production board rebuilds the artifact rather than reusing one.
        out = frame.merge(step["block"], on=list(step["keys"]), how="left")
        if len(out) != len(frame):
            raise ValueError(f"the {step['name']!r} join duplicated rows; the block must "
                             f"be unique on {list(step['keys'])}")
        cols = list(step["means"])
        out[step["flag"]] = out[cols[0]].isna().astype(float)
        for col, mean in step["means"].items():
            out[col] = out[col].fillna(mean)
    else:
        raise KeyError(f"unknown design step {kind!r} — the recipe was written by a "
                       f"newer version of src/models/posteriors.py than the one loading it")
    return out


@dataclass
class DesignRecipe:
    """Everything needed to turn a raw design frame into this head's design matrix.

    `builder` names the function that produces the *raw* frame this recipe expects
    (`stan_components.build_design`, `stan_minutes.build_design`, ...). It is recorded as a
    string rather than a callable so the pickle does not pin an import path, and so a
    consumer building a 2026-27 frame can see which builder it has to satisfy.
    """

    variant: str
    features: list[str]
    scaler: object
    steps: tuple[dict, ...] = ()
    builder: str = ""

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Apply every fitted step, in order."""
        out = frame
        for step in self.steps:
            out = _apply_step(step, out)
        return out

    def matrix(self, frame: pd.DataFrame, transformed: bool = False) -> np.ndarray:
        """The standardized design matrix — mirrors every head's private `_design`."""
        if not self.features:
            # The intercept-only duration arm. `features = []` is legal in Stan and the
            # head stores no scaler, so an empty (rows x 0) block is the honest design.
            return np.zeros((len(frame), 0))
        out = frame if transformed else self.transform(frame)
        missing = [c for c in self.features if c not in out.columns]
        if missing:
            raise KeyError(f"the recipe produced no column for {missing}; the frame is "
                           f"not the output of `{self.builder or 'the head builder'}`")
        X = out[self.features].to_numpy(dtype=float)
        return self.scaler.transform(np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0))


# ── The artifact ──────────────────────────────────────────────────────────────

@dataclass
class PosteriorArtifact:
    """One head's thinned posterior, its design recipe, and where it came from.

    `family` selects the likelihood; `response` selects which mean the head itself reports,
    because they are not all the same function of the draws. The three beta-binomial heads
    inside the games-played process report a **plug-in** mean (`sigmoid` at the posterior-mean
    coefficients) while every other beta-binomial head reports the posterior mean of `mu`.
    Storing which one is right is the difference between a round-trip that checks something
    and one that checks the loader against itself.
    """

    head: str
    head_label: str
    family: str
    response: str
    recipe: DesignRecipe
    draws: dict = field(default_factory=dict)
    extras: dict = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)
    reference: dict = field(default_factory=dict)

    # ── Scoring ───────────────────────────────────────────────────────────────

    @property
    def n_draws(self) -> int:
        return len(self.draws["alpha_draws"])

    def eta_draws(self, frame: pd.DataFrame, transformed: bool = False) -> np.ndarray:
        """(draws x rows) linear predictor — the same expression every head evaluates.

        Deliberately `X @ beta.T + alpha[None, :]` then transposed, matching the heads
        exactly rather than the mathematically equivalent `beta @ X.T`: the two differ in
        the last bits and this is checked at 1e-9.
        """
        X = self.recipe.matrix(frame, transformed=transformed)
        eta = X @ self.draws["beta_draws"].T + self.draws["alpha_draws"][None, :]
        return eta.T

    def mu_draws(self, frame: pd.DataFrame, transformed: bool = False) -> np.ndarray:
        """(draws x rows) on the response scale of the link, before any exposure."""
        eta = self.eta_draws(frame, transformed=transformed)
        if self.family == "negbinomial":
            return np.exp(np.clip(eta, -30, 30))
        return 1.0 / (1.0 + np.exp(-np.clip(eta, -30, 30)))

    def predict(self, frame: pd.DataFrame, transformed: bool = False) -> np.ndarray:
        """The head's own reported mean, per row.

        Not a general-purpose predictive — the simulator draws from `mu_draws` and the
        dispersion. This exists so the round-trip has something to compare against, and so a
        consumer can sanity-check an artifact against a stored figure.
        """
        if self.response == "eta":
            return self.eta_draws(frame, transformed=transformed).mean(axis=0)
        if self.response == "plug_in_mu":
            X = self.recipe.matrix(frame, transformed=transformed)
            eta = X @ self.draws["beta_draws"].mean(axis=0) + float(
                self.draws["alpha_draws"].mean())
            return 1.0 / (1.0 + np.exp(-np.clip(eta, -30, 30)))
        if self.response == "mean_count":
            mu = self.mu_draws(frame, transformed=transformed)
            exposure = frame[self.extras["exposure"]].to_numpy(dtype=float)
            return np.clip(mu * exposure[None, :], 1e-9, None).mean(axis=0)
        mu = self.mu_draws(frame, transformed=transformed)
        # The beta-geometric head reaches its mean through `games_played.beta_shapes`,
        # which clips `mu` before splitting it into `a` and `b`. `a / (a + b)` is the
        # clipped mu, so the bound is part of the head's reported mean rather than a
        # numerical guard on the way to it — and it is stored rather than assumed.
        clip = self.extras.get("mu_clip")
        if clip is not None:
            mu = np.clip(mu, float(clip[0]), float(clip[1]))
        mu = mu.mean(axis=0)
        if self.response == "mean_mu_x_trials":
            return mu * frame[self.extras["trials"]].to_numpy(dtype=float)
        if self.response != "mean_mu":
            raise KeyError(f"unknown response {self.response!r}")
        return mu

    # ── The gate ──────────────────────────────────────────────────────────────

    def roundtrip(self) -> dict:
        """Reproduce the stored predictions from the saved draws and recipe alone.

        This is the gate `docs/simulations-plan.md` names. It refits nothing and imports no
        Stan: the probe rows are the **raw** design frame, the recipe rebuilds the head's
        columns, and the result must match what the fitted head itself reported on the same
        rows. Two errors are reported, not one, because they fail differently — a design
        mismatch is a broken recipe and a prediction mismatch with a clean design is a
        broken link function or a mis-thinned draw block.
        """
        probe = self.reference["frame"]
        if not len(probe):
            raise ValueError(
                f"{self.head} carries an empty round-trip probe, so the gate would pass "
                f"vacuously. The head's evaluation frame produced no scorable rows — "
                f"check the filter it applies before predicting.")
        design = self.recipe.matrix(probe)
        stored_design = np.asarray(self.reference["design"], dtype=float)
        if design.shape != stored_design.shape:
            design_error = float("inf")
        elif not design.size:
            # The intercept-only arm has a (rows x 0) design. Agreeing on nothing is not a
            # failure, but the prediction comparison is then carrying the whole check —
            # which it can, since alpha is all there is to get wrong.
            design_error = 0.0
        else:
            design_error = float(np.max(np.abs(design - stored_design)))
        prediction = self.predict(probe)
        stored = np.asarray(self.reference["prediction"], dtype=float)
        pred_error = float(np.max(np.abs(prediction - stored))) \
            if prediction.shape == stored.shape else float("inf")
        scale = float(np.max(np.abs(stored))) if stored.size else 1.0
        return {
            "head": self.head,
            "n_probe_rows": len(probe),
            "design_shape_ok": design.shape == stored_design.shape,
            "max_design_error": design_error,
            "max_prediction_error": pred_error,
            "relative_prediction_error": pred_error / max(scale, 1e-12),
            "passes": bool(design_error <= DESIGN_TOL and pred_error <= PREDICTION_TOL),
        }


# ── Persistence ───────────────────────────────────────────────────────────────

def posteriors_dir(cfg: dict, window: str | None = None) -> Path:
    """`data/features/posteriors/<window>/`, one directory per fit window.

    **Namespaced by window rather than flat**, because the three windows are not versions
    of one artifact — they are three artifacts with three different consumers, and all
    three are wanted at once. `train` is what a backtest scored against realized 2022-23 /
    2023-24 may read, since those seasons are in `train_val`'s fit. `train_val` is what the
    one-shot test readout reads, for the same reason `src/final_evaluation.py` refits on
    train plus validation before scoring test: a held-out figure should describe the model
    that would actually deploy. `full` is the production board.

    A flat layout made the second window silently overwrite the first, which is the
    opposite of what `docs/simulations-plan.md` asks for when it says one posterior per fit
    window is what makes widening the backtest affordable.
    """
    root = Path(cfg["data"]["features_dir"]) / POSTERIOR_DIR
    return root if window is None else root / window


def save(artifact: PosteriorArtifact, dest_dir: Path) -> Path:
    """Mirrors `features.encode.save_artifacts`: one file, `mkdir` first, print the line."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{artifact.head}.pkl"
    with open(dest, "wb") as fh:
        pickle.dump(artifact, fh)
    return dest


def load(head: str, dest_dir: Path | str) -> PosteriorArtifact:
    """One head's posterior. No CmdStan, no refit, no `src.models.stan_*` import."""
    path = Path(dest_dir) / f"{head}.pkl"
    if not path.exists():
        raise FileNotFoundError(
            f"no posterior artifact at {path}. Run `make posteriors` — it is the only "
            f"target in the simulation layer that needs a CmdStan toolchain.")
    with open(path, "rb") as fh:
        return pickle.load(fh)


def load_all(dest_dir: Path | str, heads: list[str] | None = None
             ) -> dict[str, PosteriorArtifact]:
    """Every artifact on disk, keyed by head name."""
    directory = Path(dest_dir)
    names = heads if heads is not None else sorted(
        p.stem for p in directory.glob("*.pkl"))
    return {name: load(name, directory) for name in names}


def require_window(artifacts: dict, window: str = FIT_WINDOW) -> None:
    """Refuse artifacts fitted on a wider window than the caller is allowed to consume.

    The leak this prevents is quiet, and it is quiet in both directions. A backtest scoring
    *validation* with `train_val` heads has read 2022-23 and 2023-24 through the
    coefficients; one scoring *test* with `full` heads has read 2024-25 and 2025-26 the same
    way. No split guard can see either, because the guards sit on frames rather than on
    parameters — which is exactly why the window is stamped in and checkable.
    """
    wrong = {name: art.provenance.get("fit_window")
             for name, art in artifacts.items()
             if art.provenance.get("fit_window") != window}
    if wrong:
        raise ValueError(
            f"posterior artifacts were fitted at {sorted(set(wrong.values()))} but this "
            f"consumer requires `{window}`: {sorted(wrong)}. Re-run "
            f"`make posteriors WINDOW={window}`.")


# ── Provenance ────────────────────────────────────────────────────────────────

def _git_sha() -> str:
    """`HEAD`, marked `-dirty` when the tree has uncommitted changes.

    The marker matters more than the SHA: an artifact built from a dirty tree cannot be
    reproduced from the SHA alone, and saying so is cheaper than discovering it later.
    """
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                             text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True,
                               text=True, check=True).stdout.strip()
        return f"{sha}-dirty" if dirty else sha
    except Exception:                                             # pragma: no cover
        return "unknown"


def _provenance(window: str, frame: pd.DataFrame, n_draws: int, n_full: int,
                cfg_stan: dict, diagnostics: dict, seconds: float) -> dict:
    # The duration head's fitting frame is collapsed spells, which carry a season only
    # when the arm has covariates — so the season span is reported when it exists rather
    # than assumed.
    seasons = (sorted(frame["season"].unique()) if "season" in frame.columns else [])
    return {
        "fit_window": window,
        "n_fit_rows": int(len(frame)),
        "n_fit_seasons": len(seasons),
        "first_season": seasons[0] if seasons else "",
        "last_season": seasons[-1] if seasons else "",
        "posterior_draws": int(n_draws),
        "posterior_draws_before_thinning": int(n_full),
        "chains": int(cfg_stan.get("chains", 4)),
        "warmup": int(cfg_stan.get("warmup", 1000)),
        "samples": int(cfg_stan.get("samples", 1000)),
        "seed": int(cfg_stan.get("seed", 42)),
        "max_rhat": float(diagnostics.get("max_rhat", float("nan"))),
        "divergences": int(diagnostics.get("divergences", -1)),
        "converged": bool(diagnostics.get("converged", False)),
        "cmdstan": cmdstan_version(),
        "git_sha": _git_sha(),
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "fit_seconds": float(seconds),
        "module": "src/models/posteriors.py",
    }


# ── Splits ────────────────────────────────────────────────────────────────────

def windowed(design: pd.DataFrame, window: str, test_seasons: int
             ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """`(fit_frame, probe_frame)` for a fit window. The probe is always validation.

    `train_val` goes through `split_seasons` and drops the guarded right-hand side, which is
    exactly the legal `train, _ = split_seasons(...)` idiom `held_out.py` documents. `full`
    is the only window that reads the test seasons, and it says so out loud.
    """
    if window not in WINDOWS:
        raise ValueError(f"fit window must be one of {WINDOWS}; got {window!r}")
    train, val = selection_split(design, test_seasons)
    if window == "train":
        return train, val
    full_train, _ = split_seasons(design, test_seasons)
    if window == "train_val":
        return full_train, val
    assert_unlocked("posteriors at the `full` fit window")
    return design, val


def probe_rows(frame: pd.DataFrame, n: int = PROBE_ROWS) -> pd.DataFrame:
    """Evenly spaced rows through the frame — `thin`, applied to rows instead of draws.

    Evenly spaced rather than a head slice for the same reason the draws are: the frame is
    sorted by season, so the first `n` rows are one era of the league.
    """
    if len(frame) <= n:
        return frame
    return frame.iloc[thin(len(frame), n)]


def _thinned(model, keep: int) -> tuple[object, dict, int]:
    """A shallow copy of the head carrying only the kept draws.

    The reference predictions must come from the **same** thinned posterior the artifact
    stores, or the round-trip tolerance is dominated by thinning Monte Carlo error rather
    than by whether the recipe is right. Copying the head and swapping its draw arrays gets
    that without a second implementation of any head's predict method — the heads then thin
    `keep` out of `keep`, which is the identity.
    """
    import copy as _copy

    n_full = len(model.alpha_draws)
    idx = thin(n_full, keep)
    out = _copy.copy(model)
    draws = {"alpha_draws": np.asarray(model.alpha_draws)[idx],
             "beta_draws": np.asarray(model.beta_draws)[idx]}
    out.alpha_draws = draws["alpha_draws"]
    out.beta_draws = draws["beta_draws"]
    for name in ("rho_draws", "phi_draws", "kappa_draws"):
        values = getattr(model, name, None)
        if values is None:
            continue
        values = np.asarray(values)
        draws[name] = values[idx]
        setattr(out, name, draws[name])
    # The composition's per-(player, season) random effect. Only its SCALE is kept: the
    # fitted `u_z` describe player-seasons that are over, and the predictive integrates over
    # a fresh `z` per posterior draw — the same treatment `YearTerm` gets, one index down.
    # The term object is copied rather than shared, or thinning the artifact's draws would
    # mutate the live head's.
    ps = getattr(model, "ps", None)
    if ps is not None and getattr(ps, "enabled", False):
        out.ps = _copy.copy(ps)
        draws["sigma_u_draws"] = np.asarray(ps.sigma_draws)[idx]
        out.ps.sigma_draws = draws["sigma_u_draws"]
    for attr in ("predictive_samples", "predictive_draws"):
        if hasattr(out, attr):
            setattr(out, attr, len(idx))
    return out, draws, n_full


# ── Head builders ─────────────────────────────────────────────────────────────
#
# One function per head family. Each calls the head's own variant ladder for the frames and
# the feature list, expresses only the fitted state that ladder leaves implicit, fits once at
# the selected variant, and hands back a verified artifact.

def _finish(head: str, head_label: str, family: str, response: str, variant: str,
            features: list[str], model, fit_frame: pd.DataFrame,
            probe_transformed: pd.DataFrame, probe_raw: pd.DataFrame,
            steps: list[dict], builder: str, extras: dict, window: str,
            cfg_stan: dict, draws_kept: int, seconds: float) -> PosteriorArtifact:
    """Thin, capture the reference from the live head, assemble, and verify.

    **A fitted random effect is persisted as its SCALE or refused outright, never dropped.**
    Both kinds this project has are the same object: the fitted per-level values describe
    levels that are over, and the predictive integrates over a fresh `z` per posterior draw,
    so an artifact that carried only `alpha`/`beta` would be *quietly narrower* than the
    head it claims to persist — the one failure mode a round-trip on the mean cannot see.
    The composition's `sigma_u` is wired through (`extras["player_season_effect"]`,
    `draws["sigma_u_draws"]`, consumed by `StanComposition.predict_samples` via its
    rehydrated `PlayerSeasonTerm`); the year effect is not, and still raises.
    """
    year = getattr(model, "year", None)
    if year is not None and getattr(year, "enabled", False):
        raise NotImplementedError(
            f"{head} was fitted with a year random effect, which the recipe does not "
            f"carry: `YearTerm.shift` is a fresh N(0, 1) draw per posterior draw shared "
            f"across rows, so persisting it means persisting `sigma_year` and drawing "
            f"`z` in the simulator. No head in `make stan` enables it today; wire that "
            f"through before shipping one that does — `sigma_u` below is the worked "
            f"example.")
    ps = getattr(model, "ps", None)
    if ps is not None and getattr(ps, "enabled", False):
        extras = {**extras, "player_season_effect": True, "dispersion_u": "sigma_u_draws",
                  "u_sd_scale": float(ps.scale), "u_stream": ps.stream, **ps.summary()}
    thinned, draws, n_full = _thinned(model, draws_kept)
    recipe = DesignRecipe(variant=variant, features=list(features),
                          scaler=model.scaler, steps=tuple(steps), builder=builder)

    # The reference is taken through the head's OWN methods on its OWN transformed frame,
    # so the round-trip compares two genuinely different paths to the same number.
    stored_design = _stored_design(model, probe_transformed)
    prediction = _reference_prediction(thinned, response, probe_transformed, extras)

    artifact = PosteriorArtifact(
        head=head, head_label=head_label, family=family, response=response,
        recipe=recipe, draws=draws, extras=dict(extras),
        provenance=_provenance(window, fit_frame, len(draws["alpha_draws"]), n_full,
                               cfg_stan, getattr(model, "diagnostics", {}), seconds),
        reference={"frame": probe_raw.copy(), "design": stored_design,
                   "prediction": prediction})

    check = artifact.roundtrip()
    if not check["passes"]:
        raise AssertionError(
            f"{head}: the saved draws and recipe do NOT reproduce the head's own "
            f"predictions — design error {check['max_design_error']:.3e} (bar "
            f"{DESIGN_TOL:.0e}), prediction error {check['max_prediction_error']:.3e} "
            f"(bar {PREDICTION_TOL:.0e}). The recipe has drifted from the head's variant "
            f"ladder; do not loosen the tolerance.")
    return artifact


def _stored_design(model, frame: pd.DataFrame) -> np.ndarray:
    """The head's own standardized design matrix.

    Every head but one exposes `_design`. `BetaGeometricHead` standardizes *inline* inside
    `shapes` instead, so for that head alone the expression is written out here — which
    makes the design half of its round-trip weaker than the others by construction. Its
    prediction half is not: that still goes through `shapes` and `beta_shapes`, a genuinely
    separate path, and it is the half that would catch a wrong scaler anyway.
    """
    if hasattr(model, "_design"):
        return model._design(frame)
    if not getattr(model, "features", []):
        return np.zeros((len(frame), 0))
    X = np.nan_to_num(frame[model.features].to_numpy(dtype=float),
                      nan=0.0, posinf=0.0, neginf=0.0)
    return model.scaler.transform(X)


def _reference_prediction(model, response: str, frame: pd.DataFrame,
                          extras: dict) -> np.ndarray:
    """The head's own reported mean, through the head's own public method.

    Four heads, four different means, and they are not interchangeable: the conversion
    heads expose `predict_p`, the count and minutes heads `predict_mean`, the games-played
    binomial heads a plug-in `mean`, and the beta-geometric head nothing at all — its mean
    only exists as `a / (a + b)` from the per-draw shapes.
    """
    if response == "mean_mu":
        if hasattr(model, "predict_p"):
            return model.predict_p(frame)
        if hasattr(model, "mu_draws"):
            mus, _ = model.mu_draws(frame, len(model.alpha_draws))
            return mus.mean(axis=0)
        mus = []
        for draw in range(len(model.alpha_draws)):
            a, b = model.shapes(frame, draw)
            mus.append(a / (a + b))
        return np.mean(mus, axis=0)
    if response in ("mean_mu_x_trials", "mean_count"):
        return model.predict_mean(frame)
    if response == "plug_in_mu":
        return model.mean(frame)
    if response == "eta":
        eta, _ = model._eta_base(frame)
        return eta.mean(axis=1)
    raise KeyError(f"unknown response {response!r}")


def availability_artifact(cfg: dict, window: str, draws_kept: int) -> PosteriorArtifact:
    """The beta-binomial games-played-out-of-team-games head. No variant ladder.

    **Two season axes, and they are orthogonal.** `window` is the *split* axis — which of
    train / train_val / full is eligible to be fitted, the thing `require_window` guards and
    `posteriors/<window>/` is namespaced by. `fit_first_season` is the *truncation* axis —
    which recent suffix of that split the head actually fits, `stan.availability.first_season`
    in config and `StanAvailability.fitting_rows` in code. A `train` artifact truncated at
    2012-13 and a `train` artifact over the full history are the same window and different
    heads; conflating them would let a consumer that correctly refuses the wrong split accept
    the wrong model. So they carry different names, live in different places (provenance
    against extras), and both reach the manifest.

    The provenance is recorded over the **truncated** frame, which is what makes
    `model_cards.verify`'s population anchor a real check: it compares the rebuilt fitting
    frame against `n_fit_rows` and the season span, and recording the pre-truncation frame
    would anchor the card to a population the coefficients never saw.

    The dispersion is a vector over prior-MPG role buckets, so the artifact also has to say
    which entry applies to which row. That goes in as a `cut` recipe step writing `rho_bin`,
    the same door `src/sim/season.py` already opens on the composition
    (`art.recipe.transform(frame)["rho_bin"]`) — a consumer reconstructs the assignment for
    any frame, including a 2026-27 board, without refitting and without importing the head.
    """
    from src.models.availability import FEATURE_COLS
    from src.models.stan_availability import (FIRST_SEASON, ROLE_BIN_COL, ROLE_COL,
                                              StanAvailability, availability_design,
                                              role_edges)

    cfg_stan = cfg.get("stan", {})
    cfg_head = cfg_stan.get("availability", {})
    test_seasons = int(cfg.get("features", {}).get("availability", {})
                       .get("test_seasons", 2))
    l2 = float(cfg.get("features", {}).get("availability", {}).get("glm_l2", 1.0))

    design = availability_design(cfg)
    offered, val = windowed(design, window, test_seasons)
    probe = probe_rows(val)

    started = time.perf_counter()
    model = StanAvailability(
        l2=l2, pmf_mode="posterior", chains=int(cfg_stan.get("chains", 4)),
        warmup=int(cfg_stan.get("warmup", 1000)),
        samples=int(cfg_stan.get("samples", 1000)),
        seed=int(cfg_stan.get("seed", 42)),
        first_season=cfg_head.get("first_season", FIRST_SEASON),
        role_rho=bool(cfg_head.get("role_rho", True))).fit(offered)
    seconds = time.perf_counter() - started

    # The rows the head fitted, taken from the head's own `fitting_rows` rather than
    # re-derived from config here: the truncation is the head's, and a second expression of
    # it is a second thing that can drift out of step with the coefficients.
    fit_frame = model.fitting_rows(offered)
    steps = [{"kind": "cut", "column": ROLE_COL, "edges": role_edges(model.role_rho),
              "name": ROLE_BIN_COL}]

    return _finish(
        head="availability", head_label="availability", family="betabinomial",
        response="mean_mu", variant="base", features=list(FEATURE_COLS), model=model,
        fit_frame=fit_frame, probe_transformed=probe, probe_raw=probe, steps=steps,
        builder="src.models.stan_availability.availability_design",
        extras={"successes": "gp", "trials": "team_games", "dispersion": "rho_draws",
                # The dispersion axis: which `rho_draws` column applies to which row.
                "n_rho": int(model.n_rho), "rho_bin_column": ROLE_BIN_COL,
                "rho_bin_source": ROLE_COL,
                "rho_bin_edges": np.asarray(role_edges(model.role_rho), dtype=float),
                "rho_labels": list(model.rho_labels),
                "role_rho": bool(model.role_rho),
                # The season-truncation axis. NOT `provenance["fit_window"]`, and not
                # `provenance["first_season"]` either — that one is a *readout* of the
                # frame's own span, where this is the knob that produced it.
                "fit_first_season": str(model.first_season or ""),
                "n_rows_before_truncation": int(len(offered))},
        window=window, cfg_stan=cfg_stan, draws_kept=draws_kept, seconds=seconds)


def minutes_artifact(cfg: dict, window: str, draws_kept: int) -> PosteriorArtifact:
    """`min | available`, season-collapsed as successes out of real game length."""
    from src.models.availability import FEATURE_COLS
    from src.models.stan_minutes import (OWN, SPLINE_KNOTS, StanMinutes, build_design,
                                         variants)

    cfg_stan = cfg.get("stan", {})
    test_seasons = int(cfg.get("features", {}).get("availability", {})
                       .get("test_seasons", 2))
    n_knots = int(cfg_stan.get("minutes", {}).get("spline_knots", SPLINE_KNOTS))
    variant = selected_variant(cfg, "stan_minutes_metrics.csv", "logit_own_spline")

    design = build_design(cfg)
    fit_frame, val = windowed(design, window, test_seasons)
    probe_raw = probe_rows(val)

    tr, probe, features = variants(fit_frame, probe_raw, n_knots)[variant]
    steps = _minutes_steps(tr, variant, features, OWN, n_knots)

    started = time.perf_counter()
    model = StanMinutes(features, name=f"posteriors/minutes/{variant}",
                        chains=int(cfg_stan.get("chains", 4)),
                        warmup=int(cfg_stan.get("warmup", 1000)),
                        samples=int(cfg_stan.get("samples", 1000)),
                        seed=int(cfg_stan.get("seed", 42))).fit(tr)
    seconds = time.perf_counter() - started

    return _finish(
        head="minutes", head_label="min|available", family="betabinomial",
        response="mean_mu_x_trials", variant=variant, features=features, model=model,
        fit_frame=fit_frame, probe_transformed=probe, probe_raw=probe_raw, steps=steps,
        builder="src.models.stan_minutes.build_design",
        extras={"successes": "successes", "trials": "trials",
                "dispersion": "rho_draws", "base_features": list(FEATURE_COLS)},
        window=window, cfg_stan=cfg_stan, draws_kept=draws_kept, seconds=seconds)


def _minutes_steps(tr: pd.DataFrame, variant: str, features: list[str], own: str,
                   n_knots: int) -> list[dict]:
    """`stan_minutes.variants`' fitted state. `linear` and `logit_own` carry none."""
    if variant in ("linear", "logit_own"):
        return []
    if variant == "logit_own_quadratic":
        return [{"kind": "square", "columns": {own: f"{own}__sq"}}]
    if variant != "logit_own_spline":
        raise KeyError(f"no recipe for the minutes variant {variant!r}")
    names = [c for c in features if c.startswith(f"{own}__s")]
    return [_spline_step(tr, own, n_knots, names)]


def component_artifacts(cfg: dict, window: str, draws_kept: int):
    """The seven negative-binomial counts and four beta-binomial conversions."""
    from src.models.component_rates import (BIO_COLS, CONTEXT_COLS, CONVERSION_HEADS,
                                            COUNT_HEADS, build_design)
    from src.models.stan_components import (SPLINE_KNOTS, StanConversion, StanCount,
                                            conversion_variants, count_variants)

    cfg_stan = cfg.get("stan", {})
    test_seasons = int(cfg.get("features", {}).get("availability", {})
                       .get("test_seasons", 2))
    n_knots = int(cfg_stan.get("components", {}).get("spline_knots", SPLINE_KNOTS))
    features_dir = Path(cfg["data"]["features_dir"])

    targets = pd.read_parquet(features_dir / "component_targets.parquet")
    design = build_design(targets, cfg["data"]["seasons"], cfg["data"]["raw_dir"])
    fit_frame, val = windowed(design, window, test_seasons)
    probe_raw = probe_rows(val)
    chosen = selected_specs(cfg)

    for component in COUNT_HEADS:
        variant = chosen.get(component, "log_own")
        tr, probe, features = count_variants(fit_frame, probe_raw, component,
                                             n_knots)[variant]
        base = [f"{component}_p36_lag1"] + CONTEXT_COLS + BIO_COLS
        steps = _count_steps(fit_frame, tr, variant, features, base,
                             f"{component}_p36_lag1", n_knots)

        started = time.perf_counter()
        model = StanCount(features, component,
                          name=f"posteriors/{component}/{variant}",
                          chains=int(cfg_stan.get("chains", 4)),
                          warmup=int(cfg_stan.get("warmup", 1000)),
                          samples=int(cfg_stan.get("samples", 1000)),
                          seed=int(cfg_stan.get("seed", 42))).fit(tr)
        seconds = time.perf_counter() - started

        yield _finish(
            head=component, head_label=component, family="negbinomial",
            response="mean_count", variant=variant, features=features, model=model,
            fit_frame=fit_frame, probe_transformed=probe, probe_raw=probe_raw,
            steps=steps, builder="src.models.component_rates.build_design",
            extras={"component": component, "exposure": "total_minutes",
                    "dispersion": "phi_draws"},
            window=window, cfg_stan=cfg_stan, draws_kept=draws_kept, seconds=seconds)

    for made, attempted in CONVERSION_HEADS:
        label = f"{made}|{attempted}"
        variant = chosen.get(label, "logit_own_spline")
        # Conversion heads fit on rows with attempts only — the same filter the head
        # applies internally, applied to the probe so the reference rows are scorable.
        live_probe = probe_raw[probe_raw[attempted].to_numpy(dtype=float) > 0]
        tr, probe, features = conversion_variants(fit_frame, live_probe, made,
                                                  attempted, n_knots)[variant]
        own = f"{made}_pct_lag1"
        base = [own, f"{attempted}_p36_lag1"] + CONTEXT_COLS + BIO_COLS
        steps = _conversion_steps(fit_frame, tr, variant, features, base, own, n_knots)

        started = time.perf_counter()
        model = StanConversion(features, made, attempted,
                               name=f"posteriors/{label}/{variant}",
                               chains=int(cfg_stan.get("chains", 4)),
                               warmup=int(cfg_stan.get("warmup", 1000)),
                               samples=int(cfg_stan.get("samples", 1000)),
                               seed=int(cfg_stan.get("seed", 42))).fit(tr)
        seconds = time.perf_counter() - started

        yield _finish(
            head=f"{made}_given_{attempted}", head_label=label, family="betabinomial",
            response="mean_mu", variant=variant, features=features, model=model,
            fit_frame=fit_frame, probe_transformed=probe, probe_raw=live_probe,
            steps=steps, builder="src.models.component_rates.build_design",
            extras={"made": made, "attempted": attempted, "trials": attempted,
                    "dispersion": "rho_draws"},
            window=window, cfg_stan=cfg_stan, draws_kept=draws_kept, seconds=seconds)


def _count_steps(train: pd.DataFrame, tr: pd.DataFrame, variant: str,
                 features: list[str], base: list[str], own: str,
                 n_knots: int) -> list[dict]:
    """`stan_components.count_variants`' fitted state: impute, then log, then spline."""
    steps = [_impute_step(train, base, tr)]
    if variant == "linear":
        return steps
    log_name = f"log_{own}"
    steps.append({"kind": "log1p", "columns": {own: log_name}})
    if variant == "log_own":
        return steps
    if variant != "log_own_spline":
        raise KeyError(f"no recipe for the count variant {variant!r}")
    names = [c for c in features if c.startswith(f"{log_name}__s")]
    steps.append(_spline_step(tr, log_name, n_knots, names))
    return steps


def _conversion_steps(train: pd.DataFrame, tr: pd.DataFrame, variant: str,
                      features: list[str], base: list[str], own: str,
                      n_knots: int) -> list[dict]:
    """`stan_components.conversion_variants`' fitted state: impute, logit, spline."""
    steps = [_impute_step(train, base, tr)]
    if variant == "linear":
        return steps
    logit_name = f"logit_{own}"
    steps.append({"kind": "logit", "columns": {own: logit_name}, "clip": 1e-3})
    if variant == "logit_own":
        return steps
    if variant != "logit_own_spline":
        raise KeyError(f"no recipe for the conversion variant {variant!r}")
    names = [c for c in features if c.startswith(f"{logit_name}__s")]
    steps.append(_spline_step(tr, logit_name, n_knots, names))
    return steps


def composition_artifact(cfg: dict, window: str, draws_kept: int) -> PosteriorArtifact:
    """The team-game minutes allocation. One fit, and it is the expensive one.

    `stan.composition.player_season_effect` turns on the per-(player, season) random effect
    added for item 3d. It is read here rather than baked in because it changes what the
    artifact carries — `sigma_u_draws`, and the predictive that consumes it — and because
    the arm that ships is decided by `make composition-effects`, not by this module.
    """
    from src.models.stan_composition import (OWN, PILOT_FIRST_SEASON, RHO_BIN_COL,
                                             RHO_BINS, TEAM_COLS, StanComposition,
                                             composition_frame, effect_variants,
                                             team_context, variants)

    cfg_stan = cfg.get("stan", {})
    comp_cfg = cfg_stan.get("composition", {})
    first_season = str(comp_cfg.get("first_season", PILOT_FIRST_SEASON))
    keep = int(comp_cfg.get("predictive_samples", 200))
    ps_effect = bool(comp_cfg.get("player_season_effect", False))
    team_block = bool(comp_cfg.get("team_context", False))
    test_seasons = int(cfg.get("features", {}).get("availability", {})
                       .get("test_seasons", 2))
    variant = selected_variant(cfg, "stan_composition_metrics.csv",
                               "betabinom_ot_graded")

    frame = composition_frame(cfg)
    pilot = frame[frame["season"] >= first_season].reset_index(drop=True)
    fit_frame, val = windowed(pilot, window, test_seasons)
    # Whole team-game blocks, because the composition's design is per-row but its
    # simulator is per block — a probe cut mid-block would be unusable downstream.
    probe_raw = team_game_probe(val)

    block = None
    if team_block:
        block = team_context(Path(cfg["data"]["features_dir"]))
        arm = "ps_team" if ps_effect else "team"
        tr, probe, features, dispersed, n_rho, _, _ = effect_variants(
            fit_frame, probe_raw, block, variant, RHO_BINS)[0][arm]
        variant = f"{variant}+{arm}"
    else:
        tr, probe, features, dispersed, n_rho = variants(fit_frame, probe_raw,
                                                         RHO_BINS)[variant]
        if ps_effect:
            variant = f"{variant}+ps"
    steps = _composition_steps(fit_frame, tr, features, OWN, RHO_BIN_COL, RHO_BINS,
                               team_block=block)

    started = time.perf_counter()
    model = StanComposition(features, dispersed, n_rho,
                            name=f"posteriors/composition/{variant}",
                            chains=int(cfg_stan.get("chains", 4)),
                            warmup=int(cfg_stan.get("warmup", 1000)),
                            samples=int(cfg_stan.get("samples", 1000)),
                            seed=int(cfg_stan.get("seed", 42)),
                            predictive_samples=keep,
                            player_season_effect=ps_effect,
                            u_sd_scale=float(comp_cfg.get("effects", {})
                                             .get("u_sd_scale", 1.0)),
                            u_centered=bool(comp_cfg.get("u_centered", False))).fit(tr)
    seconds = time.perf_counter() - started

    return _finish(
        head="composition", head_label="minutes composition", family="composition",
        response="eta", variant=variant, features=features, model=model,
        fit_frame=fit_frame, probe_transformed=probe, probe_raw=probe_raw, steps=steps,
        builder="src.models.stan_composition.composition_frame",
        extras={"dispersed": int(dispersed), "n_rho": int(n_rho),
                "rho_bin_column": RHO_BIN_COL, "dispersion": "rho_draws",
                "first_season": first_season, "group_keys": ["game_id", "team_id"],
                "team_context": team_block, "team_cols": list(TEAM_COLS),
                "predictive_samples": keep},
        window=window, cfg_stan=cfg_stan, draws_kept=draws_kept, seconds=seconds)


def team_game_probe(val: pd.DataFrame, n: int = PROBE_ROWS) -> pd.DataFrame:
    """Whole team-games spanning the frame, never a partial block.

    `ragged_arrays` asserts that rows are contiguous team-game blocks, so a probe that
    slices through one would be rejected by the very code the artifact exists to feed.

    Public because the round-trip probe is not the only bounded cut anyone takes of this
    frame: `model_cards.predictive_frame` needs the same block-aware subsample for the same
    reason, and a second implementation of it would be a second chance to slice a block.
    """
    starts = np.flatnonzero(val["position"].to_numpy(dtype=np.int64) == 0)
    if len(starts) == 0:
        return val.iloc[:n]
    ends = np.append(starts[1:], len(val))
    per_block = max(int(np.median(ends - starts)), 1)
    n_blocks = max(int(n // per_block), 1)
    pick = thin(len(starts), min(n_blocks, len(starts)))
    keep = np.concatenate([np.arange(starts[b], ends[b]) for b in pick])
    return val.iloc[keep].reset_index(drop=True)


def _composition_steps(train: pd.DataFrame, tr: pd.DataFrame, features: list[str],
                       own: str, bin_col: str, n_bins: int,
                       team_block: pd.DataFrame | None = None) -> list[dict]:
    """`stan_composition.variants`' fitted state, in the order that function applies it.

    `design_missing` is set **before** the imputation, because it flags the rookie who lost
    every design column at once and the imputation is what erases the evidence.
    """
    from src.models.availability import FEATURE_COLS
    from src.models.stan_composition import (TEAM_COLS, TEAM_MISSING, UNIT_KEYS,
                                             rho_bin_edges)

    base = [c for c in FEATURE_COLS if c != "minutes_per_game_lag1"] + [own]
    steps: list[dict] = [
        {"kind": "isna_flag", "column": "gp_share_lag1", "name": "design_missing"},
        _impute_step(train, base, tr),
    ]
    if "ot_x_own" in features:
        steps.append({"kind": "product", "left": "n_overtimes", "right": own,
                      "name": "ot_x_own"})
    if team_block is not None:
        # `attach_team_context` joins AFTER `variants` has built `design_missing` and
        # imputed, so the step goes here. **The block is the RAW one, not `tr`'s columns**:
        # `tr` has already had its holes filled from train means, so rebuilding the block
        # from it would persist imputed values as if they were observed and leave
        # `team_missing` identically zero on every rebuilt frame — every column present,
        # every shape right, and only the numbers wrong, which is the failure the round-trip
        # exists for. The means still come from `tr`'s covered rows, which are unimputed.
        covered = tr[tr[TEAM_MISSING] == 0]
        steps.append({"kind": "join", "name": "team_context", "keys": list(UNIT_KEYS),
                      "flag": TEAM_MISSING,
                      "block": team_block.drop_duplicates(UNIT_KEYS)
                                         .reset_index(drop=True),
                      "means": {c: float(covered[c].mean()) for c in TEAM_COLS}})
    # Not a design column — the graded arm's dispersion is indexed by it, so the
    # simulator needs the same edges the fit used.
    steps.append({"kind": "bins", "column": bin_col,
                  "edges": rho_bin_edges(tr, n_bins), "name": "rho_bin"})
    return steps


def game_length_artifacts(cfg: dict, window: str, draws_kept: int):
    """The two halves of the game-length draw: does a game go to OT, and how deep.

    Two artifacts rather than one, because they are two likelihoods on two frames — the
    onset head fits season *cells* and the depth head fits collapsed *depth rows*. The
    simulator recombines them through `stan_game_length.draw_inputs`, which is the same
    factorization the rest of the chain runs on.

    This is the one head family whose fitting frame is games rather than player-seasons, so
    its probe frames are small by construction: at the shipped arm the validation cells are
    one row per validation season and the depth rows are one per observed depth. That is
    thin for a round-trip and it is the honest probe — there is nothing else on that frame.
    """
    from src.models.games_played import MU_MAX, MU_MIN
    from src.models.stan_game_length import (ARMS, KAPPA_SCALE, MATCHUP_BINS,
                                             MATCHUP_COL, SEASON_COL, depth_rows,
                                             fit_depth, fit_overtime, game_frame,
                                             matchup_edges, overtime_cells)

    cfg_stan = cfg.get("stan", {})
    cfg_gl = cfg_stan.get("game_length", {})
    test_seasons = int(cfg.get("features", {}).get("availability", {})
                       .get("test_seasons", 2))
    arm = selected_variant(cfg, "stan_game_length_metrics.csv", "season_trend")
    features = ARMS[arm]["features"]

    games = game_frame(cfg)
    fit_games, val_games = windowed(games, window, test_seasons)
    edges = matchup_edges(fit_games, int(cfg_gl.get("matchup_bins", MATCHUP_BINS)))
    if MATCHUP_COL in features:
        fit_games = fit_games[fit_games[MATCHUP_COL].notna()]
        val_games = val_games[val_games[MATCHUP_COL].notna()]

    cells = overtime_cells(fit_games, features, edges)
    probe_cells = overtime_cells(val_games, features, edges)
    started = time.perf_counter()
    onset = fit_overtime(cells, features, "posteriors/game_length_ot", cfg_stan)
    seconds = time.perf_counter() - started
    yield _finish(
        head="game_length_ot", head_label="overtime onset", family="betabinomial",
        response="plug_in_mu", variant=arm, features=list(features), model=onset,
        fit_frame=cells, probe_transformed=probe_cells, probe_raw=probe_cells, steps=[],
        builder="src.models.stan_game_length.overtime_cells",
        extras={"successes": "y", "trials": "n", "dispersion": "rho_draws",
                "arm": arm, "season_column": SEASON_COL,
                "matchup_edges": np.asarray(edges, dtype=float)},
        window=window, cfg_stan=cfg_stan, draws_kept=draws_kept, seconds=seconds)

    spells = depth_rows(fit_games)
    probe_depth = depth_rows(val_games)
    started = time.perf_counter()
    depth = fit_depth(spells, cfg_stan,
                      float(cfg_gl.get("kappa_scale", KAPPA_SCALE)),
                      name="posteriors/game_length_depth")
    seconds = time.perf_counter() - started
    yield _finish(
        head="game_length_depth", head_label="overtime depth", family="betageometric",
        response="mean_mu", variant=arm, features=[], model=depth, fit_frame=spells,
        probe_transformed=probe_depth, probe_raw=probe_depth, steps=[],
        builder="src.models.stan_game_length.depth_rows",
        extras={"length": "t", "weight": "w", "dispersion": "kappa_draws",
                "arm": arm, "mu_clip": (MU_MIN, MU_MAX)},
        window=window, cfg_stan=cfg_stan, draws_kept=draws_kept, seconds=seconds)


def games_played_artifacts(cfg: dict, window: str, draws_kept: int):
    """Entry, exit, onset and the spell-duration head of the games-played process."""
    from src.models.availability import FEATURE_COLS
    from src.models.games_played import MU_MAX, MU_MIN
    from src.models.stan_games_played import (BetaBinomialHead, BetaGeometricHead,
                                              GLM_L2, KAPPA_SCALE, games_played_design,
                                              onset_rate_lags, spell_rows_for)

    cfg_stan = cfg.get("stan", {})
    cfg_gp = cfg_stan.get("games_played", {})
    test_seasons = int(cfg.get("features", {}).get("availability", {})
                       .get("test_seasons", 2))
    features_dir = Path(cfg["data"]["features_dir"])
    variant = selected_variant(cfg, "stan_games_played_metrics.csv",
                               "duration_covariates")

    panel = pd.read_parquet(features_dir / "availability_panel.parquet")
    design = games_played_design(cfg, panel)
    design = design.merge(onset_rate_lags(design, cfg["data"]["seasons"]),
                          on=["season", "player_id"], how="left")
    fit_all, val = windowed(design, window, test_seasons)
    fit_frame = fit_all[fit_all["fittable"]]
    probe_raw = probe_rows(val[val["fittable"]])

    common = dict(l2=float(cfg_gp.get("glm_l2", GLM_L2)),
                  chains=int(cfg_stan.get("chains", 4)),
                  warmup=int(cfg_stan.get("warmup", 1000)),
                  samples=int(cfg_stan.get("samples", 1000)),
                  seed=int(cfg_stan.get("seed", 42)))

    for head, y_col, n_col in (("gp_entry", "pre_tenure", "entry_trials"),
                               ("gp_exit", "post_tenure", "exit_trials"),
                               ("gp_onset", "onsets", "at_risk_played")):
        started = time.perf_counter()
        model = BetaBinomialHead(list(FEATURE_COLS), f"posteriors/{head}",
                                 **common).fit(fit_frame, y_col, n_col)
        seconds = time.perf_counter() - started
        yield _finish(
            head=head, head_label=head.replace("gp_", "games-played "),
            family="betabinomial", response="plug_in_mu", variant=variant,
            features=list(FEATURE_COLS), model=model, fit_frame=fit_frame,
            probe_transformed=probe_raw, probe_raw=probe_raw, steps=[],
            builder="src.models.stan_games_played.games_played_design",
            extras={"successes": y_col, "trials": n_col, "dispersion": "rho_draws",
                    "arm": variant},
            window=window, cfg_stan=cfg_stan, draws_kept=draws_kept, seconds=seconds)

    # `duration_covariates` is the selected arm, and its covariates are the availability
    # block joined onto the collapsed spell rows — so the duration head's design frame is a
    # *spell* frame, not a player-season frame, and the artifact records that.
    dur_features = list(FEATURE_COLS) if variant == "duration_covariates" else []
    spell_fit = spell_rows_for(panel, fit_frame,
                               dur_features if dur_features else None)
    spell_probe = probe_rows(spell_rows_for(panel, probe_raw,
                                            dur_features if dur_features else None))
    started = time.perf_counter()
    duration = BetaGeometricHead(dur_features, "posteriors/gp_duration",
                                 kappa_scale=float(cfg_gp.get("kappa_scale",
                                                              KAPPA_SCALE)),
                                 **common).fit(spell_fit)
    seconds = time.perf_counter() - started
    yield _finish(
        head="gp_duration", head_label="absence-spell duration",
        family="betageometric", response="mean_mu", variant=variant,
        features=dur_features, model=duration, fit_frame=spell_fit,
        probe_transformed=spell_probe, probe_raw=spell_probe,
        steps=[], builder="src.models.stan_games_played.spell_rows_for",
        extras={"length": "t", "weight": "w", "dispersion": "kappa_draws",
                "arm": variant, "mu_clip": (MU_MIN, MU_MAX)},
        window=window, cfg_stan=cfg_stan, draws_kept=draws_kept, seconds=seconds)


# ── Which variant shipped ─────────────────────────────────────────────────────

def selected_variant(cfg: dict, artifact: str, fallback: str) -> str:
    """Read the shipped variant from the sweep that chose it, never re-decide it.

    The same rule `season_terms.selected_specs` follows and `src/final_evaluation.py`
    follows: this module persists what shipped, so re-running a selection here would be a
    second opinion competing with the artifact of record.
    """
    path = Path(cfg["evaluation"]["predictions_dir"]) / artifact
    if not path.exists():
        print(f"  /!\\  {path} is missing — falling back to `{fallback}`. Re-run the "
              f"head's own make target before trusting this artifact.")
        return fallback
    table = pd.read_csv(path)
    chosen = table[table["selected"]] if "selected" in table.columns else table.iloc[:0]
    if not len(chosen):
        print(f"  /!\\  {path} selects no variant — falling back to `{fallback}`.")
        return fallback
    return str(chosen["variant"].iloc[0])


def selected_specs(cfg: dict) -> dict[str, str]:
    """Every component head's selected variant, from `stan_component_metrics.csv`."""
    from src.models.season_terms import selected_specs as _specs

    specs, _ = _specs(Path(cfg["evaluation"]["predictions_dir"]))
    return specs


# ── Entry point ───────────────────────────────────────────────────────────────

def fit_first_season(artifact: PosteriorArtifact) -> str:
    """The season truncation a head applied *inside* its fit window, or `""` for none.

    One reader for two spellings: the availability head writes `fit_first_season` and the
    composition — which had the truncation first, under the flatter name — writes
    `first_season`. Renaming the composition's key would silently reinterpret every artifact
    already on disk, so the reader takes both instead. Do not add a third spelling.

    Distinct from `provenance["first_season"]`, which is the observed span of the fitted
    frame rather than the knob that produced it.
    """
    return str(artifact.extras.get("fit_first_season")
               or artifact.extras.get("first_season") or "")


def manifest_row(artifact: PosteriorArtifact, check: dict, dest: Path) -> dict:
    p = artifact.provenance
    return {
        "head": artifact.head,
        "head_label": artifact.head_label,
        "family": artifact.family,
        "variant": artifact.recipe.variant,
        "n_features": len(artifact.recipe.features),
        "n_draws": artifact.n_draws,
        "n_draws_before_thinning": p["posterior_draws_before_thinning"],
        "fit_window": p["fit_window"],
        # The OTHER season axis — the truncation inside the window. Empty for the eighteen
        # heads that fit whatever the window hands them. See the module docstring's table:
        # `first_season` below is the fitted frame's observed span, not this.
        "fit_first_season": fit_first_season(artifact),
        "n_fit_rows": p["n_fit_rows"],
        "first_season": p["first_season"],
        "last_season": p["last_season"],
        "max_rhat": p["max_rhat"],
        "divergences": p["divergences"],
        "converged": p["converged"],
        "fit_seconds": p["fit_seconds"],
        # A head with a random effect whose scale is not on the manifest is a head whose
        # predictive is quietly narrower than its fit, so the scale is a manifest column
        # rather than something a consumer has to unpickle to discover.
        "player_season_effect": bool(artifact.extras.get("player_season_effect", False)),
        "sigma_u": float(artifact.extras.get("sigma_u", 0.0)),
        "max_design_error": check["max_design_error"],
        "max_prediction_error": check["max_prediction_error"],
        "roundtrip_passes": check["passes"],
        "cmdstan": p["cmdstan"],
        "git_sha": p["git_sha"],
        "built_at": p["built_at"],
        "artifact": str(dest),
    }


def run(cfg: dict, window: str = FIT_WINDOW, groups: tuple[str, ...] = GROUPS,
        draws_kept: int = POSTERIOR_DRAWS) -> dict[str, Path]:
    dest_dir = posteriors_dir(cfg, window)
    dest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = dest_dir / "manifest.csv"

    print(f"Persisting fitted posteriors — fit window `{window}`, {draws_kept:,} draws "
          f"per head,\n  thinned across the WHOLE posterior via `stan_utils.thin`; never "
          f"sliced off the front.")
    if window == "full":
        print("  /!\\  The `full` window fits on the held-out test seasons. Every consumer "
              "must\n       treat these artifacts as production-only — see "
              "`require_window`.")

    builders = {
        "game-length": lambda: game_length_artifacts(cfg, window, draws_kept),
        "availability": lambda: [availability_artifact(cfg, window, draws_kept)],
        "games-played": lambda: games_played_artifacts(cfg, window, draws_kept),
        "minutes": lambda: [minutes_artifact(cfg, window, draws_kept)],
        "components": lambda: component_artifacts(cfg, window, draws_kept),
        "composition": lambda: [composition_artifact(cfg, window, draws_kept)],
    }
    unknown = [g for g in groups if g not in builders]
    if unknown or not groups:
        raise KeyError(f"unknown or empty head group(s) {unknown or list(groups)}; "
                       f"choose from {GROUPS}")

    # Each artifact AND its manifest row are written the moment the head is built, for the
    # reason `stan_composition._checkpoint` exists: the composition head alone is hours of
    # sampler time, and a crash in it must not cost the seventeen cheaper heads — nor the
    # record of what they are. The manifest merges by head, so re-running one group leaves
    # every other row in place.
    rows, paths = [], {}

    def flush() -> pd.DataFrame:
        """Merge this run's rows over whatever is on disk *now*, by head.

        Re-read at every flush rather than snapshotted at the start, so two `--groups`
        runs in parallel — which is the sane way to spend a 14-core machine when the
        composition head is hours and the other seventeen are not — take the union
        instead of the last writer clobbering the first. The `.pkl` files were always
        safe; the manifest was not.
        """
        built = pd.DataFrame(rows)
        merged = built
        if manifest_path.exists():
            existing = pd.read_csv(manifest_path)
            if len(existing):
                merged = pd.concat(
                    [existing[~existing["head"].isin(set(built["head"]))], built],
                    ignore_index=True)
        merged = merged.sort_values("head").reset_index(drop=True)
        merged.to_csv(manifest_path, index=False)
        return merged

    for group in groups:
        print(f"\n── {group} ──")
        for artifact in builders[group]():
            dest = save(artifact, dest_dir)
            # Re-loaded from disk before the check, so the manifest attests to the
            # artifact a consumer will actually open rather than to the one in memory —
            # that is what makes the pickled `StandardScaler` and `SplineTransformer`
            # part of the gate instead of an assumption about them.
            check = load(artifact.head, dest_dir).roundtrip()
            rows.append(manifest_row(artifact, check, dest))
            paths[artifact.head] = dest
            flush()
            p = artifact.provenance
            print(f"Saved {artifact.n_draws:,} posterior draws x "
                  f"{len(artifact.recipe.features)} features → {dest}")
            print(f"    {artifact.head_label} · {artifact.recipe.variant} · "
                  f"{p['n_fit_rows']:,} rows {p['first_season']}–{p['last_season']} · "
                  f"R-hat {p['max_rhat']:.4f} · {p['divergences']} divergences · "
                  f"{p['fit_seconds'] / 60:.1f} min")
            print(f"    round-trip without refitting: design "
                  f"{check['max_design_error']:.2e}, prediction "
                  f"{check['max_prediction_error']:.2e} over "
                  f"{check['n_probe_rows']:,} probe rows → "
                  f"{'PASS' if check['passes'] else 'FAIL'}")

    built = pd.DataFrame(rows)
    frame = flush()
    print(f"\nSaved {len(frame):,} manifest rows → {manifest_path}")

    # Judged on the rows built in THIS run, not on the merged frame: a `roundtrip_passes`
    # column that has been through a CSV comes back as a string on a mixed concat, and
    # `bool("False")` is True — the exact class of silent pass this gate exists to prevent.
    failed = built[~built["roundtrip_passes"]]
    print(f"\n{len(frame)} heads on disk at window(s) "
          f"{', '.join(sorted(set(frame['fit_window'].astype(str))))}; "
          f"{int((~built['converged']).sum())} of {len(built)} built here failed a "
          f"convergence bar; worst round-trip prediction error "
          f"{built['max_prediction_error'].max():.2e}")
    if len(failed):
        raise AssertionError(
            f"{len(failed)} head(s) do not round-trip: {', '.join(failed['head'])}. "
            f"Saved draws plus recipe must reproduce the head's stored predictions "
            f"without refitting.")
    print("Everything downstream of here is numpy — `make posteriors` is the only target "
          "in the\nsimulation layer that needs a CmdStan toolchain.")
    paths["manifest"] = manifest_path
    return paths


if __name__ == "__main__":
    import argparse

    # Imported through the package path rather than used directly, because this module
    # defines classes that get PICKLED: run as `__main__` they would pickle as
    # `__main__.PosteriorArtifact` and no other process could load them (CLAUDE.md).
    from src.models.posteriors import run as _run

    parser = argparse.ArgumentParser(description="Persist the fitted Stan posteriors.")
    # `None` rather than the module constant, so `configs/default.yaml`'s `sim:` block is
    # the default and the flag is the override — not the other way round.
    parser.add_argument("--window", default=None, choices=list(WINDOWS),
                        help="fit window; `full` reads the held-out seasons and is "
                             "guarded")
    parser.add_argument("--groups", default=",".join(GROUPS),
                        help=f"comma-separated subset of {','.join(GROUPS)}")
    parser.add_argument("--draws", type=int, default=None,
                        help="draws kept per head, thinned across the whole posterior")
    args = parser.parse_args()

    cfg = yaml.safe_load(open("configs/default.yaml"))
    cfg_sim = cfg.get("sim", {})
    _run(cfg,
         window=args.window or str(cfg_sim.get("fit_window", FIT_WINDOW)),
         groups=tuple(g.strip() for g in args.groups.split(",") if g.strip()),
         draws_kept=args.draws or int(cfg_sim.get("posterior_draws", POSTERIOR_DRAWS)))
