"""Component rate heads, season-collapsed, benchmarked against a no-fit floor.

Predicts each box-score component's **season total** from prior-season information, using
the collapsed likelihood argued for in `docs/predictions-plan.md`: for a log-link Poisson
with minutes exposure and a linear predictor constant within player-season, the game-level
and season-collapsed likelihoods differ only by a multinomial factor free of β, so fitting
10,194 player-seasons recovers the same coefficients as fitting 731,906 player-games. The
same identity holds for the binomial conversion heads with a hypergeometric factor.

## The benchmark this module exists to enforce

**`carry_forward`: prior per-36 rate × actual minutes / 36. No fitting at all.** It scores
held-out R² of **0.82–0.94** across the eight count heads, and the best fitted model here
beats it by only +0.001 to +0.019. Any proposed component head must be quoted against it —
a head that does not clear it is not a model, it is a worse version of arithmetic. This is
the sharpest available statement of the project's "attempts persist" finding (`fg3a` 0.908
in `persistence.csv`) and it says the rate side is close to saturated from prior-season
information alone, which is why availability carries the larger share of season-total error.

`beats_floor` is on every output row, and `run` prints a loud warning for any head that
fails it, because that is the signature of the bug described next.

## The trap that produced a false finding here

`sklearn`'s `PoissonRegressor` minimizes `deviance / (2·Σw) + alpha·‖coef‖²` — the data term
is **averaged by the weight sum**. Fitting a rate with `sample_weight = minutes` makes
`Σw ≈ 1e7`, so `alpha=1.0` is an enormous penalty that shrinks every coefficient to nearly
zero, *quietly*: the fit converges and a flexible basis partially compensates, so splines
and interactions appear to buy real signal. Measured on `reb`: R² **0.662 at alpha=1.0
against 0.928 at alpha ≤ 0.01**. `LogisticRegression` is the reverse — `Σ w·logloss +
‖coef‖²/(2C)`, not averaged, so the same weights make `C=1.0` weak. Hence `POISSON_ALPHA`
below, and hence the floor being mandatory rather than optional.

## What the specification should be

**Scale, not curvature.** A log link wants a multiplicative predictor: `log E[rate] =
β·log(prior rate)` gives `rate ∝ prior_rate^β`. Linear-in-raw-rate inside `exp()` is
misspecified, catastrophically for the zero-heavy skewed heads (`fg3a` 0.520, `blk` 0.638
against 0.879 / 0.820 on the log scale). Splines add a further +0.030 (`fg3a`) and +0.041
(`blk`) and ≤ +0.003 elsewhere; `age × own` and `mpg × own` interactions are a null once the
scale is right.

Usage:
    python -m src.models.component_rates
"""

from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import minimize_scalar
from scipy.special import gammaln
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.preprocessing import SplineTransformer, StandardScaler

from src.eda.availability import load_ages, with_lags
from src.models.availability import _neg_loglik as beta_binomial_nll
from src.models.availability import fit_dispersion

COUNT_HEADS = ["fg2a", "fg3a", "fta", "reb", "ast", "stl", "blk", "tov"]
CONVERSION_HEADS = [("fg2m", "fg2a"), ("fg3m", "fg3a"), ("ftm", "fta")]
PER36 = 36.0

# See the module docstring: this is *not* a tuning choice, it is the value at which the
# penalty stops dominating a minutes-weighted fit. Anything at or above ~0.01 silently
# destroys the coefficients.
POISSON_ALPHA = 1e-8
LOGISTIC_C = 1e6

MIN_PRIOR_MINUTES = 200      # the project's qualification threshold; reliability ~0.75
TEST_SEASONS = 2
SPLINE_KNOTS = 5
PCA_COMPONENTS = 10

# The prior-season columns the design carries. The own-rate column is per head.
CONTEXT_COLS = ["mpg_lag1", "total_minutes_lag1", "gp_lag1"]
BIO_COLS = ["age", "age_sq", "career_year"]


# ── Design ────────────────────────────────────────────────────────────────────

def season_totals(targets: pd.DataFrame) -> pd.DataFrame:
    """Collapse player-games to player-seasons: totals, per-36 rates, conversion pcts."""
    made = [m for m, _ in CONVERSION_HEADS]
    agg = {c: "sum" for c in COUNT_HEADS + made}
    agg |= {"min": "sum", "played": "sum"}
    out = (targets.groupby(["player_id", "season"], as_index=False).agg(agg)
           .rename(columns={"min": "total_minutes", "played": "gp"}))
    for c in COUNT_HEADS:
        out[f"{c}_p36"] = out[c] / out["total_minutes"].replace(0, np.nan) * PER36
    for m, a in CONVERSION_HEADS:
        out[f"{m}_pct"] = out[m] / out[a].replace(0, np.nan)
    out["mpg"] = out["total_minutes"] / out["gp"].replace(0, np.nan)
    return out


def build_design(targets: pd.DataFrame, seasons: list[str],
                 raw_dir: str | Path) -> pd.DataFrame:
    """One row per (player, target season) with lag-1 prior-season columns.

    Rows need a prior season of at least `MIN_PRIOR_MINUTES`, so every own-rate feature is
    measured over enough minutes to be worth something, and a current season with minutes,
    since minutes are the exposure.
    """
    s = season_totals(targets)
    lag_cols = ([f"{c}_p36" for c in COUNT_HEADS]
                + [f"{m}_pct" for m, _ in CONVERSION_HEADS]
                + [f"{a}_p36" for _, a in CONVERSION_HEADS]
                + [m for m, _ in CONVERSION_HEADS]
                + [a for _, a in CONVERSION_HEADS]
                + ["mpg", "total_minutes", "gp"])
    lag_cols = list(dict.fromkeys(lag_cols))
    d = with_lags(s, seasons, lag_cols, max_lag=1)

    ages = load_ages(seasons, raw_dir)
    d = (d.merge(ages, on=["season", "player_id"], how="left") if not ages.empty
         else d.assign(age=np.nan))
    d["age_sq"] = d["age"] ** 2
    d["career_year"] = d.sort_values("season_index").groupby("player_id").cumcount()

    d = d.dropna(subset=["age", "mpg_lag1", "total_minutes_lag1"])
    d = d[(d["total_minutes_lag1"] >= MIN_PRIOR_MINUTES) & (d["total_minutes"] > 0)]
    return d.reset_index(drop=True)


def split_seasons(design: pd.DataFrame, test_seasons: int = TEST_SEASONS
                  ) -> tuple[pd.DataFrame, pd.DataFrame]:
    order = sorted(design["season"].unique())
    held = set(order[-test_seasons:])
    return design[~design["season"].isin(held)], design[design["season"].isin(held)]


# ── The benchmark ─────────────────────────────────────────────────────────────

def carry_forward(frame: pd.DataFrame, component: str) -> np.ndarray:
    """The no-fit floor: prior per-36 rate × actual minutes / 36.

    No parameters, no fitting, no training data. Everything else in this module has to
    beat it, and most of the achievable skill is already here.
    """
    rate = frame[f"{component}_p36_lag1"].to_numpy(dtype=float)
    return np.clip(rate * frame["total_minutes"].to_numpy(dtype=float) / PER36, 0.0, None)


def carry_forward_conversion(train: pd.DataFrame, frame: pd.DataFrame, made: str,
                             attempted: str) -> np.ndarray:
    """The conversion floor: prior percentage **shrunk** toward the training league mean.

    The raw carry-forward is not a usable floor here and that is a fact about proportions,
    not a coding choice. A player who went 0-for-3 from three has a prior 3P% of exactly
    0.000; carrying it forward onto 200 attempts gives a beta-binomial NLL of ~1.3e9 and
    makes the benchmark meaningless. `CLAUDE.md` already settles the fix — conversion
    percentages must be **shrunk hard toward player/league means**, unlike the share and
    count columns. So the floor is empirical-Bayes:

        p_hat = (made_{S-1} + k·league_mean) / (attempts_{S-1} + k)

    with `k` pseudo-attempts and `league_mean` both estimated on **train** only. Still no
    regression and no features — one shrinkage constant — which is the honest proportion
    analogue of "carry the rate forward".
    """
    lag_made, lag_att = f"{made}_lag1", f"{attempted}_lag1"
    m_train = train[lag_made].to_numpy(dtype=float)
    a_train = train[lag_att].to_numpy(dtype=float)
    ok = np.isfinite(m_train) & np.isfinite(a_train) & (a_train > 0)
    league = float(m_train[ok].sum() / a_train[ok].sum())

    def nll_for(k: float) -> float:
        p = (m_train[ok] + k * league) / (a_train[ok] + k)
        y = train[made].to_numpy(dtype=float)[ok]
        n = train[attempted].to_numpy(dtype=float)[ok]
        live = n > 0
        rho = fit_dispersion(y[live].astype(int), n[live].astype(int), p[live])
        return float(beta_binomial_nll(y[live].astype(int), n[live].astype(int),
                                       p[live], rho))

    best = minimize_scalar(nll_for, bounds=(1.0, 2000.0), method="bounded")
    k = float(best.x)
    m = frame[lag_made].to_numpy(dtype=float)
    a = frame[lag_att].to_numpy(dtype=float)
    p = np.where(np.isfinite(m) & np.isfinite(a),
                 (np.nan_to_num(m) + k * league) / (np.nan_to_num(a) + k), league)
    return np.clip(p, 1e-3, 1 - 1e-3)


# ── Feature construction ──────────────────────────────────────────────────────

def impute(train: pd.DataFrame, test: pd.DataFrame,
           cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Fill NaN from **train** means and flag it.

    A player with no prior 3PA has a genuinely undefined prior 3P%; the indicator keeps
    that distinguishable from "league-average shooter", and splines cannot take NaN.
    """
    tr, te, flags = train.copy(), test.copy(), []
    for col in cols:
        if not train[col].isna().any() and not test[col].isna().any():
            continue
        mean = float(train[col].mean())
        flag = f"{col}__miss"
        tr[flag] = train[col].isna().astype(float)
        te[flag] = test[col].isna().astype(float)
        tr[col], te[col] = train[col].fillna(mean), test[col].fillna(mean)
        flags.append(flag)
    return tr, te, flags


def add_log(train: pd.DataFrame, test: pd.DataFrame,
            cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """`log1p` of a non-negative column — the multiplicative scale a log link wants."""
    tr, te, names = train.copy(), test.copy(), []
    for col in cols:
        name = f"log_{col}"
        for frame, src in ((tr, train), (te, test)):
            frame[name] = np.log1p(np.clip(src[col].to_numpy(dtype=float), 0.0, None))
        names.append(name)
    return tr, te, names


def add_spline(train: pd.DataFrame, test: pd.DataFrame, cols: list[str],
               n_knots: int = SPLINE_KNOTS
               ) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Cubic B-spline basis with linear extrapolation, knots from **train** quantiles."""
    tr, te, names = train.copy(), test.copy(), []
    for col in cols:
        st = SplineTransformer(n_knots=n_knots, degree=3, extrapolation="linear",
                               include_bias=False)
        basis_tr = st.fit_transform(train[[col]].to_numpy(dtype=float))
        basis_te = st.transform(test[[col]].to_numpy(dtype=float))
        for j in range(basis_tr.shape[1]):
            name = f"{col}__s{j}"
            tr[name], te[name] = basis_tr[:, j], basis_te[:, j]
            names.append(name)
    return tr, te, names


def add_interactions(train: pd.DataFrame, test: pd.DataFrame,
                     pairs: list[tuple[str, str]]
                     ) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    tr, te, names = train.copy(), test.copy(), []
    for a, b in pairs:
        name = f"{a}__x__{b}"
        tr[name] = tr[a].to_numpy(dtype=float) * tr[b].to_numpy(dtype=float)
        te[name] = te[a].to_numpy(dtype=float) * te[b].to_numpy(dtype=float)
        names.append(name)
    return tr, te, names


# ── Walk-forward PCA ──────────────────────────────────────────────────────────

def matrix_feature_cols(matrix: pd.DataFrame) -> list[str]:
    """Numeric season-matrix feature columns, excluding keys and volume."""
    drop = {"player_id", "season", "season_start_year", "team_id", "gp", "min",
            "reliability", "stats_source", "archetype", "draft_bucket",
            "team_abbreviation", "player_name", "roster_coverage"}
    return [c for c in matrix.columns
            if c not in drop and pd.api.types.is_numeric_dtype(matrix[c])]


def walk_forward_pca(design: pd.DataFrame, matrix: pd.DataFrame,
                     n_components: int = PCA_COMPONENTS
                     ) -> tuple[pd.DataFrame, list[str]]:
    """PC scores of each row's **prior-season** stat vector, fitted point-in-time.

    For a target season S the features describe season S-1, and everything through S-1 is
    observable at the prediction date — so the scaler and the PCA are refitted for every
    target season on matrix rows from **S-1 and earlier only**. Fitting one PCA on all 30
    seasons would leak the future into the basis, which is the same class of error as
    filling a historical row from a current-status source, and it is invisible in the
    output: the scores look completely normal and simply score too well.

    This is why the repo's cached `pca_features_tier*.pkl` artifacts cannot be used here —
    they are fitted pooled or within-season across the whole sample.
    """
    feats = matrix_feature_cols(matrix)
    m = matrix[["player_id", "season_start_year"] + feats].copy()
    m[feats] = m[feats].apply(pd.to_numeric, errors="coerce")
    m = m.dropna(subset=["season_start_year"])
    m[feats] = m[feats].fillna(m[feats].median())

    design = design.copy()
    design["prior_start_year"] = design["season_start_year"] - 1
    pc_names = [f"pc{i + 1}" for i in range(n_components)]
    out = []
    for target_year, block in design.groupby("season_start_year", sort=True):
        prior_year = int(target_year) - 1
        history = m[m["season_start_year"] <= prior_year]
        rows = block.merge(
            m[m["season_start_year"] == prior_year],
            left_on=["player_id", "prior_start_year"],
            right_on=["player_id", "season_start_year"],
            how="left", suffixes=("", "_m"))
        if len(history) <= n_components or rows[feats].isna().all(axis=None):
            continue
        scaler = StandardScaler().fit(history[feats].to_numpy(dtype=float))
        pca = PCA(n_components=n_components, random_state=0).fit(
            scaler.transform(history[feats].to_numpy(dtype=float)))
        X = rows[feats].to_numpy(dtype=float)
        keep = ~np.isnan(X).any(axis=1)
        scores = np.full((len(rows), n_components), np.nan)
        if keep.any():
            scores[keep] = pca.transform(scaler.transform(X[keep]))
        block = block.copy()
        for j, name in enumerate(pc_names):
            block[name] = scores[:, j]
        out.append(block)
    if not out:
        raise ValueError("walk-forward PCA produced no rows; check season_start_year")
    return pd.concat(out, ignore_index=True), pc_names


# ── Heads ─────────────────────────────────────────────────────────────────────

def _design_matrix(frame: pd.DataFrame, features: list[str]) -> np.ndarray:
    X = frame[features].to_numpy(dtype=float)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


def nb_nll(y: np.ndarray, mu: np.ndarray, phi: float) -> np.ndarray:
    """Negative binomial (NB2) negative log-likelihood, `var = mu + mu^2/phi`."""
    mu = np.clip(mu, 1e-9, None)
    return -(gammaln(y + phi) - gammaln(phi) - gammaln(y + 1)
             + phi * np.log(phi / (phi + mu))
             + y * np.log(np.where(y > 0, mu / (phi + mu), 1.0)))


def fit_nb_dispersion(y: np.ndarray, mu: np.ndarray) -> float:
    r = minimize_scalar(lambda lp: nb_nll(y, mu, float(np.exp(lp))).sum(),
                        bounds=(-3.0, 8.0), method="bounded")
    return float(np.exp(r.x))


def fit_count_head(train: pd.DataFrame, test: pd.DataFrame, features: list[str],
                   component: str, alpha: float = POISSON_ALPHA
                   ) -> tuple[np.ndarray, float]:
    """Poisson rate GLM with minutes exposure; returns test means and NB dispersion.

    Exposure is handled by the standard identity — fitting `y/M` with `sample_weight = M`
    is exactly a Poisson with offset `log M`.
    """
    m_train = train["total_minutes"].to_numpy(dtype=float)
    m_test = test["total_minutes"].to_numpy(dtype=float)
    y_train = train[component].to_numpy(dtype=float)
    scaler = StandardScaler().fit(_design_matrix(train, features))
    model = PoissonRegressor(alpha=alpha, max_iter=5000).fit(
        scaler.transform(_design_matrix(train, features)), y_train / m_train,
        sample_weight=m_train)
    mu_train = np.clip(
        model.predict(scaler.transform(_design_matrix(train, features))) * m_train,
        1e-6, None)
    mu_test = np.clip(
        model.predict(scaler.transform(_design_matrix(test, features))) * m_test,
        1e-6, None)
    return mu_test, fit_nb_dispersion(y_train, mu_train)


def fit_conversion_head(train: pd.DataFrame, test: pd.DataFrame, features: list[str],
                        made: str, attempted: str, C: float = LOGISTIC_C
                        ) -> tuple[np.ndarray, np.ndarray, float]:
    """Grouped-binomial logistic GLM; returns the test mask, probabilities and rho.

    A grouped binomial is exactly a weighted Bernoulli on two rows per group — one with
    weight `made`, one with weight `attempted - made` — which is what lets `sklearn`'s
    solver fit it without expanding to one row per shot.
    """
    n_train = train[attempted].to_numpy(dtype=float)
    y_train = train[made].to_numpy(dtype=float)
    ok_train = n_train > 0
    ok_test = test[attempted].to_numpy(dtype=float) > 0

    scaler = StandardScaler().fit(_design_matrix(train, features)[ok_train])
    X = scaler.transform(_design_matrix(train, features)[ok_train])
    X2 = np.vstack([X, X])
    labels = np.r_[np.ones(int(ok_train.sum())), np.zeros(int(ok_train.sum()))]
    weights = np.r_[y_train[ok_train], (n_train - y_train)[ok_train]]
    keep = weights > 0
    model = LogisticRegression(C=C, max_iter=5000).fit(X2[keep], labels[keep],
                                                       sample_weight=weights[keep])
    p_train = model.predict_proba(X)[:, 1]
    p_test = model.predict_proba(
        scaler.transform(_design_matrix(test, features)[ok_test]))[:, 1]
    rho = fit_dispersion(y_train[ok_train].astype(int), n_train[ok_train].astype(int),
                         p_train)
    return ok_test, p_test, rho


# ── Evaluation ────────────────────────────────────────────────────────────────

def _r2(y: np.ndarray, pred: np.ndarray) -> float:
    return float(1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def count_variants(train: pd.DataFrame, test: pd.DataFrame, component: str,
                   pc_names: list[str] | None = None
                   ) -> dict[str, tuple[pd.DataFrame, pd.DataFrame, list[str]]]:
    """Feature sets to compare for one count head, raw and PCA."""
    own = f"{component}_p36_lag1"
    base = [own] + CONTEXT_COLS + BIO_COLS
    tr, te, flags = impute(train, test, base)
    base = base + flags
    out = {"linear": (tr, te, list(base))}

    tl, el, log_names = add_log(tr, te, [own])
    log_base = [c for c in base if c != own] + log_names
    out["log_own"] = (tl, el, list(log_base))

    ts, es, spline_names = add_spline(tr, te, [own])
    out["log_own_spline"] = (ts, es, [c for c in base if c != own] + spline_names)

    ti, ei, inter_names = add_interactions(tl, el, [("age", log_names[0]),
                                                   ("mpg_lag1", log_names[0])])
    out["log_own_inter"] = (ti, ei, log_base + inter_names)

    if pc_names:
        # PCA replaces every prior-season stat EXCEPT the head's own response variable.
        pca_base = [c for c in log_base if c not in CONTEXT_COLS] + pc_names
        tp, ep, _ = impute(tl, el, pc_names)
        out["pca"] = (tp, ep, list(pca_base))
        tps, eps, spl = add_spline(tp, ep, [own]) if own in tp else (tp, ep, [])
        out["pca_spline"] = (tps, eps,
                             [c for c in pca_base if c not in log_names] + spl)
        tpi, epi, pinter = add_interactions(
            tp, ep, [(log_names[0], pc_names[0]), (log_names[0], pc_names[1])])
        out["pca_inter"] = (tpi, epi, pca_base + pinter)
    return out


def conversion_variants(train: pd.DataFrame, test: pd.DataFrame, made: str,
                        attempted: str, pc_names: list[str] | None = None
                        ) -> dict[str, tuple[pd.DataFrame, pd.DataFrame, list[str]]]:
    own = f"{made}_pct_lag1"
    vol = f"{attempted}_p36_lag1"
    base = [own, vol] + CONTEXT_COLS + BIO_COLS
    tr, te, flags = impute(train, test, base)
    base = base + flags
    out = {"linear": (tr, te, list(base))}
    ts, es, spline_names = add_spline(tr, te, [own])
    out["spline_own"] = (ts, es, [c for c in base if c != own] + spline_names)
    ti, ei, inter = add_interactions(tr, te, [("mpg_lag1", own), (vol, own)])
    out["inter"] = (ti, ei, base + inter)
    if pc_names:
        tp, ep, _ = impute(tr, te, pc_names)
        pca_base = [c for c in base if c not in CONTEXT_COLS + [vol]] + pc_names
        out["pca"] = (tp, ep, list(pca_base))
        tpi, epi, pinter = add_interactions(tp, ep, [(own, pc_names[0])])
        out["pca_inter"] = (tpi, epi, pca_base + pinter)
    return out


def evaluate(train: pd.DataFrame, test: pd.DataFrame,
             pc_names: list[str] | None = None) -> pd.DataFrame:
    """Every head × variant, with the no-fit floor always present as a row."""
    rows = []
    for component in COUNT_HEADS:
        y = test[component].to_numpy(dtype=float)
        floor = carry_forward(test, component)
        floor_r2 = _r2(y, floor)
        floor_phi = fit_nb_dispersion(train[component].to_numpy(dtype=float),
                                      np.clip(carry_forward(train, component), 1e-6, None))
        rows.append({"analysis": "variant_sweep", "head": component,
                     "kind": "count", "variant": "carry_forward",
                     "n_features": 0, "r2": floor_r2,
                     "mae": float(np.abs(y - floor).mean()),
                     "nll": float(nb_nll(y, np.clip(floor, 1e-6, None), floor_phi).mean()),
                     "dispersion": floor_phi, "beats_floor": True})
        for label, (tr, te, features) in count_variants(train, test, component,
                                                        pc_names).items():
            mu, phi = fit_count_head(tr, te, features, component)
            r2 = _r2(y, mu)
            rows.append({"analysis": "variant_sweep", "head": component,
                         "kind": "count", "variant": label,
                         "n_features": len(features), "r2": r2,
                         "mae": float(np.abs(y - mu).mean()),
                         "nll": float(nb_nll(y, mu, phi).mean()),
                         "dispersion": phi, "beats_floor": bool(r2 > floor_r2)})

    for made, attempted in CONVERSION_HEADS:
        head = f"{made}|{attempted}"
        n_test = test[attempted].to_numpy(dtype=float)
        y_test = test[made].to_numpy(dtype=float)
        ok = n_test > 0
        p_floor = carry_forward_conversion(train, test, made, attempted)[ok]
        live = train[attempted].to_numpy(dtype=float) > 0
        rho_floor = fit_dispersion(
            train[made].to_numpy(dtype=float)[live].astype(int),
            train[attempted].to_numpy(dtype=float)[live].astype(int),
            carry_forward_conversion(train, train, made, attempted)[live])
        floor_nll = float(beta_binomial_nll(y_test[ok].astype(int), n_test[ok].astype(int),
                                           p_floor, rho_floor) / int(ok.sum()))
        realised = y_test[ok] / n_test[ok]
        rows.append({"analysis": "variant_sweep", "head": head,
                     "kind": "conversion", "variant": "carry_forward",
                     "n_features": 0, "r2": _r2(realised, p_floor),
                     "mae": float(np.abs(realised - p_floor).mean()),
                     "nll": floor_nll, "dispersion": rho_floor, "beats_floor": True})
        for label, (tr, te, features) in conversion_variants(train, test, made, attempted,
                                                             pc_names).items():
            mask, p, rho = fit_conversion_head(tr, te, features, made, attempted)
            nll = float(beta_binomial_nll(y_test[mask].astype(int),
                                          n_test[mask].astype(int), p, rho)
                        / int(mask.sum()))
            rows.append({"analysis": "variant_sweep", "head": head,
                         "kind": "conversion", "variant": label,
                         "n_features": len(features),
                         "r2": _r2(y_test[mask] / n_test[mask], p),
                         "mae": float(np.abs(y_test[mask] / n_test[mask] - p).mean()),
                         "nll": nll, "dispersion": rho, "beats_floor": bool(nll < floor_nll)})
    return pd.DataFrame(rows)


# ── Regularization sensitivity ────────────────────────────────────────────────
#
# The `sklearn` alpha trap cost a full set of published figures, and it failed *quietly*:
# `PoissonRegressor` averages the deviance by the weight sum, so with `sample_weight =
# minutes` (Sum w ~ 1e7) `alpha=1.0` crushes every coefficient while the fit still converges
# and a flexible basis partially compensates. `reb` read R² 0.662 instead of 0.928, and the
# spline and interaction variants looked like they were buying real signal.
#
# So the trap gets a curve rather than an anecdote, and the curve is a regression guard: a
# future refactor that reintroduces a default-looking alpha shows up as the high-alpha end of
# the fitted line dropping below the no-fit floor. That crossing is the visual statement of
# why the floor is mandatory — arithmetic beats an over-penalized GLM.

# Spans the correct and the broken regimes. `POISSON_ALPHA` sits at the bottom; 1.0 is the
# value that looks like a default and is not one.
ALPHA_GRID = [1e-8, 1e-6, 1e-4, 1e-2, 1e-1, 1.0, 10.0]

# Swept on both the misspecified raw-rate spec (which is what the recorded 0.662 was measured
# on) and the shipped log spec, because "does over-regularization hurt the spec we ship" is
# the question the guard actually needs to answer.
ALPHA_VARIANTS = ("linear", "log_own")


def alpha_feature_sets(train: pd.DataFrame, test: pd.DataFrame, component: str
                       ) -> dict[str, tuple[pd.DataFrame, pd.DataFrame, list[str]]]:
    """The two feature sets the alpha sweep runs on — no splines, no PCA.

    A deliberately trimmed `count_variants`: the sweep is about the penalty, so building a
    spline basis and a walk-forward PCA for every alpha would cost time to measure nothing.
    """
    own = f"{component}_p36_lag1"
    base = [own] + CONTEXT_COLS + BIO_COLS
    tr, te, flags = impute(train, test, base)
    base = base + flags
    tl, el, log_names = add_log(tr, te, [own])
    return {"linear": (tr, te, list(base)),
            "log_own": (tl, el, [c for c in base if c != own] + log_names)}


def alpha_sensitivity(train: pd.DataFrame, test: pd.DataFrame,
                      alphas: list[float] = None, heads: list[str] = None,
                      variants: tuple[str, ...] = ALPHA_VARIANTS) -> pd.DataFrame:
    """Held-out R² per count head across an alpha grid, with the no-fit floor beside it.

    `floor_r2` rides on every row so the crossing is readable without a join, and
    `beats_floor` keeps the same meaning it has everywhere else in this module.
    """
    alphas = list(ALPHA_GRID if alphas is None else alphas)
    heads = list(COUNT_HEADS if heads is None else heads)
    rows = []
    for component in heads:
        y = test[component].to_numpy(dtype=float)
        floor_r2 = _r2(y, carry_forward(test, component))
        sets = alpha_feature_sets(train, test, component)
        for variant in variants:
            if variant not in sets:
                continue
            tr, te, features = sets[variant]
            for alpha in alphas:
                mu, phi = fit_count_head(tr, te, features, component, alpha=alpha)
                r2 = _r2(y, mu)
                rows.append({"analysis": "alpha_sensitivity", "head": component,
                             "kind": "count", "variant": variant, "alpha": alpha,
                             "n_features": len(features), "r2": r2,
                             "mae": float(np.abs(y - mu).mean()),
                             "nll": float(nb_nll(y, mu, phi).mean()),
                             "dispersion": phi, "floor_r2": floor_r2,
                             "beats_floor": bool(r2 > floor_r2)})
    return pd.DataFrame(rows)


# ── Entry point ───────────────────────────────────────────────────────────────

def run(cfg: dict) -> dict[str, Path]:
    features_dir = Path(cfg["data"]["features_dir"])
    raw_dir = cfg["data"]["raw_dir"]
    seasons = cfg["data"]["seasons"]

    targets = pd.read_parquet(features_dir / "component_targets.parquet")
    design = build_design(targets, seasons, raw_dir)
    design["season_start_year"] = design["season"].str.slice(0, 4).astype(int)
    print(f"Component rates: {len(design):,} player-seasons "
          f"(prior season >= {MIN_PRIOR_MINUTES} min), "
          f"{design['season'].nunique()} target seasons")

    matrix = pd.read_parquet(features_dir / "season_matrix_roster_tierA.parquet")
    with_pcs, pc_names = walk_forward_pca(design, matrix)
    print(f"  walk-forward PCA: {len(pc_names)} components, refitted per target season on "
          f"S-1 and earlier only")
    print(f"  {with_pcs[pc_names[0]].notna().mean():.1%} of rows have PC scores "
          f"({len(matrix_feature_cols(matrix))} season-matrix columns in)")

    train, test = split_seasons(with_pcs)
    print(f"  {len(train):,} train / {len(test):,} test "
          f"({', '.join(sorted(test['season'].unique()))} held out)\n")

    table = evaluate(train, test, pc_names)
    alpha_cfg = cfg.get("evaluation", {}).get("alpha_grid", ALPHA_GRID)
    alphas = alpha_sensitivity(train, test, alpha_cfg)
    out_dir = Path(cfg["evaluation"]["predictions_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / "component_rate_metrics.csv"
    pd.concat([table, alphas], ignore_index=True).to_csv(dest, index=False)

    for kind, metric in (("count", "r2"), ("conversion", "nll")):
        sub = table[table["kind"] == kind]
        piv = sub.pivot_table(index="head", columns="variant", values=metric)
        cols = [c for c in ["carry_forward", "linear", "log_own", "log_own_spline",
                            "log_own_inter", "spline_own", "inter", "pca", "pca_spline",
                            "pca_inter"] if c in piv.columns]
        print(f"Held-out {metric} by {kind} head "
              f"({'higher' if metric == 'r2' else 'lower'} is better):")
        print(piv[cols].round(4).to_string())
        gain = piv.drop(columns=["carry_forward"]).sub(piv["carry_forward"], axis=0)
        best = (gain.max(axis=1) if metric == "r2" else -gain.min(axis=1))
        print(f"  best variant vs the no-fit floor: "
              f"{', '.join(f'{h} {v:+.4f}' for h, v in best.items())}\n")

    # A single variant losing to the floor is often a real finding — `linear` loses on
    # six of eight count heads because linear-in-raw-rate inside exp() is misspecified.
    # What would signal the regularization trap is the *best* variant for a head losing.
    fitted = table[table["variant"] != "carry_forward"]
    beaten = sorted(set(table["head"]) - set(fitted.loc[fitted["beats_floor"], "head"]))
    if beaten:
        print(f"⚠️  NO variant beats the no-fit floor for: {', '.join(beaten)}")
        print("   Two possible causes, and they need different responses. Either the floor "
              "is genuinely\n   unbeatable for that head — `ftm|fta` is the known case, "
              "since free-throw percentage is\n   pure player skill with no context to "
              "add — or the fit is over-regularized. Check alpha\n   against the "
              "module docstring before concluding it is a null.")
    else:
        print("Every head has at least one variant beating the no-fit floor.")
    lost = fitted[~fitted["beats_floor"]]
    if len(lost):
        print(f"\n{len(lost)} individual fits lose to the floor (expected for `linear`, "
              "which is misspecified on the raw rate scale):")
        print(lost.pivot_table(index="head", columns="variant", values="r2",
                              aggfunc="first").round(4).to_string())

    # ── the alpha curve, as a permanent regression guard ─────────────────────
    print("\nRegularization sensitivity — held-out R² by alpha under exposure weights.\n"
          "The floor is arithmetic with no parameters, so a fitted line dropping BELOW it "
          "is the\nsignature of the penalty dominating the data term:")
    for variant in ALPHA_VARIANTS:
        sub = alphas[alphas["variant"] == variant]
        if sub.empty:
            continue
        piv = sub.pivot_table(index="head", columns="alpha", values="r2")
        piv.insert(0, "floor", sub.groupby("head")["floor_r2"].first())
        print(f"\n  variant = {variant}")
        print(piv.round(4).to_string())
    top = max(alpha_cfg)
    at_top = alphas[alphas["alpha"] == top]
    print(f"\n  at alpha={top:g}, {int((~at_top['beats_floor']).sum())} of "
          f"{len(at_top)} fits fall below the no-fit floor.")

    # The guard is each head against its OWN optimum on the grid, not against the floor:
    # `blk` and `fg3a` lose to the floor at every alpha because they need splines, which is
    # a documented modelling finding rather than the penalty misbehaving.
    best = alphas.loc[alphas.groupby(["variant", "head"])["r2"].idxmax()]
    worst_best_alpha = float(best["alpha"].max())
    print(f"  every head's best alpha on the grid is <= {worst_best_alpha:g}, and the loss "
          "from the shipped\n  "
          f"alpha={POISSON_ALPHA:g} to alpha=1.0 is:")
    for variant in ALPHA_VARIANTS:
        sub = alphas[alphas["variant"] == variant]
        shipped = sub[sub["alpha"] == min(alpha_cfg)].set_index("head")["r2"]
        broken = sub[sub["alpha"] == 1.0].set_index("head")["r2"]
        if shipped.empty or broken.empty:
            continue
        loss = (shipped - broken).sort_values(ascending=False)
        print(f"    {variant:9s} median {loss.median():.3f} R², worst "
              f"{loss.index[0]} {loss.iloc[0]:.3f}")
    reb = alphas[(alphas["head"] == "reb") & (alphas["variant"] == "linear")]
    lo = reb[reb["alpha"] <= 0.01]
    one = reb[reb["alpha"] == 1.0]
    if len(lo) and len(one):
        print(f"  `reb` linear reproduces the recorded pair: R² "
              f"{lo['r2'].max():.4f} at alpha <= 0.01 (recorded 0.928) against "
              f"{float(one['r2'].iloc[0]):.4f} at alpha=1.0 (recorded 0.662)\n  — the trap "
              "as a curve rather than an anecdote.")

    print(f"\nSaved {len(table) + len(alphas):,} metric rows "
          f"({len(alphas):,} of them the alpha sweep) → {dest}")
    return {"metrics": dest}


if __name__ == "__main__":
    cfg = yaml.safe_load(open("configs/default.yaml"))
    run(cfg)
