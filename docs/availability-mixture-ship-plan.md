# Availability mixture ship plan

**Ephemeral scaffolding**, written 2026-08-11. Three sessions with one prepared prompt each,
to take `docs/availability-window-plan.md` §7's `mixture` arm from a point-MLE ladder result
to a shipped Stan head. **Delete this file when the round lands**, as
`dashboard-build-prompts.md` was.

`availability-window-plan.md` is the measurement and stays; this is only the work plan. Read
its §7 (what was measured), §7f (the metric set) and §8 (the decisions) before starting.

---

## 1. The decisions, settled 2026-08-11

All five of §8's open questions are now closed. Four were the user's call and are recorded as
taken; the fifth is a default I chose and flagged as such.

### D1. The head is selected on **calibration, with a CRPS guard**

An arm ships if it improves the tail calibration metrics **and** its CRPS is *non-inferior* —
the paired-bootstrap interval must exclude a material loss. `mixture` passes: CRPS **+0.011**
with an interval of [−0.028, +0.051] spanning zero, against `boundary_tail_error` −0.0089
[−0.0099, −0.0042] and `shoulder_error` −0.0021 [−0.0086, −0.0010] on validation, and
−0.0128 [−0.0138, −0.0064] on the rolling harness.

**This is a change of rule and it is deliberate.** Every prior decision on this head was taken
on mean CRPS. The reason to move is in `README.md` §4 — *"A model that improves marginal CRPS
by 1% and gets the correlation structure wrong is worth less here than one that does the
reverse"* — and in §1 of the window plan: a dead roster slot and an iron man are the two events
a Round-1 knockout turns on, and both errors make a drafted roster look more reliable than it
is. The guard is what stops a future arm trading real accuracy for tails.

**The rule is stated before the next arm is measured**, which is the point of writing it down
here rather than inferring it from the arm that happened to win.

### D2. The Stan port is **not** gated on a contest-level readout

Port first; measure the contest value afterwards with `make strategy-sweep`, as a value
measurement rather than a gate. Nobody has shown that moving `P(missed ≤ 5)` from 16.9%
toward the observed 12.1% changes a draft, and that remains an open question — it is just not
a blocking one.

### D3. The mixture lives in **`betabinomial_glm.stan`**, with `π = 0` nesting asserted

Not a fork. The file serves six heads (availability, minutes, four conversions, overtime
onset), and the discipline that makes that safe already exists: `n_rho = 1` reproduces the
shared-dispersion target **bit for bit**, asserted on Stan's own `log_prob`. The mixture block
gets the same treatment — `π = 0` must reproduce the current target exactly, and that
assertion is the rollback path.

### D4. Session order: **simulator fix + `l2` sweep first**, then the port

Both cheap items in one session. The simulator break is live and independent of the mixture,
and removing the `l2` confound before porting means the port is not carrying an unresolved
question. The port then has one job.

### D5. `π`'s covariate block — **my default, not the user's decision**

`PI_COLS` in `src/models/availability_window.py`: `age`, `age_sq`, `gp_share_lag1`,
`trailing_missed_lag1`, `n_spells_lag1`, `longest_spell_lag1`, `playoff_games_lag1`,
`career_minutes_lag1`. Eight columns, as one reading of `docs/potential-to-dos.md` item 5's
"age, prior absence, playoff workload". It is a shipped choice that lands in the persisted
`DesignRecipe`, and the fitted `π` runs from **1.2%** to **10.8%** across the 10th and 90th
percentiles of players, so the block carries real signal. **Override it in session 2 if you
want a different one** — it is the least-examined of the five.

---

## 2. Session 1 — the simulator's draw path, and the `l2` confound

> ✅ **Done 2026-08-11. Both parts landed; neither changed a decision.**
>
> **Part A.** `make simulate-season` was run first and reproduced the failure *in the target*
> — `ValueError: could not broadcast input array from shape (4,) into shape (539,)` at sim 0
> of 2,000 — so the break was a broken target and not only a broken line. The fix is
> `season.availability_rho_bin` + `season.availability_rates`: the dispersion axis is
> reconstructed from the artifact's own `cut` step, both artifact shapes are handled,
> `rho_bin`'s 1-based convention and the no-design player's lowest-bucket rule are each
> pinned by a test, and a shared-`rho` artifact reproduces the old scalar draw **bit for
> bit** (also a test). Six tests added. All four seasons were re-simulated at 2,000 sims, so
> the Gate A artifact holds no rows from the pre-window posterior; every row improved and the
> two realized-minutes bonus rows reproduced to four decimals as the control. Details in
> `availability-window-plan.md` §8, figures in `simulations-plan.md`.
>
> **Part B is a null**, which is the useful kind here: eight penalties from 0 to 256 for
> every arm at the shipped window move the reference by **0.00034** CRPS at its best, so
> **99.4%** of `beta_rect`'s margin survives and every ordering in §7c stands. The mechanism
> is that at 4,027 rows the pinned penalty is 1.5 parts in 100,000 of the objective. §7c
> result 3, §7d, §7g and §8 decision 5 are updated. **D1 is unaffected** — `mixture` ties
> CRPS at every penalty on the grid — so session 2 starts with nothing outstanding.
>
> One thing session 3 inherits: the tensors were rebuilt, so `make bracket` / `make draft` /
> `make strategy-sweep` are now scored against a superseded tensor. §4 re-runs them anyway.

### What is actually wrong with the simulator, corrected

`src/sim/season.py:645-648` draws availability by **re-implementing** the head:

```python
a, b = beta_shapes(ctx["avail_mu"][draw],
                   np.full(ctx["n_players"], ctx["avail_rho"][draw]))
p_available = rng.beta(a, b)
gp = rng.binomial(ctx["cell_games"], p_available[ctx["cell_player"]])
```

`FIT_WINDOW = "train"` (`season.py:176`) and the persisted `train` posterior has carried
`rho_draws` of shape **(1000, 4)** with `n_rho: 4` since the role-graded head shipped.
`np.full(n_players, <4-vector>)` raises `ValueError: could not broadcast input array from
shape (4,) into shape (500,)` — verified against the loaded artifact. The `train_val`
artifact is still `(1000,)` with `role_rho: None`, so it predates the window round: **one
window raises and the other is silently stale.**

> ⚠️ **An earlier recommendation in this repo said the fix is to route through the head's
> `predict_samples`. That is wrong, and the reason matters.** `predict_samples` returns
> **games played** for the design rows, one row per player-season. The simulator needs the
> **rate**, because it applies that rate to each of a player's *cells* — a player traded
> mid-season has more than one — and then hands the count to `allocate_spells`. A head-level
> predictive cannot be split across cells. The whole-board coupling is not the problem either:
> `predict_samples` preserves it per output row.

**So the fix is to gather each player's dispersion bin**, mirroring what `season.py:819`
already does for the composition head:

```python
row_rho_bin = comp_art.recipe.transform(players)["rho_bin"].to_numpy(np.int64)[row_player]
```

The availability artifact carries the same recipe step — a `cut` on `minutes_per_game_lag1`
at `[0, 12, 24, 30, 60]` named `rho_bin` — so `avail_art.recipe.transform(avail)["rho_bin"]`
gives the head's own bucket per player.

### Three traps in that one line

1. **`rho_bin` is 1-based.** `recipe.transform` and `stan_availability.role_bins` both return
   `[1, 2, 3, 4]`, verified. Indexing `rho_draws[draw]` needs `bin - 1`. An off-by-one gives a
   player the *wrong bucket's* dispersion and raises nothing.
2. **Both artifact shapes must work.** `rho_draws` is `(draws,)` on a shared-`rho` artifact
   and `(draws, K)` on a graded one, and `train_val` is currently the former. Handle both, or
   the fix trades one broken window for the other.
3. **Players with no design row.** `~present` rows already get `no_design_availability`'s
   empirical rate for their mean; they need a bin too. The head's own rule is the **lowest**
   bucket — the widest dispersion, the conservative direction — which `role_bins` documents.

### Then the `l2` sweep

`docs/availability-window-plan.md` §7d: `l2 = 1.0` is pinned across arms, the penalty reaches
`beta[1:]` only, and `mixture` carries eleven unpenalized parameters against the reference's
zero. Sweep `l2` for the **reference arm** at the shipped window and re-read how much of
`beta_rect`'s −0.056 CRPS margin survives a reference regularized as favourably as the
alternatives are. Point MLE, minutes to run. It does not change D1 — `mixture` is selected on
calibration and ties CRPS — but it decides whether §7c result 3 stands as written.

### Prompt for session 1

> Read `docs/availability-mixture-ship-plan.md` §1 and §2, then
> `docs/availability-window-plan.md` §7d and §8.
>
> **Part A — fix the simulator's availability draw.** `src/sim/season.py:645-648` inlines the
> beta-binomial with a scalar dispersion and raises against the shipped role-graded posterior
> (`rho_draws` is `(1000, 4)`; `FIT_WINDOW = "train"`). Gather each player's `rho_bin` from
> the availability artifact's own recipe, the way line 819 already does for the composition
> head. **Do not route this through `predict_samples`** — it returns games played for design
> rows and the simulator needs the *rate*, applied per cell. Handle `(draws,)` and
> `(draws, K)` artifacts both, remember `rho_bin` is 1-based, and give no-design players the
> lowest bucket. Add a test that fails on the current code and one that pins the 1-based
> indexing, then run `make simulate-season` end to end — nobody has, and the failure has only
> been reproduced at the expression level.
>
> **Part B — the `l2` confound.** Sweep `l2` for the `betabinom` reference arm in
> `src/models/availability_window.py` at the shipped window, and report how much of
> `beta_rect`'s CRPS margin survives. Update §7c result 3 and §7d with the answer, including
> if it is a null.
>
> Do **not** port the mixture; that is session 2. Do not filter `availability_design` — six
> modules import it. Finish with `.venv/bin/python -m pytest tests/` and `make docs-audit`
> both green.

---

## 3. Session 2 — the Stan port

Add the mixture to `src/stan/betabinomial_glm.stan` behind data-supplied switches, exactly as
`n_rho`/`rho_bin` was added. `π = 0` must reproduce the current target bit for bit, asserted
on Stan's own `log_prob` so the other five heads on that file are provably untouched.

The reference implementation is `MixtureFrailty` in `src/models/availability_window.py`:
`π_i = θ·σ(γ'z_i)` with `θ` bounded in [0, 1] — **not** `σ(γ₀ + γ'z_i)`, because `θ = 0` has
to be attainable at a finite parameter value rather than in a limit. The low component is a
beta-binomial with a scalar mean bounded at `MU_LOW_MAX = 0.5` and its own dispersion.

Expected fitted values, from the point MLE on the same rows: `θ ≈ 0.112`, mean `π ≈ 4.9%`,
`μ_low ≈ 0.0999` (about 8 games of 82, not at its bound), `ρ_low ≈ 0.044`.

**Two risks worth planning for.** The point MLE needed **multi-start** — begun at its own
nesting point a three-class mixture sat on the bound and reproduced the incumbent to four
decimals — so NUTS from random inits may find different modes across chains; check per-chain
agreement on `θ` and `μ_low` rather than only R̂. And the availability head fits in ~94
seconds today, so a 2-component marginalized mixture should stay cheap; if it does not, that
is a signal about the geometry, not a budget problem.

### Prompt for session 2

> Read `docs/availability-mixture-ship-plan.md` §1 and §3, then
> `docs/availability-window-plan.md` §7b, §7c and §8, then `src/models/availability_window.py`
> (`MixtureFrailty`) and `src/models/stan_availability.py`.
>
> Port the `mixture` arm to `src/stan/betabinomial_glm.stan` and `stan_availability.py`,
> behind config switches in `configs/default.yaml`, with **`π = 0` reproducing the current
> target bit for bit, asserted on Stan's own `log_prob`** — the same discipline `n_rho = 1`
> carries. Keep `θ` bounded in [0, 1] so the nesting point is attainable, not a limit.
> `PI_COLS` is the covariate block unless you have a reason to change it (see D5).
>
> Verify the port against the point MLE the way the window round did: the Stan posterior's
> CRPS, PIT and the four tail-region errors should sit beside `availability_likelihood.csv`'s
> `mixture` row, and the point MLE refitted on the same rows should reproduce that row to four
> decimals. Check per-chain agreement on `θ` and `μ_low`, not only R̂ — this likelihood is
> multimodal and the point MLE needed multi-start.
>
> Do not touch the simulator or re-run `stan-components`, `stan-composition` or
> `stan-minutes`. Finish with `pytest` and `make docs-audit` green, and record the result in
> `availability-window-plan.md` including if the port does not reproduce the ladder.

---

## 4. Session 3 — propagation, and the gates already owed

Once the posterior exists: `make posteriors` (the recipe needs `π`'s design block and its
scaler), `make model-cards` (the predictive is drawn through the head's own
`predict_samples`), and the simulator, which session 1 has already made able to read a
graded artifact.

**Two gates are owed from the *window* round and have not been re-run**
(`availability-window-plan.md` §5.1): `season-total`'s Gate E and `stan-games-played`, both of
which hold this head as a floor. Re-run them once here rather than twice.

Then `make strategy-sweep` as the D2 value measurement — what the shape win is worth in
advance probability, now that it is not a gate.

### Prompt for session 3

> Read `docs/availability-mixture-ship-plan.md` §4 and `docs/availability-window-plan.md` §8.
>
> Propagate the shipped mixture head: `make posteriors` (the persisted `DesignRecipe` needs
> `π`'s covariate block and scaler), `make model-cards`, then `make simulate-season`. Re-run
> the two gates that hold this head as a floor and have been outstanding since the window
> round — `season-total`'s Gate E and `make stan-games-played` — and report whether either
> verdict moves. Then run `make strategy-sweep` and report what the calibration win is worth
> in Round-1 advance probability, as a value measurement rather than a gate (decision D2).
>
> Update `availability-window-plan.md`, the decision registry and `README.md`'s Results if a
> quoted headline moved. Delete `docs/availability-mixture-ship-plan.md` and its router line
> in `CLAUDE.md` when the round lands. Finish with `pytest` and `make docs-audit` green.

---

## 5. Standing traps for every session in this round

1. **Do not filter `availability_design`.** Six modules import it; a filter inside it silently
   re-scopes the minutes head, the composition head, the games-played spell process and the
   simulator, and none of them would raise. Window and mixture both cut the head's **own
   fitting rows** inside `fit`.
2. **`make docs-audit` is a gate**, not a report. Every session ends with it green.
3. **`betabinomial_glm.stan` serves six heads.** Any change to it must nest the current target
   exactly, asserted on `log_prob`.
4. **Selection reads validation only**, and the rolling harness (13 origins, fitting half) is
   the confirmation instrument — never a second selection.
5. **Nothing ships from the point-MLE ladder.** It selects; `stan_availability.py` ships.
