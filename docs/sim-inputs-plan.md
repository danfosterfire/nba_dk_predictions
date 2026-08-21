# Simulator inputs plan — the consumed constants, their checks, and five work items

**Opened 2026-08-16. A workplan: nothing below has been built.** Not itself in
`make docs-audit` — but **item 1 edits documents that are**, so that item ends by running
`make docs-audit` and `make dashboard-audit` and treating a failure as part of the item.
Figures quoted from artifacts name their artifact; the Gate A diagnostic readings are from
`outputs/predictions/sim_season_gate_a.csv` as of this writing.

**Item 1 goes first, deliberately.** The README still describes "four numbers the simulator
will consume as direct inputs" — a list written before two of the four graduated to
diagnostics and before the σ injection existed. That stale frame cost real time this
session: the owner (and the assistant, initially) had to reverse-engineer from `sim/season.py`
which values are actually consumed. Items 2–5 all touch these values; doing the accuracy
pass first means every later measurement is written against a correct map.

## The inventory (the source of truth for item 1)

**Consumed at draw time — three constants, plus one at the contest layer:**

| constant | value today | provenance | enters the draw at | its check |
|---|---|---|---|---|
| residual copula, count block | matrix, per `fit_window` | `src/eda/residual_correlation.py` → `residual_correlation.csv`, minutes-conditioned | `sim/season.count_copula` inverts it to the **frailty** scale (see below), then correlates the per-game lognormal frailties | Gate A `cross_component_correlation` diagnostic |
| per-game overdispersion `v = 0.025` | `targets.BONUS_GAME_OVERDISPERSION` | `make component-targets`, calibrated to zero bias on the realized per-game bonus rate | variance of those same frailties, **and** the scale of the copula inversion — one mechanism, two roles | Gate A **gated** bonus-rate comparison |
| injected σ, **graded by role** — 0.600 / 0.375 / 0.375 / 0.300 | `sim.minutes.player_season_sigma_by_role`, with `player_season_sigma = 0.375` the shared rung it was selected against | per-bucket train CRPS grid (`minutes_role_sigma`), confirmed on validation; **shipped 2026-08-16** | `rehydrate_composition` → per-(player, season) logit shift, each unit at its own `rho_bin`'s σ | `make minutes-role-sigma` — the per-role PIT is a target now, not scratch |
| Gate C `rho` (contest layer, not this doc's scope) | solved per season **per arm** | `sim/draft.py` bisection against the realized skill gap | the ADP field's rank-noise rotation | `docs/preseason-plan.md` P5/6b carries its caveat |

**Diagnostics only — measured, reported, never imposed:**

| diagnostic | target | current reading | status |
|---|---|---|---|
| game-level minutes dispersion | 4.72× (`stan_minutes_dispersion.csv`) | **7.9** | superseded as an input **by decision**: the composition's fitted role-graded ρ (in the posterior) owns this; the overshoot is item 2's business |
| ten-game minutes block inflation | 2.42× (`serial_correlation.csv`) | **1.40–1.54** | never imposed; *produced* by the season-constant frailties, which carry ~60% of it; item 2 |
| cross-component correlation | the copula's own target | mean abs cell error 0.015–0.019, bar 0.0216 | within bar; max cell ~0.07 |

The `fit_window` discipline rides with all of these: which window to consume is decided by
what the number will be scored against, never by which is widest — the rule the
`simulator-inputs-calibrate-on-train-plus-validation` decision records.

---

## Item 1 — the accuracy pass (FIRST)

Bring every description of the simulator's inputs into agreement with the inventory above.
Named corrections:

- **`README.md`**, two places: the train/validation section's caveat ("The four numbers the
  simulator will consume as direct inputs…") — rewrite around consumed-vs-diagnostic and add
  the σ injection to the consumed list; and the Simulation section's minutes bullet, which
  says the 2.43× enters "for serial dependence between games" — it is a diagnostic, and the
  serial dependence that exists comes from the season-constant frailties.
- **`docs/simulations-plan.md`**: the live spec inherits the inventory table (or a pointer
  to it) and loses any four-inputs framing.
- **Module docstrings** where the same stale frame appears — `src/eda/serial_correlation.py`
  says the block inflation "is a number the simulator consumes"; sweep the others named in
  the inventory. Docstring-only changes, no behavior.
- **`dashboard/inputs.py`** — the page that shows "the inputs the simulator is given": split
  it visibly into **"consumed by the draw"** and **"diagnostic only"** sections, each row
  carrying provenance (artifact, module, the config key or constant) and, for consumed rows,
  *where it enters the draw*; for diagnostic rows, the current reading against its target.
  **Read `docs/dashboard-plan.md` before touching the page** — its charter rules apply — and
  `docs/dashboard-revision-plan.md` inherits them.
- **`dashboard/decisions.py`**: `make dashboard-audit` reports entries whose source language
  drifted; review any entry describing the four-inputs frame (the "diagnostic by decision"
  status of the game-level dispersion should already be an entry — mark it `reviewed` or fix
  its `source`).
- Close by running `make docs-audit` (gate) and `make dashboard-audit` (report); both clean
  is the item's done condition.

## Item 2 — put the missing serial structure at the right timescale

**The two failing diagnostics have one suspect: variance placed at the wrong timescale.**
The sim's minutes carry too much iid per-game noise (dispersion 7.9 against 4.72) and too
little serially-correlated variance (block inflation 1.40–1.54 against 2.42), while the
season-unit total is roughly right (pooled sd_ratio 1.19, per-role PIT scratch measurement,
2026-08-16). The draw structure has only two timescales — iid per-game and season-constant —
and the data's minutes ACF decays ("slow role change plus short-range shocks",
`serial_correlation.py`).

⚠️ **Two corrections from 2026-08-16, and the first one is to this paragraph.** The
1.40–1.54 above was already stale when it was written: the Gate A artifact on disk read block
inflation **1.7285** (2022-23) and **1.5283** (2023-24) before anything changed, so the gap
to 2.42 was smaller than this item is specified against on one season and not the other.

Second, grading the injected σ by role (`docs/draw-time-calibration-plan.md`) moved both
diagnostics and **did not move them together**, which is worth knowing because co-movement is
this item's own gate:

| diagnostic | target | 2022-23 pre → post | 2023-24 pre → post |
|---|---|---|---|
| block inflation | 2.4167 | 1.7285 → **1.8104** | 1.5283 → 1.5309 |
| game-level dispersion | 4.7163 | 7.4985 → **7.5867** | 7.3001 → 7.3745 |

Block inflation improved on one season and stood still on the other; dispersion moved *away*
from target on both. Raising the fringe bucket's season-constant variance adds serial
structure and per-game spread at once, so a σ change cannot separate the two — which is
precisely the argument for this item's AR(1), a mechanism that **redistributes** variance
rather than adding it. **Re-measure before building**: the curve is now read under a graded
σ, and both endpoints have moved.

**Measure first**: block inflation at block lengths 5/10/20/40 plus the lag ACF, per fit
window, through `serial_correlation`'s own machinery — the *curve*, not the ten-game point.
Optionally role-graded, since everything else on this axis has been.

**Then build, draw-time only**: an AR(1) component in the minutes draw that **redistributes**
existing per-game variance into a correlated component (total per-game variance held fixed by
construction), its ρ_ar calibrated to the measured curve. Nesting: ρ_ar = 0 is iid (today's
draw exactly); ρ_ar = 1 degenerates to a season-constant shift (the injection's timescale).
Not a Stan change — an AR latent per game cannot be marginalized by the quadrature device
and would break the season-collapse identities; draw time is the right home.

**Gates**: both diagnostics move toward target *together* (the co-movement is the built-in
check); season-unit CRPS/PIT does not degrade; Gate A's gated rows do not move (the season
unit is orthogonal to within-season arrangement). **Falsifiers**: the measured curve needs
two timescales AR(1) cannot fit (then record the shape and stop — do not stack processes);
or redistribution degrades the season unit (then the "total variance is right" premise was
wrong, which is worth knowing on its own).

**Sequencing**: composes with `docs/draw-time-calibration-plan.md` (graded σ) — same draw
path. Do graded σ first or together; both nest independently.

## Item 3 — per-head `v_h`, and re-invert the copula

**The saturation finding is the smoking gun** (`count_copula`'s own docstring): at the shared
`v = 0.025` the measured residual coupling is *at the ceiling* the frailty can produce — the
inversion asks for off-diagonals above 1 on the strongest pairs and clips at 0.999. One
scalar is setting seven heads' per-game variance and the copula scale at once, calibrated to
a single pooled moment (bonus-rate bias).

**The work**: measure per-head per-game residual overdispersion `v_h` by moments on the
fitting half (each head's own player-game mean–variance relation, beyond the season frailty);
re-run the inversion with the `v_h` vector; re-confirm the bonus-rate gate at zero bias
(re-calibrate a shared level factor if needed — the bonus rate is one moment and stays the
anchor). No Stan anywhere.

**Gates**: the `saturated` count falls; the cross-component diagnostic's max cell shrinks;
the bonus-rate gate holds. **Falsifier**: `v_h` comes back ≈flat at 0.025 — then the ceiling
is real structure (the "two views of one per-game big-night factor" reading), and the honest
follow-up is an explicit one-factor model for the per-game frailties, which gets its own
entry rather than scope creep here.

## Item 4 — two cheap artifact reads that license (or close) the rest

1. **Bonus rate by role/archetype**: realized vs simulated per-game bonus rate split by role
   bucket (the tensor and `bonus_calibration.csv` exist). The pooled calibration is zero-bias
   *by construction*; bonus equity concentrates in double-double bigs, and a compensating
   role error would be invisible pooled. A miss here raises item 3's priority and adds a
   role dimension to it; a clean read closes the question.
2. **Copula cell stability by archetype**: split the residual correlations by role/archetype
   buckets with bootstrap bands (the residual frames exist; the PCA archetypes exist). If the
   big cells (`fga`–`reb` +0.133) are stable across archetypes, the pooled matrix is fine and
   that is worth recording; if they move materially, grading the copula becomes a real item
   with this read as its evidence.

An afternoon each, pandas over existing artifacts, no builds.

## Item 5 — the one missing copula cell, priced before built

`fg3a | fga` is the single conversion head with real per-game structure (block inflation
**1.575** against the other conversions' 1.007–1.103 nulls) and its −0.083 coupling against
`fga` is deliberately not carried (`sim/season.py`, "The copula, and the one cell it does not
carry"). **Price it first at the dk_pts level**: bound the effect of the missing cell on
per-game dk_pts variance and the bonus rate (the 3-vs-2 substitution largely cancels in the
DK sum, so the prior is *small*). Only if the bound clears the noise floor of the metrics it
would move does the mechanism get built: a per-game jitter on the mix share, correlated with
`fga`'s frailty through the copula, moment-matched the same way `count_copula` already works.
If the bound is below the floor, record the null and close the entry — that is a result.

---

## Order

1 (the accuracy pass, first by direction) → 4 (cheap reads; they inform 3 and 5) → 2 and 3
in either order (independent mechanisms, both measurement-first) → 5 last, and only through
its price gate. Items 2–5 touch no fitted head and no audited record; item 1 touches audited
prose and ends by proving the audits still pass.
