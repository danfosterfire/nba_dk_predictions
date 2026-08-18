# Entry collection plan — the first real entries, designed for data

**Opened 2026-08-16, decided the same day: 20 ADP-null autodrafts plus 5–10 live drafts on
top, all at the $1 tier.** An operational runbook, not a results doc; not in
`make docs-audit`. The working contest is the **$1 `15k_and_one`** (the stated likely tier —
confirm against the actual 2026 contest lineup when the lobby opens). Budget $25–30.
**The first entries are in — 2026-08-16 and 2026-08-17, two of the 20 ADP-null autodraft
pods, both in the $1 `NBA Best Ball $15K And-One [150 Entry Max]`** — which confirms the
assumed $1 tier exists in the 2026 lineup. The decision registry carries the entry and the
`no tournaments entered` framing in docs and memory is updated accordingly (that framing
had drifted once before and was scrubbed on 2026-08-11). Both pick logs are captured and
validated — `docs/draft-board-ingestion-plan.md` is the capture workflow, which closes
this plan's "capture pre-step".

## The design, and why

**20 × ADP-null autodraft pods — the power base, not to be traded away.**
Upload the **DK-recalibrated consensus board** as the custom ranking and let DK's autodraft
execute it. Our seat then *is* the shipped opponent model — one simulated field agent
inserted into each real pod — so the other eleven drafters' deviations from the model read
maximally cleanly. The statistical argument: opponent behavior *conditional on board state*
is validly sampled whatever we do, but our picks decide **which board states get sampled**;
off-ADP play pushes pods into regions rare under typical play, and at ~20 pods
(~220 opponent drafters, ~3,500 opponent picks) the budget should be spent near the typical
region, where the questions live.

**5–10 × live drafts with real strategy — additional pods, not carved out of the 20.**
Drafted through `draft_room` (`make draft-room`) with the **block-off production rankings**
(see the two-fit schedule below). What these buy that autodraft cannot: the only rehearsal
of the tool against a real clock, real DK interface and the 8G/8F/3C caps before the
real-stakes drafts; and the only observation of whether a room *reacts* to off-ADP behavior
(position runs after a reach), which no simulation can produce. The measured safety net
applies: submitting our ranking to DK's autodraft is identical to clicking it under the
caps, so a rehearsal going sideways degrades to a known-safe fallback.

## What the pick logs are for

The full 12-entrant draft history per pod is the capture target — the field-calibration
data the opponent model has never had (it is built from ADP boards, not observed drafts).
The questions, stated before the data exists:

1. **Does the real field carry a slot-reaching lean?** Gate B fit it at **0 picks** from
   the ADP curve; the logs test that on real pods.
2. **How much rank noise does the field actually have?** Gate C's `rho` is currently
   *solved* per season against the realized skill gap — the known worst calibrated number
   in the project (`docs/preseason-plan.md` P5). An observed pick-level dispersion is the
   first direct measurement it could ever be checked against.
3. **Field drift across the entry window** — see timing below.
4. **Room reaction to off-ADP picks** (live pods only), and execution rehearsal.

## The two-fit production schedule

The live-draft rankings cannot come from the shipped model: **the preseason blocks require
a complete preseason, and these entries happen before it exists.** Imputing preseason data
is empty by construction — every block is difference-coded against prior-season features,
so the predictable part of a preseason is exactly the part with a delta of ~0. The honest
answer is the exact rollbacks, which are first-class config and were exercised end-to-end
by `make preseason-contest` on 2026-08-15:

| fit | when | flags | serves |
|---|---|---|---|
| **Fit 1 — blocks off** | early fall, whenever convenient | `stan.availability.preseason: false`, `stan.minutes.preseason: false`, `stan.composition.preseason.adopt: false`, `stan.components.preseason: false` | the 5–10 live drafts and their `draft_room` engine — the correct model for the pre-preseason information set (the pre-2026-08-13 problem statement, not a degraded model) |
| **Fit 2 — blocks on** | the Oct 17–20 runbook window (`docs/preseason-plan.md`) | shipped config | the real-stakes entries only |

Two shortcuts recorded as **do-nots**: scoring the block-on model with zeroed preseason
columns is *not* the pre-block model (difference-coding recovers zero only at fixed
coefficients, and the block-on heads were jointly refitted on cut windows); and scoring the
full pool through the missing indicators injects a coefficient fitted on injury/late-signing
cases into everyone.

The 20 null pods need neither fit — only the consensus board, which requires the
`make daily-capture` cron to be alive (the ADP sources are not backfillable;
`docs/adp-plan.md`).

## Timing

Field behavior drifts across the offseason (casual early, news-informed late). **Spread the
20 null pods across the entry window** — the drift is then a finding rather than a
confound — and **run the 5–10 live pods late**, closest to the regime the real entries will
face, after Fit 1 exists. Accept the trade this makes explicit: late pods leave little time
to incorporate their logs before the real drafts; their value is rehearsal and observation,
not model updates.

## Before the first entry — the capture pre-step

✅ **Built 2026-08-17 — `docs/draft-board-ingestion-plan.md`.** The artifact is one row
per pick — draft date, contest, seat, entrant, pick number, resolved player id, and **which
ranking our seat used (`ranking_file` in the boards manifest, auto vs live)** — landed under
`data/raw/adp_autodrafted_boards_<year>/` beside the other capture programs and pooled into
`data/features/draft_pick_log.parquet` by `make draft-boards`. The DK draft room is login-gated with no archive, so whatever is not captured at
draft time is gone; that is the same deadline logic the ADP capture already lives under.
Provenance rule: a pod whose ranking version is unrecorded is uninterpretable — record it
at entry time, per pod.

## What would change the plan

- The 2026 contest lineup not offering the $1 tier as assumed → re-decide the tier, keep
  the design.
- The ADP capture cron having gapped → the null board's provenance breaks; fix capture
  before entering.
- DK changing autodraft/custom-ranking mechanics → re-verify the "autodraft equals capped
  click" equivalence before relying on it as the fallback.
