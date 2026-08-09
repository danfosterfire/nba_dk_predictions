"""The decision registry — one structured record of every load-bearing decision.

The decisions themselves live as prose across `README.md` and the `docs/` files —
the plan docs, plus `project-spec.md`, `facts-archive.md`, `data-quirks.md`,
`model-development-notes.md` and `train-validate-test-split.md`. `CLAUDE.md` is a
router and holds none of them. Prose cannot be filtered, counted or cross-referenced, and
re-typing it into nine tab functions would make nine more copies of it. So this
module is **one distillation** of that prose, and every tab renders its own slice.

`dashboard/README.md` states the precedence rule that governs this file: the docs
are the source of truth and this registry is a browsing aid. Where the two
disagree, the registry is stale.

Pure data — **no Streamlit import**, so `audit.py` and the tests exercise it
directly. `status` comes from a closed vocabulary; a reversal becomes `withdrawn`
and keeps its entry rather than being deleted, because the reversals are the most
useful thing on the decision-log tab.
"""

from dataclasses import dataclass, field

# ── Vocabulary, fixed and closed ──────────────────────────────────────────────

STATUSES: dict[str, str] = {
    "built": "code exists and its artifact is on disk",
    "settled": "decided against evidence, not to be relitigated",
    "measured": "a number we have, not yet a decision",
    "null": "tested and found to be worth nothing — recorded so it is not rebuilt",
    "withdrawn": "previously believed, then falsified by a better measurement",
    "open": "known gap with no answer yet",
    "blocked": "cannot proceed, with the unblocking condition named",
    "deadline": "will be permanently lost if not done by a date",
    "incident": "a one-time diagnosis of an external system, not reproducible by design",
}

# Statuses that legitimately have no artifact on disk, so the audit's existence
# check skips them. `incident` is the deliberate exception to the provenance rule:
# re-deriving "223 of 228 games came back empty on 2025-04-10" would mean re-probing
# a third party to no purpose. See docs/provenance-plan.md, "figures versus
# incident records".
UNBACKED_STATUSES: tuple[str, ...] = ("open", "blocked", "deadline", "incident")

# One topic per walkthrough tab, in tab order. Tab 9 renders all of them.
TOPICS: tuple[str, ...] = ("problem", "data", "eda", "availability", "minutes",
                           "components", "simulations", "drafting")

TOPIC_LABELS: dict[str, str] = {
    "problem": "Problem & constraints",
    "data": "Data collection",
    "eda": "Exploratory data analysis",
    "availability": "The availability head",
    "minutes": "The minutes head",
    "components": "The DK component-rate heads",
    "simulations": "Season simulations",
    "drafting": "Drafting strategy",
}


@dataclass(frozen=True)
class Decision:
    """One load-bearing decision, distilled from the doc named in `source`.

    `reproduce` is the provenance link, written as `"make <target> → <artifact>"`
    with several artifacts comma-separated. `artifact_paths` and `make_target`
    parse it, and `audit.py` checks the paths exist.
    """

    id: str
    topic: str
    claim: str
    because: str
    status: str
    source: str                      # the doc this was distilled from
    reviewed: str                    # when a human last checked it against that doc
    date: str                        # when the decision was taken
    reproduce: str = ""              # "make <target> → <artifact>[, <artifact>]"
    replaced_by: str = ""            # withdrawn: what the better measurement says
    caught_by: str = ""              # withdrawn: what caught it
    unblocks: str = ""               # blocked: the condition that would clear it
    due: str = ""                    # deadline: the date past which it is lost
    tags: tuple[str, ...] = field(default_factory=tuple)


# ── Parsing helpers ───────────────────────────────────────────────────────────

_ARROW = "→"


def make_target(d: Decision) -> str:
    """The command half of `reproduce`, or "" when the entry carries no figure."""
    if _ARROW not in d.reproduce:
        return d.reproduce.strip()
    return d.reproduce.split(_ARROW, 1)[0].strip()


def artifact_paths(d: Decision) -> tuple[str, ...]:
    """The repo-relative artifact paths `reproduce` names, in order.

    Comma-separated so one entry can cite the several files a target writes
    together — the Stan port verification is three CSVs from one fit.
    """
    if _ARROW not in d.reproduce:
        return ()
    tail = d.reproduce.split(_ARROW, 1)[1]
    return tuple(p.strip() for p in tail.split(",") if p.strip())


def needs_artifact(d: Decision) -> bool:
    """Whether the provenance rule obliges this entry to name a live artifact."""
    return d.status not in UNBACKED_STATUSES


def by_topic(topic: str, registry: tuple[Decision, ...] | None = None
             ) -> tuple[Decision, ...]:
    reg = REGISTRY if registry is None else registry
    return tuple(d for d in reg if d.topic == topic)


def by_status(status: str, registry: tuple[Decision, ...] | None = None
              ) -> tuple[Decision, ...]:
    reg = REGISTRY if registry is None else registry
    return tuple(d for d in reg if d.status == status)


def status_mix(registry: tuple[Decision, ...] | None = None) -> dict[str, int]:
    """How much of this project is settled against open — as a number."""
    reg = REGISTRY if registry is None else registry
    return {s: sum(1 for d in reg if d.status == s) for s in STATUSES}


def all_artifacts(registry: tuple[Decision, ...] | None = None) -> tuple[str, ...]:
    """Every artifact path the registry references, deduped, in first-seen order."""
    reg = REGISTRY if registry is None else registry
    seen: dict[str, None] = {}
    for d in reg:
        for p in artifact_paths(d):
            seen.setdefault(p, None)
    return tuple(seen)




# ── The registry ──────────────────────────────────────────────────────────────
#
# `reproduce` names the make target and the artifact(s) it writes. A **glob** is
# used where one target owns a whole family — `make pca` writes twenty files and
# deserves one line, not twenty. `audit.py` resolves globs for both the existence
# check and the orphan check, so a family that stops being written shows up either
# way.

REGISTRY: tuple[Decision, ...] = (

    # ══ Problem & constraints ════════════════════════════════════════════════
    Decision(
        id="cross-season-join",
        topic="problem",
        claim="The information set is a cross-season join: current-season roster "
              "membership × prior-season statistics.",
        because="Before the season starts we know the schedule, the season-start "
                "rosters and *previous*-season stats. We do not know mid-season "
                "trades, current-season minutes, injuries or form. So team "
                "composition aggregates the season-S roster described by S-1 stats — "
                "not the S-1 roster — minutes weights must come from S-1, and every "
                "feature except the schedule-derived ones is constant within a "
                "player-season. The only per-game variation available is opponent, "
                "home/away and rest.",
        status="settled",
        reproduce="make team-context → data/features/team_context_tier*.parquet, "
                  "data/features/team_composition_tier*.parquet",
        source="docs/project-spec.md",
        reviewed="2026-08-08",
        date="2026-07-27",
        tags=("constraint",),
    ),
    Decision(
        id="components-not-dk-pts",
        topic="problem",
        claim="Predict the twelve components; never predict dk_pts directly.",
        because="Team-context effects move components hard and in opposite directions, "
                "then cancel in the DK sum — 2.11 gross into 0.25 net, an 8.3× "
                "cancellation on `teammate_assist_supply`, with `role_crowding` at "
                "8.2×, `teammate_spacing` 5.0× and `team_pace` 4.7×. And the "
                "double-double bonus is a threshold on five components, so "
                "E[bonus] ≠ bonus(E[x]): a single dk_pts head cannot represent it.",
        status="settled",
        reproduce="make context-value → outputs/eda/team_context_value_tier*.csv",
        source="docs/predictions-plan.md",
        reviewed="2026-07-30",
        date="2026-07-29",
        tags=("architecture",),
    ),
    Decision(
        id="decompose-pts-into-shot-classes",
        topic="problem",
        claim="Decompose `pts` further into shot classes — FT / 2PT / 3PT, attempts "
              "and makes. `pts` is not a primitive count.",
        because="The weighting is the entire source of `pts`'s overdispersion: "
                "within-player var/mean is 2.16–2.31 for `pts` in every minutes "
                "bucket but 0.86–1.08 for the shot classes, because a 2× coefficient "
                "squares into the variance and doubles var/mean while leaving the "
                "mean alone. So a Poisson head is *misspecified* on `pts` and "
                "correctly specified on the classes — decomposing removes the "
                "misspecification rather than patching it with a negative binomial. "
                "Free throws are the exception that stays overdispersed because they "
                "arrive in pairs.",
        status="settled",
        reproduce="make target-profile → outputs/eda/target_profile.csv",
        source="docs/facts-archive.md",
        reviewed="2026-08-08",
        date="2026-07-28",
        tags=("architecture",),
    ),
    Decision(
        id="joint-draw-not-marginals",
        topic="problem",
        claim="The deliverable is a joint draw over the components, not twelve "
              "marginals.",
        because="The double-double bonus is a simultaneous threshold on "
                "pts/reb/ast/stl/blk, so it cannot be computed from marginal means. "
                "Independent sampling measures 22.7% too low against the realized "
                "mean bonus.",
        status="settled",
        reproduce="make component-targets → outputs/eda/bonus_calibration.csv",
        source="docs/predictions-plan.md",
        reviewed="2026-07-30",
        date="2026-07-29",
        tags=("architecture",),
    ),
    Decision(
        id="regular-season-only",
        topic="problem",
        claim="Every fitting frame is regular season only; playoff logs are features, "
              "never targets.",
        because="The contest is over before the playoffs begin — Round 4 ends 4/4, "
                "ahead of a mid-April playoff start — so it is out of scope by the "
                "rules of the product before it is by statistics. The statistics "
                "agree: playoff minutes are a role interaction with a *sign change*, "
                "median playoff-to-regular MPG 0.505 for bench against 1.054 for "
                "starters, so a pooled indicator would fit one coefficient to a −50% "
                "and a +5% effect at once.",
        status="settled",
        reproduce="make availability-profile → outputs/eda/availability_profile.csv",
        source="docs/dk_best_ball_rules.md",
        reviewed="2026-07-30",
        date="2026-07-29",
        tags=("scope",),
    ),
    Decision(
        id="variance-budget-are-ceilings",
        topic="problem",
        claim="The variance budget is a table of **in-sample ceilings**, not of "
              "achievable gains, and the two bases must not be conflated.",
        because="Player-season identity is a share of *total* per-game variance; "
                "everything else is a share of the *within-player residual*. The "
                "artifact carries `basis` per row so misreading the denominator is "
                "impossible, and it carries both null constructions for the "
                "interaction row, since the same statistic reads +0.96% permuting "
                "opponent and +0.31% permuting archetype.",
        status="measured",
        reproduce="make variance-budget → outputs/eda/variance_budget.csv",
        source="docs/provenance-plan.md",
        reviewed="2026-07-30",
        date="2026-07-29",
        tags=("frame",),
    ),
    Decision(
        id="own-minutes-18-percent",
        topic="problem",
        claim="Own minutes played explains 18.6% of the within-player-season residual "
              "variance in dk_pts.",
        because="Measured by conditioning the within-player-season residual on the "
                "raw minutes level, pooled across players.",
        status="withdrawn",
        replaced_by="**46.4%.** Conditioning on the within-player minutes *deviation* "
                    "— 'he played eight more minutes than he usually does', the "
                    "contrast the residual is actually defined by — gives 46.399%, "
                    "stable at 46.189% nonparametrically, against a saturated upper "
                    "bound of 59.398%.",
        caught_by="`make variance-budget` reproduced the recorded 18.6% to three "
                  "digits under the raw-level construction, which is how the "
                  "construction was identified. Pooling across players attenuates by "
                  "more than half: a 30-minute game is below average for a 34-mpg "
                  "starter and far above it for an 18-mpg reserve, so their residuals "
                  "cancel inside the cell. Direction is safe — minutes matter more, "
                  "which strengthens the shared-`min` draw.",
        reproduce="make variance-budget → outputs/eda/variance_budget.csv",
        source="docs/provenance-plan.md",
        reviewed="2026-07-30",
        date="2026-07-29",
        tags=("reversal",),
    ),
    Decision(
        id="opponent-interaction-over-main-effect",
        topic="problem",
        claim="Encode opponent as an interaction with player style rather than as a "
              "team-quality main effect.",
        because="An in-sample ANOVA put opponent × archetype × season above a shuffled "
                "null while the opponent main effect looked small.",
        status="withdrawn",
        replaced_by="The main effect beats the interaction **~17×** out of sample on "
                    "dk_pts — 0.368% against +0.022%, held out on 2024-25/25-26 with "
                    "prior-season-only inputs. Build the main effect properly; keep "
                    "the interaction for the *components*, where it adds +0.203% on "
                    "`blk_per36`, nearly doubling that head's main effect.",
        caught_by="The original comparison put an in-sample ceiling beside an "
                  "achievable held-out gain. Measuring both out of sample on the same "
                  "rows reversed the ordering.",
        reproduce="make opponent → outputs/eda/opponent_matchup_tierA.csv",
        source="docs/facts-archive.md",
        reviewed="2026-08-08",
        date="2026-07-29",
        tags=("reversal",),
    ),
    Decision(
        id="opponent-cancellation-1-41x",
        topic="problem",
        claim="The opponent effect cancels 1.41× across components — 1.105 gross "
              "against 0.785 net.",
        because="Measured once in a planning session as the per-component weighted "
                "opponent sd against the sd measured on dk_pts directly.",
        status="withdrawn",
        replaced_by="**~2×.** Four internally consistent constructions all land "
                    "between 1.84× and 2.11×; the shipped one reads 1.981 gross "
                    "against 0.987 net. The components move about twice as much as "
                    "their DK sum does — a *stronger* argument for component heads "
                    "than the recorded figure made.",
        caught_by="Promoting the figure to `make opponent` showed no single "
                  "construction can produce 1.41×: gross and net had been measured on "
                  "different bases. `cross_component_cancellation` now takes both from "
                  "the same `sd_col` and a test pins that the ratio is invariant to "
                  "the column.",
        reproduce="make opponent → outputs/eda/opponent_matchup_tierA.csv",
        source="docs/provenance-plan.md",
        reviewed="2026-07-30",
        date="2026-07-29",
        tags=("reversal",),
    ),
    Decision(
        id="lstm-trunk-deprioritized",
        topic="problem",
        claim="Deprioritize the LSTM/Transformer trunk over prior-season game logs. "
              "Do not relitigate without new evidence.",
        because="The prior-season game *sequence* is worth ~+0.6 pp R² — 0.7657 with "
                "five order features against 0.7599 for four season aggregates, and "
                "0.7584 for the same five features computed on **shuffled** game "
                "order. So +0.0059 over aggregates and +0.0074 above its own null. "
                "The trunk is mostly re-deriving a season mean `season_matrix.py` "
                "already holds.",
        status="settled",
        reproduce="make feature-diagnostics → outputs/eda/feature_diagnostics.csv",
        source="docs/model-development-notes.md",
        reviewed="2026-08-08",
        date="2026-07-29",
        tags=("architecture", "do-not-relitigate"),
    ),
    Decision(
        id="five-games-settle-most-of-the-season",
        topic="problem",
        claim="Five current-season games settle 86% of the season total — the "
              "measured price of the prior-season-only constraint.",
        because="Extrapolating the first-k mean over the games actually played gives "
                "R² 0.859 / 0.905 / 0.942 / 0.974 at k = 5 / 10 / 20 / 41. Separately, "
                "log(season total) is 84.5% explained by log(per-game rate) alone and "
                "73.4% by log(games) alone — overlapping shares, not a partition, "
                "since the two factors correlate at +0.585. Set against +0.0086 R² for "
                "the entire own-team block, this is the honest caveat on the whole "
                "project.",
        status="measured",
        reproduce="make target-profile → outputs/eda/target_profile.csv, "
                  "outputs/eda/target_season_totals.parquet, "
                  "outputs/eda/target_trajectories.parquet",
        source="docs/model-development-notes.md",
        reviewed="2026-08-08",
        date="2026-07-28",
        tags=("frame",),
    ),

    # ══ Data collection ══════════════════════════════════════════════════════
    Decision(
        id="daily-capture-deadline",
        topic="data",
        claim="`make daily-capture` must stay on a scheduler. A day it does not run "
              "is a day permanently lost.",
        because="Both sources are current-status feeds that cannot be backfilled: the "
                "NBA injury-report PDFs age out of the CDN after ~7 months (a 403 "
                "thereafter, forever) and the ESPN feed has no history at all. It runs "
                "under launchd rather than cron because macOS cron silently skips a "
                "run whose time passed while the machine was asleep, and launchd "
                "catches missed runs up on wake.",
        status="deadline",
        due="standing — every day",
        source="docs/availability-plan.md",
        reviewed="2026-07-30",
        date="2026-07-27",
        tags=("capture",),
    ),
    Decision(
        id="capture-before-backfill",
        topic="data",
        claim="Archive the raw source and re-parse offline; never treat the parsed CSV "
              "as the artifact worth keeping.",
        because="`make injury-reports` is idempotent and re-parsing is offline "
                "(`--reparse`), so a parser bug costs nothing as long as the PDF is on "
                "disk. The PDFs are the irreplaceable half.",
        status="settled",
        reproduce="make report-calibration → outputs/eda/report_calibration.csv, "
                  "data/features/report_transfer.parquet",
        source="docs/availability-plan.md",
        reviewed="2026-07-30",
        date="2026-07-27",
        tags=("capture",),
    ),
    Decision(
        id="point-in-time-discipline",
        topic="data",
        claim="Only dated-at-publication sources may fill history. Never fill a "
              "historical row from a current-status feed.",
        because="The largest correctness risk in the availability head. The ESPN feed "
                "and Basketball-Reference describe *today*, so a 2019 row filled from "
                "them encodes the resolved outcome — which is the target. "
                "`return_date` is a **forecast made on the snapshot date** and must "
                "never be overwritten with the realized return. Read the ESPN log only "
                "through `injuries.snapshot_as_of`, and every training row through "
                "`models.availability.assert_point_in_time`.",
        status="settled",
        reproduce="make availability → data/features/availability_panel.parquet, "
                  "data/features/availability_features.parquet",
        source="docs/availability-plan.md",
        reviewed="2026-07-30",
        date="2026-07-27",
        tags=("discipline", "leakage"),
    ),
    Decision(
        id="two-guards-beat-one-matching-rule",
        topic="data",
        claim="A fuzzy name match needs two independent guards, and every non-exact "
              "match must be listed and read.",
        because="**An unmatched rate is monotonically increasing in the error it is "
                "supposed to detect** — every fabricated match improves the score. The "
                "rejected surname-initial rule scores 0.00% unmatched against the "
                "cascade's 0.50% while making 31 matches, 23 of which the cascade "
                "refuses: Cameron Boozer → Carlos Boozer, Darryn Peterson → Drew "
                "Peterson, Mikel Brown Jr. → Moses Brown. Either guard alone still "
                "invents people, so a first name must be a genuine ≥3-char prefix "
                "**and** the candidate must have played within ~3 seasons.",
        status="settled",
        reproduce="make adp-panel → outputs/eda/adp_match_audit.csv",
        source="docs/model-development-notes.md",
        reviewed="2026-08-08",
        date="2026-07-29",
        tags=("joins", "discipline"),
    ),
    Decision(
        id="no-nba-history-is-not-unmatched",
        topic="data",
        claim="Keep 'no such entity' apart from 'join failed'.",
        because="162 DraftKings pool entries have no `player_id` because they have "
                "never played an NBA game. Reported as `no_nba_history` rather than "
                "`unmatched`, they are the 2026 draft class existing; collapsed "
                "together they become a fake defect that hides real ones. The same "
                "distinction `report_calibration.py` draws between `absent` and "
                "`unmatched`.",
        status="settled",
        reproduce="make adp-panel → outputs/eda/adp_match_audit.csv",
        source="docs/model-development-notes.md",
        reviewed="2026-08-08",
        date="2026-07-29",
        tags=("joins", "discipline"),
    ),
    Decision(
        id="name-joins-in-exactly-two-places",
        topic="data",
        claim="Name-based joins exist in exactly two places; everything else keys on "
              "ids. And `normalize_name` strips digits.",
        because="`report_calibration.py` (`game_id` + `name_key`) and the ADP modules "
                "(`player_key`) are the only two; `preprocess.py` and `dataset.py` "
                "group by `player_id` only. `injuries.py` and `injury_reports.py` "
                "store names and never join, so any future consumer inherits the whole "
                "problem and should get the ADP cascade rather than a fresh `merge`. "
                "Digit-stripping makes `P0`…`P39` a **single key** — a synthetic test "
                "fixture built that way silently exercised a 40-way collision and "
                "*passed*, so test names must stay distinct after normalization.",
        status="settled",
        reproduce="make report-calibration → outputs/eda/report_calibration.csv",
        source="docs/model-development-notes.md",
        reviewed="2026-08-08",
        date="2026-07-29",
        tags=("joins", "discipline"),
    ),
    Decision(
        id="status-is-a-measurement-not-a-bracket",
        topic="data",
        claim="The panel's `status` column replaces the roster-window bracket with a "
              "measurement — and `not_rostered` must stay apart from `unknown`.",
        because="`played` / `dnp` (dressed and available, not used — a *rotation* fact, "
                "not a health one) / `inactive` / `not_rostered` / `unknown`, wherever "
                "the box-score backfill has run. Coverage is tracked **per game**, so "
                "only inside a backfilled game does a missing box-score row mean 'not "
                "on that roster'; collapsing the two makes a half-finished backfill "
                "read as a league-wide roster collapse. Absences are keyed on "
                "`(player_id, team_id, game_id)` because a `game_id` belongs to both "
                "teams, and a two-key merge credits a traded player's new-team games "
                "to his old team.",
        status="built",
        reproduce="make availability → data/features/availability_panel.parquet, "
                  "data/features/availability_features.parquet",
        source="docs/availability-plan.md",
        reviewed="2026-07-30",
        date="2026-07-28",
        tags=("capture",),
    ),
    Decision(
        id="four-coverage-boundaries",
        topic="data",
        claim="Four coverage boundaries, not one: core families 30 seasons, "
              "tracking/`pt_shot` 13, estimated 12, hustle 11 — and the box-score "
              "inactive list at 2006-07.",
        because="Tier A is the 30-season box-score frame and Tier B the 13-season "
                "tracking superset. The inactive-list boundary was probed on both "
                "sides — 0 rows for 2005-06, 6 on 2006-07 opening night — rather than "
                "assumed. 2015-16 hustle covers only 147 players, which is a partial "
                "first-season rollout and not a fetch failure.",
        status="measured",
        reproduce="make season-matrix → data/features/coverage_report.csv, "
                  "data/features/season_matrix_*.parquet",
        source="docs/data-quirks.md",
        reviewed="2026-08-08",
        date="2026-07-27",
        tags=("coverage",),
    ),
    Decision(
        id="roster-description-coverage",
        topic="data",
        claim="14.7% of realized roster minutes have no usable prior-season row. Never "
              "drop them.",
        because="8.73% true rookies, 5.07% sub-threshold, 0.91% returnees, on the "
                "season-start roster the model actually aggregates. Per team-season "
                "the mean is 14.8%, p90 29.0% and the worst 50.4%. Dropping them "
                "silently biases every aggregate toward veterans; the unfiltered "
                "`season_matrix_roster_tier*` twin exists for exactly this, with "
                "`stats_source` and `reliability` per row. The head-count and "
                "minutes-weighted splits differ — a rookie is 14.1% of roster rows and "
                "8.7% of roster minutes — and neither substitutes for the other.",
        status="measured",
        reproduce="make context-value → outputs/eda/roster_coverage_profile_tier*.csv",
        source="docs/provenance-plan.md",
        reviewed="2026-07-30",
        date="2026-07-29",
        tags=("coverage",),
    ),
    Decision(
        id="roster-coverage-15-9",
        topic="data",
        claim="15.9% of roster minutes have no usable S-1 row — 9.4% rookies, 5.4% "
              "sub-threshold, 1.1% returnees.",
        because="Measured once on the roster as recorded, without stating which roster "
                "window it used.",
        status="withdrawn",
        replaced_by="**14.71%** on the season-start roster (`roster_window_games = "
                    "10`), which is the only window knowable before the season and "
                    "therefore the only one `team_context` aggregates. The recorded "
                    "figure is the **whole-season** roster: window 82 reproduces it at "
                    "15.79% / 9.16% / 5.52% / 1.11%.",
        caught_by="Measuring across seven window sizes. The gap is mechanical — "
                  "rookies and players absent the previous season *arrive late*, so "
                  "widening the window pulls in disproportionately many of them. Both "
                  "windows are emitted so the difference stays visible. The recorded "
                  "figure was pessimistic by 1.2 pp and 'never drop them' is "
                  "unaffected.",
        reproduce="make context-value → outputs/eda/roster_coverage_profile_tier*.csv",
        source="docs/provenance-plan.md",
        reviewed="2026-07-30",
        date="2026-07-29",
        tags=("reversal",),
    ),
    Decision(
        id="game-id-padding",
        topic="data",
        claim="`game_id` is `int64` in the game logs and a zero-padded 10-char string "
              "in the box-score files — normalize with `boxscore_status.pad_game_id`.",
        because="Without it the merge matches nothing, **silently**, while still "
                "returning a full panel. The minutes feasibility assertion has a "
                "second arm for exactly this: an unmatched player-game means the join "
                "is broken, which would otherwise show up only as a quietly smaller "
                "fitting frame.",
        status="settled",
        reproduce="make game-length → outputs/eda/game_length_coverage.csv, "
                  "data/features/game_length.parquet",
        source="docs/model-development-notes.md",
        reviewed="2026-08-08",
        date="2026-07-29",
        tags=("joins", "failure-mode"),
    ),
    Decision(
        id="boxscore-summary-v2-silent-failure",
        topic="data",
        claim="`BoxScoreSummaryV2` returns 200 with a populated `GameSummary` while "
              "silently dropping the `InactivePlayers` result set after 2025-04-10 — "
              "223 of 228 games in 2025-26 came back empty.",
        because="A backfill built on it records 'nobody was inactive' rather than "
                "failing, which is indistinguishable from a real roster fact. Its "
                "sibling fails the opposite way and is therefore safe: "
                "`BoxScoreTraditionalV2` returns **0 rows** from 2025-26 on — loud and "
                "obvious. `BoxScoreSummaryV3` covers 2006-07 → 2025-26 and agrees with "
                "V2 exactly where V2 works.",
        status="incident",
        source="docs/data-quirks.md",
        reviewed="2026-08-08",
        date="2026-07-28",
        tags=("failure-mode", "nba-api"),
    ),
    Decision(
        id="v3-parser-raises-on-stub-payloads",
        topic="data",
        claim="`nba_api`'s own V3 parser raises on three stub payloads, and the guard "
              "in `_inactive_rows` cannot catch it. Those 3 of 25,709 games are left "
              "unfetched deliberately.",
        because="Three games on 2025-11-19 return a `boxScoreSummary` whose `arena`, "
                "`teamId` and `inactives` are all null; the parser calls "
                "`arena.get('arenaId')` unguarded and throws **inside the "
                "constructor**, before `get_dict()` is reached — so the error text "
                "reads like a bug in this repo when it is upstream. There is nothing "
                "to recover: the raw JSON is a stub and V2 is inside its own silent "
                "window. Recording the dressed rows without the inactive list would "
                "mislabel those games' inactives as `not_rostered`, so with no status "
                "rows the game is `status_covered = 0` and every row is `unknown`, "
                "which is correct.",
        status="incident",
        source="docs/data-quirks.md",
        reviewed="2026-08-08",
        date="2026-07-28",
        tags=("failure-mode", "nba-api"),
    ),
    Decision(
        id="poisoned-server-side-cache",
        topic="data",
        claim="Empty `LeagueDashPlayerStats` responses are poisoned server-side cache "
              "entries keyed on the full query, not rate limiting.",
        because="Backoff does not help; cycling `per_mode_detailed` does. `fetch.py` "
                "handles it and records the winning mode in "
                "`data/raw/_fetch_manifest.csv`. Diagnosed once against a live "
                "endpoint; the *defence* is in the code and the manifest is on disk, "
                "which is what makes leaving the diagnosis as prose honest.",
        status="incident",
        source="docs/data-quirks.md",
        reviewed="2026-08-08",
        date="2026-07-27",
        tags=("failure-mode", "nba-api"),
    ),
    Decision(
        id="append-only-manifest-overstates-damage",
        topic="data",
        claim="The backfill manifest is append-only, so counting `error` rows "
              "overstates the damage — 77 error rows against **3** games that actually "
              "lack data.",
        because="A retried game keeps its old failure row and gains a new success row, "
                "so the count must dedupe to games with no successful attempt. Every "
                "transient network fault — 68 of them, clustered in 2012-13 and "
                "2013-14 — cleared on one retry. The backfill finished at 25,706 of "
                "25,709 games with mean `status_coverage` 0.696 and ~99.8% per season "
                "from 2006-07 on.",
        status="incident",
        source="docs/data-quirks.md",
        reviewed="2026-08-08",
        date="2026-07-28",
        tags=("failure-mode",),
    ),
    Decision(
        id="espn-tcc-outage",
        topic="data",
        claim="A macOS Full Disk Access lapse cost 28 days of ESPN injury snapshots, "
              "permanently.",
        because="The feed has no history at all, so a day the scheduler does not run "
                "is unrecoverable — the concrete cost of the `daily-capture` deadline "
                "rather than a hypothetical one. The launchd job now holds the grant, "
                "and `make capture-status` reports the archive count without making a "
                "request.",
        status="incident",
        source="docs/availability-plan.md",
        reviewed="2026-07-30",
        date="2026-07-28",
        tags=("failure-mode", "capture"),
    ),
    Decision(
        id="bbref-cannot-supply-injuries",
        topic="data",
        claim="Basketball-Reference cannot supply historical injury data — checked, "
              "not assumed.",
        because="`/leagues/NBA_2024_transactions.html` is 323 KB with **zero** "
                "occurrences of 'injur' (the formal Injured List ended in 2005); "
                "`/friv/injuries.fcgi` is a 38-row current-status page with no "
                "history, so it carries the same point-in-time prohibition as the ESPN "
                "feed; `*/gamelog/` is `Disallow`ed. It *is* a legitimate dated source "
                "of **roster movement**, which is the right way to disambiguate "
                "`inactive` — a roster-membership source, not an injury one.",
        # `incident`, not `null`: a dated diagnosis of a third party's site, so
        # re-deriving it would mean re-scraping to no purpose. `null` is reserved for
        # nulls that are *measurements* and therefore carry an artifact — the
        # figures-versus-incidents line in docs/provenance-plan.md.
        status="incident",
        source="docs/availability-plan.md",
        reviewed="2026-07-30",
        date="2026-07-28",
        tags=("sources",),
    ),

    # ══ Exploratory data analysis ════════════════════════════════════════════
    Decision(
        id="absorb-season-always",
        topic="eda",
        claim="Always absorb season when regressing across the 30 pooled seasons.",
        because="Pooled, `teammate_spacing` correlates +0.315 with next-season per-36 "
                "points and `team_pace` +0.336; with season fixed effects they are "
                "+0.035 and +0.091. Spacing, pace and scoring all roughly doubled over "
                "the sample, so anything built from them tracks anything rising. **This "
                "has already produced one false finding that reached the README.**",
        status="settled",
        reproduce="make context-value → outputs/eda/team_context_value_tier*.csv",
        source="docs/facts-archive.md",
        reviewed="2026-08-08",
        date="2026-07-27",
        tags=("method",),
    ),
    Decision(
        id="minutes-weight-per36-rates",
        topic="eda",
        claim="Minutes-weight any regression on per-36 rates — it changes the feature "
              "*ranking*, not just the coefficients.",
        because="sd(pts_per36) is 42.4 in sub-5-minute games against 6.6 above 24 "
                "minutes, so unweighted, garbage time dominates and the opponent "
                "signal vanishes under it. Median +0.214 r across the 72 "
                "minutes-weighted Tier A columns: `stl` reads 0.743 weighted against "
                "0.401 unweighted, `tov` 0.790/0.456, `blk` 0.902/0.626. Unweighted, "
                "the low-count defensive stats look like noise when they are among the "
                "stickiest things a player has.",
        status="settled",
        reproduce="make persistence → outputs/eda/persistence.csv",
        source="docs/model-development-notes.md",
        reviewed="2026-08-08",
        date="2026-07-27",
        tags=("method",),
    ),
    Decision(
        id="share-versus-conversion",
        topic="eda",
        claim="Persistence splits on **share vs conversion**, not on 'count vs "
              "percentage'.",
        because="Shot-mix and usage *shares* persist nearly as well as counts — "
                "`sco_pct_fga_3pt` 0.886, `usg_pct_fg3a` 0.870, `adv_ast_pct` 0.845 — "
                "while *conversion* percentages do not: `fg3_pct` 0.500, `fg_pct` "
                "0.435, `ft_pct` 0.365, `efg_pct` 0.281. Shares are first-class "
                "features; only the conversion side gets shrunk. Attempts persist "
                "(`fg3a` 0.908, `fga` 0.862), so spend the model's capacity there.",
        status="settled",
        reproduce="make persistence → outputs/eda/persistence.csv",
        source="docs/model-development-notes.md",
        reviewed="2026-08-08",
        date="2026-07-28",
        tags=("method",),
    ),
    Decision(
        id="whole-families-are-noise",
        topic="eda",
        claim="Whole feature families are noise and must not be fed — and a *pooled* "
              "ranking promotes exactly the columns worth dropping.",
        because="Clutch (`clu_*`, median r 0.25), the rating columns (`def_rating` "
                "0.116, `net_rating` 0.180, `off_rating` 0.285), `plus_minus` 0.477, "
                "`sl_backcourt_*` ≈ 0.01, and every tracking `*_pct` conversion column. "
                "These are also the largest **era gaps** — `adv_e_pace` reads 0.574 "
                "pooled against 0.212 within-season — so ranking on the pooled figure "
                "recommends them.",
        status="settled",
        reproduce="make persistence → outputs/eda/persistence.csv",
        source="docs/model-development-notes.md",
        reviewed="2026-08-08",
        date="2026-07-28",
        tags=("method",),
    ),
    Decision(
        id="reliability-is-per-column",
        topic="eda",
        claim="Reliability is per column, not one curve. The global "
              "`0.924 · m/(m+66)` over-credits the median column.",
        because="`r(m) = r_inf · m/(m+m0)` refitted for every column gives a median m0 "
                "of **364** minutes (p10 112, p90 ≥ 2000) against the global 66, and "
                "minutes-for-r=0.75 spans 192 (`fg3a`) to > 20,000 (clutch), median "
                "1,416. The global curve was fitted on eight *style* stats — the "
                "fast-stabilizing end. Lag decay is column-specific too, so the flat "
                "0.96/season staleness decay is too fast for shares and too slow for "
                "volume.",
        status="measured",
        reproduce="make persistence → outputs/eda/persistence.csv",
        source="docs/model-development-notes.md",
        reviewed="2026-08-08",
        date="2026-07-28",
        tags=("method",),
    ),
    Decision(
        id="archetypes-partition-a-continuum",
        topic="eda",
        claim="Archetypes are a partition of a continuum, not discovered clusters — "
              "use the **soft** membership vector.",
        because="k-means silhouette decreases monotonically in k and peaks at 0.181 "
                "(Tier A) / 0.241 (Tier B) at k=4; no k shows real separation. GMM BIC "
                "picks k=9 / k=6, which is what is fitted. A hard archetype label "
                "claims structure the data does not have.",
        status="settled",
        reproduce="make archetypes → outputs/eda/archetype_sweep_tier*.csv, "
                  "data/features/archetype*.parquet, data/features/kmeans_tier*.pkl, "
                  "data/features/gmm_tier*.pkl, data/features/cluster_meta_tier*.pkl",
        source="docs/model-development-notes.md",
        reviewed="2026-08-08",
        date="2026-07-27",
        tags=("dimension-reduction",),
    ),
    Decision(
        id="two-dr-budgets",
        topic="eda",
        claim="Two dimension-reduction budgets, and they are not the same problem. "
              "Player's own stats: n=10,900, p=150 — barely reduce. Team composition: "
              "n=**892 team-seasons** — reduce hard, 2–4 dims per side.",
        because="`within_season` standardization is era-neutral and is what feeds "
                "archetypes and modeling; `pooled` lets era show up as a visible "
                "trajectory and is for looking, not fitting. Linear DR is also "
                "degenerate for team aggregation — the minutes-weighted mean of PC "
                "scores is *exactly* the PCA projection of the mean stat line, so "
                "averaging and projection commute and all the value lives in the "
                "nonlinear step.",
        status="settled",
        reproduce="make pca → data/features/pca_*",
        source="docs/eda-plan.md",
        reviewed="2026-07-30",
        date="2026-07-27",
        tags=("dimension-reduction",),
    ),
    Decision(
        id="dr-figures-still-prose-only",
        topic="eda",
        claim="Six figures in the PCA / archetype / dimension-reduction line still "
              "have no runnable source.",
        because="kNN leaking identity (18.8% nearest, 37.7% within 5); the mean "
                "centroid collapsing (336 team-season pairs in the closest 1%); the "
                "averaging-equals-projection identity to 3.6e-15; mid-season churn at "
                "78/572 players; 98.4% of minutes rows non-integer; and the global "
                "reliability curve. None is load-bearing for a decision that is still "
                "open — the DR line is not feeding the pipeline, churn is explicitly "
                "out of scope, and `persistence.csv`'s per-column curves already "
                "supersede the global one. Listed so the count is a number rather than "
                "an impression, and so a future sweep starts from a list.",
        status="open",
        source="docs/provenance-plan.md",
        reviewed="2026-07-30",
        date="2026-07-29",
        tags=("provenance",),
    ),
    Decision(
        id="feature-matrix-is-singular",
        topic="eda",
        claim="The raw feature matrix is **singular**, not merely collinear — so any "
              "unpenalized GLM/OLS on it fails outright.",
        because="Tier A rank 144/149, Tier B 277/285, condition number infinite. "
                "Sixteen Tier A columns have exactly infinite VIF because families "
                "ship literal duplicates — `adv_def_rating` == `def_def_rating`, "
                "`usg_pct_blk` == `def_pct_blk` — and others are exact complements "
                "(`sco_pct_fga_2pt` + `sco_pct_fga_3pt` = 1). 104/149 columns have "
                "VIF > 10. Ridge/elastic-net or a pruned set is mandatory.",
        status="settled",
        reproduce="make feature-diagnostics → outputs/eda/feature_diagnostics.csv, "
                  "outputs/eda/feature_correlation_tier*.parquet",
        source="docs/model-development-notes.md",
        reviewed="2026-08-08",
        date="2026-07-28",
        tags=("method",),
    ),
    Decision(
        id="key-team-features-on-team-id",
        topic="eda",
        claim="Key team features on `team_id`, never `team_abbreviation`.",
        because="`team_id` has 30 categories, min 317 rows, no cold start. The "
                "abbreviation has **36**, min 24 rows, with NOK/VAN/CHH/NOH/SEA "
                "spanning as few as 2 seasons — relocations split their own history. "
                "`archetype` (9 / 6) and `draft_bucket` (5) are safe to one-hot.",
        status="settled",
        reproduce="make feature-diagnostics → outputs/eda/feature_diagnostics.csv",
        source="docs/model-development-notes.md",
        reviewed="2026-08-08",
        date="2026-07-28",
        tags=("method",),
    ),
    Decision(
        id="aging-lives-in-availability",
        topic="eda",
        claim="Aging lives in availability, not in rates — put age in the availability "
              "head.",
        because="Delta-method, era-adjusted, minutes-weighted, indexed to age 23: "
                "per-36 DK-linear peaks at 26 (1.041) and is still 0.895 at 34, while "
                "minutes/game peaks at 27 (1.111) then falls to 0.792 at 34 and "
                "**0.515 at 37**. A ±15% rate arc against a −54% availability arc. "
                "Aging is also component-specific: at 34 `reb/36` 0.977 and `blk/36` "
                "1.011 against `pts/36` 0.844 — a single dk_pts age curve averages two "
                "opposite shapes.",
        status="settled",
        reproduce="make aging → outputs/eda/aging_curves.csv",
        source="docs/model-development-notes.md",
        reviewed="2026-08-08",
        date="2026-07-28",
        tags=("method",),
    ),
    Decision(
        id="cross-sectional-age-curves-are-worthless",
        topic="eda",
        claim="Cross-sectional age curves are worthless here — pure survivorship.",
        because="Mean per-36 DK-linear reads 31.1 at 19, 31.9 at 23, 31.6 at 30, 31.4 "
                "at 34 and **34.3 at 39** — flat, then rising, because weak veterans "
                "leave the league. `cross_sectional_mean` sits beside `cumulative` in "
                "the output only to make the gap visible; never build on it.",
        status="null",
        reproduce="make aging → outputs/eda/aging_curves.csv",
        source="docs/model-development-notes.md",
        reviewed="2026-08-08",
        date="2026-07-28",
        tags=("method",),
    ),
    Decision(
        id="teammate-usage-load-carries-the-block",
        topic="eda",
        claim="`teammate_usage_load` is the strongest own-team feature, and it exists "
              "*only* under the corrected roster construction.",
        because="ΔR² = +0.0066 alone; the whole own-team block is +0.0086, of which the "
                "usage family is +0.0080 and crowding/spacing/pace is +0.0000. It is "
                "the minutes-weighted mean of teammates' prior usage × 5. On the "
                "superseded S-1 roster it measures +0.0001. Prefer it to the raw "
                "`teammate_usage_sum`, which grows with roster size on the inclusive "
                "frame.",
        status="measured",
        reproduce="make context-value → outputs/eda/team_context_value_tier*.csv",
        source="docs/facts-archive.md",
        reviewed="2026-08-08",
        date="2026-07-28",
        tags=("features",),
    ),
    Decision(
        id="role-crowding-is-a-settled-null",
        topic="eda",
        claim="`role_crowding` is a settled null. Do not revive archetype-similarity "
              "crowding.",
        because="Retested under the corrected construction: r = +0.048 vs next-season "
                "per-36 pts with season absorbed (+0.020 once P's own prior style is "
                "controlled), −0.005 vs per-game dk_pts, ΔR² = +0.00001. The old "
                "endogeneity explanation — 'prior-season minutes already absorbed it' "
                "— is **falsified**, since crowding on a season-S lineup that did not "
                "exist in S-1 is still a null. Rosters are built to avoid redundancy, "
                "and `teammate_usage_load` measures the same mechanism far better.",
        status="null",
        reproduce="make context-value → outputs/eda/team_context_value_tier*.csv",
        source="docs/facts-archive.md",
        reviewed="2026-08-08",
        date="2026-07-28",
        tags=("features",),
    ),
    Decision(
        id="own-team-features-leave-one-out",
        topic="eda",
        claim="Own-team features must be leave-one-out, and never use same-game "
              "teammate stats.",
        because="Otherwise they encode P's own style rather than his teammates', and "
                "same-game teammate stats are contemporaneous with the outcome. The "
                "roster is derived from a player's first appearance inside his team's "
                "first 10 games: 10 games gives a ~15-man roster covering 96% of "
                "realized team minutes, against 11 players and 84% at 0.",
        status="settled",
        reproduce="make team-context → data/features/team_context_tier*.parquet",
        source="docs/facts-archive.md",
        reviewed="2026-08-08",
        date="2026-07-27",
        tags=("features", "leakage"),
    ),
    Decision(
        id="opponent-profile-from-team-families",
        topic="eda",
        claim="Build the opponent profile from the team families directly — never by "
              "aggregating player rows.",
        because="`team_stats_opponent`'s `OPP_*` columns are *what opponents recorded "
                "against this team* and are already the per-component opponent "
                "profile. Reading them straight off the team families also avoids all "
                "of the roster-coverage bias. Team files break the per-36 rule, "
                "though: counting stats are per game but `MIN` and `POSS` are season "
                "totals, so normalize per-100 possessions with `POSS / GP`.",
        status="settled",
        reproduce="make opponent → data/features/opponent_*, "
                  "outputs/eda/opponent_matchup_tierA.csv",
        source="docs/data-quirks.md",
        reviewed="2026-08-08",
        date="2026-07-28",
        tags=("features",),
    ),
    Decision(
        id="shuffled-null-must-name-its-marginal",
        topic="eda",
        claim="'Above a shuffled null' is null-dependent — always state which marginal "
              "is permuted.",
        because="Same data, same cells: **+0.969%** permuting opponent within season "
                "against **+0.311%** permuting archetype. With ~2,700 cells over 198k "
                "rows the expected chance R² is ~1.4% against a raw statistic of "
                "2.31%, so the headline was largely cell count. The generalized helper "
                "reproduces the one-off it came from to 0.005 pp, so any future "
                "importance ranking ships with its own chance level.",
        status="settled",
        reproduce="make feature-diagnostics → outputs/eda/feature_diagnostics.csv",
        source="docs/facts-archive.md",
        reviewed="2026-08-08",
        date="2026-07-28",
        tags=("method",),
    ),
    Decision(
        id="season-matrix-serves-the-pca",
        topic="eda",
        claim="The season matrix serves the PCA, not roster aggregation — build roster "
              "aggregates on the unfiltered twin.",
        because="Its `GP≥20 & MIN≥10` filter keeps garbage-time per-36 outliers out of "
                "the PCA, but it also drops real teammates who consume real minutes. "
                "`season_matrix_roster_tier*` is the unfiltered twin built for this — "
                "14,569 / 6,942 rows against 10,900 / 5,077 — with reliability "
                "shrinkage applied.",
        status="settled",
        reproduce="make season-matrix → data/features/season_matrix_*.parquet",
        source="docs/data-quirks.md",
        reviewed="2026-08-08",
        date="2026-07-27",
        tags=("method",),
    ),

    # ══ The availability head ════════════════════════════════════════════════
    Decision(
        id="availability-beta-binomial-glm",
        topic="availability",
        claim="The availability head is a beta-binomial GLM, and nothing beats it by a "
              "margin a paired interval can distinguish.",
        because="On validation (2022-23/2023-24), CRPS in games: GBM **9.876**, ridge "
                "10.004, GLM **10.006**, league/age 13.387. Two things confirm the "
                "*distribution* is the working part: the fitted dispersion lands at "
                "20–30× implied overdispersion, independently recovering the ~20× "
                "measured in `availability_profile.csv`, and PIT is near-uniform "
                "(KS 0.07–0.11 against the baseline's 0.15).",
        status="built",
        reproduce="make availability-model → "
                  "outputs/predictions/availability_metrics.csv, "
                  "outputs/predictions/availability_pit.csv, "
                  "outputs/predictions/availability_predictions.csv",
        source="docs/availability-plan.md",
        reviewed="2026-08-08",
        date="2026-07-28",
        tags=("head",),
    ),
    Decision(
        id="gbm-does-not-beat-the-glm",
        topic="availability",
        claim="**Gradient boosting does not beat a 19-feature GLM** — withdrawn as "
              "stated. On validation the GBM leads on mean CRPS.",
        because="The claim was measured on the held-out split, where the GLM read "
                "10.795 against a GBM's 10.888. Converting this head on 2026-08-08 "
                "reversed the ordering: **9.876 against 10.006**. What replaces it is "
                "narrower and holds — nothing beats the GLM *distinguishably*.",
        status="withdrawn",
        replaced_by="**Nothing beats the GLM by a margin a paired interval can "
                    "distinguish.** The GBM's lead is −0.1297 CRPS with a 95% interval "
                    "of [−0.3154, +0.0672], and it is +0.333 *worse* on the "
                    "fewest-games quartile — see `an-ordering-is-not-a-verdict`.",
        caught_by="Converting `src/models/availability.py` to "
                  "`held_out.selection_split` on 2026-08-08 — the last head in the "
                  "project still scoring the test seasons, and the one that took three "
                  "decisions on them.",
        reproduce="make availability-model → "
                  "outputs/predictions/availability_metrics.csv",
        source="docs/model-development-notes.md",
        reviewed="2026-08-08",
        date="2026-08-08",
        tags=("head", "reversal", "held-out-split"),
    ),
    Decision(
        id="an-ordering-is-not-a-verdict",
        topic="availability",
        claim="The GLM keeps the head even though the GBM leads, and the two reasons "
              "are measured rather than argued.",
        because="**One — the margin is not distinguishable from zero.** Paired over the "
                "same 883 validation rows the GBM is **−0.1297** CRPS with a 95% "
                "interval of **[−0.3154, +0.0672]**; the ridge's **−0.0014** is the "
                "same order as the 0.0013-CRPS margin that decided the games-played "
                "Gate D, reversed, and wrote `src/models/held_out.py`. Only the "
                "league/age baseline separates. **Two — the challengers lose where the "
                "head exists to work.** By realized games-played quartile the GBM is "
                "**+0.333** CRPS *worse* on the fewest-games quartile (ridge +0.251) "
                "while winning the other three. A model that is better on average by "
                "being better at ordinary seasons is not the one to ship.",
        status="settled",
        reproduce="make availability-model → "
                  "outputs/predictions/availability_ladder_comparison.csv",
        source="docs/availability-plan.md",
        reviewed="2026-08-08",
        date="2026-08-08",
        tags=("head", "method", "held-out-split"),
    ),
    Decision(
        id="do-not-minutes-weight-availability",
        topic="availability",
        claim="Do not minutes-weight the availability head — the explicit exception to "
              "the rule everywhere else in this project.",
        because="Minutes weighting suppresses per-36 rates measured over a few "
                "garbage-time minutes, which is real measurement error. Games played "
                "has none — 'he played 12 games' is exact — so weighting down-weights "
                "exactly the injured seasons the head exists to predict. It halves the "
                "ceiling (R² 0.236 → 0.116) and flips the MPG-vs-GP ordering. "
                "`availability_profile.csv` reports both columns.",
        status="settled",
        reproduce="make availability-profile → outputs/eda/availability_profile.csv",
        source="docs/availability-plan.md",
        reviewed="2026-07-30",
        date="2026-07-28",
        tags=("method",),
    ),
    Decision(
        id="games-played-least-persistent",
        topic="availability",
        claim="Games played is the least persistent quantity in the project — and "
              "there is no durability latent to extract.",
        because="r = 0.317 season-absorbed and minutes-weighted, against `min` per "
                "game 0.779 and `dk_pts` per game 0.869. GP is ~20× overdispersed "
                "against a binomial, so the head must emit a *distribution*. Two "
                "nulls: a 3-year availability average does not beat 1 year (0.392 vs "
                "0.398), and longest absence spell persists at r = 0.090. Prior MPG "
                "predicts next-season GP about as well as prior GP does.",
        status="measured",
        reproduce="make availability-profile → outputs/eda/availability_profile.csv",
        source="docs/availability-plan.md",
        reviewed="2026-07-30",
        date="2026-07-28",
        tags=("measurement",),
    ),
    Decision(
        id="availability-worth-211-dk-pts",
        topic="availability",
        claim="The head is worth ~211 dk_pts of season-total MAE, and the oracles "
              "settle that availability is the larger half of the remaining error.",
        because="Composing `season_total = gp × dk_per_game_played` while holding a "
                "**fixed** rate model gives −211.1 MAE (−32.7%) against assuming a "
                "full season and −39.9 against carrying prior GP forward, with bias "
                "falling from +541.9 to +6.1. Perfect games played gives 221.3 MAE "
                "against perfect rate's 302.7, so availability carries the larger "
                "share. `oracle_gp` is invariant to the head by construction, which "
                "makes it the check that a GP change moved only what it should.",
        status="built",
        reproduce="make season-total → outputs/predictions/season_total_metrics.csv, "
                  "outputs/predictions/season_total_predictions.csv",
        source="docs/availability-plan.md",
        reviewed="2026-07-30",
        date="2026-07-28",
        tags=("head", "downstream"),
    ),
    Decision(
        id="season-total-r2-column",
        topic="availability",
        claim="The season-total table's R² column reads 0.10 / 0.47 / 0.55 / 0.59 / "
              "0.78 / 0.88 across the six games-played treatments.",
        because="Recorded alongside the MAE, RMSE, bias and CRPS columns, all of "
                "which reproduce exactly.",
        status="withdrawn",
        replaced_by="**0.141 / 0.493 / 0.559 / 0.595 / 0.773 / 0.885.** Recomputing "
                    "R² straight from `season_total_predictions.csv` reproduces "
                    "`season_total_metrics.csv` exactly on all six rows, so the "
                    "artifact is self-verifying and the prose was wrong. No decision "
                    "changes: the head's value rests on MAE, and −211.1 dk_pts "
                    "(−32.7%), the oracle split of 213.8 against 132.5, and the "
                    "rotation-player figures are all exact as recorded.",
        caught_by="Building the dashboard, which reads the artifact live and put the "
                  "two side by side. **The obvious hypothesis — a pre-playoff-workload "
                  "leftover — is falsified**: `full_season`, `prior_gp` and `oracle_gp` "
                  "never touch the availability head, so that change cannot move their "
                  "R², yet `full_season` was the most wrong of the six by 0.041. Since "
                  "equal RMSE pins SSE and `r2 = 1 − SSE/ss_tot`, only `ss_tot` could "
                  "differ — and solving per row gives six mutually inconsistent "
                  "denominators (0.954× to 1.033× the actual). It was hand-typed and "
                  "never recomputed when the MAE side of the table was refreshed.",
        reproduce="make season-total → outputs/predictions/season_total_metrics.csv, "
                  "outputs/predictions/season_total_predictions.csv",
        source="docs/availability-plan.md",
        reviewed="2026-07-30",
        date="2026-07-30",
        tags=("reversal", "provenance"),
    ),
    Decision(
        id="gains-shrink-on-rotation-players",
        topic="availability",
        claim="Gains shrink on established rotation players but the ordering holds — "
              "−23.3% against −32.7% overall.",
        because="651.3 naive → 499.4 with the head, because regulars miss less time. "
                "Do not quote the aggregate figure as if it applied to the players a "
                "DFS user cares about most.",
        status="measured",
        reproduce="make season-total → outputs/predictions/season_total_metrics.csv",
        source="docs/availability-plan.md",
        reviewed="2026-07-30",
        date="2026-07-28",
        tags=("downstream",),
    ),
    Decision(
        id="fit-beta-binomial-with-analytic-gradient",
        topic="availability",
        claim="Fit the beta-binomial with an analytic gradient, and check the "
              "denominator first. Both failures here were silent.",
        because="(1) 13 player-seasons (0.12%) have `gp > team_games` — traded players "
                "whose two teams' schedules overlap — and since the log-likelihood is "
                "a *sum*, those rows make it non-finite at **every** ρ. Use "
                "`n = max(team_games, gp)`. (2) A joint L-BFGS-B fit over 17 "
                "parameters with *numeric* gradients does not converge: the objective "
                "is ~1e5 while a finite-difference step moves it by ~1e-3, so it stops "
                "on gradient noise and returns coefficients near zero — a fitted model "
                "with R² −0.09. Supply the analytic gradient and alternate β with ρ.",
        status="settled",
        reproduce="make availability-model → "
                  "outputs/predictions/availability_metrics.csv",
        source="docs/model-development-notes.md",
        reviewed="2026-08-08",
        date="2026-07-28",
        tags=("failure-mode",),
    ),
    Decision(
        id="absence-reasons-are-not-a-null",
        topic="availability",
        claim="Splitting absences by reason is **not** the null the plan expected, and "
              "the carrying reasons invert the intuition.",
        because="On 7,673 season pairs: prior `gp_share` alone R² 0.2353 → + "
                "`missed_games` 0.2365 → + the 8-way reason split **0.2648**, against "
                "a shuffled-row null of 0.2363 (sd 0.0004) — **+0.0285 above chance, "
                "≈69 sd**. The aggregate is worth only +0.0012, so the split is "
                "essentially the entire effect. `missed_scratch` −0.353 and "
                "`missed_inactive` −0.233 against next-season `gp_share`, while "
                "**`missed_injury` is +0.034**: the *rotation* reasons predict "
                "availability and the injury reason does not. `missed_scratch` "
                "persists at 0.469, higher than `gp_share` itself.",
        status="measured",
        reproduce="make availability-profile → outputs/eda/availability_profile.csv",
        source="docs/availability-plan.md",
        reviewed="2026-07-30",
        date="2026-07-28",
        tags=("features",),
    ),
    Decision(
        id="feature-cols-consumes-no-reason-column",
        topic="availability",
        claim="`models.availability.FEATURE_COLS` does not consume any absence-reason "
              "column yet.",
        because="The one evidence-backed feature change outstanding on the head: the "
                "reason split is worth +0.0285 above a shuffled null in sample and "
                "nothing reads it.",
        status="open",
        source="docs/availability-plan.md",
        reviewed="2026-07-30",
        date="2026-07-28",
        tags=("next",),
    ),
    Decision(
        id="playoff-workload-is-selection-not-fatigue",
        topic="availability",
        claim="Playoff workload earns its place on the head — but by **selection**, "
              "not fatigue. Every sign is the opposite of the mechanism it was built "
              "for.",
        because="Validation CRPS 10.104 → **10.006** for four features. But every "
                "single-season playoff column predicts "
                "*better* next-season availability: `playoff_mpg` r = +0.282 raw, "
                "+0.055 controlled. Playoff participation marks a good player on a "
                "good team, and that selection effect beats fatigue outright. The only "
                "column pointing the way fatigue would is **`career_minutes` at "
                "−0.067** — cumulative mileage. The block was adopted on the *test* "
                "split (10.914 → 10.795) and re-decided on validation 2026-08-08: "
                "same sign, same ordering of all four variants, decision unchanged — "
                "the contrast with the model ladder, which reversed.",
        status="measured",
        reproduce="make availability-model → "
                  "outputs/predictions/availability_workload_ablation.csv",
        source="docs/availability-plan.md",
        reviewed="2026-08-08",
        date="2026-07-29",
        tags=("features", "held-out-split"),
    ),
    Decision(
        id="total-minutes-incl-playoffs-is-a-null",
        topic="availability",
        claim="`total_minutes_incl_playoffs` is a measured null and is deliberately "
              "**not** a feature.",
        because="It was built on the true observation that `total_minutes` undercounts "
                "real mileage by 10.8% for playoff players. Swapping it in *lowers* "
                "in-sample R² (0.2843 → 0.2816), because folding playoff minutes into "
                "the total mixes team quality into a clean regular-season workload "
                "measure. Keep the two effects in separate columns — pinned by a test "
                "so it does not get 'fixed' back in.",
        status="null",
        reproduce="pytest tests/test_availability_model.py → "
                  "tests/test_availability_model.py",
        source="docs/availability-plan.md",
        reviewed="2026-07-30",
        date="2026-07-29",
        tags=("features", "regression-guard"),
    ),
    Decision(
        id="select-on-validation-not-test",
        topic="availability",
        claim="Select on validation, and do not print a test column beside it. A "
              "*paired* bootstrap does not rescue a test-set selection.",
        because="The test column preferred every curved variant over the linear one "
                "and **none of them replicated**. A paired bootstrap on those 911 "
                "rows put the quadratic gain at −0.047, 95% CI [−0.079, −0.015], "
                "P(Δ<0) = 99.7% — and it was still a false positive, because a paired "
                "interval says a difference is consistent *within one sample*, not "
                "that the sample was representative. This ablation always selected on "
                "validation; since 2026-08-08 it no longer computes the test column at "
                "all, and every validation figure reproduced to five decimals on the "
                "move — a determinism check on the conversion, not evidence for the "
                "verdict.",
        status="settled",
        reproduce="make availability-model → "
                  "outputs/predictions/availability_nonlinearity.csv",
        source="docs/availability-plan.md",
        reviewed="2026-08-08",
        date="2026-07-29",
        tags=("method", "reversal-adjacent", "held-out-split"),
    ),
    Decision(
        id="clustering-is-not-the-overdispersion",
        topic="availability",
        claim="Availability is strongly autocorrelated, but that is **not** where the "
              "overdispersion comes from — and the simple Markov chain is falsified.",
        because="`P(play|played) = 0.905`, `P(play|missed) = 0.308`, so lag-1 ρ = 0.597 "
                "and a 2-state chain inflates variance (1+ρ)/(1−ρ) = **3.96×** against "
                "the **22.7×** measured. Clustering is ~a sixth of it; the rest is "
                "between-player heterogeneity, which no AR process can generate. And a "
                "constant hazard implies **geometric** spells, which matches the mean "
                "(3.25) and misses both tails — 0.483 of spells are 1 game against "
                "0.308 predicted, 0.0635 are 10+ against 0.0365. Use a 2-component or "
                "semi-Markov process, for the season-total joint distribution and the "
                "preseason initial state, not for GP CRPS.",
        status="measured",
        reproduce="make availability-profile → outputs/eda/availability_profile.csv",
        source="docs/availability-plan.md",
        reviewed="2026-07-30",
        date="2026-07-28",
        tags=("simulator-input",),
    ),
    Decision(
        id="stan-availability-port-verified",
        topic="availability",
        claim="The Stan port reproduces the MLE, and that is a **defined check** "
              "rather than a hopeful comparison.",
        because="An L2 penalty of `l2` on standardized coefficients *is* a "
                "`normal(0, 1/sqrt(2·l2))` prior, so the posterior **mode** is exactly "
                "the penalized optimum. Measured: held-out CRPS 10.7947 (plug-in) / "
                "10.7953 (posterior) against the MLE's 10.7952, max coefficient gap "
                "0.0127, largest gap 0.095 posterior sd, and the MLE inside the 95% "
                "credible interval for **21/21** terms. R̂ 1.0025, min ESS 2,402, "
                "**0 divergences**.",
        status="built",
        reproduce="make stan-availability → outputs/predictions/stan_availability_*",
        source="docs/predictions-plan.md",
        reviewed="2026-07-30",
        date="2026-07-29",
        tags=("head", "verification"),
    ),
    Decision(
        id="board-correlation-scales-as-sqrt-n",
        topic="availability",
        claim="The posterior's shared-β term is worth almost nothing on one roster, "
              "and the **size of the portfolio** is what decides.",
        because="What the posterior buys is `Var_θ(Σ E[Y|θ])`: every player shares β, "
                "so one draw moves the whole board together, and that term is exactly "
                "0 under any point estimate. But the independent term grows as √N and "
                "the shared-β term as N, so their ratio scales as √N — inflation is "
                "**+0.2% at 12 players** and **+6.4% across all 911**. Quoting the "
                "full-board figure for a 15-man roster is the over-claim to avoid; it "
                "was made and corrected in the session that built this.",
        status="measured",
        reproduce="make stan-availability → "
                  "outputs/predictions/stan_availability_board.csv",
        source="docs/predictions-plan.md",
        reviewed="2026-07-30",
        date="2026-07-29",
        tags=("verification",),
    ),
    Decision(
        id="integrated-predictive-is-not-wider",
        topic="availability",
        claim="The integrated predictive is **not** necessarily wider per player, and "
              "expecting it to be is a trap.",
        because="By the law of total variance the mixture adds `Var_θ(E[Y|θ])` but "
                "replaces `Var(Y|θ̄)` with `E_θ[Var(Y|θ)]`, and `n·μ(1−μ)·[1+(n−1)ρ]` "
                "is **concave in μ**, so Jensen pushes the other way. Measured: +0.046 "
                "against −0.082 games², i.e. the mixture is marginally *narrower*, "
                "verified against the pmf to 1.3e-10. The marginal width is a red "
                "herring; the covariance is not.",
        status="measured",
        reproduce="make stan-availability → "
                  "outputs/predictions/stan_availability_metrics.csv",
        source="docs/predictions-plan.md",
        reviewed="2026-07-30",
        date="2026-07-29",
        tags=("verification",),
    ),
    Decision(
        id="stan-numerical-traps",
        topic="availability",
        claim="Two numerical facts about the Stan heads, both of which cost a run.",
        because="(1) `1 - inv_logit(eta)` is exactly 0 in double precision by "
                "`eta ≈ 37`, making a beta shape parameter 0 and rejecting the whole "
                "target; `s * inv_logit(-eta)` is algebraically identical and survives "
                "to `eta ≈ 745`. (2) **Always pass `inits`** — Stan's default "
                "uniform(−2, 2) on the unconstrained scale puts the starting linear "
                "predictor near `2·sqrt(K)`, which at K = 19 is already in the "
                "saturation region. Every head inits at the intercept-only solution "
                "with zero slopes. `n = max(team_games, gp)` matters *more* under HMC, "
                "not less: a non-finite target poisons the trajectory rather than "
                "merely stopping an optimizer.",
        status="settled",
        reproduce="make stan-availability → "
                  "outputs/predictions/stan_availability_diagnostics.csv",
        source="docs/predictions-plan.md",
        reviewed="2026-07-30",
        date="2026-07-29",
        tags=("failure-mode",),
    ),
    Decision(
        id="designation-scale-is-not-monotone",
        topic="availability",
        claim="The injury-report designation scale is **not monotone** — treat "
              "Probable and Available as one designation and condition on reason.",
        because="P(play): Out 0.002, Doubtful 0.030, Questionable 0.498, Probable "
                "0.914, **Available 0.855** — `Available` plays *less* than "
                "`Probable`. That inversion is reason mix, not label noise: excluding "
                "G-League rows the scale reads 0.001 / 0.016 / 0.559 / 0.920 / 0.903 "
                "and the residual sits inside a ~1.6 pp standard error. 12,338 of "
                "12,406 archive rows joined to a realized outcome, 0.0% unmatched.",
        status="measured",
        reproduce="make report-calibration → outputs/eda/report_calibration.csv",
        source="docs/availability-plan.md",
        reviewed="2026-07-30",
        date="2026-07-29",
        tags=("features",),
    ),
    Decision(
        id="out-is-sticky-questionable-resolves",
        topic="availability",
        claim="`Out` is near-deterministic and sticky; `Questionable` is a coin flip "
              "that resolves — and a stale Questionable is worth about what a fresh "
              "one is.",
        because="Out → 89.8% inactive / 9.4% dnp / 0.2% played, and **98.5% of Out "
                "designations are unchanged** in the next day's report. Questionable is "
                "unchanged only 38.9% of the time, but reads p_play 0.475 at lead 1 "
                "against 0.498 pooled — the encouraging read for a preseason snapshot, "
                "which is taken weeks ahead. Minutes barely move: a Questionable who "
                "plays gets 23.5 against a Probable's 24.7, so there is **no "
                "meaningful minutes haircut** to model.",
        status="measured",
        reproduce="make report-calibration → outputs/eda/report_calibration.csv",
        source="docs/availability-plan.md",
        reviewed="2026-07-30",
        date="2026-07-29",
        tags=("features",),
    ),
    Decision(
        id="reason-disambiguates-inactive",
        topic="availability",
        claim="The stated reason disambiguates `inactive` directly — no "
              "Basketball-Reference scrape needed for the preseason snapshot.",
        because="The plan wanted BBRef transaction logs to separate 'unavailable "
                "because hurt' from 'unavailable because not on the team'. The PDFs "
                "state it: `Out|Not With Team` is **12.5% absent** from the box score "
                "and `Out|Trade Pending` **14.3%**, against 0.2% for "
                "`Out|Injury/Illness`. Roster mechanics are the only reasons that "
                "generate `absent` at all. Forward-only — the archive starts "
                "2025-12-29 — so BBRef remains the only option for history.",
        status="measured",
        reproduce="make report-calibration → outputs/eda/report_calibration.csv, "
                  "data/features/report_transfer.parquet",
        source="docs/availability-plan.md",
        reviewed="2026-07-30",
        date="2026-07-29",
        tags=("sources",),
    ),
    Decision(
        id="preseason-snapshot-ablation-blocked",
        topic="availability",
        claim="The preseason-snapshot ablation cannot be run yet.",
        because="The injury-report archive is forward-only and starts 2025-12-29, so "
                "there is not yet a preseason snapshot on one side of a season "
                "boundary and a realized games-played total on the other. The transfer "
                "function is already measured; what is missing is a second season.",
        status="blocked",
        unblocks="The archive crossing a season boundary — an October 2026 snapshot "
                 "plus the realized 2026-27 season.",
        source="docs/availability-plan.md",
        reviewed="2026-07-30",
        date="2026-07-29",
        tags=("next",),
    ),

    # ══ The minutes head ═════════════════════════════════════════════════════
    Decision(
        id="minutes-trials-are-game-length",
        topic="minutes",
        claim="The trials denominator for `min` is the real game length, never 48.",
        because="Five players are on court at every moment, so a team's summed minutes "
                "are exactly 5 × game length, and **the two teams are two independent "
                "estimates of the same quantity — that is the validation**: 0 "
                "disagreements over all 37,986 games, worst rounding residual 0.617 "
                "min against a 2.5 min decision boundary. 5.93% of games go to "
                "overtime, so truncating at 48 discards ~6% of games and censors the "
                "top of the minutes distribution exactly where stars play most. The "
                "join is feasible on 100.0% of 731,906 player-games with **zero** rows "
                "where `min > game_length`. Build it from the raw logs, never from the "
                "filtered `component_targets.parquet`.",
        status="built",
        reproduce="make game-length → outputs/eda/game_length_coverage.csv, "
                  "data/features/game_length.parquet",
        source="docs/predictions-plan.md",
        reviewed="2026-07-30",
        date="2026-07-29",
        tags=("head",),
    ),
    Decision(
        id="minutes-spec-is-the-opposite-of-the-counts",
        topic="minutes",
        claim="For minutes the logit scale is a dead wash and **curvature** is what "
              "pays — the opposite of the count heads, and the two answers must not be "
              "pooled into one rule.",
        because="`logit(own)` reads validation R² 0.8819 against linear's 0.8826 and is "
                "worse on CRPS too (145.45 against 144.71), while `logit(own) + spline` "
                "is selected at 143.93 CRPS and 0.8835 R², clearing the no-fit floor by "
                "+0.0299 R² and −17.5 minutes. The selected variant survived the move to "
                "validation-only scoring *and* the raising of selection to full-length "
                "chains, which few selections in this project have — contrast the "
                "season-term ablation, where 9 of 13 heads flipped their arm.",
        status="built",
        reproduce="make stan-minutes → outputs/predictions/stan_minutes_metrics.csv, "
                  "outputs/predictions/stan_minutes_diagnostics.csv",
        source="docs/predictions-plan.md",
        reviewed="2026-08-06",
        date="2026-07-29",
        tags=("head",),
    ),
    Decision(
        id="minutes-curvature-is-a-floor-not-a-ceiling",
        topic="minutes",
        claim="The nonlinearity that pays on minutes is a **floor at the bottom** of "
              "the prior-MPG range, not a ceiling at the top — and it is *not* the age "
              "arc.",
        because="Splining one column at a time: only `minutes_per_game_lag1` moves it "
                "materially (**+0.0124** validation R²), with `total_minutes_lag1` "
                "marginal at +0.0008 and everything else at or below zero. A spline on "
                "**`age` is actively worse** (−0.0008), because `age + age_sq` already "
                "absorbs the arc. Mean "
                "next-season MPG runs 4.3 → 10.5 (+6.1) at the bottom against a roughly "
                "parallel −2.0 decline from 26 mpg up. Part of that is survivorship. "
                "The probe carried a test column and a `replicates` flag until "
                "2026-08-08; both are withdrawn, because those two columns differed in "
                "training data as well as scored rows and so could not certify a "
                "replication. The contrast that carries the finding is between the two "
                "**targets** on one frame — a null for games played, real for minutes.",
        status="measured",
        reproduce="make availability-model → "
                  "outputs/predictions/availability_minutes_nonlinearity.csv",
        source="docs/availability-plan.md",
        reviewed="2026-08-08",
        date="2026-07-29",
        tags=("features", "held-out-split"),
    ),
    Decision(
        id="three-minutes-numbers-compose",
        topic="minutes",
        claim="The simulator needs **three** minutes numbers and they compose, they do "
              "not substitute. The season-level ρ is not one of the two the simulator "
              "draws with.",
        because="(1) the season-level mean from the head, fitted ρ = 0.05025; (2) the "
                "**game-level** dispersion, ρ = 0.0776 or **4.65× binomial** at a "
                "48-minute game, for the marginal spread of a single game; (3) the "
                "**2.43× ten-game block variance inflation** for serial dependence "
                "*between* games. A season total cannot separate a per-game random "
                "effect from a per-season one — iid game noise is diluted by ~1/G while "
                "a shared season multiplier passes through in full — so drawing "
                "per-game minutes from the season-level ρ would make every simulated "
                "game far too close to the player's average.",
        status="measured",
        reproduce="make stan-minutes → "
                  "outputs/predictions/stan_minutes_dispersion.csv",
        source="docs/predictions-plan.md",
        reviewed="2026-08-06",
        date="2026-07-29",
        tags=("simulator-input",),
    ),
    Decision(
        id="minutes-head-held-out-bias",
        topic="minutes",
        claim="The fitted minutes heads carry a −33 to −41 minute held-out bias against "
              "the floor's −5.7, because the floor is unbiased and shrinkage is not.",
        because="Measured on the held-out seasons only, and the second half of it is "
                "false. Re-scored on validation the **floor** is the biased arm at "
                "+23.91 minutes while the fitted arms run −3.26 to −14.00, so 'the "
                "floor is unbiased because it does not shrink' does not survive a change "
                "of seasons. The signed level was a property of which seasons were "
                "scored, not of the head.",
        status="withdrawn",
        replaced_by="minutes-shrinkage-gap-below-the-floor",
        caught_by="Re-running `make stan-minutes` on validation under "
                  "`src/models/held_out.py` (2026-08-06), which added a `val_bias` "
                  "column the artifact had never carried.",
        reproduce="make stan-minutes → outputs/predictions/stan_minutes_metrics.csv",
        source="docs/availability-plan.md",
        reviewed="2026-08-06",
        date="2026-07-29",
        tags=("defect",),
    ),
    Decision(
        id="minutes-shrinkage-gap-below-the-floor",
        topic="minutes",
        claim="The minutes head's calibration defect is **relative, not signed**: every "
              "fitted arm sits 27–38 minutes *below* the no-fit floor, and that gap is "
              "what reproduces across splits.",
        because="On validation the fitted arms run −27.2 (linear) to −37.9 (spline) "
                "minutes below the carry-forward floor, against −27.1 to −35.3 on the "
                "retired held-out column — while the absolute levels moved by ~29 "
                "minutes across the same change. Shrinkage moves every arm about the "
                "same distance down from the floor; where that lands depends on the "
                "seasons scored. It still compounds through the eleven component heads "
                "that take these minutes as exposure, so it is still worth correcting — "
                "but a signed level must not be quoted for it, and the two columns are "
                "not a controlled contrast in any case (different rows, different "
                "training data, different chain lengths).",
        status="open",
        source="docs/availability-plan.md",
        reviewed="2026-08-06",
        date="2026-08-06",
        tags=("next", "defect"),
    ),
    Decision(
        id="composition-beats-independent-minutes",
        topic="minutes",
        claim="The team-game **composition** — a multinomial decomposed into sequential "
              "binomial trials — beats the independent per-player minutes draw on its "
              "own marginal metric, *and* makes the team total exact by construction.",
        because="Fitted on all 30 seasons and scored on validation (2022-23/23-24, 52,295 "
                "player-rows / 4,920 team-games), the selected variant scores **4.4945** "
                "minutes of CRPS against the no-fit floor's 4.6776 (−0.1832) and the "
                "incumbent independent draw's **4.7842** (−0.2898, −6.06%) — not the "
                "expected wash. On top of that the incumbent misses the team's "
                "`5 × game_length` total by **33.89** minutes per team-game on average "
                "where the composition is exact on every draw of every game. Zero-sum is "
                "what makes teammate-absence redistribution a *fitted* quantity rather "
                "than a hand-set rule. **Gate E taken 2026-08-04**: the head is now in "
                "`make stan`. The `independent_comparator` row is the control — it never "
                "trains on the composition window and reproduced to six decimals, which is "
                "what makes the rest readable as a window effect. (The pilot read −0.406.) "
                "✅ **Re-measured on validation 2026-08-08**, closing the last "
                "code/artifact disagreement in the repo: the module went validation-only on "
                "2026-08-05 and the artifact was regenerated three days later. **Nothing "
                "reversed** — same selected arm, same ordering, and no arm moved more than "
                "0.0019 CRPS across a doubling of chain length. The retired test column "
                "(4.5592 selected against 4.9140, −0.3548 / −7.2%, 36.87 minutes of "
                "team-sum error) is preserved in `docs/minutes-composition-plan.md` as "
                "presence-checked historical claims.",
        status="built",
        reproduce="make stan-composition → "
                  "outputs/predictions/stan_composition_metrics.csv, "
                  "outputs/predictions/stan_composition_ppc.csv",
        source="docs/minutes-composition-plan.md",
        reviewed="2026-08-06",
        date="2026-07-31",
        tags=("simulator-input",),
    ),
    Decision(
        id="composition-needs-dispersion-not-just-decomposition",
        topic="minutes",
        claim="The **pure** stick-breaking decomposition is worse than the no-fit floor. "
              "The dispersion is the model, not a refinement.",
        because="The `binomial` arm — the demo's model with the cap fixed — scores "
                "**4.9388** validation CRPS against the floor's 4.6776, with PIT KS "
                "**0.1919** against 0.0496: far too tight, exactly as the measured "
                "game-level ρ (4.65× binomial) predicted. The beta-binomial arm at "
                "4.5417 clears the floor comfortably. Same shape as the NB-vs-Poisson "
                "finding on the count heads — the likelihood family decides whether "
                "there is a model at all. ✅ **The PIT failure gained a validation twin "
                "on 2026-08-08**: the pre-lock artifact wrote `test_pit_ks` and no "
                "`val_pit_ks`, so the sharpest statement of this arm's failure used to be "
                "a held-out number (0.1948 against 0.0354). It now reproduces on the split "
                "that selects.",
        status="measured",
        reproduce="make stan-composition → "
                  "outputs/predictions/stan_composition_metrics.csv",
        source="docs/minutes-composition-plan.md",
        reviewed="2026-08-08",
        date="2026-07-31",
        tags=("specification",),
    ),
    Decision(
        id="composition-shared-rho-is-role-graded",
        topic="minutes",
        claim="The allocation dispersion is **role-graded**: ρ per prior-share quartile "
              "runs 0.1768 fringe to 0.0855 star, a 2.07× spread that one shared ρ of "
              "0.1211 was splitting the difference on.",
        because="A 34-mpg starter's allocation step is genuinely steadier than a "
                "reserve's, and the direction was predicted from the shared-ρ arm's "
                "variance ratios (1.22 fringe to 0.63 star) before it was fitted. "
                "`betabinom_ot_graded` differs from its twin in the **dispersion "
                "alone** — same features, same mean function — so the contrast is "
                "clean. Worth −0.049 validation CRPS. **The calibration fix is the "
                "point**: mean |variance ratio − 1| falls **0.2730 → 0.1782**, a 35% cut. "
                "⚠️ The pilot reported 0.2571 → 0.1055 (59%) with the star tier at 0.9880; "
                "at full window the star tier lands at 0.8136 and three of four tiers sit "
                "below 1, so the head is mildly over-dispersed in aggregate. `n_rho = 1` "
                "is the shared model exactly, so the "
                "graded arm strictly generalizes it; bin edges come from **train** "
                "quantiles only, since leakage in ρ never touches the mean and would "
                "be invisible. (The pre-lock test-split artifact read 0.1751 / 0.0839, a "
                "2.09× spread against a shared 0.1195, and a 0.2928 → 0.1796 / 39% cut; "
                "every ρ rose ~0.0016 at full chain length and the ordering is untouched.)",
        status="built",
        reproduce="make stan-composition → "
                  "outputs/predictions/stan_composition_dispersion.csv, "
                  "outputs/predictions/stan_composition_ppc.csv",
        source="docs/minutes-composition-plan.md",
        reviewed="2026-08-08",
        date="2026-07-31",
        tags=("specification",),
    ),
    Decision(
        id="step-dispersion-is-not-marginal-variance",
        topic="minutes",
        claim="Grading a **step** dispersion does not map one-to-one onto **marginal** "
              "variance by tier — q2's calibration got *worse* while the two extremes "
              "got much better.",
        because="Realized/simulated variance ratio by tier went 1.2224 / 0.8430 / "
                "0.6589 / 0.6285 shared to 1.0465 / **0.8132** / 0.7071 / 0.8136 "
                "graded: three of four tiers improved, and q2 was pushed further off a "
                "mark it happened to hit. The fitted ρ "
                "is the dispersion of a sequential step, while the ratio is measured "
                "on a player's marginal minutes — and because the order is prior-share "
                "*descending*, a low-share player breaks his stick last and inherits "
                "the accumulated remainder variation from everyone ahead of him. So a "
                "finer binning will not straightforwardly fix the fringe tier; the "
                "entanglement is between sequence position and dispersion.",
        status="measured",
        reproduce="make stan-composition → outputs/predictions/stan_composition_ppc.csv",
        source="docs/minutes-composition-plan.md",
        reviewed="2026-08-04",
        date="2026-07-31",
        tags=("next", "methodology"),
    ),
    Decision(
        id="composition-joint-nll-is-not-a-bijection",
        topic="minutes",
        claim="The composition's joint-NLL win over independent draws is **not** the "
              "same kind of comparison as the 3PA/2PA reparameterization's, and must "
              "not be quoted as if it were.",
        because="Composition **32.862** against independent **37.984** per team-game on "
                "validation — but the map is not a bijection with unit Jacobian. "
                "The composition's last step is deterministic, so it concentrates all "
                "its mass on the simplex slice the data always satisfy and wins partly "
                "by *knowing the constraint* rather than by fitting better. `fga` × "
                "`fg3a|fga` was legitimate because `(fg2a, fg3a) ↔ (fga, fg3a)` is the "
                "same point in different coordinates. Here the decision metrics are "
                "CRPS, the team-sum error and the PPCs; the joint NLL is contrast only.",
        status="measured",
        reproduce="make stan-composition → "
                  "outputs/predictions/stan_composition_joint_nll.csv",
        source="docs/minutes-composition-plan.md",
        reviewed="2026-08-04",
        date="2026-07-31",
        tags=("methodology",),
    ),
    Decision(
        id="game-length-is-a-two-parameter-geometric-tail",
        topic="minutes",
        claim="Game length needs exactly **two** parameters — P(any OT) and a constant "
              "continuation probability — and it covers 3OT/4OT for free.",
        because="Fitted on 30,626 regular-season training games: p_any = **0.0608**, "
                "p_more = **0.1408**. The continuation probability is near-constant in "
                "depth over the full sample (1,942 → 264 → 41 → 6 games at 1/2/3/4 OT), "
                "which is what makes the geometric form enough. On validation it predicts "
                "**128.4** single-OT games against **120** observed — the shape holds "
                "but it overpredicts OT by ~7% on recent seasons, which is one more "
                "entry for the season-effects ledger rather than a defect in the form. "
                "A covariate model is not worth it at a 6% base rate, and game "
                "closeness is not knowable preseason. The fitted parameters are invariant "
                "to the split by construction — `fit_ot_tail` receives 1996-97 → 2021-22 "
                "either way — and reproduced to six decimals on the 2026-08-08 refit; the "
                "retired test reading was 256.9 against 222 on twice as many team-games.",
        status="built",
        reproduce="make stan-composition → "
                  "outputs/predictions/stan_composition_ot_tail.csv",
        source="docs/minutes-composition-plan.md",
        reviewed="2026-08-08",
        date="2026-07-31",
        tags=("simulator-input",),
    ),
    Decision(
        id="composition-sampling-cost-is-four-specific-traps",
        topic="minutes",
        claim="The composition head's sampling cost was four separate numerical traps, "
              "not one — the probe went from **60+ minutes without a draw** to 65 s.",
        because="(1) A hard clip on the carry-forward offset put near-zero beta shape "
                "parameters on ~1% of rows where proportional carry-forward exceeds the "
                "cap; saturating at 0.93 with an `offset_clipped` indicator fixes it. "
                "(2) `beta_binomial_lccdf` routes through `grad_F32`: `optimize(iter=30)` "
                "took >600 s with it and 0.8 s with a pmf-ratio recurrence. (3) Summing "
                "that recurrence as `1 − head` gives `log1m(1) = −inf` on a tiny tail "
                "and killed every chain at init; sum the **tail upward in log space**. "
                "(4) `metric=\"dense_e\"` drops treedepth 8–9 → 4, cutting the probe "
                "645 s → 65 s. Also: per-column imputation flags are exact duplicates "
                "here, because a rookie loses the whole design block at once.",
        status="measured",
        reproduce="make stan-composition → "
                  "outputs/predictions/stan_composition_diagnostics.csv",
        source="docs/minutes-composition-plan.md",
        reviewed="2026-08-04",
        date="2026-07-31",
        tags=("performance", "failure-mode"),
    ),
    Decision(
        id="bspline-bases-are-ill-conditioned-for-hmc",
        topic="minutes",
        claim="B-spline bases are badly conditioned for HMC — valid, just expensive.",
        because="The spline variants sample at treedepth 8 (255 leapfrog steps per "
                "iteration) with a step size of 0.011, against treedepth 3–4 for the "
                "linear ones — **721 s against 171 s** for the same data on the minutes "
                "head. An orthogonalized (QR-whitened) basis is the fix if spline "
                "variants ever become the shipped spec.",
        status="measured",
        reproduce="make stan-minutes → "
                  "outputs/predictions/stan_minutes_diagnostics.csv",
        source="docs/predictions-plan.md",
        reviewed="2026-08-06",
        date="2026-07-29",
        tags=("performance",),
    ),
    Decision(
        id="no-head-carries-a-season-term",
        topic="components",
        claim="No head carries a season term, and the league moves — the two usable "
              "forms are a year-on-year **trend** and a year-level **random effect**, "
              "and they are complementary rather than alternatives.",
        because="A season *fixed* effect does not exist at prediction time. "
                "Subtracting a linear trend shifts the **mean** of the year-over-year "
                "changes and leaves their **variance** exactly unchanged, since "
                "`diff(a + b·x)` is the constant `b` — so a trend fixes bias and only a "
                "year effect addresses spread. The three-point **mix** (`fg3a_pct`) is "
                "the only quantity worth extrapolating a trend for (R² 0.93 at "
                "+3.58%/season); the shot-attempt basis decomposed the retired `fg3a` "
                "count series (R² 0.93 at +4.07%/season) into +0.47%/season of total "
                "volume and +3.58%/season of mix, so the revolution is almost entirely "
                "*which* shots are taken. Everything else "
                "is shock, `stl` most starkly at trend R² 0.03. `fta` is the sharpest "
                "case and it is refereeing — a 1.21× band, 4.4% yoy sd, past ±5% in 9 "
                "of 29 transitions. The cost is measured on the no-fit floor, which "
                "lags any league move by exactly one season — and since 2026-08-08 the "
                "lag is **checked** rather than asserted: on the validation seasons a "
                "component's bias carries the opposite sign to that season's league "
                "move in **13 of 14** cells, correlating at **−0.944**. `fta` runs "
                "−4.4% into the league's +7.3% rise and +10.7% into its −7.5% fall. "
                "⚠️ The retired held-out reading quoted `fta` −7.0% (−10.7% in 2025-26) "
                "and `blk` +6.2% in *both* seasons as 'drift, not noise'; on validation "
                "`fta` pools to +2.9% and **`blk` reverses sign** (+4.3% → −5.4%), "
                "against a trend R² of 0.118. Never quote the pooled column alone — it "
                "reports a lag as a level. This "
                "outranks the shared-β correlation the Stan work was built for, "
                "because a league shift is perfectly correlated across every player "
                "and so does not diversify — against +0.2% for shared-β on a 15-man "
                "roster. **The ablation has now run and the decision is recorded "
                "separately** — see `season-term-ablation`, `no-trend-on-any-head` and "
                "`year-effect-is-a-simulator-input`.",
        status="measured",
        reproduce="make season-effects → outputs/eda/season_effects_summary.csv, "
                  "outputs/eda/season_effects_league_rates.csv, "
                  "outputs/eda/season_effects_carry_forward_bias.csv",
        source="docs/predictions-plan.md",
        reviewed="2026-08-08",
        date="2026-07-30",
        tags=("method", "era"),
    ),
    Decision(
        id="no-trend-on-any-head",
        topic="components",
        claim="No head carries a year-on-year trend — its apparent win on season-total "
              "dk_pts is cross-component cancellation, confirmed on validation.",
        because="On the 773-row validation frame an all-trend composition scores MAE "
                "**105.76** against base's **105.71** — it does not win at all — while "
                "flipping bias from **−16.44** to **+10.21**. The component biases behind "
                "it move in both directions and cancel in the DK sum, the same mechanism "
                "recorded at 8.30× on `teammate_assist_supply`. ⭐ This RESTORES the "
                "original reading after the held-out column briefly undermined it: on test "
                "the same table read trend 106.06 / −13.57 against base 108.56 / −36.89, "
                "i.e. a trend improving *both* MAE and bias, which the cancellation story "
                "could not explain and which `CLAUDE.md` flagged as needing a human "
                "decision. On the split that decides, the anomaly is gone and no decision "
                "is owed. The sharpest single refutation remains the **retired** `fg3a` "
                "head — trend R² 0.93 at +4.07%/season in the league series, yet it "
                "selected `base` (val CRPS 33.247 against trend 35.443) while a trend "
                "flipped its bias from −3.74% to +8.44% — because the three-point climb "
                "decelerated to +1.61%/season over the last six seasons and the head's "
                "`log(fg3a_p36_lag1)` feature already carries the league level forward, so "
                "a trend adds a second correction on top of one already there. That head "
                "is no longer fitted; its successor `fg3a|fga` selects `year` on a 0.046% "
                "margin, which is noise.",
        status="settled",
        reproduce="make season-terms → outputs/predictions/season_term_metrics.csv, "
                  "outputs/predictions/season_term_season_total.csv",
        source="docs/predictions-plan.md",
        reviewed="2026-08-08",
        date="2026-07-31",
        tags=("era", "method"),
    ),
    Decision(
        id="season-term-ablation",
        topic="components",
        claim="A perfect league-level override is worth at most **~5% of MAE**, which "
              "bounds every form of season term — fitted or manual.",
        because="`oracle_league` rescales each scored season by its own realized total: "
                "a perfect per-season league multiplier, the most general form any "
                "league-level term can take, and unusable as a model because it reads the "
                "season it forecasts. On validation it is worth `fta` **4.97%**, `reb` "
                "2.58%, `tov` **2.36%**, `ast` 1.29%, `blk` **1.17%**, `stl` 0.36%, `fga` "
                "−0.00% — median **1.29%**. `fga` is negative because a perfect rescale "
                "can cost a fraction on a head whose league level barely moves, which is "
                "the cleanest statement that there is nothing to win. **Quote the "
                "magnitude, never the order**: on the held-out split the same ceiling read "
                "`stl` 3.11% first and `fta` 2.16% third with a median of 1.71%, so the "
                "ranking fully reorders across two scored seasons while its size does not. "
                "So the whole season-term question is bounded small on point accuracy, "
                "which is why the answer is 'no term' despite the league movement being "
                "real. 54 fits, 0 divergences, max R̂ 1.0105, 79.3 min — half the fits of "
                "the both-splits run, and affordable at all only because "
                "`metric=\"dense_e\"` cut the `blk` spline base from 236.6 s to 13.4 s.",
        status="settled",
        reproduce="make season-terms → outputs/predictions/season_term_metrics.csv, "
                  "outputs/predictions/season_term_diagnostics.csv, "
                  "outputs/predictions/season_term_bonus.csv",
        source="docs/predictions-plan.md",
        reviewed="2026-08-08",
        date="2026-07-31",
        tags=("era", "method"),
    ),
    Decision(
        id="year-effect-is-a-simulator-input",
        topic="simulations",
        claim="The year random effect belongs in the SIMULATOR as a variance component, "
              "not in any component head as a feature — it is worth ~50× the shared-β "
              "term on a 15-man roster.",
        because="It is mean-zero at prediction time, so it cannot move point accuracy and "
                "measurably does not: median validation ΔR² against base is **−0.00027** "
                "across thirteen heads. What it does is widen the JOINT distribution, and "
                "a league shift is perfectly correlated across players so it grows as N "
                "while independent error grows as sqrt(N). Roster season-total dk_pts sd "
                "inflation: **+7.95%** at 12 players, **+10.4% at 15**, +21.0% at 30, "
                "+83.5% at 150, **+254%** across all 773 — against shared-β's +0.2% / "
                "+0.2% / +0.3% / +1.1% / +6.4%, which is a different board and a different "
                "head, so the ratio is indicative rather than a like-for-like division. "
                "(Superseded test-board reading: +8.9% / +11.6% / +22.1% / +91.1% / +278% "
                "across 791.) It is also a defined check rather than a hopeful one: "
                "`sigma_year`, fitted by NUTS on player-season rows, lands within 20% of "
                "the directly measured league yoy sd on 3 of 7 count heads (`blk` 0.85×, "
                "`tov` 0.89×, `reb` 1.07×) and within a factor of 1.4 on all seven. **The "
                "shot-mix head at 1.84× is the tell** — with no trend term it absorbs "
                "drift as a sequence of shocks and then zeroes it. ⚠️ Every σ moved on "
                "2026-08-07 because the old `year_*` columns were written from the TEST "
                "arm's fit (27 training seasons) into a row whose CRPS came from the "
                "validation arm (25); `year_n_train_seasons` shows it. Take σ from the "
                "measured league movement rather than the fitted value, and draw one per "
                "head: the cross-component shock correlation is +0.011 on average, so "
                "there is no common factor. Exception: the **minutes** head, where a year "
                "effect wins at val 143.81 against base 144.09 — the one head that adopts "
                "a season term.",
        status="settled",
        reproduce="make season-terms → "
                  "outputs/predictions/season_term_roster_spread.csv, "
                  "outputs/predictions/season_term_sigma_vs_league.csv",
        source="docs/predictions-plan.md",
        reviewed="2026-08-08",
        date="2026-07-31",
        tags=("era", "correlation", "simulation"),
    ),
    Decision(
        id="availability-season-x-role-is-a-null",
        topic="availability",
        claim="The availability season × role interaction is a **validation null** — the "
              "worst two arms of seven — and the era effect does not transfer into a "
              "better forecast.",
        because="`trend_x_role` and `trend_x_role_year` are the **worst two of seven arms "
                "on validation** (10.078, 10.087 against base 10.007): the most expensive "
                "arms in the ablation at 26 features, buying a loss of 0.08 games. They "
                "were simultaneously the best two on test (10.742, 10.736 against base "
                "10.797), which is the exact shape of the false positive this project "
                "already shipped once — the nonlinearity arm whose paired bootstrap on "
                "test read [−0.079, −0.015] with P(Δ<0) = 99.7% and did not replicate — "
                "and it was caught only because selection never reads the test column. "
                "⚠️ Since 2026-08-07 `src/models/held_out.py` stops that test column being "
                "computed at all, so the val/test contrast is preserved as the record of "
                "why the lock exists rather than as a re-runnable measurement; the verdict "
                "does not depend on it. The selected arm, `trend`, is worth **0.010 "
                "games** of validation CRPS, which is nothing. The era effect itself is "
                "real and independently confirmed (the 2023-24 policy break is −4.63% at "
                "p = 0.008 on `gp_share [30+ mpg]`); it simply does not survive as a "
                "feature, because a perfect league-level correction is worth ≤5% of MAE "
                "anywhere.",
        status="null",
        reproduce="make season-terms → outputs/predictions/season_term_metrics.csv",
        source="docs/availability-plan.md",
        reviewed="2026-08-08",
        date="2026-07-31",
        tags=("era", "availability"),
    ),
    Decision(
        id="minutes-bias-is-shrinkage-not-era",
        topic="minutes",
        claim="The minutes head's −33 to −41 minute held-out bias is shrinkage toward a "
              "30-season mean, **not** an era effect.",
        because="`docs/availability-plan.md` proposed the bias as 'the signature an era "
                "effect would leave', with the obvious test being a trend. The test ran "
                "and falsifies it: on validation a trend makes the bias **worse by 10.8 "
                "minutes** (−14.2 on `base` and −13.0 on `year` against **−25.0** on "
                "`trend` and −26.9 on `trend_year`). An era effect the head was failing "
                "to track would have been corrected by a trend, not amplified by it. The "
                "year random effect does help — it wins at val 143.81 against base 144.09 "
                "and is the least biased fitted arm — but that is a partial improvement, "
                "so a genuine bias correction is still owed before the simulator consumes "
                "these minutes as exposure for eleven other heads. **The −33 to −41 level "
                "itself is withdrawn** (see `minutes-head-held-out-bias`): that was the "
                "held-out column, and on validation the FLOOR is the biased one at "
                "**+23.9** while the fitted arms run −13 to −27. What reproduces across "
                "both splits is the GAP — the fitted arms sit 27–51 minutes below the "
                "carry-forward — so quote the distance, never a signed level. The "
                "conclusion here is unaffected, since a trend making the bias worse is a "
                "statement about the trend, not about which seasons it was measured on. "
                "Re-measured on validation 2026-08-07 (superseded held-out figures: "
                "−41.0 / −56.6 / −38.2, worse by 15.5 minutes).",
        status="withdrawn",
        replaced_by="The bias is shrinkage toward the pooled mean; the year effect "
                    "recovers ~1 minute of it and no trend helps.",
        caught_by="make season-terms — the trend arm doubled the bias instead of "
                  "removing it",
        reproduce="make season-terms → outputs/predictions/season_term_metrics.csv",
        source="docs/availability-plan.md",
        reviewed="2026-08-08",
        date="2026-07-31",
        tags=("era", "minutes"),
    ),
    Decision(
        id="player-participation-policy-is-a-real-break",
        topic="availability",
        claim="The 2023-24 Player Participation Policy is a measurable level break in "
              "games-played share, and it is **role-graded in the direction the "
              "availability plan predicted** — heavy-minute players lose, fringe "
              "players do not.",
        because="A level-only break at 2023-24 reads **−4.63%** on `gp_share [30+ mpg]` "
                "(p = 0.008) and −4.43% on `gp_share [all]` (p = 0.024), while "
                "`gp_share [<12 mpg]` is **+5.37% and not significant** (p = 0.28). "
                "COVID gets a different treatment because it is a different shape: "
                "2019-20 and 2020-21 are a transient regime, so they get an indicator "
                "that does not carry into the forecast, and on that arm **0 of 17** "
                "series are significant. The decisive column is neither p-value but "
                "`next_season_shift_pct` — how far the one-season-ahead extrapolation "
                "moves once the regime is handled — and beside it `regime_seasons`, "
                "which is **3** for every break arm. A level+slope break fits its slope "
                "on those three seasons alone and then moves the forecast by up to "
                "**+22.1%**, which is noise, not a better trend. So the break test "
                "**disqualifies extrapolating a trend across 2023-24** rather than "
                "supplying a corrected one.",
        status="measured",
        reproduce="make season-effects → outputs/eda/season_effects_regimes.csv",
        source="docs/availability-plan.md",
        reviewed="2026-07-31",
        date="2026-07-31",
        tags=("era", "method"),
    ),
    Decision(
        id="year-shocks-are-not-one-common-factor",
        topic="simulations",
        claim="League year-to-year shocks are **not** one common factor, so a simulator "
              "draws an independent year effect per head rather than one shared draw.",
        because="Detrended log league rates correlate at a mean of **−0.009** across "
                "136 pairs over 30 seasons — no common factor at all. But mean |r| is "
                "**0.307** and 20.6% of pairs exceed 0.5, so they are not independent "
                "either: the structure is in specific PAIRS. The largest is "
                "`fg2a`–`fg3a` at **−0.833**, which is the 3PA/2PA substitution the "
                "component heads already remove by reparameterizing into "
                "`fga` × `fg3a | fga`. Detrending is load-bearing — two series that "
                "both drift upward would otherwise correlate through their trends, "
                "which is drift and not shock. So the year effect needs a correlation "
                "matrix, the same shape the residual copula already takes, and "
                "`YearTerm.stream` gives each head an independent draw as the default "
                "until one is fitted.",
        status="measured",
        reproduce="make season-effects → "
                  "outputs/eda/season_effects_shock_correlation.csv",
        source="docs/predictions-plan.md",
        reviewed="2026-07-31",
        date="2026-07-31",
        tags=("era", "correlation"),
    ),
    Decision(
        id="no-shooting-hot-hand",
        topic="minutes",
        claim="There is no shooting hot hand, so the successes/trials heads collapse "
              "for free. The serial structure is all in the **exposure**.",
        because="Both field-goal conversion rows are dead nulls — `fg2m|fg2a` lag-1 "
                "excess +0.002 (z = 1.9) and `fg3m|fg3a` −0.002 (z = −1.3) on ~600k "
                "pairs, block inflation 1.03× and 1.01×. Constant-θ-within-season, "
                "exactly what the binomial collapse assumes, is what the data looks "
                "like. What *is* dependent is minutes at 2.43× and shot volume at "
                "~1.46× **on top of** minutes. The shuffle-within-player-season null is "
                "load-bearing: it carries the ≈ −0.016 bias that within-season "
                "demeaning induces, without which `stl` and `tov` read as negative "
                "dependence when both are mildly positive.",
        status="null",
        reproduce="make serial-correlation → outputs/eda/serial_correlation.csv",
        source="docs/model-development-notes.md",
        reviewed="2026-08-08",
        date="2026-07-29",
        tags=("simulator-input",),
    ),

    # ══ The DK component-rate heads ══════════════════════════════════════════
    Decision(
        id="component-no-fit-floor-is-mandatory",
        topic="components",
        claim="Every component head must be quoted against the no-fit floor: prior "
              "per-36 rate × actual minutes / 36, with no fitting at all.",
        because="The floor scores validation R² **0.81–0.95** and the best of seven "
                "fitted variants beats it by only +0.0013 to +0.0334. A head that does "
                "not clear it is not a model — and the floor is what caught the sklearn "
                "alpha artifact, which is why it is mandatory rather than advisory. "
                "Every output row carries `beats_floor` and the runner warns when no "
                "variant clears it for a head, because that is also the signature of "
                "the regularization trap. Moved from the held-out seasons to validation "
                "on 2026-08-05, where it read 0.82–0.94 and +0.0019 to +0.0203; this "
                "module had been the only head in the project with **no split guard at "
                "all**, because it defined its own `split_seasons` instead of importing "
                "the shared one.",
        status="built",
        reproduce="make component-rates → "
                  "outputs/predictions/component_rate_metrics.csv",
        source="docs/predictions-plan.md",
        reviewed="2026-08-06",
        date="2026-07-29",
        tags=("benchmark",),
    ),
    Decision(
        id="scale-not-curvature-for-counts",
        topic="components",
        claim="For the component count heads the answer is **scale**, not curvature: "
              "put the own prior rate in on the log scale.",
        because="A log link wants a multiplicative predictor — "
                "`log E[rate] = β·log(prior rate)` makes the model "
                "`rate ∝ prior_rate^β`, which is the right shape. Linear-in-raw-rate "
                "inside `exp()` is badly misspecified, catastrophically so for the "
                "zero-heavy skewed heads — and the negative-binomial fits are far "
                "harsher than the Poisson ones: held-out R² **−19.00 on `fg3a` and "
                "−1.393 on `blk`**, against 0.520/0.638 under Poisson. Linear-in-raw-"
                "rate is not merely misspecified, it is unusable.",
        status="settled",
        reproduce="make component-rates → "
                  "outputs/predictions/component_rate_metrics.csv",
        source="docs/model-development-notes.md",
        reviewed="2026-08-08",
        date="2026-07-29",
        tags=("specification",),
    ),
    Decision(
        id="splines-help-only-fg3a-and-blk",
        topic="components",
        claim="Splines over `log(own)` are worth ≤ +0.003 outside `fg3a` and `blk`, so "
              "spend flexibility on those two heads only.",
        because="Measured on the season-collapsed sklearn Poisson sweep, where "
                "`log(own)` recovers nearly everything in one term and a spline adds "
                "+0.030 on `fg3a` and +0.041 on `blk` and almost nothing elsewhere.",
        status="withdrawn",
        replaced_by="**The guidance is Poisson-specific and does not carry to the "
                    "shipped negative-binomial heads.** Under NB the identical "
                    "`log_own` spec *collapses* — `blk` 0.8204 → **0.6794** and "
                    "`fg3a` 0.8791 → **0.3719**, both far below their floors — and "
                    "only a spline recovers them (0.8579, 0.9046). The validation "
                    "split picks a spline for **four** heads (`fg3a`, `blk`, `ast`, "
                    "`stl`), and on two of them it is the difference between a model "
                    "and a failure.",
        caught_by="`make stan-components` fitting the same specs under the likelihood "
                  "that actually ships. NB2's `var = μ + μ²/φ` down-weights large "
                  "counts, so the fit is driven by the low-count mass — exactly where "
                  "the log-scale relation is most curved.",
        reproduce="make stan-components → "
                  "outputs/predictions/stan_component_metrics.csv",
        source="docs/predictions-plan.md",
        reviewed="2026-07-30",
        date="2026-07-30",
        tags=("reversal", "specification"),
    ),
    Decision(
        id="component-interactions-and-pca-are-nulls",
        topic="components",
        claim="Three nulls on the component heads: the `age × own` and `mpg × own` "
              "interactions, walk-forward PCA of all 156 columns, and `ftm|fta`.",
        because="Interactions are ≤ +0.001 and *negative* for `blk` and `ast` once the "
                "scale is right. Walk-forward PCA — refitting scaler and PCA per target "
                "season on S-1 and earlier only, never pooled — lands within ±0.003 of "
                "`log_own` on every head, so the 121 style/tracking columns add nothing "
                "once you have the player's own prior rate and his minutes. And "
                "`ftm|fta` is the head where *nothing* beats the floor, because "
                "free-throw percentage is pure player skill with no context to add.",
        status="null",
        reproduce="make component-rates → "
                  "outputs/predictions/component_rate_metrics.csv",
        source="docs/model-development-notes.md",
        reviewed="2026-08-08",
        date="2026-07-29",
        tags=("specification",),
    ),
    Decision(
        id="conversion-floor-must-be-shrunk",
        topic="components",
        claim="The conversion floor has to be a **shrunk** carry-forward, and that is a "
              "fact about proportions.",
        because="A player who went 0-for-3 from three has a prior 3P% of exactly 0.000; "
                "carrying it onto 200 attempts gives a beta-binomial NLL of **1.3e9** "
                "and makes the benchmark meaningless. The floor is therefore "
                "`p = (made + k·league_mean)/(attempts + k)` with `k` and `league_mean` "
                "fitted on train only — one shrinkage constant, no features. This is "
                "the standing 'shrink conversion percentages hard' rule showing up as a "
                "benchmark requirement.",
        status="settled",
        reproduce="make component-rates → "
                  "outputs/predictions/component_rate_metrics.csv",
        source="docs/predictions-plan.md",
        reviewed="2026-07-30",
        date="2026-07-29",
        tags=("benchmark",),
    ),
    Decision(
        id="sklearn-alpha-averages-by-weight-sum",
        topic="components",
        claim="`sklearn`'s two regularization conventions are opposite, and with "
              "exposure weights the difference is ~7 orders of magnitude.",
        because="`PoissonRegressor` minimizes `deviance / (2·Σw) + alpha·‖coef‖²` — the "
                "data term is **averaged by the weight sum**. Fitting a rate with "
                "`sample_weight = minutes` makes Σw ≈ 1e7, so a default-looking "
                "`alpha=1.0` is an enormous penalty: `reb` reads R² **0.6620 at "
                "alpha=1.0 against 0.9322 at alpha=0.01**. It fails *quietly* — the fit "
                "converges, coefficients are finite, and a flexible basis partially "
                "compensates, so splines looked like they were buying real signal. "
                "`LogisticRegression` is the reverse: its objective is not averaged, so "
                "`C=1.0` is *weak*. The trap now ships as a permanent alpha-sensitivity "
                "ablation, and at alpha = 10 every fit falls below the no-fit floor.",
        status="settled",
        reproduce="make component-rates → "
                  "outputs/predictions/component_rate_metrics.csv",
        source="docs/data-quirks.md",
        reviewed="2026-08-08",
        date="2026-07-29",
        tags=("failure-mode", "regression-guard"),
    ),
    Decision(
        id="zero-inflation-is-a-minutes-artifact",
        topic="components",
        claim="Zero-inflation is a minutes artifact, not a property of the target — but "
              "three heads are genuinely low-count.",
        because="`dk_pts` is 0 in 35.4% of sub-5-minute games and **0.0% above 18 "
                "minutes**, so conditional on minutes the zeros are Poisson zeros and "
                "there is nothing to zero-inflate. But in 30–48 minute games `blk` is "
                "still 0 in 58.6% of games, `fg3m` 42.2% and `stl` 33.8%: those three "
                "are misspecified under MSE at *any* minutes level. Separately, "
                "dispersion is worst at **low usage**, not low minutes — so if a "
                "dispersion term is added, key it on usage.",
        status="measured",
        reproduce="make target-profile → outputs/eda/target_profile.csv",
        source="docs/model-development-notes.md",
        reviewed="2026-08-08",
        date="2026-07-28",
        tags=("specification",),
    ),
    Decision(
        id="stan-component-heads-built",
        topic="components",
        claim="The eleven Stan component heads are built and fitted separately — eight "
              "negative-binomial counts and three beta-binomial conversions, from two "
              "`.stan` files.",
        because="Two files serve every head because availability, minutes and the three "
                "conversions are the same likelihood with different `y`/`n`, and the "
                "counts are the other one. That is the factorization argument as code "
                "rather than as prose. Sources are checked in; cmdstanpy compiles a "
                "*copy* into `outputs/stan/` so no binary and no generated `.hpp` "
                "enters the repo.",
        status="built",
        reproduce="make stan-components → outputs/predictions/stan_component_*",
        source="docs/predictions-plan.md",
        reviewed="2026-07-30",
        date="2026-07-30",
        tags=("head",),
    ),
    Decision(
        id="the-free-throw-family-fails-its-floor",
        topic="components",
        claim="~~Under the shipped negative binomial the **whole free-throw family** "
              "falls below its no-fit floor — `fta` now joins `ftm|fta`.~~ "
              "**Withdrawn 2026-08-06: `fta` clears its floor on validation. Only "
              "`ftm|fta` fails.**",
        because="The finding rested on `fta` scoring **test** R² 0.8649 against the "
                "carry-forward floor's 0.8673 — a shortfall of **0.0024**, which was "
                "never distinguishable from zero. Re-run on the split that is allowed "
                "to select (`src/models/held_out.py`), `fta` selects `log_own` and "
                "scores **0.8909** against a floor of **0.8765**, clearing by "
                "**+0.0144**. `ftm|fta` still fails at every variant and is now the "
                "only head in the project that does — which was always the "
                "better-founded half, since free-throw *percentage* has a "
                "pure-player-skill argument that trips to the line never had. The "
                "sklearn probe agrees on validation (0.8922 against the same floor), "
                "so both instruments now say the same thing. **The floor stays "
                "mandatory** — that argument is untouched; what changed is that a "
                "two-parts-in-a-thousand shortfall was being reported as a head-level "
                "defect. Fourth reversal of a sub-1% test margin since the lock.",
        status="withdrawn",
        replaced_by="**Only `ftm|fta` fails.** On validation `fta` selects `log_own` and "
                    "scores **0.8909** against a carry-forward floor of **0.8765**, "
                    "clearing by **+0.0144** — where the recorded reading had it 0.0024 "
                    "*below* a test-split floor. `ftm|fta` still loses at every variant "
                    "(3.0804 against 3.0541) and is now the only head in the project that "
                    "fails its floor, which is the case that always had a mechanism behind "
                    "it: free-throw percentage is pure player skill, so shrinking the prior "
                    "is already optimal. There is still no equivalent argument for trips to "
                    "the line — and now none is needed.",
        caught_by="`src/models/held_out.py` moved the sweep off the test split, and `fta` "
                  "changed sides. The margin that carried the original finding was 0.0024 "
                  "R² — two parts in a thousand, on 791 rows — so it was never "
                  "distinguishable from zero, and the entry read as a head-level defect "
                  "rather than as noise. The sklearn probe independently agrees on "
                  "validation (0.8922 against the same floor), so the two instruments that "
                  "had disagreed now do not. Fourth reversal of a sub-1% test margin since "
                  "the lock landed; the no-fit floor itself is untouched and still "
                  "mandatory.",
        reproduce="make stan-components → "
                  "outputs/predictions/stan_component_metrics.csv",
        source="docs/predictions-plan.md",
        reviewed="2026-08-06",
        date="2026-07-30",
        tags=("defect",),
    ),
    Decision(
        id="fg3a-share-reparameterization",
        topic="components",
        claim="Handle the 3PA/2PA substitution by reparameterizing into the chain — "
              "model `fga` as the count and `fg3a | fga` as a binomial **share**.",
        because="It enforces the substitution *by construction*, keeps the posterior "
                "factorization exact, and is better specified anyway, since shot-mix "
                "shares persist like counts (`sco_pct_fga_3pt` at 0.886) while the two "
                "raw counts trade off at −0.11 residual correlation. **Re-measured "
                "un-handicapped (Gate 0, 2026-08-03) and it still wins**: arm A at each "
                "head's own selected variant against arm B swept for real gives "
                "**−0.501 nats on validation**, per player-season. The "
                "recorded −0.771 / −0.793 was measured against a straw man — both arms "
                "fitted at `log_own`, where `fg3a` reads R² 0.3719 with "
                "`beats_floor = False` against 0.9046 for the spline it ships — and "
                "0.306 of that margin was the handicap. The comparison is legitimate "
                "because `(fg2a, fg3a) ↔ (fga, fg3a)` is a **bijection with unit "
                "Jacobian on the integers**, so the two joint log-densities are "
                "directly comparable. **Re-run validation-only on 2026-08-06** with "
                "`src/models/held_out.py`. The test column it used to carry (−0.494, and "
                "−0.492 against arm A's best-of-16) is retired: the best-of-16 grid was "
                "read from `stan_component_metrics.csv`'s `test_nll`, and adoption removed "
                "those head rows while the split move removed the column. That grid was "
                "worth 0.001640 nats over arm A's own selected pair, so the loss is "
                "bookkeeping rather than evidence. Adopted — see "
                "`shot-attempt-basis-adopted`.",
        status="built",
        reproduce="make stan-substitution → "
                  "outputs/predictions/stan_component_substitution_sweep.csv, "
                  "outputs/predictions/stan_component_substitution_sweep_diagnostics.csv, "
                  "outputs/predictions/stan_component_substitution.csv",
        source="docs/shot-attempt-basis-plan.md",
        reviewed="2026-08-06",
        date="2026-08-03",
        tags=("specification",),
    ),
    Decision(
        id="season-term-selection-is-noise-dominated",
        topic="components",
        claim="Re-running the season-term ablation on the shot-attempt basis flipped the "
              "selected arm for **9 of 13 heads**, almost all on margins under **1%** of "
              "the selection metric. That strengthens the no-season-term verdict rather "
              "than overturning it.",
        because="On the July head list nearly every head selected `base`; on the new one "
                "11 of 13 select a term (`trend` x5, `year` x3, `trend_year` x3, `base` "
                "only for `reb` and `stl`). Read as a verdict that is a reversal. Read as "
                "a **replication test** it is the opposite: the flips are 0.05% "
                "(`fg3a|fga`), 0.01% (`fg3m|fg3a`), 0.09% (`ftm|fta`), 0.11% (`gp`), "
                "0.20% (`min`), 0.26% (`fga`), 0.35% (`blk`) and 0.79% (`fta`), on a "
                "change that does not touch most of those heads at all. An ablation whose "
                "winner is that unstable is measuring noise. Only `ast` (2.21%) and "
                "`fg2m|fg2a` (1.88%) move enough to be worth a second look, and neither is "
                "a shot-attempt head. The oracle ceiling is unchanged in kind — median "
                "**1.71%** of MAE, max `stl` 3.11% — which is why none of it is worth "
                "adopting. The minutes head is the tell in the other direction: it selects "
                "`year` and reproduced val 143.81 / test 146.54 **exactly** across the "
                "basis change, which is what a real selection looks like next to nine "
                "unstable ones. ⚠️ One recorded argument did weaken: the season total used "
                "to show `trend` winning MAE while flipping bias to +7.60 (read as "
                "cross-component cancellation producing a false positive). It now reads "
                "106.06 / −13.57 against base's 108.56 / −36.89, so `trend` improves both "
                "and cancellation no longer explains the aggregate win on its own. That "
                "part needs a human decision, not a doc edit.",
        status="measured",
        reproduce="make season-terms → outputs/predictions/season_term_metrics.csv, "
                  "outputs/predictions/season_term_season_total.csv",
        source="docs/predictions-plan.md",
        reviewed="2026-08-04",
        date="2026-08-04",
        tags=("methodology", "next"),
    ),
    Decision(
        id="shot-attempt-basis-adopted",
        topic="components",
        claim="**Adopted 2026-08-04**: `COUNT_HEADS` is now `fga, fta, reb, ast, stl, blk, "
              "tov` and `fg3a | fga` is a conversion head. `fg2a` is derived, and the head "
              "count is still eleven.",
        because="Gate 0 measured the reparameterization un-handicapped and it won by "
                "−0.4935 nats on test, so the ablation became the specification. Three "
                "things the migration turned up that the plan did not predict. (1) "
                "`season_totals` computed per-36 rates over `COUNT_HEADS` alone, so "
                "`fg2a_p36` and `fg3a_p36` would have vanished and two conversion heads "
                "would have lost their volume feature — `rate_columns()` now derives the "
                "list from both head lists, which is a no-op in the old basis. (2) The "
                "`fg3a_pct_lag1` hazard the plan warned about is **not real** after "
                "adoption: `season_totals` builds `fg3a_pct = fg3a / fga`, which is the "
                "attempt mix and exactly right, while shooting percentage stays "
                "`fg3m_pct`. (3) `fga` and `fg3a` are drawn from different posteriors, so "
                "the derived `fg2a` needs a zero-clip. **`fga` is now the best-behaved "
                "count head in the project**: the highest floor (0.9514), selected at "
                "0.9584, and a linear predictor costs it 0.0025 R² where it cost `fg3a` "
                "−19.00. (Those three read 0.9464 / 0.9505 / 0.0068 on the held-out "
                "split, before the sweep moved to validation on 2026-08-06.)",
        status="built",
        reproduce="make stan-components → "
                  "outputs/predictions/stan_component_metrics.csv",
        source="docs/shot-attempt-basis-plan.md",
        reviewed="2026-08-06",
        date="2026-08-04",
        tags=("specification",),
    ),
    Decision(
        id="shot-mix-drifts-but-shooting-does-not",
        topic="components",
        claim="`fg3a | fga` is a conversion head by **likelihood**, not by subject matter — "
              "it is the largest non-minutes serial dependence in the project, while the "
              "three shooting heads remain clean nulls.",
        because="`make serial-correlation` on the new heads puts the shot-mix share at a "
                "lag-1 excess of **+0.101** (z = 94) with a 10-game block variance "
                "inflation of **1.57×** — above the `fga` count it splits — against a max "
                "|excess| of **0.0122** across `fg2m|fg2a`, `fg3m|fg3a` and `ftm|fta`. "
                "Shot *selection* drifts within a season the way minutes do; shooting "
                "*accuracy* does not. The module's summary line used to take one maximum "
                "over all conversion heads, which after adoption would have reported that "
                "drift as a hot hand **and** hidden that the shooting heads are still "
                "nulls; it now reports the two groups separately. This is a finding the "
                "two-count basis could not surface, because the mix was not a modelled "
                "quantity.",
        status="measured",
        reproduce="make serial-correlation → outputs/eda/serial_correlation.csv",
        source="docs/shot-attempt-basis-plan.md",
        reviewed="2026-08-04",
        date="2026-08-04",
        tags=("next", "simulator-input"),
    ),
    Decision(
        id="gate-a-cost-model-is-a-lower-bound",
        topic="minutes",
        claim="The composition's timing gate **under-predicts, and by a factor that is not "
              "constant** — 1.63× two-pass, **1.17×** one-pass — because per-row sampler "
              "cost is superlinear in rows.",
        because="The one-pass probe extrapolated **8.3 h** for the four-arm sweep; it took "
                "**9.78 h** of sampler time. Per-row cost runs 5.95 ms on the 26k-row "
                "probe against **14.83 ms** on the 631k-row `betabinom` fit: more data "
                "sharpens the posterior, which shrinks the step size, which buys more "
                "leapfrog steps per iteration on top of an already-linear per-gradient "
                "cost. **Gate A is a lower bound, not an estimate.** ⚠️ The two-pass sweep "
                "missed by 1.63× (12.8 h against 20.9 h), and the shrink is not the sampler "
                "becoming predictable — the old figure was inflated by a second pass on a "
                "larger frame that the linear model handled badly. "
                "`stan_games_played.probe_timing` hard-codes `raw_hours * 1.63` and keeps "
                "it deliberately: lowering it makes that gate more permissive, and "
                "admitting an unaffordable run is far worse than aborting an affordable "
                "one. Its recorded fallback order is also wrong: it "
                "says cut `binomial` first, but `binomial` is the *cheapest* arm (20.5% of "
                "sweep time against the graded arm's 27.3%) and cutting it removes the "
                "result that dispersion is load-bearing. That spread narrowed from 2.6× to "
                "1.33× when the test pass went, so the argument is weaker than it was. "
                "Shorten chains first, subsample train "
                "second, cut arms last — and never cut `betabinom_ot`, which `run()` reads "
                "after the whole sweep and before any CSV is written.",
        status="measured",
        reproduce="make stan-composition → "
                  "outputs/predictions/stan_composition_diagnostics.csv",
        source="docs/minutes-composition-plan.md",
        reviewed="2026-08-08",
        date="2026-08-04",
        tags=("performance", "methodology"),
    ),
    Decision(
        id="the-coordinate-change-beats-the-fitting",
        topic="components",
        claim="For the shot-attempt pair, **the basis is worth more than the model**: the "
              "reparameterized no-fit floor beats the canonical basis's best fitted "
              "configuration.",
        because="At their no-fit floors — prior per-36 rate × minutes for the count, a "
                "shrunk carry-forward for the share, no features anywhere — the two bases "
                "score **11.174** (canonical) against **10.064** (reparameterized), a "
                "**−1.110** nat gap. Arm A's fitted configuration scores 10.505, so the "
                "reparameterized floor beats it by **−0.441** with zero features, and arm "
                "B's own fitted heads add only −0.060 on top of their floor. This is the "
                "same shape as the standing finding that the component "
                "rate side is nearly saturated by a carry-forward: when the floor is that "
                "strong, the parameterization is where the remaining leverage is, not the "
                "feature set. It also reframes the substitution result — it is not a "
                "better model of shot attempts, it is the same information in coordinates "
                "where the dependence is structural instead of residual. **Measured on "
                "validation since 2026-08-06**, where the gap is wider than the test "
                "column it replaces (11.024 / 10.086 / −0.938 / −0.391 / −0.101). The "
                "floors are arithmetic and reproduce to the digit across the refit, so the "
                "widening is the fitted side moving, not the benchmark.",
        status="measured",
        reproduce="make stan-substitution → "
                  "outputs/predictions/stan_component_substitution_sweep.csv",
        source="docs/shot-attempt-basis-plan.md",
        reviewed="2026-08-06",
        date="2026-08-03",
        tags=("specification", "methodology"),
    ),
    Decision(
        id="conversion-own-rate-column-must-be-named",
        topic="components",
        claim="`conversion_variants` takes its own-rate column **explicitly**; the "
              "`{made}_pct_lag1` naming convention does not generalize.",
        because="The `fg3a | fga` share head's own rate is the attempt-**mix** share "
                "`fg3a_share_lag1`, but the convention resolves to `fg3a_pct_lag1` — "
                "three-point *shooting* percentage. That column does not exist today, so "
                "the call raises; the hazard is that adopting the reparameterization "
                "**creates** it (`(\"fg3a\", \"fga\")` becomes a conversion head, and "
                "`build_design`'s lag loops generate it for free), at which point the "
                "head would silently fit on shooting accuracy and its entire rationale — "
                "shot-mix shares persist at 0.886, conversion percentages at 0.500 — "
                "would be gone with no error anywhere. Fixed before the Gate 0 "
                "measurement rather than after, and the refactored arm reproduces the "
                "recorded 9.991042 joint NLL to nine decimal places, which is how we know "
                "the fix changed nothing else.",
        status="built",
        reproduce="make stan-substitution → "
                  "outputs/predictions/stan_component_substitution_sweep.csv",
        source="docs/shot-attempt-basis-plan.md",
        reviewed="2026-08-03",
        date="2026-08-03",
        tags=("failure-mode",),
    ),

    Decision(
        id="held-out-split-is-locked-in-code",
        topic="problem",
        claim="The test seasons are a **capability**, not a convention: reaching them "
              "raises unless `src/final_evaluation.py` has explicitly unlocked them.",
        because="A rule that lives only in prose gets followed until it is inconvenient, "
                "and this one already failed. The games-played head's Gate D was "
                "specified with the incumbent's *test* figures as its bars, run on the "
                "test split, and settled which model ships — on a 0.0013 CRPS margin a "
                "paired bootstrap could not distinguish from zero, which **reversed** "
                "when re-decided on validation. Nothing in the code objected because "
                "nothing in the code knew. The guard now sits inside "
                "`availability.split_seasons`, the one choke point every head goes "
                "through, so a new head cannot forget to add it; it fires on *use* "
                "rather than on carving, so `train, _ = split_seasons(...)` stays legal "
                "and `score(model, test)` does not. Nine modules were converted and the "
                "sweeps stopped emitting a test column at all, which cut roughly half "
                "the sampler time and removed a confound — the test side used to refit "
                "on train + validation at double the iterations, so a val/test "
                "disagreement conflated the rows, the training data and the chain "
                "length, and was never the replication check it looked like.",
        status="built",
        # The lock's own artifact is `final_evaluation.csv`, which deliberately does not
        # exist yet — running it would spend the split. So this entry cites the artifact
        # the lock *produced*: a gate that passed on test by 0.03 dk_pts and fails on
        # validation by 6.34. See `final-evaluation-has-not-run` for the other half.
        reproduce="make season-total → outputs/predictions/season_total_gate_e.csv",
        source="docs/train-validate-test-split.md",
        reviewed="2026-08-08",
        date="2026-08-05",
        tags=("discipline",),
    ),
    Decision(
        id="final-evaluation-has-not-run",
        topic="problem",
        claim="`make final-evaluation` has **never been run**, and the test seasons are "
              "therefore unspent. Three heads are registered for it — availability, "
              "games played and the season total.",
        because="It is the one end-of-project measurement of the whole workflow, and the "
                "workflow is not finished: the simulator, the ranking layer and the "
                "backtested draft strategies are all still ahead. Running it now would "
                "spend the split for a number that describes a partial system, and any "
                "modelling decision taken afterwards would make the estimate biased. "
                "Registering a head there is what makes its held-out number *takeable*, "
                "which is why the registry is asserted by a test rather than assumed — a "
                "head dropped from it silently loses its final measurement. The heads "
                "that are not registered yet (minutes, components, composition, season "
                "terms) have simply not had their refit wired.",
        status="open",
        unblocks="the simulator, ranking and drafting layers being finished",
        source="docs/train-validate-test-split.md",
        reviewed="2026-08-08",
        date="2026-08-05",
        tags=("discipline",),
    ),
    Decision(
        id="component-rates-had-no-split-guard",
        topic="components",
        claim="`component_rates.py` defined its own `split_seasons`, which routed around "
              "the held-out lock entirely. Deleted; the shared split is now the only one.",
        because="The lock lives inside `availability.split_seasons` precisely so a head "
                "cannot forget to add it — and a private copy of the same six lines "
                "defeats that completely, silently, while looking like ordinary "
                "duplication. The copy was behaviourally identical, so folding it onto "
                "the shared function changed nothing but the guard. A test now walks "
                "every converted module with `ast` and fails if one names "
                "`split_seasons` instead of `held_out.selection_split`.",
        status="built",
        reproduce="make component-rates → "
                  "outputs/predictions/component_rate_metrics.csv",
        source="docs/train-validate-test-split.md",
        reviewed="2026-08-08",
        date="2026-08-05",
        tags=("discipline",),
    ),
    # ══ Season simulations ═══════════════════════════════════════════════════
    Decision(
        id="fit-stan-heads-separately",
        topic="simulations",
        claim="Fit the eleven Stan heads separately, not as one joint model — the "
              "posterior factorizes exactly.",
        because="The generative structure is a chain of conditionals with distinct "
                "parameter blocks and independent priors, so the joint posterior "
                "factorizes into independent blocks and eleven separate fits recover "
                "the *identical* posterior — an identity, not an approximation. "
                "`megamodel.stan` is the proof from this project's own history: every "
                "head had its own `beta_*` with an independent prior and no parameter "
                "was shared, so it paid the full joint-fit price and ran on "
                "`sample_frac(0.01)`. **99% of the data was given up for a coupling "
                "that was not in the model.** Eleven small models are also "
                "independently diagnosable and parallel; in one joint model divergences "
                "in the `blk` block degrade every other block's sampler.",
        status="settled",
        reproduce="make stan-components → outputs/predictions/stan_component_*",
        source="docs/predictions-plan.md",
        reviewed="2026-07-30",
        date="2026-07-29",
        tags=("architecture",),
    ),
    Decision(
        id="correlation-enters-at-draw-time",
        topic="simulations",
        claim="The correlation the simulator needs enters at **draw time**, not fit "
              "time: draw `min` once and push it through all eleven heads.",
        because="Minutes are the largest common factor by far, at **46.4%** of "
                "within-player residual variance. Beyond that, residual "
                "cross-component correlation is **small** — conditioning minutes out, "
                "the off-diagonals average **+0.0070** with a max of **+0.1329** "
                "(`fga`–`reb`). Impose that matrix with a Gaussian copula at simulation "
                "time if the shared minutes draw misses; do not fit jointly for it. The "
                "matrix is PSD with minimum eigenvalue **+0.7853**, so it needs no "
                "nearest-PSD correction. ⚠️ **Every figure here was corrected on "
                "2026-08-07**, and by two separate events rather than one. The minutes "
                "share read 18.6%, which conditioned on the raw minutes *level* and is "
                "attenuated by construction; and the off-diagonals read +0.012 / +0.142 "
                "(`fg2a`–`reb`) / minimum eigenvalue +0.756, which are the retired "
                "two-count basis. Adopting the shot-attempt basis removed the 3PA/2PA "
                "substitution from this matrix **by construction** — the recorded "
                "−0.125 pair no longer exists here, and it ships under "
                "`basis == \"legacy_two_count_basis\"` at −0.1248 instead. What replaces "
                "it is `fga`–`fg3a|fga` at −0.0836, a genuine residual relation between "
                "shot volume and shot mix rather than an accounting identity. The "
                "simulator must read `fit_window == \"train_val\"`, not the `full` "
                "window quoted here.",
        status="measured",
        reproduce="make residual-correlation → outputs/eda/residual_correlation.csv",
        source="docs/simulations-plan.md",
        reviewed="2026-08-08",
        date="2026-07-29",
        tags=("simulator-input",),
    ),
    Decision(
        id="simulator-inputs-calibrate-on-train-plus-validation",
        topic="simulations",
        claim="The four numbers the simulator is **given** — the residual copula, the "
              "game-level minutes dispersion, the ten-game block inflation and the bonus "
              "overdispersion — are calibrated per fit window rather than over every "
              "season. Their artifacts carry a `fit_window` column and the consumers "
              "default to `train_val`. **A third window, `train`, was added 2026-08-08** "
              "— which window to consume is decided by what the number will be scored "
              "against, not by which is widest.",
        because="None of the four is *fitted*, so no train/test split guard has ever "
                "covered them, and all four were being measured over every season "
                "including the two the heads hold out. That would calibrate the "
                "simulator on the seasons it is later backtested against. The measured "
                "differences are small — game-level dispersion 4.65× → 4.68×, `min` "
                "block inflation 2.4321 → 2.4206, copula off-diagonal mean +0.0070 → "
                "+0.0070 with min eigenvalue +0.7853 → +0.7839, bonus overdispersion "
                "0.0248 → 0.0248 — which is exactly why this had to be fixed rather "
                "than argued about: every difference is below the precision the figures "
                "are quoted at, so the leak could never have announced itself in a "
                "backtest. `to_matrix` defaults to `train_val` because a default is what "
                "an unthinking consumer gets. The 2026-08-08 addition closes the half of "
                "this that `train_val` alone could not: it excludes the TEST seasons and "
                "nothing else, so it is clean for the one-shot test readout and NOT for "
                "the realized 2022-23 / 2023-24 backtest, which scores the very seasons "
                "it contains. That gap only became reachable once `make posteriors` "
                "started emitting coefficients per window — a backtest could then hold "
                "clean coefficients beside a noise shape calibrated on the seasons it was "
                "scoring. `train` drops twice TEST_SEASONS, the same two-step carve "
                "`held_out.selection_split` performs, and three of the four modules "
                "already looped over `FIT_WINDOWS` so they picked it up unchanged. The "
                "measured move is again below the quoted precision — `min` block "
                "inflation 2.4206 → 2.4167 — which is again the argument for fixing it "
                "rather than arguing about it.",
        status="built",
        reproduce="make residual-correlation / serial-correlation / component-targets "
                  "/ stan-minutes → outputs/eda/residual_correlation.csv, "
                  "outputs/eda/serial_correlation.csv, "
                  "outputs/eda/bonus_calibration.csv, "
                  "outputs/predictions/stan_minutes_dispersion.csv",
        source="README.md",
        reviewed="2026-08-08",
        date="2026-08-04",
        tags=("simulator-input",),
    ),
    Decision(
        id="copula-must-use-the-conditioned-matrix",
        topic="simulations",
        claim="Build the copula on the **minutes-conditioned** matrix, never the raw "
              "one — a 16× difference.",
        because="Off-diagonals average +0.112 raw against +0.007 conditioned, with the "
                "largest raw cell `fg2a`–`fta` at +0.492 — two shot-volume counts both "
                "scaling with the minutes they were accumulated over, not a "
                "cross-component dependence. Building on the raw matrix would impose "
                "16× the intended coupling *on top of* the shared minutes draw that "
                "produced it, which is why `minutes_conditioned` is an explicit column "
                "rather than an implicit filename convention.",
        status="settled",
        reproduce="make residual-correlation → outputs/eda/residual_correlation.csv",
        source="docs/provenance-plan.md",
        reviewed="2026-07-30",
        date="2026-07-29",
        tags=("simulator-input",),
    ),
    Decision(
        id="block-inflation-is-the-decision-relevant-column",
        topic="simulations",
        claim="Block variance inflation is the decision-relevant serial-correlation "
              "column: it is the factor by which an independent-draws simulator "
              "understates the variance of an aggregate.",
        because="`min` at 2.43×, shot volume ~1.46× **on top of** minutes, and the "
                "conversion heads at 1.01–1.10×. Put the sequential model on minutes, "
                "beside the availability spell process, and leave the other eleven "
                "heads collapsed.",
        status="measured",
        reproduce="make serial-correlation → outputs/eda/serial_correlation.csv",
        source="docs/simulations-plan.md",
        reviewed="2026-07-30",
        date="2026-07-29",
        tags=("simulator-input",),
    ),
    Decision(
        id="bonus-overdispersion-is-calibrated",
        topic="simulations",
        claim="The bonus overdispersion is calibrated, and independent sampling is "
              "**22.7% too low**.",
        because="Against 11,938 player-seasons with ≥200 minutes: independent sampling "
                "gives E[bonus] 0.0951 against a realized 0.1231, while the shipped "
                "0.10 gives 0.1239 — a bias of +0.0009. `BONUS_OVERDISPERSION` is the "
                "variance of the shared per-game Gamma frailty, which does two jobs at "
                "once: it makes each category's marginal negative binomial *and* "
                "induces the positive dependence the bonus needs. It is not a "
                "dispersion of `dk_pts` and it is not fitted by any head.",
        status="measured",
        reproduce="make component-targets → outputs/eda/bonus_calibration.csv",
        source="docs/provenance-plan.md",
        reviewed="2026-07-30",
        date="2026-07-29",
        tags=("simulator-input",),
    ),
    Decision(
        id="bonus-calibration-holds-across-buckets",
        topic="simulations",
        claim="The bonus calibration at overdispersion 0.10 holds with good fit across "
              "minutes buckets.",
        because="Asserted alongside the aggregate calibration, which does reproduce.",
        status="withdrawn",
        replaced_by="**It does not.** At the shipped value the per-bucket bias runs "
                    "−0.0142 at 12–18 mpg against +0.0291 at 30–48 — a 0.0433 spread "
                    "that cancels to +0.0009 in aggregate. The buckets do not fit; "
                    "their errors offset. And the value the **simulator** needs is "
                    "**0.025, not 0.10**, because the simulator draws per *game*: at "
                    "the player-game unit minutes are no longer hidden inside the "
                    "frailty, the fitted optimum is 0.0248, and at that value the fit "
                    "holds across every bucket (bias −0.0004 to +0.0007). Using 0.10 "
                    "per game over-predicts the bonus by +0.036 dk_pts/game for 30+ "
                    "minute players — exactly the players it is worth most for. "
                    "`BONUS_GAME_OVERDISPERSION = 0.025` is added beside the original "
                    "with the unit stated in both docstrings.",
        caught_by="The bucket break, which `docs/provenance-plan.md` correctly called "
                  "'checkable and currently unchecked'. The reason is structural rather "
                  "than a tuning miss: one scalar frailty variance is standing in for "
                  "minutes variation whose *relative* size differs by bucket.",
        reproduce="make component-targets → outputs/eda/bonus_calibration.csv",
        source="docs/provenance-plan.md",
        reviewed="2026-07-30",
        date="2026-07-29",
        tags=("reversal", "simulator-input"),
    ),
    Decision(
        id="never-plug-in-the-expectation",
        topic="simulations",
        claim="Never plug in `E[min]` or `E[gp]` — draw them. And never cap minutes at "
              "48.",
        because="The bonus is a *threshold*, so plugging in an expectation gives "
                "`bonus(E[x])` where the deliverable is `E[bonus(x)]`, and those differ "
                "by construction. The same reasoning is why the deliverable is a joint "
                "draw rather than twelve marginals.",
        status="settled",
        reproduce="make component-targets → outputs/eda/bonus_calibration.csv",
        source="docs/simulations-plan.md",
        reviewed="2026-07-30",
        date="2026-07-29",
        tags=("simulator-input",),
    ),
    Decision(
        id="games-played-is-a-tenure-decomposition",
        topic="simulations",
        claim="The games-played process is **entry index × exit index × a within-tenure "
              "two-state chain**, not one recurrent chain over the schedule. A departure "
              "is an absorbing hitting time, not a low recovery rate.",
        because="A waived player's cell has a recovery hazard of about zero, so a "
                "recurrent chain makes him absorbing from his *first* absence rather than "
                "from the game he was actually cut — relocating the departure earlier in "
                "the season and dragging the left tail out. Gate 0 measures it at the most "
                "generous parameterization available, each cell running at its own "
                "observed hazards, so the failure is a property of the **process class** "
                "and no covariate block can rescue it: the plain full-window chain "
                "over-predicts `P(GP < 41)` by 21.9% (z = +5.29 against the sampling error "
                "of the observed proportion) where the tenure decomposition misses by 3.8% "
                "(z = +0.93). The fix keeps all 30 seasons, because "
                "`team_games = pre-tenure + tenure + post-tenure` is identifiable "
                "structurally from `in_appearance_window`. Two things fall out free: every "
                "within-tenure spell is **interior**, so the duration head needs no "
                "censoring branch; and the initial state is **known**, which is the one "
                "thing the collapse to sufficient statistics cannot supply.",
        status="settled",
        reproduce="make games-played → outputs/predictions/stan_games_played_gate.csv, "
                  "outputs/predictions/stan_games_played_collapse.csv, "
                  "outputs/predictions/stan_games_played_spells.csv",
        source="docs/games-played-plan.md",
        reviewed="2026-08-05",
        date="2026-08-05",
        tags=("architecture", "simulator-input"),
    ),
    Decision(
        id="games-played-collapses-to-four-counts",
        topic="simulations",
        claim="The game-level Markov likelihood collapses **exactly** to four transition "
              "counts per player-season — 1,297,766 transitions into 32,944 binomial "
              "rows, a 39.4× reduction.",
        because="Every feature the head uses is constant within a player-season, which is "
                "the prediction-time constraint rather than a modelling choice, so for "
                "observed states the per-transition probabilities are constant too and "
                "`(onsets, at-risk, recoveries, at-risk-missed)` are sufficient. The same "
                "algebraic collapse the count heads already use — an identity, not an "
                "approximation, and the reason fitting a game-level process costs no more "
                "than the season-level head it extends. Right-censoring needs no term at "
                "all on this side: the product runs over *observed* transitions and the "
                "terminal state contributes no factor.",
        status="built",
        reproduce="make games-played → "
                  "outputs/predictions/stan_games_played_collapse.csv",
        source="docs/games-played-plan.md",
        reviewed="2026-08-05",
        date="2026-08-05",
        tags=("architecture",),
    ),
    Decision(
        id="clustering-supplies-42-percent-of-the-gp-overdispersion",
        topic="availability",
        claim="Serial clustering explains roughly **a sixth** of the ~22.7× games-played "
              "overdispersion; the rest is between-player heterogeneity.",
        because="A 2-state chain with lag-1 ρ = 0.597 inflates variance 3.96× against the "
                "22.7× measured, and 3.96 / 22.7 is about a sixth.",
        status="withdrawn",
        replaced_by="**Two independent errors, and they compound.** The composition is "
                    "**additive**, not multiplicative — `inflation = C + ρ(n − C)`, which "
                    "returns `1 + (n−1)ρ` at C = 1 — so dividing one figure by the other "
                    "is not a decomposition of anything. And the two figures are measured "
                    "on different frames: the 3.96× runs on the appearance window over "
                    "**all** players, the 22.7× on the full window over **established "
                    "rotation** players. Matched, the same population reads "
                    "`P(play|played) = 0.9443` and `P(play|missed) = 0.1333`, giving "
                    "**C = 9.58** — clustering supplies **42%** of the budget, not 17%. "
                    "The direction of the correction matters more than its size: there is "
                    "**no dispersion hole to fill, there is a surplus to avoid**. Stacking "
                    "the incumbent's fitted ρ = 0.2757 on the measured clustering predicts "
                    "29.55 against a 22.70 target, a 30.2% overshoot — which is why this "
                    "head was decided in writing to ship on tail calibration with GP CRPS "
                    "as a non-regression bar.",
        caught_by="Building `docs/games-played-plan.md`, which needed the clustering term "
                  "as an actual input to a simulator rather than as a rhetorical share. "
                  "`serial_structure` now emits `*_rotation` keys on the matched frame so "
                  "the two can never again be divided into each other; the original rows "
                  "are untouched and still correct measurements of what they measure.",
        reproduce="make availability-profile → outputs/eda/availability_profile.csv",
        source="docs/availability-plan.md",
        reviewed="2026-08-05",
        date="2026-07-29",
        tags=("reversal", "simulator-input"),
    ),
    Decision(
        id="games-played-ships-the-calibration-not-the-fit",
        topic="simulations",
        claim="The games-played spell process ships **option (b)** — hazards inverted from "
              "the incumbent's marginal — because it beats all four fitted arms on CRPS and "
              "the tail.",
        because="Inverting `inflation = C + rho(n - C)` gives a two-state chain whose "
                "stationary play rate is exactly the incumbent's per-player predictive mean "
                "and whose lag-1 autocorrelation is exactly the measured clustering, so it "
                "reproduces a validated marginal by construction. On the TEST split it read "
                "CRPS 10.7939 against 10.7952 and cut the tail error from 0.0264 to 0.0174.",
        status="withdrawn",
        replaced_by="**No arm ships; the incumbent stands.** The decision was made on the "
                    "test split, and it reverses on validation. There the incumbent wins "
                    "CRPS (**10.0057** against the fallback's 10.0173 and "
                    "`duration_covariates`' 10.1625, the latter at P(better) = 0.1% on a "
                    "paired bootstrap over 883 rows) **and** the tail (**0.0406** against "
                    "0.0437 and 0.0537). The tail was the declared win condition and it "
                    "flips sign on both challengers. Separately the fallback's one input "
                    "leaked: `measured_clustering` ran over all 30 seasons giving C = "
                    "9.5806 where train+validation gives 9.8126, and with the leak removed "
                    "its test CRPS is 10.8008, failing that bar too — so both of its Gate D "
                    "passes were artifacts. What survives is the oracle-tenure result "
                    "(validation CRPS 7.2265 against 10.0057), everything descriptive, and "
                    "`duration_covariates`' better PIT KS, which is the one metric that "
                    "replicates in a challenger's favour on both splits.",
        caught_by="A direct question about whether such small CRPS differences could be "
                  "overfitting, followed by the standing instruction that test data must "
                  "not inform modelling decisions. Gate D as specified in the plan defines "
                  "its bars as the incumbent's *test* figures while the same plan says "
                  "'select on validation'; the implementation resolved that conflict the "
                  "wrong way instead of flagging it. Gates B and C are computed on the test "
                  "frame too and must move before this is re-decided.",
        reproduce="make stan-games-played → "
                  "outputs/predictions/stan_games_played_metrics.csv, "
                  "outputs/predictions/stan_games_played_gates.csv, "
                  "outputs/predictions/spell_process.csv, "
                  "outputs/predictions/stan_games_played_gp_pmf.csv, "
                  "outputs/predictions/stan_games_played_coefficients.csv, "
                  "outputs/predictions/stan_games_played_diagnostics.csv, "
                  "outputs/predictions/stan_games_played_pit.csv, "
                  "outputs/predictions/stan_games_played_predictions.csv",
        source="docs/games-played-plan.md",
        reviewed="2026-08-08",
        date="2026-08-05",
        tags=("reversal", "simulator-input"),
    ),
    Decision(
        id="the-tenure-is-the-games-played-bottleneck",
        topic="availability",
        claim="Given the **observed** tenure the within-tenure chain scores CRPS 7.0391 "
              "against the season-level beta-binomial's 10.9870 — 36% better. The "
              "bottleneck is not the absence process, it is knowing when a player joins "
              "and leaves a roster.",
        because="The oracle-tenure arm is not shippable — it reads the realized season — "
                "but it separates two questions the aggregate metric fuses. The process "
                "class is dramatically better than the incumbent *given* the tenure, and "
                "every bit of that advantage is destroyed by having to predict entry and "
                "exit from preseason covariates: the fitted full-window arms land at "
                "10.8981 and 10.8026 against the incumbent's 10.7952. That is mid-season "
                "roster churn, which `CLAUDE.md` already scopes out as irreducible — and "
                "this is a far more specific statement of where the remaining value sits "
                "than 'availability is hard'. It also says where NOT to spend: more "
                "flexibility on the absence process cannot recover what the tenure factors "
                "lose.",
        status="measured",
        reproduce="make stan-games-played → "
                  "outputs/predictions/stan_games_played_metrics.csv",
        source="docs/games-played-plan.md",
        reviewed="2026-08-05",
        date="2026-08-05",
        tags=("next",),
    ),
    Decision(
        id="three-state-tenure-is-a-null",
        topic="availability",
        claim="Identifying the tenure from the box-score `status` rather than structurally "
              "does **not** help — it is the only arm that fails its own floor.",
        because="The three-state arm reaches the 16.74% of missed games that fall outside "
                "the appearance window while the player is still rostered — season-ending "
                "and preseason injury, the highest-value population in the head, and the "
                "one the structural proxy files under 'not on the team'. It still loses: "
                "test CRPS 11.2470 against a floor refit on the same 2006-07+ rows at "
                "10.9795, +0.2676. Recorded so it is not rebuilt on the strength of the "
                "mechanism, which is real, rather than the result, which is negative.",
        status="null",
        reproduce="make stan-games-played → "
                  "outputs/predictions/stan_games_played_metrics.csv",
        source="docs/games-played-plan.md",
        reviewed="2026-08-05",
        date="2026-08-05",
        tags=(),
    ),
    Decision(
        id="gate-e-fails-on-validation",
        topic="availability",
        claim="**Gate E of the games-played plan fails.** Composed through "
              "`season_total.py`, the spell process scores 406.80 dk_pts of MAE and "
              "291.80 CRPS against the incumbent beta-binomial's 400.46 / 287.26 — "
              "worse on both.",
        because="It had been recorded as a ✅ on the **test** split, at 435.1053 MAE "
                "against a 435.1352 bar: a margin of 0.03 dk_pts on a ~435 quantity, "
                "seven parts in a hundred thousand, which was never evidence of "
                "anything. Re-run on validation it lands 6.34 the wrong side and loses "
                "CRPS and bias too. That is the third gate in this head to reverse on "
                "moving off the test split. The bars are no longer written down: "
                "`season_total.gate_e` reads the incumbent's own row out of whichever "
                "table it is scoring, the same fix `stan_games_played._gate_d` took. "
                "**It does not speak to the `hybrid` arm**, whose games-played pmf is "
                "the incumbent's by construction and which would therefore tie Gate E "
                "exactly — the same marginal-gate category error as Gate D, one level "
                "down.",
        status="settled",
        reproduce="make season-total → outputs/predictions/season_total_gate_e.csv",
        source="docs/games-played-plan.md",
        reviewed="2026-08-06",
        date="2026-08-05",
        tags=("gate",),
    ),
    Decision(
        id="spell-simulator-not-built",
        topic="simulations",
        claim="**The spell process is now built** (`make stan-games-played`, 2026-08-05). "
              "The residual copula over a shared `min` draw is not, and is the next piece "
              "of work.",
        because="The specification was already pinned by measurements — block variance "
                "inflation per component, a falsified 2-state Markov chain, a PSD "
                "residual correlation matrix ready to use as a copula input, and the "
                "per-game bonus overdispersion. The half that consumes the *availability* "
                "measurements now exists: an entry x exit x within-tenure chain whose "
                "hazards are calibrated to the incumbent's marginal, emitting a per-player "
                "games-played pmf and a game-level absence sequence. What remains is the "
                "composition step that turns eleven marginal posteriors into one "
                "correlated season. Validation will be posterior-predictive checks on "
                "held-out team-total variance and same-team pairwise covariance — not "
                "point accuracy.",
        status="open",
        reproduce="make stan-games-played → outputs/predictions/spell_process.csv",
        source="docs/simulations-plan.md",
        reviewed="2026-08-05",
        date="2026-07-29",
        tags=("next",),
    ),
    Decision(
        id="no-head-persists-its-posterior",
        topic="simulations",
        claim="**Every Stan head now writes its thinned posterior to disk** — `make "
              "posteriors`, one pickle per head plus a manifest. Simulating from the "
              "joint posterior no longer means refitting, and this is the only target in "
              "the simulation layer that needs a CmdStan toolchain.",
        because="`make stan` writes metrics, diagnostics and per-row predictions and "
                "threw the draws away. The one exception was `stan_composition`'s crash "
                "checkpoint, which pickles alpha/beta/rho draws incidentally to survive a "
                "9.9 h loop rather than as a consumable. Refitting cost ~137 min for the "
                "component heads and ~2.7 h for one composition arm — which a draft room "
                "cannot do once and a strategy sweep cannot do a hundred times. "
                "`src/models/posteriors.py` generalizes that checkpoint into a contract "
                "over 18 heads: 1,000 draws thinned across the WHOLE posterior via "
                "`stan_utils.thin` (never sliced off the front, for the reason "
                "`season_terms._draw_components` records), the design recipe as an "
                "ordered list of fitted steps (imputation means, log/logit transforms, "
                "the fitted SplineTransformer and its knots, the dispersion bin edges), "
                "the feature list, the fitted scaler, the selected variant read from the "
                "sweep that chose it, and provenance. Each head is verified at build "
                "time: the recipe applied to raw probe rows must reproduce the head's own "
                "design matrix and predictions, and it does bit for bit (0.00e+00 against "
                "bars of 1e-9 and 1e-8) — so a variant ladder that changes shape fails the "
                "build rather than writing a wrong artifact. The fit window is stamped in "
                "and enforceable via `require_window`, because a backtest scoring "
                "validation with heads fitted at the `full` window has read the test "
                "seasons THROUGH THE COEFFICIENTS, which no frame-level split guard can "
                "see. It also makes a walk-forward backtest affordable later without "
                "re-deciding anything.",
        status="built",
        reproduce="make posteriors → data/features/posteriors/train/manifest.csv, "
                  "data/features/posteriors/train/availability.pkl",
        source="docs/simulations-plan.md",
        reviewed="2026-08-08",
        date="2026-08-08",
        tags=("architecture",),
    ),
    Decision(
        id="game-level-dispersion-is-not-a-fit",
        topic="minutes",
        claim="`stan_minutes.game_level_dispersion` is a **data measurement, not a model "
              "output** — the 4.65x figure does not depend on the Stan fit at all.",
        because="It reads `targets` and `lengths`, computes each player-season's own realized "
                "share as `mu`, and fits a dispersion to that; the `StanMinutes` object never "
                "appears, so every Stan fit in the module could be deleted and it would still "
                "return 4.65x. It lives there by convenience. That matters because `README.md` "
                "cites it as one of two things the marginal head owns that the composition "
                "does not produce — and the composition's selected arm already fits its own "
                "game-level dispersion role-graded over four bins (rho 0.177 fringe to 0.085 "
                "star). The two sit on different parameterizations, so they are not the same "
                "number, but they are the same kind of quantity and only one can govern a "
                "draw. If the simulator draws minutes from the composition, 4.65x is a "
                "diagnostic to check those draws against rather than an input to them.",
        status="measured",
        reproduce="make stan-minutes → outputs/predictions/stan_minutes_dispersion.csv, "
                  "outputs/predictions/stan_composition_dispersion.csv",
        source="docs/simulations-plan.md",
        reviewed="2026-08-08",
        date="2026-08-08",
        tags=("architecture",),
    ),
    Decision(
        id="minutes-head-supersession-is-open",
        topic="minutes",
        claim="Whether `stan_composition` **supersedes** `stan_minutes` in the production "
              "chain is open, and one gate settles it: the composition's season-total sums "
              "against the marginal head's season-total predictions, on validation.",
        because="`README.md` says the two heads 'compose rather than compete', with the "
                "marginal head still owning the season-level mean and the game-level "
                "dispersion. Audited 2026-08-08, that sentence asserts three things and only "
                "one holds. The dispersion claim is false as stated (see "
                "[[game-level-dispersion-is-not-a-fit]]). The season-mean claim is simply "
                "**untested** — the two heads score at different units (minutes at "
                "season-total, CRPS 143.9 / MAE 199.6 / R2 0.883; composition per-team-game, "
                "CRPS 4.494 / MAE 6.33 / R2 0.474) and the composition's per-game predictions "
                "sum to a season total by construction, so it can produce the season mean and "
                "nobody has measured whether it is better. What IS load-bearing is the year "
                "effect: `season_terms` selected the `year` arm for `min` (val MAE 199.03 "
                "against base 199.72, sigma_year 0.0231), it is the only head in the project "
                "shipping a season term, and it is worth +10.4% on a 15-man roster's "
                "season-total sd — the spread a 2-of-12 knockout is decided on. "
                "`composition_glm.stan` carries no year block, so retiring the marginal head "
                "today would drop the project's only era correction. Coverage cuts the other "
                "way (the composition cannot drop rookies, since the team sum must be "
                "complete), and cost is a non-argument: 0.341 h against the composition's "
                "9.92 h.",
        status="open",
        unblocks="make minutes-unification → outputs/predictions/minutes_unification.csv",
        source="docs/simulations-plan.md",
        reviewed="2026-08-08",
        date="2026-08-08",
        tags=("next", "architecture"),
    ),
    Decision(
        id="game-length-is-drawn-not-looked-up",
        topic="simulations",
        claim="**Game length is a random variable in a forward simulation**, not a lookup, "
              "and it needs its own Stan head with a full posterior.",
        because="Every backtest so far reads `game_length` from "
                "`data/features/game_length.parquet`, because in a replay the games already "
                "happened. In the production run and in every simulated-truth season the "
                "sweep draws, nothing knows how long a game will be — and both minutes heads "
                "need it: the composition head allocates exactly 5 x game_length per "
                "team-game and the marginal head uses it as binomial trials. A point-MLE "
                "version already exists in the wrong module and the wrong form "
                "(`stan_composition.fit_ot_tail`, two floats). It needs no new .stan source: "
                "`betabinomial_glm.stan` for whether a game goes to overtime, collapsed to "
                "season cells, and `betageometric_duration.stan` for how deep — the same "
                "frailty device the absence-spell process uses one level down, and the "
                "natural fix for the plain geometric's only miss (it over-predicts 3OT+ by 3 "
                "games in 2,460). Roughly 30 collapsed rows and 2-4 parameters: the cheapest "
                "head in the project.",
        status="open",
        unblocks="make stan-game-length, registered in make posteriors",
        source="docs/simulations-plan.md",
        reviewed="2026-08-08",
        date="2026-08-08",
        tags=("next", "architecture"),
    ),
    Decision(
        id="overtime-rate-is-a-trend-not-a-wander",
        topic="simulations",
        claim="The league overtime rate carries a **real season trend** — logit slope "
              "−0.00893 per season (z = −3.38) — against season dispersion of only 1.11x "
              "binomial. The train window's 0.0608 overstates 2026-27 by ~17% relative.",
        because="Measured over 35,546 regular-season games: the fitted rate falls from "
                "0.0670 in 1996-97 to 0.0526 in 2025-26 and extrapolates to 0.0521 for "
                "2026-27; pooled over the last five seasons it is 0.0504 against 0.0613 over "
                "the first twenty-five. Near-binomial season dispersion is what makes this a "
                "trend rather than a wander, the distinction `src/eda/season_effects.py` "
                "exists to draw, so this would be the second head to ship a season term "
                "after minutes. **Size it honestly**: the trend is worth about 0.09% of "
                "total minutes, so it is not a mean-effects story. It is a tail story — "
                "overtime is where 40+ minute games come from (1,650 player-games exceed 48 "
                "minutes, maximum 63.0), and under a best-ball weekly max plus a threshold "
                "bonus an OT frequency 17% too high inflates every star's simulated ceiling, "
                "which is the statistic a 2-of-12 pod is most sensitive to.",
        status="measured",
        reproduce="make game-length → data/features/game_length.parquet, "
                  "outputs/eda/game_length_coverage.csv",
        source="docs/simulations-plan.md",
        reviewed="2026-08-08",
        date="2026-08-08",
        tags=("season-effects",),
    ),
    Decision(
        id="overtime-is-shared-by-both-teams",
        topic="simulations",
        claim="The game-length draw happens **once per game, shared by both teams** — never "
              "per team-game and never per player.",
        because="Overtime is a property of the game: every player on the floor gets the extra "
                "minutes together. Drawing it per team-game would silently destroy that, and "
                "it is a real source of the correlated upside the tournament objective "
                "rewards — a same-team stack, which the strategy layer explicitly considers, "
                "shares its overtimes. It joins the four existing rules the season simulator "
                "must not violate, and it is the kind of wiring error that produces a "
                "plausible marginal and a wrong joint, which is exactly what this layer is "
                "built to get right.",
        status="settled",
        reproduce="make game-length → data/features/game_length.parquet",
        source="docs/simulations-plan.md",
        reviewed="2026-08-08",
        date="2026-08-08",
        tags=("architecture",),
    ),
    Decision(
        id="posteriors-fit-on-train-not-train-val",
        topic="simulations",
        claim="Persisted posteriors default to the **`train`** fit window, not `train_val`, "
              "and artifacts are namespaced by window. The plan's original "
              "`fit_window: train_val` was reversed on 2026-08-08 before anything consumed "
              "it.",
        because="The reasoning for `train_val` was that it matches the four simulator "
                "inputs already calibrated that way — the residual copula, the game-level "
                "minutes dispersion, the block variance inflation and the bonus "
                "overdispersion. That analogy does not transfer. Those four are GIVEN to "
                "the simulator and never scored against realized data; they set the shape "
                "of the noise. The posterior coefficients generate the board, and the "
                "realized backtest replays portfolios drafted from that board against real "
                "2022-23 and 2023-24 box scores — the validation seasons. At `train_val` "
                "every one of those rows is in the fit: 883 of 10,361 availability rows "
                "(8.5%) and 773 of 9,403 component rows (8.2%). Per row the leverage on a "
                "~12-parameter GLM is tiny, but as a class this is the shape of all four "
                "sub-1% reversals already logged in this repo. `train_val` is not "
                "discarded, it moves to the consumer it fits: the one-shot test readout on "
                "2024-25 / 2025-26, which should describe the model that would actually "
                "deploy — the rule src/final_evaluation.py already follows. `full` remains "
                "the production board and is guarded. All three are wanted at once, so "
                "data/features/posteriors/<window>/ replaces a flat layout in which the "
                "second window silently overwrote the first.",
        status="settled",
        reproduce="make posteriors → data/features/posteriors/train/manifest.csv",
        source="docs/simulations-plan.md",
        reviewed="2026-08-08",
        date="2026-08-08",
        tags=("methodology", "leakage"),
    ),
    Decision(
        id="sim-tensor-is-player-by-period",
        topic="simulations",
        claim="The simulator's contract to everything downstream is a "
              "`player x scoring_period x sim` tensor of dk_pts — **never a player-game "
              "array**.",
        because="Best ball scores by scoring period, and there are only 20 of them (Round "
                "1's 17 weeks plus three double weeks). At ~550 players and 2,000 sims "
                "that is ~88 MB in float32 — small enough to hold for a whole strategy "
                "sweep and to load into a draft room in under a second. Per-game draws "
                "still happen inside the simulator, because the double-double bonus is a "
                "per-game threshold on five components and E[bonus] != bonus(E[x]), but "
                "they are summed into periods immediately. Fixing this contract is the "
                "difference between a draft sweep that runs in minutes and one that runs "
                "in hours, and it is what makes the sub-second in-draft recompute "
                "achievable.",
        status="settled",
        reproduce="make component-targets → outputs/eda/bonus_calibration.csv",
        unblocks="src/sim/season.py writes data/features/sim_tensor_<season>.npz",
        source="docs/simulations-plan.md",
        reviewed="2026-08-08",
        date="2026-08-08",
        tags=("architecture",),
    ),
    Decision(
        id="simulated-truth-needs-error-injection",
        topic="simulations",
        claim="Strategy tuning runs on **simulated truth with the model's measured "
              "out-of-sample error injected**. An uninjected simulated backtest cannot "
              "price ADP, exposure caps, or any other hedge against model error.",
        because="A season drawn from the model's own posterior is a world where the model "
                "is perfectly calibrated by construction, so ADP can only add noise and "
                "the sweep drives alpha to zero for reasons that have nothing to do with "
                "whether the market knows something. The same failure hits every "
                "error-hedging strategy. So the truth draw is perturbed to reproduce the "
                "measured miss — availability CRPS 10.006 games, component R2 0.81-0.95 "
                "against the no-fit floors, season-total MAE 400.5 dk_pts — before "
                "anything is scored against it. An uninjected sweep is not a conservative "
                "version of this; it answers a different question and its alpha is not "
                "transportable. Realized 2022-23 / 2023-24 remains the honest readout, at "
                "N = 2 seasons.",
        status="settled",
        reproduce="make stan-components → outputs/predictions/stan_component_metrics.csv",
        unblocks="Gate C in docs/simulations-plan.md",
        source="docs/simulations-plan.md",
        reviewed="2026-08-08",
        date="2026-08-08",
        tags=("methodology",),
    ),
    Decision(
        id="scoring-periods-are-nba-weeks",
        topic="simulations",
        claim="DK's scoring periods are **NBA week ranges** — derive them from "
              "`ScheduleLeagueV2`'s `weekNumber`, do not re-derive weeks from raw dates.",
        because="The NBA's own week numbering runs Monday-Sunday and partitions game "
                "dates with zero dates in more than one week. `weekNumber` is populated "
                "from 2017-18 only, so the 21 older seasons need a derivation, and the "
                "one that ships — dense-rank the Mondays that carry games — reproduces "
                "the NBA's own numbering on **10,749 of 10,749 games across all nine "
                "seasons that publish one**. Dense-ranking rather than counting elapsed "
                "weeks is the whole trick: in 2019-20 the NBA numbered the bubble "
                "restart weeks 22-24, consecutive with March, where elapsed calendar "
                "weeks give 41-43. The module owns the three edge cases once — a "
                "postponed game scores in the period it is played (the schedule endpoint "
                "serves realized dates, so this is 0 of 7,380 games today and stops "
                "being free the moment a forward schedule is read), the NBA Cup final "
                "scores nowhere, and the all-star gap moves no Monday and so shifts no "
                "period. Round 1 = 17 weeks and Rounds 2-4 = one double week each is "
                "asserted for all 27 full-length seasons; the 1998-99 and 2011-12 "
                "lockouts and 2020-21 are too short to close every round, which is the "
                "case DK's own shortened-season rule already covers.",
        status="built",
        reproduce="make scoring-periods → data/features/scoring_periods.parquet, "
                  "outputs/eda/scoring_periods_audit.csv, "
                  "outputs/eda/scoring_periods_rounds.csv",
        source="docs/simulations-plan.md",
        reviewed="2026-08-08",
        date="2026-08-08",
        tags=("rules",),
    ),
    Decision(
        id="production-schedule-not-published",
        topic="simulations",
        claim="The **2026-27 regular-season schedule is not published**, which gates the "
              "production run only. Rosters are already live.",
        because="`ScheduleLeagueV2` returns 20 rows for 2026-27 — 19 preseason games plus "
                "the 12/11/2026 NBA Cup final that does not score — against 1,400 rows "
                "for 2025-26. `commonteamroster` already returns 2026-27 rosters, with "
                "POSITION null for unsigned and two-way players, which the DK board "
                "covers. Because the backtest seasons all have realized schedules, every "
                "backtest piece can be built now and only the production season waits. "
                "The schedule is normally released in mid-August.",
        status="blocked",
        unblocks="the NBA publishes the 2026-27 schedule; poll ScheduleLeagueV2",
        source="docs/simulations-plan.md",
        reviewed="2026-08-08",
        date="2026-08-08",
        tags=("capture", "production"),
    ),

    # ══ Drafting strategy ════════════════════════════════════════════════════
    Decision(
        id="round-one-is-a-zero-consolation-knockout",
        topic="drafting",
        claim="Round 1 is a zero-consolation knockout in all five tournaments, so the "
              "objective is **P(advance)**, not E[score].",
        because="Top 2 of 12 advance and ranks 3–12 receive $0. That changes the "
                "objective function outright — variance is worth paying for near the "
                "cut line — and it is why the cascading tie-break mechanics stop being "
                "a footnote. **Sharpened 2026-08-08: Round 1 is the *only* "
                "zero-consolation round**, and that reframes the objective rather than "
                "overturning it. Surviving it guarantees a cash in both target "
                "tournaments — `600k_shootaround` pays 11 of 12 Round-2 entries at $30 "
                "minimum on a $20 entry, and `20k_spin_move` pays or advances all 6 at "
                "$80 minimum on a $52 entry. So P(any return) = P(top 2 of 12) exactly, "
                "and everything past Round 1 sets the size of the return rather than its "
                "sign. 'Convex, therefore chase the tail' holds only for "
                "`600k_shootaround`, and only above that Round-2 floor.",
        status="measured",
        reproduce="make adp-profile → outputs/eda/adp_profile.csv",
        source="docs/dk_best_ball_rules.md",
        reviewed="2026-08-08",
        date="2026-07-28",
        tags=("economics",),
    ),
    Decision(
        id="rake-is-the-break-even-edge-hurdle",
        topic="drafting",
        claim="Quote rake as the break-even **edge hurdle**, `1/(1−rake) − 1`, not as a "
              "raw percentage.",
        because="A raw rake percentage is not denominated like a measured edge, so the "
                "two cannot be compared. On the hurdle scale the cheap tournaments "
                "demand substantially more edge just to return the fee than the "
                "expensive ones do — which inverts the intuition that a $1 entry is the "
                "forgiving one.",
        status="measured",
        reproduce="make adp-profile → outputs/eda/adp_profile.csv",
        source="docs/dk_best_ball_rules.md",
        reviewed="2026-07-30",
        date="2026-07-28",
        tags=("economics",),
    ),
    Decision(
        id="adp-in-strategy-layer",
        topic="drafting",
        claim="ADP belongs in the strategy layer, not in the GLMM.",
        because="Under a knockout payout the edge *is* model-minus-market, so a model "
                "fit on ADP reproduces its own benchmark and the edge goes to zero by "
                "construction. The one narrow exception is an ADP prior for thin-data "
                "players only, with a pre-registered test.",
        status="settled",
        reproduce="make adp-profile → outputs/eda/adp_profile.csv, "
                  "data/features/adp_transfer.parquet",
        source="docs/adp-plan.md",
        reviewed="2026-07-30",
        date="2026-07-28",
        tags=("architecture",),
    ),
    Decision(
        id="dk-id-is-a-persistent-player-key",
        topic="drafting",
        claim="The DraftKings `ID` is a **persistent** player key across boards — 667 "
              "shared with 100% name agreement.",
        because="That makes the DK-side join an id join rather than a name join, which "
                "is the only reason the board data escapes the name-matching hazard. "
                "249 of 698 rows carry ADP and the board is right-censored near pick "
                "186.",
        status="measured",
        reproduce="make adp-draftkings → data/features/adp_draftkings.parquet, "
                  "data/features/adp_dk_id_map.parquet",
        source="docs/adp-plan.md",
        reviewed="2026-07-30",
        date="2026-07-28",
        tags=("joins",),
    ),
    Decision(
        id="adp-freeze-rule",
        topic="drafting",
        claim="A snapshot's **calendar date is not its season** — the ADP table is "
              "frozen for ~11 months.",
        because="The 2025-09-06 snapshot is 2024-25 ADP, and any `month >= 10` rule "
                "misassigns six 2020 snapshots because 2020-21 tipped off in December. "
                "Getting this wrong silently shifts an entire season of market data by "
                "one year, which would look like a modelling result rather than a "
                "join bug.",
        status="settled",
        reproduce="make adp-panel → data/features/adp_panel.parquet, "
                  "data/features/adp_fantasypros.parquet",
        source="docs/adp-plan.md",
        reviewed="2026-07-30",
        date="2026-07-28",
        tags=("joins", "failure-mode"),
    ),
    Decision(
        id="adp-recalibration-ladder",
        topic="drafting",
        claim="Recalibrating FantasyPros consensus onto the DK board is worth a large "
              "share of the raw gap — but it rests on **one** anchor.",
        because="Raw 24.38 → monotone 17.32 picks of mean absolute error, with the "
                "position offset adding a further −2.0. Centers sit at +13.88 and the "
                "rounds-9+ tier gap at 32.31, which is where 9 of 16 picks are made. "
                "`n_anchors = 1` is the caveat that has to be rendered loudly: a single "
                "timing-matched board pair is one observation of the mapping, not a "
                "fitted transfer function. The offset figure read −0.3 until 2026-07-31 "
                "— the planning-session value on 218 pairs, mixed into an entry whose "
                "other numbers came from the 226-pair artifact. It weakens 'one anchor "
                "identifies a shape' without overturning it: the offset is 11.6% of the "
                "recalibrated error against the monotone step's own −7.1 picks.",
        status="measured",
        reproduce="make adp-profile → outputs/eda/adp_profile.csv, "
                  "data/features/adp_transfer.parquet",
        source="docs/adp-plan.md",
        reviewed="2026-07-31",
        date="2026-07-28",
        tags=("market",),
    ),
    Decision(
        id="october-2026-dk-board",
        topic="drafting",
        claim="An early-to-mid October 2026 DraftKings board must be downloaded by hand "
              "from the draft lobby while contests are open.",
        because="The board is login-gated, has **zero** Wayback snapshots, exposes no "
                "API, and is live only while contests are open (~Oct). It cannot be "
                "scraped or backfilled: a board not downloaded while it is open is gone "
                "permanently. Two are captured (2025-10-17, 2026-07-28); the "
                "load-bearing one is timing-matched to the October 2025 anchor, which "
                "is what would take `n_anchors` from 1 to 2.",
        status="deadline",
        due="2026-10-15",
        source="docs/adp-plan.md",
        reviewed="2026-07-30",
        date="2026-07-28",
        tags=("capture",),
    ),
    Decision(
        id="wayback-adp-backfill",
        topic="drafting",
        claim="The FantasyPros Wayback backfill is at 23 of 259 archived snapshots and "
              "needs re-running.",
        because="The sweep was blocked on 2026-07-28 by Wayback throttling (HTTP 498, "
                "its rate-limit code), which clears on its own. The run is resumable "
                "and skips anything already archived, so re-running costs nothing and a "
                "partial run is safe to repeat. Unlike the DK board this one *can* be "
                "backfilled — but the archive is not permanent, so it is not "
                "indefinitely deferrable either.",
        status="deadline",
        due="2026-08-01",
        source="docs/adp-plan.md",
        reviewed="2026-07-30",
        date="2026-07-28",
        tags=("capture",),
    ),
    Decision(
        id="two-strategies-two-tiers",
        topic="drafting",
        claim="**Two strategies ship this year**: 10 entries at $20 "
              "(`600k_shootaround`) and 4 entries at $52 (`20k_spin_move`).",
        because="Near-equal stake — $200 against $208 — across two structures whose "
                "objectives differ in shape rather than scale, so comparing them is "
                "itself a result. `20k_spin_move`'s path is three successive shallow cuts "
                "(2/12 → 2/6 → 2/6) into a nearly flat final table where all 8 finalists "
                "clear $750 on a $52 entry, so it rewards survival and durability; "
                "P(reach round 4) at random is 1.85%. `600k_shootaround` narrows "
                "brutally after round 1 (2/12 → 1/12 → 1/10) into a $200,000 top prize at "
                "10,000x entry, so above its round-2 floor it rewards correlated upside "
                "and differentiation from the field; P(reach round 4) at random is "
                "0.139%. Its rake hurdle is also 43% higher (+17.60% against +12.32%). "
                "Confirming the sweep actually selects different rosters for the two is "
                "Gate D.",
        status="settled",
        reproduce="make dashboard → dashboard/economics.py, "
                  "data/raw/dk_best_ball_tournament_metadata.csv, "
                  "data/raw/dk_best_ball_tournament_prize_structure.csv",
        source="docs/simulations-plan.md",
        reviewed="2026-08-08",
        date="2026-08-08",
        tags=("economics", "strategy"),
    ),
    Decision(
        id="test-split-is-a-pure-readout",
        topic="drafting",
        claim="The test seasons get **one** strategy backtest before going live, and it "
              "changes **nothing** — not the strategy, not the stake, not the entry "
              "decision.",
        because="The strategy is frozen on validation and written to an artifact; the "
                "test runner reads which strategy shipped rather than re-deciding, "
                "exactly as `src/final_evaluation.py` does, and emits a risk report — ROI "
                "distribution, P(advance), P(cash), worst-case drawdown across the 10 + 4 "
                "entries — with no ranking and no recommendation. It runs inside "
                "`held_out.unlocked()` so the unlock is visible in the log, and the sweep "
                "itself goes through `selection_split` and never materializes the test "
                "rows. Prose already failed once here: the games-played head's Gate D was "
                "specified with test figures as its bars and settled which model ships, "
                "on a margin a paired bootstrap could not distinguish from zero.",
        status="settled",
        reproduce="make final-evaluation → src/models/held_out.py, "
                  "src/final_evaluation.py",
        unblocks="src/sim/strategy.py and its final-evaluation counterpart",
        source="docs/simulations-plan.md",
        reviewed="2026-08-08",
        date="2026-08-08",
        tags=("methodology", "split"),
    ),
    Decision(
        id="reactive-draft-is-primary",
        topic="drafting",
        claim="**Reactive live-pick is the primary draft mode**, driven by a local "
              "draft-room UI with one click per pick. Ranking-submission is the fallback.",
        because="This resolves a contradiction the plan carried in two places — its "
                "mechanics section concluded ranking-submission should lead and its "
                "build-list said the opposite. Fewer, higher-conviction entries drafted "
                "manually is the year's plan, which makes a real-time recommender the "
                "deliverable. A local Streamlit page over the precomputed board needs no "
                "external access and works at a 30-second clock; reading the DK page via "
                "the browser extension is worth exploring for 8-hour slow drafts but is "
                "explicitly not on the critical path. Ranking-submission still gets built, "
                "because a fast clock can outrun a human and because the opponent model "
                "needs DK's documented autodraft logic (queue → ranking → 8G/8F/3C caps) "
                "regardless.",
        status="settled",
        reproduce="make adp-draftkings → "
                  "data/raw/dk_draft_rankings/DkPreDraftRankings_July28_2026.csv, "
                  "data/features/adp_draftkings.parquet",
        unblocks="dashboard/draft_room.py and src/sim/draft.py",
        source="docs/simulations-plan.md",
        reviewed="2026-08-08",
        date="2026-08-08",
        tags=("strategy", "product"),
    ),
    Decision(
        id="select-on-p-advance-report-roi",
        topic="drafting",
        claim="The in-draft objective is **payout-weighted EV over the full bracket**, "
              "but the sweep **selects on lift in P(top 2 of 12)** and reports ROI "
              "alongside it.",
        because="The two statistics are not equally measurable on the same simulation "
                "budget. ROI is dominated by rare deep runs — `600k_shootaround` reaches "
                "round 4 on 0.139% of entries — so its Monte Carlo error is enormous. "
                "P(top 2 of 12) is a 16.67% event and resolves orders of magnitude "
                "faster, and since surviving round 1 is exactly the condition for any "
                "return at all, its lift over an ADP-drafted entry is a defensible "
                "headline rather than a proxy. ROI is still reported, against the "
                "break-even hurdle and with its interval. Simulating the whole bracket "
                "also prices something scoring rounds independently cannot: the round-2 "
                "to round-4 field is not an ADP field, it is the population that already "
                "cleared a 2-of-12 cut, so an independent-field model would systematically "
                "overstate continuation value.",
        status="settled",
        reproduce="make dashboard → dashboard/economics.py, "
                  "data/raw/dk_best_ball_tournament_prize_structure.csv",
        unblocks="src/sim/bracket.py and src/sim/strategy.py",
        source="docs/simulations-plan.md",
        reviewed="2026-08-08",
        date="2026-08-08",
        tags=("methodology", "strategy"),
    ),
    Decision(
        id="dk-position-eligibility-from-rosters",
        topic="drafting",
        claim="DK position eligibility comes from `data/raw/team_rosters_*.csv`, whose "
              "`POSITION` column carries DK-shaped dual eligibility for all 30 seasons "
              "with **zero** nulls.",
        because="Without it no lineup can be filled at all — the weekly slate is 2 G / 2 F "
                "/ 1 C / 2 UTIL and a player's eligibility decides which of them he can "
                "occupy. The column already encodes duals the way DK does (`G-F`, `F-C`, "
                "`C-F`, `F-G`), and the two `data/raw/dk_draft_rankings/*.csv` boards "
                "carry DK's *own* positions for 698 and 942 players, which is what the "
                "NBA.com → DK mapping gets validated against rather than assumed. The "
                "2026-27 rosters do carry nulls, for unsigned and two-way players, and "
                "the DK board covers exactly those.",
        status="measured",
        reproduce="make fetch → data/raw/team_rosters_2023_24.csv, "
                  "data/features/adp_dk_id_map.parquet",
        source="docs/simulations-plan.md",
        reviewed="2026-08-08",
        date="2026-08-08",
        tags=("joins", "rules"),
    ),
)
