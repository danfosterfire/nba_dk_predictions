"""The live draft room — one click per pick, a ranked recommendation in ~110 ms.

**Two launches, one body.** `make draft-room` runs this file directly, and that is what
draft night uses: a thirty-second clock should not share a process with anything. The
dashboard also carries it as page 9, through the three-line `dashboard/views/draft_room.py`
that imports `render()`. The split is the whole of it — `main()` is
`st.set_page_config` plus `render()`, so the page config is set exactly once by whichever
entrypoint owns the process, and everything on screen is written once.

That a page this expensive can live in the app at all is a property of `st.navigation`
rather than a judgement call: a page's script does not run until the reader selects it, and
`@st.cache_resource` holds the ~40 MB reference field for the life of the process, so
coming *back* to the room costs a rerun rather than a rebuild. Measured in a browser rather
than assumed — see `docs/dashboard-plan.md`, "Step 7, as built".

Everything that computes anything lives in `src/sim/draft_room.py`. This file is the
surface: it holds the pick log, draws the board, and turns a click into
`draft_room.mark_pick`. **One click always means the same thing — "the seat on the clock
just took this player"** — so a drafter never has to say *whose* pick it was, and our own
pick and an opponent's are the same gesture. The snake order says the rest.

## The one rule this page breaks, and why

`dashboard/README.md`: "Nothing in this package imports from `src/`." This file does, and
it is the only one that does. The rule exists so the dashboard cannot refit, re-project or
re-cluster — so a *view* can never drift from the fit it is describing. This page does not
fit anything either; it reads `make simulate-season`'s tensor, `make draft-pool`'s board
and `make draft-room`'s cached field, and every number it prints comes out of
`src/sim/`. What it needs from `src/` is `bracket.best_lineup` and `draft.legal_mask`, and
the alternative to importing them is *reimplementing* them — a second copy of the matroid
that seats a weekly lineup and a second opinion about which players are legal. That is
precisely the drift the rule was written to prevent, arriving through the other door.

So the invariant is narrowed rather than waived: `tests/test_dashboard.py` still fails on
any other file, and it pins that this one reaches no further than `src.sim`, which is the
numpy layer over the artifacts and imports no Stan. Joining the navigation did not widen
it — `SRC_IMPORTERS` still names one file, and it now names it by *path*, so the view
wrapper cannot inherit the exemption by sharing a basename.

## What is on screen, and why each piece is there

- **The clock line** — pick, round, who is up. Wrong-seat bookkeeping is the failure that
  silently invalidates everything below it.
- **Counts against 8 G / 8 F / 3 C and against the 2 G / 2 F / 1 C slate.** DK's caps are
  maxima that bind autodraft; the slate is a minimum that binds the lineup. A roster can
  satisfy every cap and still seat nobody at centre, so both are shown and `at_risk` warns
  a round before the choice disappears.
- **What the pick costs.** Every row carries its gap to the top of the table in the ranked
  unit, because the question at eight seconds on the clock is not "who is best" but "how
  much am I giving up by taking the guy I already wanted".
- **The runner-up, explicitly.** The recommendation is a ranking over a simulated field
  and the top two are often inside its noise; hiding that would make the page more
  confident than the model.
"""

import sys
import time
from datetime import datetime
from pathlib import Path

# The repo root on the path so `dashboard.*` and `src.*` resolve: `streamlit run` puts the
# *script's* directory on sys.path, not the project root.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import streamlit as st

from dashboard import economics, shell
from dashboard.artifacts import load_cfg
from src.sim import draft, draft_room

BOARD_ROWS = 24                # best-available rows offered for a one-click "gone"
SEARCH_ROWS = 12
ROW_WIDTHS = [3.2, 1, 1, 1.1, 1]   # player · EV · P(top 2) · cost vs best · cushion

# Streamlit's default button is a text-sized target. A draft room is used with thirty
# seconds on the clock, so the pick buttons are given a real hit area and the primary one
# is given a colour that survives a glance.
#
# **The root size is dropped to 15px and everything below is in `rem`**, so one number
# shrinks the whole page proportionally rather than each rule fighting the others. Two
# Streamlit defaults are also overridden outright, and both were cutting text rather than
# merely crowding it: a metric label is `nowrap` with an ellipsis, so "Lead over runner-up"
# rendered as "Lead over runn…" in a narrow column, and the column headers wrapped
# mid-phrase at the default body size.
PAGE_CSS = """
<style>
  html { font-size: 15px; }
  div[data-testid="stVerticalBlock"] div.stButton > button {
      min-height: 2.5rem; font-size: 0.92rem; font-weight: 600; padding: 0.2rem 0.6rem;
  }
  div.stButton > button[kind="primary"] { min-height: 3.0rem; font-size: 1.02rem; }
  [data-testid="stMetricValue"] { font-size: 1.2rem; }
  [data-testid="stMetricLabel"] { white-space: normal; overflow: visible; }
  [data-testid="stMetricLabel"] p {
      font-size: 0.72rem; white-space: normal; overflow: visible; text-overflow: clip;
      line-height: 1.15;
  }
  [data-testid="stCaptionContainer"] p { font-size: 0.78rem; }
</style>
"""

# The numeric columns of the recommendation table. Set here rather than inline so the
# header row and the cells cannot drift apart.
HEAD_STYLE = "font-size:0.78rem;font-weight:600;opacity:0.8;white-space:nowrap"
CELL_STYLE = "font-size:0.92rem"


# ── Loading ───────────────────────────────────────────────────────────────────

def seasons_with_a_tensor(features: Path) -> list[str]:
    return sorted(p.name[len("sim_tensor_"):-len(".npz")]
                  for p in features.glob("sim_tensor_*.npz"))


@st.cache_data(show_spinner=False, ttl=300)
def open_injuries(raw_dir: str) -> pd.DataFrame:
    """Both injury feeds' latest word. Short TTL so a fresh capture appears mid-draft."""
    return draft_room.load_injury_notes(raw_dir)


@st.cache_resource(show_spinner="Drafting the reference field — once per season…")
def open_room(season: str, n_sims: int, field_drafts: int):
    """`load_room`, cached for the life of the process.

    `cache_resource` rather than `cache_data` because the room is ~40 MB of numpy that
    must not be copied per rerun, and because it is read-only after construction — the
    draft's own state lives in `st.session_state` and is rebuilt from the pick log.
    """
    return draft_room.load_room(load_cfg(), season, n_sims=n_sims,
                                n_field_drafts=field_drafts)


def state_for(room, picks: list[int]) -> draft.DraftState:
    return draft_room.replay(room, picks)


# ── Formatting ────────────────────────────────────────────────────────────────

def money(value: float, places: int = 0) -> str:
    r"""A dollar figure Streamlit will not read as LaTeX.

    Streamlit's markdown treats `$…$` as inline maths, so an unescaped pair of dollar
    signs in one caption swallows the sentence between them and renders it in italic
    serif. It looked like a broken f-string in the browser and like nothing at all in
    `AppTest`, which is the case for looking at the real page.
    """
    return f"\\${value:,.{places}f}"


def injury_panel(rows: pd.DataFrame, title: str) -> None:
    """Every note in full, under an expander that says how many there are.

    The badge on the button says *that* a player is hurt; this says *what*, which is the
    part a drafter actually decides on — "Out, right Achilles, back ~April" and
    "Day-To-Day, sore calf" are the same badge and not remotely the same pick. Both feeds
    are shown separately with their capture date rather than reconciled, because they are
    two independent reports and a merged one would hide which said what.
    """
    if rows is None or rows.empty:
        return
    with st.expander(f"⚕ {title} — {rows['board_index'].nunique()} flagged"):
        for row in rows.itertuples():
            headline = f" — {row.headline}" if row.headline else ""
            st.markdown(f"**{row.player_name}** · `{row.status}`{headline}  \n"
                        f"<small>{row.source} as of {row.as_of} "
                        f"({row.age_days}d old){' · ' + row.detail if row.detail else ''}"
                        f"</small>", unsafe_allow_html=True)


def choice(label: str, options: list[str], key: str, default: int, **kwargs) -> str:
    """A selectbox whose choice outlives the reader leaving the page.

    Standalone this is exactly `st.selectbox`; as page 9 of the dashboard it is what stops
    a navigation away and back from silently re-pointing the room at a different season or
    tournament while the pick log stays put. A remembered option that is no longer on offer
    — a season whose tensor has gone — falls back to the caller's default rather than
    raising.
    """
    was = shell.recall(key, None)
    index = options.index(was) if was in options else default
    return shell.remember(key, st.selectbox(label, options, index=index, **kwargs))


def format_gap(value: float, objective: str) -> str:
    """A difference in the unit the table is ranked in, signed and readable at a glance."""
    if objective == "bracket_ev":
        return ("−" if value < 0 else "") + money(abs(value))
    if objective == "p_advance":
        return f"{value * 100:+.2f} pp"
    return f"{value:+,.0f} dk"


# ── Interaction — every click goes through here ───────────────────────────────

def take(player: int) -> None:
    """Record the pick, and record what the room had advised at the moment it was made.

    The annotation is read from the *previous* render, which is what `st.session_state`
    holds when an `on_click` callback fires — and it is the only moment the advice and the
    choice exist together. Reconstructing it afterwards is impossible: the board has moved
    on, and re-ranking from the finished log would score the pick against a state that
    never existed when it was made.
    """
    st.session_state.picks.append(int(player))
    ranking = st.session_state.get("last_ranking")
    note: dict = {"taken_at": datetime.now().isoformat(timespec="seconds")}
    if ranking is not None and len(ranking):
        row = ranking[ranking["player_id"] == st.session_state.board_ids[int(player)]]
        note |= {
            "recommended": ranking["player_name"].iloc[0],
            "recommended_value": float(ranking["value"].iloc[0]),
            "taken_value": float(row["value"].iloc[0]) if len(row) else None,
            "followed": bool(len(row) and row.index[0] == 0),
            "recompute_ms": st.session_state.get("last_recompute_ms"),
        }
    st.session_state.annotations.append(note)


def undo() -> None:
    if st.session_state.picks:
        st.session_state.picks.pop()
        if st.session_state.annotations:
            st.session_state.annotations.pop()


def pick_button(room, player: int, key: str, label: str | None = None,
                primary: bool = False, badge: str | None = None) -> None:
    """One button, one pick. The seat on the clock is inferred, never asked for.

    The injury badge goes **on the button** rather than in a column beside it, because the
    button is what the eye is already on and what the hand is about to hit. A status in a
    fifth column is a status nobody reads with eight seconds on the clock.
    """
    row = room.frame.iloc[player]
    text = label or f"{row['player_name']}  ·  {row['position']}  ·  {row['team']}"
    if badge:
        text = f"{text}   ⚕ {badge}"
    st.button(text, key=key, on_click=take, args=(player,),
              width="stretch", type="primary" if primary else "secondary")


# ── The page ──────────────────────────────────────────────────────────────────

def render() -> None:
    """The page. Called by `main()` standalone and by `views/draft_room.py` in the app.

    It sets no page config, because that is the one call a Streamlit process may make
    exactly once and it belongs to whichever entrypoint owns the process.
    """
    st.markdown(PAGE_CSS, unsafe_allow_html=True)

    cfg = load_cfg()
    sim = cfg.get("sim", {})
    room_cfg = sim.get("draft_room", {})
    features = ROOT / cfg["data"]["features_dir"]
    seasons = seasons_with_a_tensor(features)
    if not seasons:
        st.error("No `sim_tensor_<season>.npz` under "
                 f"`{cfg['data']['features_dir']}` — run `make simulate-season` first.")
        return

    # Four of these five say what the pick log *means*, so they are remembered rather than
    # defaulted — see `choice`. `top` is cosmetic and is left to reset.
    with st.sidebar:
        st.header("Draft room")
        season = choice("Season board", seasons, "season", len(seasons) - 1)
        entered = set(sim.get("tournaments", {}) or {})
        structures = economics.advance_table().merge(
            economics.load_metadata().rename(columns={"type": "tournament"}),
            on="tournament")
        fees = dict(zip(structures["tournament"],
                        structures["entry_fee_per_team"]))
        tournaments = sorted(fees, key=lambda t: -entered.__contains__(t))
        tournament = choice(
            "Tournament", tournaments, "tournament", 0,
            format_func=lambda t: f"{'★ ' if t in entered else ''}{t} · ${fees[t]:,.0f}")
        seat = st.number_input(
            "Your seat", 1, int(sim.get("pod_size", 12)),
            shell.recall("seat", int(room_cfg.get("seat", 0)) + 1)) - 1
        shell.remember("seat", seat + 1)
        objective = choice("Rank by", list(draft_room.RANK_OBJECTIVES), "objective",
                           list(draft_room.RANK_OBJECTIVES).index(
                               room_cfg.get("objective", "bracket_ev")))
        top = st.slider("Candidates shown", 5, 20, 10)
        st.divider()
        st.button("↩︎ Undo last pick", on_click=undo, width="stretch")
        if st.button("Reset draft", width="stretch"):
            st.session_state.picks = []
            st.session_state.annotations = []
            st.rerun()

    st.session_state.setdefault("picks", [])
    st.session_state.setdefault("annotations", [])
    # One log file per browser session, so resetting the draft starts a fresh file rather
    # than overwriting the one that recorded a real pod.
    st.session_state.setdefault("session_id", datetime.now().strftime("%Y%m%d-%H%M%S"))
    try:
        room = open_room(season, int(sim.get("n_sims_draft", 500)),
                         int(room_cfg.get("field_drafts", draft_room.N_FIELD_DRAFTS)))
    except Exception as error:                       # noqa: BLE001 — shown, not swallowed
        st.error(f"{type(error).__name__}: {error}")
        return

    st.session_state.board_ids = room.frame["player_id"].to_numpy()
    notes = open_injuries(str(ROOT / cfg["data"]["raw_dir"]))
    flagged, coverage = draft_room.match_injuries(room, notes)
    badges = draft_room.injury_badges(flagged)
    state = state_for(room, st.session_state.picks)
    complete = state.pick_index >= room.pod_size * draft.N_ROUNDS
    on_clock = None if complete else draft_room.seat_on_clock(state)
    ours = on_clock == seat
    health = draft_room.roster_health(room, state, seat)

    # Autosaved on every rerun rather than on a button, because a live draft is exactly
    # where a closed tab costs something no `make` target can rebuild.
    log = draft_room.pick_log(room, st.session_state.picks, seat, tournament, objective,
                              st.session_state.annotations)
    saved = draft_room.log_path(ROOT / "outputs" / "draft_logs", season,
                                st.session_state.session_id)
    if len(log):
        draft_room.save_pick_log(log, saved)

    with st.sidebar:
        st.download_button("⬇ Download pick log", log.to_csv(index=False),
                           file_name=saved.name, mime="text/csv",
                           disabled=log.empty, width="stretch")
        st.caption(f"Autosaved to `{saved.relative_to(ROOT)}` after every pick.")
        st.divider()
        st.markdown("**Injury feeds**")
        sources = draft_room.injury_sources(notes)
        if sources.empty:
            st.caption("No injury capture on disk — run `make daily-capture`.")
        for row in sources.itertuples():
            st.caption(f"`{row.source}` · {row.rows} players · as of **{row.as_of}** "
                       f"({row.age_days}d old)")
        st.caption(f"{coverage['players_flagged']} on this board; "
                   f"{coverage['unmatched']} listed players are not draftable here"
                   + (f"; **{coverage['ambiguous']} ambiguous names left unattached**"
                      if coverage["ambiguous"] else ""))
        st.caption(
            f"`{room.season}` board · {room.board.n_players:,} players, "
            f"{int(room.scorable.sum()):,} priceable · {room.n_sims:,} sims at the "
            f"`{room.fit_window}` window · field {room.field_round.shape[0]:,} entries "
            f"at `{room.field_cfg.noise_model}` noise "
            f"{room.field_cfg.rank_noise_sd:.2f}")

    # ── The clock ────────────────────────────────────────────────────────────
    clock = st.columns([1, 1, 2, 1, 1, 1])
    clock[0].metric("Pick", "—" if complete else f"{state.pick_index + 1}")
    clock[1].metric("Round", "—" if complete else f"{state.round_index + 1}")
    clock[2].metric("On the clock",
                    "draft complete" if complete
                    else ("YOU" if ours else f"seat {on_clock + 1}"))
    for i, position in enumerate(draft_room.POSITIONS):
        held, cap = health["counts"][position], health["caps"][position]
        clock[3 + i].metric(position, f"{held} / {cap}")

    owing = [f"{n} {p}" for p, n in health["owed"].items() if n]
    st.caption(f"Counts are against DK's **autodraft caps** (8 G / 8 F / 3 C), which are "
               f"maxima and bind autodraft rather than a manual pick. The **starting "
               f"slate** is 2 G / 2 F / 1 C and is a minimum: "
               + (f"still owed {', '.join(owing)}." if owing
                  else "filled — every further pick is free."))

    relevance = draft_room.injury_relevance(room, notes)
    if not relevance["describes_this_season"]:
        st.warning(
            f"⚕ The injury feeds describe **today** "
            f"({', '.join(str(y) for y in relevance['capture_years'])}), and this is a "
            f"`{room.season}` practice board — the notes are about a different season. "
            f"They are shown because they are what you would see live; do not read them "
            f"as news about this roster.")

    if health["at_capacity"]:
        st.caption(f"At DK's autodraft cap on {', '.join(health['at_capacity'])} — those "
                   f"caps bind **autodraft**, not a manual pick, so the recommendation "
                   f"still offers them.")
    if health["at_risk"]:
        st.warning(f"{health['picks_left']} picks left and "
                   f"{sum(health['owed'].values())} starting slots still owed — the next "
                   f"picks are forced to "
                   f"{', '.join(p for p, n in health['owed'].items() if n)}.")

    if complete:
        st.success("Draft complete.")
        st.dataframe(draft_room.roster_table(room, state, seat), hide_index=True,
                     width="stretch")
        return

    left, right = st.columns([3, 2], gap="large")

    # ── The recommendation ───────────────────────────────────────────────────
    with left:
        if health["picks_left"] <= 0:
            st.info("Your sixteen are in. Keep clicking picks to track the rest of the "
                    "board.")
        else:
            st.subheader("YOUR PICK" if ours else f"Seat {on_clock + 1} is on the clock")
            if not ours:
                st.caption(f"Shown so you can plan ahead — it is **not** your pick. A "
                           f"click here still means *seat {on_clock + 1} took him*.")
            start = time.perf_counter()
            try:
                table, context = draft_room.evaluate(room, state, seat, tournament,
                                                     objective=objective, top=top)
            except Exception as error:               # noqa: BLE001
                st.error(f"{type(error).__name__}: {error}")
                return
            elapsed = time.perf_counter() - start
            lead = format_gap(context["lead"], objective)
            # What the click callback reads. Stashed here because a callback fires before
            # the next render, so this is the only place the advice and the pick coexist.
            st.session_state.last_ranking = context["ranking"]
            st.session_state.last_recompute_ms = round(elapsed * 1000, 1)

            banner = st.columns([2, 1.2, 1])
            banner[0].metric("Take", table["player_name"].iloc[0])
            banner[1].metric("Lead over #2", lead,
                             help=f"Runner-up: {table['player_name'].iloc[1]}"
                             if len(table) > 1 else None)
            banner[2].metric("Recompute", f"{elapsed * 1000:.0f} ms",
                             help="Gate E's bar is 1,000 ms")

            head = st.columns(ROW_WIDTHS, vertical_alignment="bottom")
            for column, name in zip(head, ("Player", "EV", "P(top 2)", "Cost", "Cushion")):
                column.markdown(f"<span style='{HEAD_STYLE}'>{name}</span>",
                                unsafe_allow_html=True)
            for i, row in table.iterrows():
                cols = st.columns(ROW_WIDTHS, vertical_alignment="center")
                player = int(np.nonzero(
                    room.frame["player_id"].to_numpy() == row["player_id"])[0][0])
                with cols[0]:
                    pick_button(room, player, key=f"rec-{player}",
                                label=f"{'★ ' if i == 0 else ''}{row['player_name']}  ·  "
                                      f"{row['position']}  ·  {row['team']}",
                                primary=(i == 0 and ours), badge=badges.get(player))
                for column, text in zip(
                        cols[1:],
                        (money(row["ev"]), f"{row['p_advance']:.1%}",
                         "—" if i == 0 else format_gap(row["cost_vs_best"], objective),
                         f"{row['rank_cushion']:+.0f}")):
                    column.markdown(f"<span style='{CELL_STYLE}'>{text}</span>",
                                    unsafe_allow_html=True)

            shown = set(int(np.nonzero(room.frame["player_id"].to_numpy() == p)[0][0])
                        for p in table["player_id"])
            injury_panel(flagged[flagged["board_index"].isin(shown)],
                         "Injury notes on the players above")

            st.caption(
                f"Ranked by **{objective}** for `{tournament}` "
                f"({money(context['entry_fee'], 0)} entry"
                f"{', entered this year' if tournament in entered else ''}). "
                f"{context['n_candidates']:,} legal, priceable players considered. "
                f"**EV** is the expected payout of your *whole completed sixteen* with "
                f"that player in it — a level, so only the differences down the column "
                f"mean anything. **Cost** is what taking that row instead of the top one "
                f"gives up. **Cushion** is his board rank minus the number of players "
                f"expected off the board by your next pick — negative means the field "
                f"takes him first, positive means you can wait.")
            st.caption(
                "Every candidate is scored inside a **completed** roster — your picks so "
                "far, this player, and the best available at each pick you have left — "
                "because a payout is a function of a finished sixteen. **EV is a ranking "
                "device, not money**: the field drafts off ADP while this board is the "
                "model's own projection, so the level is optimistic until the strategy "
                "sweep's error injection prices that. `p_advance` is the figure that "
                "reproduces its analytic value exactly.")

    # ── The board — one click marks anyone gone ──────────────────────────────
    with right:
        st.subheader("Mark a pick")
        st.caption("One click = the seat on the clock took that player. Works for your "
                   "own pick and for everyone else's.")
        query = st.text_input("Search the board", placeholder="type a name…",
                              label_visibility="collapsed")
        available = np.nonzero(state.available()[0])[0]
        frame = room.frame.iloc[available]
        if query:
            frame = frame[frame["player_name"].str.contains(query, case=False, na=False)]
            rows = frame.head(SEARCH_ROWS)
        else:
            rows = frame.head(BOARD_ROWS)
        if rows.empty:
            st.info("Nobody on the board matches that.")
        for player in rows.index:
            pick_button(room, int(player), key=f"board-{player}",
                        badge=badges.get(int(player)))
        injury_panel(flagged[flagged["board_index"].isin(set(rows.index))],
                     "Injury notes on the board above")

    # ── Our roster ───────────────────────────────────────────────────────────
    st.divider()
    st.subheader(f"Your roster — {health['picks_made']} of {draft_room.ROSTER_SIZE}")
    roster = draft_room.roster_table(room, state, seat)
    if roster.empty:
        st.caption("Nothing drafted yet.")
    else:
        held = list(state.roster_of(seat))
        shown = roster.drop(columns=["player_id"]).round(1)
        shown.insert(len(shown.columns), "injury",
                     [badges.get(int(p), "") for p in held])
        st.dataframe(shown, hide_index=True, width="stretch")
        injury_panel(flagged[flagged["board_index"].isin(set(held))],
                     "Injury notes on your roster")

    log = pd.DataFrame({
        "pick": np.arange(1, len(st.session_state.picks) + 1),
        "seat": [int(draft.snake_order(room.pod_size, draft.N_ROUNDS)[i]) + 1
                 for i in range(len(st.session_state.picks))],
        "player": room.frame["player_name"].to_numpy()[st.session_state.picks],
        "position": room.frame["position"].to_numpy()[st.session_state.picks],
    })
    with st.expander(f"Pick log — {len(log)} picks"):
        st.dataframe(log.iloc[::-1], hide_index=True, width="stretch")


def main() -> None:
    """`make draft-room` — one process, one page, no navigation to share it with.

    Draft night runs this rather than the dashboard, and the reason is not performance
    (page 9 paints as fast) but blast radius: nothing else in the process can raise, hold
    the GIL, or load a 90 MB tensor while a clock is running.
    """
    st.set_page_config(page_title="Draft room", page_icon="🏀", layout="wide")
    render()


if __name__ == "__main__":
    main()
