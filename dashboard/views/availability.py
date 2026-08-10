"""Availability — the season-level head, and the games-played tenure decomposition.

Page 3 of `docs/dashboard-plan.md`'s expansion, and the first instantiation of the generic
model-detail renderer in `dashboard/views/model_page.py`. Everything on it — the seven
blocks, the head selector, the unit — comes from there; this module names the class and
nothing else, which is the whole point of building the renderer first.

Five heads sit behind the selector. `availability` predicts games played out of team games
directly. The other four are the tenure decomposition, which models the process instead:
`gp_entry` and `gp_exit` bound the stretch of the schedule a player is with the team, and
inside that tenure `gp_onset` starts absence spells at a fitted per-game hazard while
`gp_duration` decides how long each spell lasts.

The heads are **not all at the same unit** — `gp_duration` is fitted per absence spell and
the other four per player-season — which is why the page reads the unit off
`model_card_index.csv` and states it under the head's name rather than once at the top.
"""

from dashboard.views import model_page

CLASS_KEY = "availability"


def render() -> None:
    model_page.render(CLASS_KEY)
