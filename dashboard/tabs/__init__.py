"""The nine walkthrough tabs, in project order.

Organized **by topic rather than by module**, which is the whole point of the
revamp: the pre-split app had one tab per `src/eda/` module, and the availability
head, the minutes head, the component floor, the Stan ports, the box-score backfill,
the ADP pipeline and the tournament economics all landed afterwards with nowhere to
live.

Each module exposes exactly one `render(ctx)` taking the `Ctx` from
`dashboard.artifacts`. That the count is again nine is a coincidence worth noticing
and not relying on — `TABS` is the single declaration and `app.py` dispatches off it.
"""

from dashboard.tabs import (availability, components, data, decision_log, drafting,
                            eda, minutes, problem, simulations)

TABS: tuple[tuple[str, object], ...] = (
    ("Problem", problem.render),
    ("Data collection", data.render),
    ("EDA", eda.render),
    ("Availability", availability.render),
    ("Minutes", minutes.render),
    ("Components", components.render),
    ("Simulations", simulations.render),
    ("Drafting", drafting.render),
    ("Decision log", decision_log.render),
)

TAB_NAMES: tuple[str, ...] = tuple(name for name, _ in TABS)
