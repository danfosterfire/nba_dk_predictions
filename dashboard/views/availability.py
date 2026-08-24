"""Availability — the two heads a simulated season's absences are drawn from.

Page 3 of `docs/dashboard-plan.md`'s expansion, and the first instantiation of the generic
model-detail renderer in `dashboard/views/model_page.py`. Everything on it — the seven
blocks, the head selector, the unit — comes from there; this module names the class and its
one named block, the shared Stan program between blocks 3 and 4.

**The page carries the draw path only** (since 2026-08-23): `availability`, which supplies
the games-played count, and `gp_duration`, which supplies the spell shape the misses are
laid out in. `src/sim/season.py::_sim_one` draws the count from the first and lays it out
with `games_played.layout_tenure` at the second's fitted `(mu, kappa)` — the shipped
`tenure_merge` arm of `sim.availability.layout` (`docs/availability-window-plan.md` §13).

The class's other three heads — `gp_entry`, `gp_exit` and `gp_onset`, the games-played
tenure decomposition — are fitted, converged and carded, and are **not on this page**,
because a simulated season never calls them: `season.py` states why it does not call
`HybridProcess.sequences`, and what they supply instead is the games-played pmf Gate A
scores the drawn seasons against. `model_cards.heads_of` reads the cut off the artifact's
own `in_draw_path` column rather than restating it here, so a refactor of the simulator
moves this page through `make model-cards`.

**Nothing on this page shows the layout itself**, which is a real gap rather than an
oversight of this docstring: `availability_exchangeability.csv` and
`availability_clustering.csv` are cited by `dashboard/decisions.py` and by nothing that
draws, so `make dashboard-audit`'s orphan check is satisfied while the 2x2 that selected the
shipped arm has no picture anywhere. The layout's shipped arm is drawn on the *Inputs
beyond the heads* page.

The two heads are **not at the same unit** — `gp_duration` is fitted per absence spell and
`availability` per player-season — which is why the page reads the unit off
`model_card_index.csv` and states it under the head's name rather than once at the top.
The chain role is read from the same file for the same reason.
"""

from dashboard.views import model_page

CLASS_KEY = "availability"


def render() -> None:
    model_page.render(CLASS_KEY, extra={3: model_page.stan_block})
