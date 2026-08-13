"""Availability — the season-level head, and the games-played tenure decomposition.

Page 3 of `docs/dashboard-plan.md`'s expansion, and the first instantiation of the generic
model-detail renderer in `dashboard/views/model_page.py`. Everything on it — the seven
blocks, the head selector, the unit — comes from there; this module names the class and
nothing else, which is the whole point of building the renderer first.

Five heads sit behind the selector and **the shipped chain takes two of them**, which is
the thing this page has to say and could not until `model_card_index.csv` carried a chain
role. `src/sim/season.py::_sim_one` draws the games-played *count* from `availability` and
lays those misses out with `games_played.layout_tenure` at `gp_duration`'s fitted spell
shape — the tenure edge blocks at the ends of the schedule and `allocate_spells` on the
interior remainder, which is the shipped `tenure_merge` arm of `sim.availability.layout`
(`docs/availability-window-plan.md` §13). `gp_entry`, `gp_exit` and `gp_onset` — the tenure
decomposition, which models the generating process rather than the count — are never called
at draw time; `season.py` states why it does not call `HybridProcess.sequences`, and note
that the layout draws its edge blocks from an empirical resample rather than from those
heads, for a reason §13a gives. They stay on the page because they are fitted, converged and
carded, and because what they produce is the games-played pmf Gate A scores the simulated
seasons against.

**Nothing on this page shows the layout itself**, which is a real gap rather than an
oversight of this docstring: `availability_exchangeability.csv` and
`availability_clustering.csv` are cited by `dashboard/decisions.py` and by nothing that
draws, so `make dashboard-audit`'s orphan check is satisfied while the 2x2 that selected the
shipped arm has no picture anywhere.

The heads are **not all at the same unit** — `gp_duration` is fitted per absence spell and
the other four per player-season — which is why the page reads the unit off
`model_card_index.csv` and states it under the head's name rather than once at the top.
The chain role is read from the same file for the same reason.
"""

from dashboard.views import model_page

CLASS_KEY = "availability"


def render() -> None:
    model_page.render(CLASS_KEY)
