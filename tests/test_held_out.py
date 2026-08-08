import pandas as pd
import pytest

from src.models import held_out
from src.models.availability import split_seasons
from src.models.held_out import (HeldOutLocked, assert_unlocked, is_unlocked,
                                 selection_split, unlocked)


@pytest.fixture(autouse=True)
def _relock():
    """Undo `conftest`'s suite-wide unlock — this file tests the lock itself."""
    previous = held_out._unlocked
    held_out._unlocked = False
    try:
        yield
    finally:
        held_out._unlocked = previous


def _design(seasons: list[str], per_season: int = 5) -> pd.DataFrame:
    return pd.DataFrame({
        "season": [s for s in seasons for _ in range(per_season)],
        "player_id": list(range(per_season)) * len(seasons),
        "gp": 40,
    })


SEASONS = ["2020-21", "2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]


def test_the_held_out_frame_raises_when_read():
    """The guard is on *use*, not on carving the split.

    A module has to be able to separate the held-out rows to know what to exclude, so
    `train, _ = split_seasons(...)` must stay legal while `test["gp"]` must not.
    """
    train, test = split_seasons(_design(SEASONS), 2)
    assert len(train) == 20 and len(test) == 10      # counting is not reading
    for reach in (lambda: test["gp"], lambda: test.to_numpy(),
                  lambda: test.merge(train, on="player_id")):
        with pytest.raises(HeldOutLocked):
            reach()


def test_selection_split_never_materializes_the_held_out_rows():
    """`selection_split` is what a sweep should call: it returns train and validation and
    the test seasons are simply absent, so there is nothing to read by accident."""
    train, val = selection_split(_design(SEASONS), 2)
    held = {"2024-25", "2025-26"}
    assert not held & set(train["season"])
    assert not held & set(val["season"])
    # Validation is the two seasons before the held-out pair, and it is a plain frame.
    assert set(val["season"]) == {"2022-23", "2023-24"}
    assert list(val["gp"])                            # readable, unlike the test frame


def test_the_unlock_is_scoped_and_requires_a_reason():
    assert not is_unlocked()
    with pytest.raises(HeldOutLocked):
        assert_unlocked("a sweep")

    with unlocked("the end-of-project evaluation"):
        assert is_unlocked()
        assert_unlocked("the final evaluation")       # does not raise
    assert not is_unlocked()

    with pytest.raises(ValueError):
        with unlocked(""):
            pass


def test_the_unlock_re_locks_even_when_the_block_raises():
    """A leaked unlock would silently disarm every later guard in the process."""
    with pytest.raises(RuntimeError):
        with unlocked("deliberate failure"):
            raise RuntimeError("boom")
    assert not is_unlocked()


def test_the_error_names_the_way_out():
    """A guard that stops you without saying what to do instead gets worked around."""
    try:
        assert_unlocked("some head")
    except HeldOutLocked as exc:
        message = str(exc)
    assert "some head" in message
    assert "VALIDATION" in message
    assert "make final-evaluation" in message


# ── Every converted head, as a regression guard ───────────────────────────────
#
# The lock only works if the heads go through it, and "goes through it" is not something a
# reader can verify by looking: `split_seasons` is called several frames deep and the guard
# fires on *use*, not on the call.
#
# The obvious test — call each head's `run` with the lock on and assert it does not raise —
# is the wrong one, and the first attempt at it wrote garbage over
# `component_rate_metrics.csv` and `season_total_metrics.csv` before anyone noticed. A
# `run` is an entry point: it reads the real config paths and WRITES the real artifacts.
# Testing it in place means the test suite is a build step, which is a worse failure than
# the one being guarded against.
#
# So the property is checked where it actually lives — in the source. Each converted module
# must reach the split through `held_out.selection_split` and must not name the guarded
# `split_seasons`. Same technique as the dashboard's no-import-from-src test, and for the
# same reason: a structural invariant is cheaper and more honest to assert structurally.

import ast
from pathlib import Path

# Every module that selects, ablates or gates. `availability` is absent because it DEFINES
# `split_seasons` and is where the guard lives — it is converted too, and gets a
# function-scoped check of its own below; `held_out` and `final_evaluation` are the two
# that are allowed to reach the held-out frame.
CONVERTED = [
    "src/models/stan_availability.py",
    "src/models/stan_minutes.py",
    "src/models/stan_components.py",
    "src/models/stan_composition.py",
    "src/models/stan_games_played.py",
    "src/models/season_terms.py",
    "src/models/season_total.py",
    "src/models/component_rates.py",
    "src/eda/season_effects.py",
]


def _names(path: str) -> set[str]:
    """Every bare name and attribute tail referenced in a module."""
    tree = ast.parse(Path(path).read_text())
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            out.add(node.id)
        elif isinstance(node, ast.Attribute):
            out.add(node.attr)
        elif isinstance(node, ast.alias):
            out.add(node.asname or node.name.split(".")[-1])
    return out


def test_every_converted_head_reaches_the_split_through_selection_split():
    for path in CONVERTED:
        names = _names(path)
        assert "selection_split" in names, (
            f"{path} does not call held_out.selection_split; selection must not carve "
            f"its own split")


def test_no_converted_head_names_the_guarded_split():
    """`split_seasons` returns the held-out frame, so a head that names it is asking for it.

    Not a style rule. `component_rates` had a *private copy* of `split_seasons` for exactly
    as long as it went unguarded, and `season_terms` imported the shared one under an alias
    (`split_availability`) — both of which read as innocuous and both of which route around
    `src/models/held_out.py`.
    """
    for path in CONVERTED:
        assert "split_seasons" not in _names(path), (
            f"{path} names split_seasons, which hands back the guarded held-out frame; "
            f"use held_out.selection_split")


def _function_names(path: str, function: str) -> set[str]:
    """Every name referenced inside one top-level function."""
    tree = ast.parse(Path(path).read_text())
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                and node.name == function:
            return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)} | {
                n.attr for n in ast.walk(node) if isinstance(n, ast.Attribute)}
    raise AssertionError(f"{path} defines no top-level {function}()")


AVAILABILITY = "src/models/availability.py"

# `run` selects the mean function, `workload_ablation` selects a feature block and
# `nonlinearity_ablation` selects a basis. Three decisions, all of which were taken on the
# held-out split until 2026-08-08.
AVAILABILITY_SELECTORS = ["run", "workload_ablation", "nonlinearity_ablation",
                          "minutes_nonlinearity_probe", "gbm_shuffled_null"]


def test_availability_selects_through_selection_split():
    """`availability` cannot be checked module-wide the way the list above is, because it
    *defines* `split_seasons` — the guard lives there. So the same property is asserted per
    function: the entry point goes through `selection_split`, and nothing that selects,
    ablates or scores may name the guarded split directly.
    """
    assert "selection_split" in _function_names(AVAILABILITY, "run"), (
        f"{AVAILABILITY}::run does not call held_out.selection_split")
    for function in AVAILABILITY_SELECTORS:
        assert "split_seasons" not in _function_names(AVAILABILITY, function), (
            f"{AVAILABILITY}::{function} names split_seasons, which hands back the guarded "
            f"held-out frame; the sweeps take the frames `run` gives them")


def test_availability_no_longer_carves_a_private_validation_split():
    """`_inner_split` existed because the outer split handed back *test*, so the two
    nonlinearity sweeps had to carve their own selection frame to have anything admissible
    to select on. `selection_split` returns exactly that frame, so a second private carve
    is now a second definition of the same thing — the shape `component_rates` was in.
    """
    from src.models import availability

    assert not hasattr(availability, "_inner_split"), (
        "availability defines a private inner split again; held_out.selection_split "
        "already returns the (train, validation) pair it used to carve")


def test_component_rates_no_longer_defines_its_own_split():
    """The one head that routed around the guard, pinned so it cannot do so again."""
    from src.models import component_rates

    assert not hasattr(component_rates, "split_seasons"), (
        "component_rates defines its own split_seasons again; that routes around "
        "src/models/held_out.py, which is the failure the shared split exists to prevent")
    assert component_rates.selection_split is selection_split


def test_final_evaluation_is_the_only_registered_way_to_the_test_split():
    """`final_split` is guarded, and `src/final_evaluation.py` is what unlocks it."""
    from src import final_evaluation

    with pytest.raises(HeldOutLocked):
        held_out.final_split(_design(SEASONS), 2)
    assert final_evaluation.REASON
    # Registering a head is what makes its held-out number takeable at all, so the registry
    # is asserted rather than assumed: a head dropped from it silently loses its
    # end-of-project measurement.
    assert set(final_evaluation.HEADS) >= {"availability", "games_played",
                                           "season_total"}
    assert "selection_split" not in _names("src/final_evaluation.py"), (
        "the final evaluation must use final_split, not the selection split")


def test_a_head_that_scores_the_held_out_frame_still_raises():
    """The other half of the guard: reaching the frame fails loudly, wherever it happens.

    The structural tests above say the heads do not ask for it. This one says that if one
    ever does — through a helper, a refactor, a new module — the failure is immediate and
    named rather than a silently better-looking number.
    """
    train, test = split_seasons(_design(SEASONS), 2)

    def a_sweep_that_scores_test():
        return float(test["gp"].mean())

    with pytest.raises(HeldOutLocked):
        a_sweep_that_scores_test()
    assert len(train) == 20                       # counting stays legal
