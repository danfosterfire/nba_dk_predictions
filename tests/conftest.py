"""Shared pytest setup.

The one thing here is the held-out unlock. `src/models/held_out.py` locks the test split so
that production code cannot read it outside `src/final_evaluation.py` — but unit tests
exercise the *machinery* on synthetic four-season frames, where "the last two seasons" is a
fixture rather than real held-out data, and where checking that a split behaves correctly
means reading both sides of it.

So the suite runs unlocked, and `tests/test_held_out.py` covers the lock itself: that it
raises by default in production code, that the unlock is scoped, and that it re-locks on the
way out. Unlocking here does not weaken that — it keeps the guard from turning every test of
`split_seasons` into a test of the guard.
"""

import pytest

from src.models import held_out


@pytest.fixture(autouse=True)
def _unlock_held_out_for_tests():
    """Synthetic fixtures are not the held-out seasons; let tests read both sides."""
    previous = held_out._unlocked
    held_out._unlocked = True
    try:
        yield
    finally:
        held_out._unlocked = previous
