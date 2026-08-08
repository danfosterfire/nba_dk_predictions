"""The held-out split, locked by default — one definition, enforced rather than remembered.

`CLAUDE.md` states the rule plainly: the test seasons must not inform data preparation,
modelling decisions, model fitting, or backtested draft strategies. They exist for a single
end-of-project measurement of the whole workflow.

**A rule that lives only in prose gets followed until it is inconvenient.** It already
failed once here: the games-played head's Gate D was specified with the incumbent's *test*
figures as its bars, ran on the test split, and settled which model ships — on a margin of
0.0013 CRPS that a paired bootstrap could not distinguish from zero, and which reversed when
re-decided on validation. Nothing in the code objected, because nothing in the code knew.

So this module makes the test split a **capability** rather than a convention. Reaching it
raises unless something has explicitly unlocked it, and the only thing that does is
`src/final_evaluation.py`. The failure mode this prevents is not malice — it is a head
scoring test "just to see", printing it next to validation, and a reader treating the two
columns as equally admissible.

## Why the sweeps refit rather than predict twice

A natural alternative is to fit once on the training frame and predict on both validation
and test. Cheaper, and it makes the two columns directly comparable. It is deliberately
*not* what the final evaluation does: a test figure should describe the model that would
actually be deployed, which has seen train **and** validation, so the final evaluation
refits on `full_train` before scoring. Predicting test from a train-only fit would report a
lower bound on deployed performance rather than an estimate of it.

That refit is exactly why the two columns must not be read as a replication check. Before
this module existed, a sweep's val and test columns differed in three ways at once —
different evaluation rows, different training data, and different sampler iteration counts —
so a val/test disagreement conflated the data with the fit. Selection reads validation and
nothing else; test is read once, at the end.

Usage:
    from src.models.held_out import assert_unlocked, selection_split
"""

from contextlib import contextmanager

import pandas as pd

# Trailing target seasons held out. Matches `features.availability.test_seasons`, and is
# duplicated here rather than read from config so the lock cannot be widened by editing a
# YAML file.
TEST_SEASONS = 2

_unlocked = False


class HeldOutLocked(RuntimeError):
    """Raised when code reaches for the test split outside the final evaluation."""


def is_unlocked() -> bool:
    return _unlocked


@contextmanager
def unlocked(reason: str):
    """Unlock the held-out split for the duration of a block.

    A context manager rather than a flag so the unlock cannot leak: whatever happens inside,
    the split re-locks on the way out. `reason` is required and is printed, because an
    unlock is a thing that should be visible in a log.
    """
    global _unlocked
    if not reason:
        raise ValueError("unlocking the held-out split requires a stated reason")
    print(f"  /!\\  HELD-OUT SPLIT UNLOCKED — {reason}")
    previous, _unlocked = _unlocked, True
    try:
        yield
    finally:
        _unlocked = previous


def assert_unlocked(what: str) -> None:
    """Guard every path that scores, fits on, or otherwise reads the test seasons."""
    if not _unlocked:
        raise HeldOutLocked(
            f"{what} reaches the held-out test split, which is locked.\n"
            f"Selection, ablations and gates read the VALIDATION split; the test seasons "
            f"are for one end-of-project measurement of the whole workflow (CLAUDE.md, "
            f"'Train / val / test split').\n"
            f"If this really is that measurement, run `make final-evaluation` — or wrap the "
            f"call in `held_out.unlocked('why')`.")


def as_plain(frame: pd.DataFrame) -> pd.DataFrame:
    """Strip the held-out guard from a frame that is not actually held out.

    `split_seasons` guards whatever it puts on the right-hand side, which is correct for the
    outer split and wrong for the inner one: carving validation out of the training frame
    also produces a "held-out" side, but that side *is* the validation set and every sweep
    has to read it. Nesting the guard is the one place it would fire on the wrong frame, so
    the two callers that nest say so explicitly rather than the guard trying to guess.
    """
    return pd.DataFrame(frame)


def selection_split(design: pd.DataFrame, test_seasons: int = TEST_SEASONS
                    ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """`(train, validation)` — the only two frames a sweep is allowed to see.

    The last `test_seasons` target seasons are dropped entirely rather than returned and
    trusted not to be used, so a head that wants them has to say so out loud.
    """
    from src.models.availability import split_seasons

    full_train, _ = split_seasons(design, test_seasons)
    n_seasons = full_train["season"].nunique()
    if n_seasons < 2:
        raise ValueError(
            f"need at least 2 training seasons to carve a validation split; got "
            f"{n_seasons}")
    train, val = split_seasons(full_train, test_seasons=min(test_seasons, n_seasons - 1))
    return train, as_plain(val)


def final_split(design: pd.DataFrame, test_seasons: int = TEST_SEASONS
                ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """`(full_train, test)` for the end-of-project evaluation. Guarded.

    `full_train` is train **plus** validation: at the point the final number is taken there
    is no reason to withhold the selection split from the model, and every reason to report
    the performance of the thing that would actually ship.
    """
    from src.models.availability import split_seasons

    assert_unlocked("final_split")
    return split_seasons(design, test_seasons)
