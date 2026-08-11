"""The draft board — page 9, and the only page whose body lives outside `views/`.

`dashboard/draft_room.py` is the room itself and stays directly runnable, because
`make draft-room` is what draft night launches and a thirty-second clock should not share
a process with anything. This module is the row's owner in the navigation: it exists so
that the page a reader lands on is still a `views/` module with a `render()`, exactly like
the other six, and so that the import that reaches `src.sim` happens *here*, once, on
demand, rather than at app start.

**The import is inside `render()` deliberately, and it is worth a paragraph.** The
argument for the room being a page at all is that `st.navigation` does not run a page's
script until the reader selects it. A module-level `from dashboard import draft_room`
would half-undo that: `import src.sim.draft_room` costs 0.89 s, and `app.py` imports every
view module before it draws anything, so every reader of the fingerprint page would pay it.
Deferred, a reader who never opens the room never loads the simulation layer at all.

That also keeps the `src/` exemption where it is. `SRC_IMPORTERS` in
`tests/test_dashboard.py` names one **path**, `draft_room.py` at the package root; this
file is `views/draft_room.py` and is held to the ordinary rule — it imports nothing from
`src/` and could not, since the only thing it names is its sibling.
"""


def render() -> None:
    from dashboard import draft_room

    draft_room.render()
