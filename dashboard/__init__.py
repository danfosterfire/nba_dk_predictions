"""The Streamlit dashboard — data visualizations over the artifacts the pipeline wrote.

`app.py` is the `st.navigation` entrypoint, `shell.py` holds the state that has to
survive moving between pages, and each page is one module in `views/` behind a
`render()`. One real page today: the PCA player-style fingerprint in
`views/fingerprints.py`. See `docs/dashboard-plan.md` for what it is and what comes next,
and `dashboard/README.md` for the rules a new view has to follow.

`decisions.py` and `audit.py` live here for historical reasons and are **not** part of
the dashboard: they are the project's decision registry and its drift report, wired to
`make dashboard-audit` and to a standing instruction in `CLAUDE.md`. `economics.py` is
likewise pure contest arithmetic that the drafting layer will consume.
"""
