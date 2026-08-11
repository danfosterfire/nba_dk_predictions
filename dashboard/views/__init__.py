"""The pages `app.py` navigates between — one module per page, each exposing `render()`.

A view module is the Streamlit surface and nothing else: it reads artifacts through
`dashboard.artifacts`, builds figures with `dashboard.charts`, and takes its palette from
`dashboard.shell.current_theme()`. The logic it draws stays in a pure sibling module with
no Streamlit import (`dashboard.pca` for `fingerprints`), which is what lets
`tests/test_dashboard.py` exercise it as plain functions.

`render()` takes no arguments and returns nothing. It may write to `st.sidebar`, and what
it writes lands *below* the shell's own controls, because the entrypoint runs first.
"""
