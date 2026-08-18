# Draft-board ingestion — validated pick logs from board screenshots

**Built, stress-tested and shipped 2026-08-17.** The capture pre-step
`docs/entry-collection-plan.md` requires: the full 12-entrant pick log of every
data-collection draft, as a tabular artifact with provenance. DK's draft room is
login-gated with no archive, so each board exists only as a screenshot taken at draft
close; this workflow turns those screenshots into `data/features/draft_pick_log.parquet`
without trusting any single read of the image. `src/data/draft_boards.py` is the module,
`make draft-boards` / `draft-boards-tile` / `draft-boards-status` the targets,
`tests/test_draft_boards.py` the pins.

## The capture rules (user, at draft close)

- Screenshot the full board the moment the draft completes — it is **not re-capturable**
  (same deadline logic as the ADP archives). PNG or JPG, 1220×1439 native app export or
  anything the grid detector can read.
- Name it `draftboard_<Mon><D>_<YYYY>.png` and drop it in
  `data/raw/adp_autodrafted_boards_<year>/`. Both month spellings parse (`Aug16`,
  `July28`), mirroring the rankings filenames.
- Add one row to `boards_manifest.csv` in the same directory:
  `file, contest, our_entrant, mode(auto|live), ranking_file, notes`. The `ranking_file`
  column is the provenance record the entry plan requires — a pod whose ranking version
  is unrecorded is uninterpretable, so a board without a manifest row **fails validation**
  rather than guessing.
- Download that day's `DkPreDraftRankings_*.csv` alongside (the existing ADP capture
  habit); it is the naming authority the transcription is resolved against.

## Files per board

| file | rows | who writes it |
|---|---|---|
| `draftboard_<date>.png/.jpg` | — | user |
| `draftboard_<date>_header.csv` | 12: `seat,entrant,g,f,c` | Claude (or hand) |
| `draftboard_<date>_picks.csv` | 192: `seat,round,pick_in_round,overall,cell_name,pos,team` | Claude (or hand) |

Transcriptions record **exactly what the cell displays** — ellipses kept
(`N. Alexande...`), DK's disambiguation prefixes kept (`St. Curry`, `Jal. Williams`),
never expanded from basketball knowledge. The pool contains rookies past any knowledge
cutoff; the same-date rankings CSV is the only authority, and the matcher enforces that.

## The two-pass read protocol (Claude, ~30 image reads per board)

1. `make draft-boards-status` names boards needing transcription;
   `make draft-boards-tile` writes the crops to `outputs/draft_boards/tiles/<stem>/`.
   The 12×16 grid is **detected per image** from the dark gaps between cells, so a DK
   layout change fails loudly instead of silently mis-cropping every cell.
2. **Pass 1**: read `header.png` + the four 2× row bands (`rows_01_04.png` …); write the
   two CSVs.
3. **Pass 2**: read the 24 half-column 3× strips (`col_01_r01_08.png` …) — a different
   slicing, so band-level errors decorrelate — and diff against pass 1. Any disagreement:
   `python -m src.data.draft_boards --cell R.S --board <stem>` writes a 4× crop of that
   one cell (addressed round.seat, seat = column), which arbitrates.
4. `make draft-boards`: the validator either goes green and writes the pooled parquet, or
   names every failing cell. Fix named cells via `--cell` crops; re-run until green.

**Manual fallback:** hand-write the same two CSVs and run the same validator — downstream
is identical, so the automation decision is reversible per board.

## The validator (all hard checks block; soft notes never do)

Hard: 192 rows; `overall` a permutation of 1..192; the snake arithmetic twice over
(`overall == (round−1)·12 + pick_in_round`, `pick_in_round == seat` odd / `13−seat` even);
every cell resolves to **exactly one** rankings row matching team AND position AND
first-name prefix AND last-name prefix; no player twice; per-seat G/F/C tallies equal the
header's counts (a per-column checksum read off the image); manifest row present with a
real `ranking_file`; `mode` in {auto, live}.

Soft (printed): Spearman of pick order vs same-date ADP with the largest gaps; for `auto`
pods, our seat's agreement with cap-aware best-available under the 8G/8F/3C caps — a
whole-board integrity check, since availability at each of our picks depends on every
opponent pick before it; counts of no-ADP and censored picks.

Name resolution is near-deterministic by construction: on the full 942-player Aug-2026
rankings universe, `(first initial, last-name prefix, team, position)` leaves exactly
**2** ambiguous pairs — Stephen/Seth Curry, Jalen/Jaylin Williams — and DK's own cell
rendering separates both with longer first prefixes, which the matcher uses. A silent
error would have to survive two independent passes, resolve to a *different valid*
same-team-same-position player, and keep the G/F/C checksum, the uniqueness constraint
and the ADP ordering all intact at once.

## Stress test, 2026-08-17 (the two trial boards)

- A pre-protocol **whole-image single read is not reliable**: it misread Klay Thompson's
  position (F for G) at cell 13.1 of the Aug-16 board. The zoomed re-read caught it and
  the rankings CSV confirmed — this is why the protocol tiles.
- Under the protocol: 2 boards × 192 cells, two passes each — **0 pass-1 vs pass-2
  disagreements**, and both boards validated **green on the first run**. Soft checks:
  Spearman 0.979 / 0.987 vs same-date ADP, and the ADP-null autodraft seat agreed with
  cap-aware best-available **16/16 on both boards** — the strongest whole-board check
  available, and also a live verification of the "autodraft equals capped click"
  equivalence the entry plan leans on.
- Negative check: re-injecting the F-for-G misread makes the validator fail loudly twice
  over (`matches NO ONE` naming cell 13.1, plus the seat-1 G/F/C checksum), and exit
  non-zero. Restored, it goes green again.

These figures are reproduced by `make draft-boards` (the notes it prints) and are not
registered in `make docs-audit`.

## Known limits

- DK truncates entrant usernames at ~10 characters (`sunshinere...`). Recorded as
  displayed; `our_entrant` in the manifest matches on the visible prefix. Cross-board
  identity of *opponent* drafters is therefore approximate for long names — annotate the
  manifest `notes` if an identity ever matters.
- The seat's own picks double as data: the 16/16 agreement means the ADP-null pods are
  executing the uploaded board faithfully, so opponent deviations from ADP are the only
  signal in the log — which is the design of `docs/entry-collection-plan.md`.
- `adp` in the pick log is the ranking file's own (censored) column; the recalibrated
  consensus panel remains the modeling-grade ADP (`docs/adp-plan.md`).
