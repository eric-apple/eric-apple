# Photo consolidation + Memory Machine

This folder sorts a family's photos on this Mac. Two scripts share one SQLite catalog and one config JSON:

- `photo_consolidate.py`: inventory, plan (dates, events, exact and near duplicates), report, execute (copy).
- `memory_machine.py`: the photo AI layer: analyze (sharpness, exposure, thumbnails, CLIP embeddings), curate (best shot per burst, triage of blurry/dark/screenshots/receipts/documents, content tags), gallery (HTML contact sheet), search (plain-English photo search), highlights (top shots per event to a separate folder).

## When the user asks you to set it up or run it

The user will usually say one sentence, such as "sort Kelly's photos from the Aluminium drive into WD_Black". Do this:

1. Check prerequisites: `python3 --version` (3.10+), `which exiftool` (if missing: `brew install exiftool`). Create a venv in this folder if none exists (`python3 -m venv .venv && source .venv/bin/activate`), then `pip install -r requirements.txt -r requirements-ai.txt`. If the AI install fails, carry on without it; every stage works without AI, minus tags, utility-shot detection and search.
2. Write or update `config.<person>.json` from what the user said, modelled on `config.kelly.json`: `person`, `sources` (folders to read), `destination` (where the organized copy goes), `db_path` (`./<person>_catalog.sqlite3`). Confirm source and destination paths exist with `ls` before running anything. Optional `memory_machine` block overrides thresholds (see `DEFAULTS` in `memory_machine.py`).
3. Run, in order, reporting counts after each:
   `photo_consolidate.py --config C inventory`, `plan`, then `memory_machine.py --config C analyze`, `curate`, `gallery`, then `photo_consolidate.py --config C report`.
4. Open the gallery (`open <person>_gallery.html`) and summarize: keepers, extras, triaged by reason, highlights. Point out anything suspicious (for example a large blurry bucket suggests `blur_threshold` is too high for this camera).
5. Copy only after the user approves: `photo_consolidate.py --config C execute` (dry run) then `execute --apply`.

## Rules

- Never move, rename, edit or delete anything under a `sources` path. All output is copies.
- `execute --apply` and `highlights --apply` write to disk; run them only on the user's explicit go-ahead.
- Re-running `plan` resets destinations; always re-run `curate` after `plan`.
- Large libraries: analyze is resumable; if interrupted, run it again and it continues where it stopped.
