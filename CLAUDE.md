# Project guidance

North Sea natural-language → FELN (`layers`, `where`, `relations`) by LoRA fine-tuning of
Nemotron-3-Nano-4B via NeMo AutoModel, served as GGUF. README.md has commands and results,
TRAINING.md the exact recipe, environments and provenance. Sibling of `../feln-rag`.

- `relations[i]` connects primary `layers[0]` to `layers[i+1]`.
- FELN model, strict comparator (`FELN.same`), `FELNCompare`, relation parsing and units
  come from `../feln` (editable). Do not add a local comparator or relation parser;
  `src/feln_data.Schema` adds only what the catalog knows (fields, enum codes, casts).
- `uv sync` on the Mac; `uv run --no-sync python -m src.<module>`; tests with
  `uv run --no-sync python -m unittest discover -s tests`; `ruff` and `pyright` before done.
- Training, checkpoint scoring and export run on the RTX box in
  `/home/ubuntu/Automodel/.venv` (see TRAINING.md); never `uv sync` there, never touch
  `/home/ubuntu/gait-feln-finetuning/.venv`, never start two jobs on one GPU without
  checking `tmux ls` and `nvidia-smi`. The 2026-09-14 LoRA and 2026-09-16 QLoRA runs are
  complete: do not restart them. The AutoModel checkout there carries
  `scripts/automodel_qlora_skip_modules.patch` uncommitted; do not reset it.
- Validation selects checkpoints and quantizations; test and `tests/challenge.json` are
  regression sets. Report raw accuracy separately from schema-compiled accuracy. Never
  relax operators, codes, spatial direction, NULL semantics or the primary layer to
  improve a score. FELN validity + strict match is the objective; row counts (even zero)
  from execution are diagnostic only.
- Original ArcGIS files stay untouched. `runs/` is git-ignored; preserve adapters,
  checkpoints, prediction reports, manifests and source archives.
- Never `pkill -f`/`pgrep -f` a pattern that appears in your own ssh command; never
  commit or copy the SSH key; no HF tokens in logs or docs.
