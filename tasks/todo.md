# feln-lora

## 2026-09-15 — split from gait-feln-finetuning

- [x] Copy the Nemotron/AutoModel path (src, scripts, tests, Studio web) and clone-copy the
      run artifacts (`runs/{regen-20260914-ilike,automodel-nemotron-20260914,nemotron-mac-20260915}`)
- [x] `pyproject.toml`: `feln` + `layers-json` editable path deps (as `../feln-rag`); `uv sync`
- [x] Replace `src/feln_compare.py` with `feln` (`FELN`, `FELNCompare`, `normalize_where`,
      `parse_relation`, `to_meters`); extract `src/prompt.py` from the Qwen trainer
- [x] Prune: Qwen QLoRA trainer, Optuna, ONNX export/infer, hybrid router, SPA, finalize,
      Orin/backbone/retraining/recovery docs, old checkpoints
- [x] Proof: re-scored recorded predictions with `feln`'s comparator — 441/444 HF, 438/444
      GGUF, zero flipped verdicts; tests, ruff, pyright clean
- [x] Mac: Studio + edge client on the Q8_0 GGUF (challenge 37/40); RTX: feln installed in the
      AutoModel venv, run dir moved to `/home/ubuntu/feln-lora/runs/`, prompt prefix/suffix
      byte-identical to the export, 8/8 val-smoke with the new scorer; remote servers
      start/query/stop from the new `gpu_server.json`
- [x] README, TRAINING.md, CLAUDE.md
- [ ] GitHub remote (`mraad/feln-lora`) — on request

## Open ideas (unrequested)

- RAG shots in the fine-tuned prompt (Studio prompt override) against the remaining
  AND/OR-grouping and date-bound misses
- Blind real-user questions for a generalization claim
- Launch a fresh GPU instance from an AMI (only start/stop of the existing box exists)
