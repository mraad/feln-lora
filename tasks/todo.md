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
- [x] GitHub remote: private `mraad/feln-lora`, pushed 2026-09-15

## Open ideas (unrequested)

- RAG shots in the fine-tuned prompt (Studio prompt override) against the remaining
  AND/OR-grouping and date-bound misses
- Blind real-user questions for a generalization claim
- Launch a fresh GPU instance from an AMI (only start/stop of the existing box exists)

## 2026-09-16 — QLoRA (NF4) vs LoRA, RTX 2 GPUs, Mac inference

Findings before starting: RTX box running, both GPUs idle, no tmux. AutoModel venv lacks
`bitsandbytes`. With `quantization:` AutoModel drops its custom NemotronH class and loads HF
`NemotronHForCausalLM` (transformers 5.15.1) — whose Mamba fast path passes
`out_proj.weight` straight to the fused kernel (`modeling_nemotron_h.py:456`), so
`out_proj` must stay unquantized. AutoModel's `create_bnb_config` does not forward
`llm_int8_skip_modules` → one-line patch on the RTX checkout (`qlora.py`).

- [x] RTX: `uv pip install bitsandbytes` (0.50.2; restored `nvidia-cublas==13.4.1.1` that uv downgraded) into `/home/ubuntu/Automodel/.venv`; `python -m bitsandbytes` sanity (sm_120, cu130)
- [x] RTX: patch `nemo_automodel/components/quantization/qlora.py` to pass `llm_int8_skip_modules`; keep the diff in `scripts/automodel_qlora_skip_modules.patch`
- [x] `scripts/automodel_nemotron_feln_qlora.yaml`: same recipe + `quantization: {load_in_4bit, nf4, double_quant, compute bf16, storage bf16, llm_int8_skip_modules: [out_proj, lm_head]}`; 2 GPUs (`--nproc-per-node 2`, local 2 → effective 16 unchanged)
- [x] Smoke: 10 steps (`runs/qlora-smoke`: ~1.0 s/step, 16 GiB/GPU, 17,054,208 trainable = LoRA run, loss 0.85→0.09, adapter scores through `src.infer_feln`) on both GPUs; confirm `verify_qlora_quantization`, out_proj not Linear4bit, loss sane, memory
- [x] Full run `runs/automodel-nemotron-qlora-20260916/`: 1,332 steps in 27:15 on both GPUs, 1.04 s/step, peak 17.4 GiB/GPU, final val loss 0.0026 (data = copy of the 20260914 split), 6 epochs, in tmux
- [x] Scored all 19 checkpoints with two queue workers (mkdir claim added to `automodel_eval_queue.sh`); selected step 444 = 442/444 (LoRA: same step, 441/444)
- [x] Export merged FP16 (re-scores 442/444), F16 + Q8_0 GGUF on RTX both 442/444 (LoRA GGUFs 438); Q8_0 `ec443cbb…` rsynced to Mac `runs/nemotron-mac-qlora-20260916/`
- [ ] Mac: score val + challenge via `src.edge_client --records` on Metal; Studio `--llama qlora=…` side by side
- [ ] Compare table (val HF, val Q8_0, challenge, s/step, peak GPU mem, wall time) → TRAINING.md + README; commit
