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
- [x] Mac Metal: val 442/444, challenge 37/40 (same three misses as LoRA, control re-run); Studio side-by-side below
- [x] Compare table → TRAINING.md + README; committed

## 2026-09-16 — v2 catalog (new Layers.json aliases/hints, new 1,000-record FELN.json)

Zero-shot on the new gold (RTX, HF merged exports, new Layers.json): LoRA 832/862 and QLoRA
832/862 on geometry-layer records (misses: AND/OR grouping, `country name` alias, dropped
redundant type filter); 138/1000 target the two table layers, which FELN excludes.

- [x] `scripts/prepare_northsea.py`: gold on table layers skipped and listed in the manifest
- [x] Data `runs/regen-20260916-v2` (seed 20260916, empty-cap 1.0): 3,463 train / 410 val / 419 test, 0 dropped
- [~] RTX in parallel, one GPU each (tmux = run name), ckpt/val every 200: LoRA `runs/automodel-nemotron-v2-20260916` (GPU 0), QLoRA `runs/automodel-nemotron-qlora-v2-20260916` (GPU 1, nproc 1); each run.sh chains scoring → export → GGUF → CUDA validation
- [ ] Copy Q8_0 GGUFs + bundles to Mac `runs/nemotron-mac-v2-20260916/{lora,qlora}`; Metal val + challenge
- [ ] Table layers stay excluded (decision pending); docs + commit
