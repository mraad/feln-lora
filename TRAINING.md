# Training, scoring, export — Nemotron-3-Nano-4B LoRA and QLoRA

The LoRA run (2026-09-14/15) and the QLoRA run (2026-09-16) below are complete. Do not
restart the training or scoring jobs; rerun only to reproduce or to train on new data.

## Machines

| | Path | Notes |
|---|---|---|
| Mac | `~/GWorkspace/feln-lora` | data generation, GGUF validation on Metal, Studio, MCP; `uv sync` here |
| RTX (EC2 box in `gpu_server.json`, 2× RTX PRO 6000) | `/home/ubuntu/feln-lora` (rsync of this repo, no `.venv`) | training, checkpoint scoring, merge/export, GGUF conversion, inference servers |

The RTX Python is `/home/ubuntu/Automodel/.venv` ([NeMo AutoModel](https://github.com/NVIDIA-NeMo/Automodel.git) checkout `4e00f6be0`,
Python 3.12, torch 2.10+cu130, transformers 5.15.1, peft 0.20.0, torchao 0.18, mamba_ssm
kernels) with `uv pip install --no-deps -e /home/ubuntu/feln -e /home/ubuntu/layers-json`
on top — rsync those sibling checkouts alongside this one (or install feln without
`--no-deps` and let it fetch the public layers-json pin). A `uv sync` in
`/home/ubuntu/Automodel` would undo the `uv pip` additions (peft, sqlglot, pydantic,
torchao, feln, layers-json); never run it there. `/home/ubuntu/gait-feln-finetuning/.venv`
is the retired Qwen-era stack: it lacks mamba_ssm and OOMs on Nemotron scoring.

Two environment rules, both encoded in `scripts/automodel_nemotron_feln.yaml` and
`scripts/automodel_eval_queue.sh`: `LD_LIBRARY_PATH` must list only the venv's
`nvidia/*/lib` (the login shell's system CUDA path shadows torch's cuBLASLt →
`CUBLAS_STATUS_NOT_INITIALIZED`; unset, TransformerEngine cannot find libcudnn), and the
recipe uses `backend.attn: sdpa` (TE's cuDNN fused attention fails on this Blackwell build).

Experiment roots live under `runs/<exp>/` on both machines and hold `data/` (a
`scripts/prepare_northsea.py` output), `checkpoints/`, `exports/`, `gguf/`. Commands run
from the experiment root with `PYTHONPATH=/home/ubuntu/feln-lora`, so `src.*` and
`scripts.automodel_feln` resolve to the checkout, not to anything in the run directory.
The finished run keeps its own copy of the code it trained with under
`runs/automodel-nemotron-20260914/source-20260914/` and `source.tar.gz`.

## Data — `runs/regen-20260914-ilike/data/` (seed 20260914)

```bash
uv run --no-sync python -m scripts.prepare_northsea --output runs/<exp>   # ~1 min
```

Source of truth: `~/Documents/ArcGIS/Projects/NorthSea/{FELN.json,Layers.json,NorthSea.ddb}`
(copied, never modified; sha256 in `manifest.json`). Table layers (no geometry) are
excluded. The 1,000 gold records are kept verbatim except LIKE→ILIKE on the 12 columns whose
catalog hint says "use the SQL ILIKE" (211 rewrites, listed in the manifest). 3,600 synthetic
questions come from `src.feln_data.sample_meta`/`render`; every record is validated
against the catalog and execute-probed (`SELECT 1 … LIMIT 1` through
`src.spatial_query.plan`); synthetic records that fail to execute are dropped (0 here).
Empty result sets are kept (`--empty-cap 1.0`): row counts are irrelevant to the
objective. `grouped_split` keeps a target and its paraphrases in one split:
**3,550 train / 444 val / 435 test**, target groups disjoint (asserted).

## Recipe — `scripts/automodel_nemotron_feln.yaml`

`nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16` @ `dfaf35de…` (Mamba-2 hybrid, `nemotron_h`),
BF16 LoRA rank 16 / alpha 32 / dropout 0 on all linear modules except `*.out_proj` (the
fused Mamba kernel reads the raw weight) and `lm_head`; AdamW 2e-4, wd 0.01, cosine, 40
warmup steps, global batch 16 (local 2), six epochs = 1,332 steps (~1.1 s/step, ~25 min
plus loss-only validation passes), FSDP2 dp=1, checkpoints every 100 steps and at epoch
ends. Dataset rows come from `src.prompt.build_dataset` (prompt masked, completion + EOS
supervised, `<think></think>` non-thinking template) re-packaged by
`scripts/automodel_feln.FELNDataset` into AutoModel's shifted layout.

```bash
cd /home/ubuntu/feln-lora/runs/<exp>
SP=/home/ubuntu/Automodel/.venv/lib/python3.12/site-packages
LD_LIBRARY_PATH=$(ls -d $SP/nvidia/*/lib | tr '\n' ':') CUDA_VISIBLE_DEVICES=1 \
  PYTHONPATH=/home/ubuntu/feln-lora HF_HUB_OFFLINE=1 \
  /home/ubuntu/Automodel/.venv/bin/automodel /home/ubuntu/feln-lora/scripts/automodel_nemotron_feln.yaml \
  --nproc-per-node 1            # in tmux; restore_from: LATEST resumes
CUDA_VISIBLE_DEVICES=0 bash /home/ubuntu/feln-lora/scripts/automodel_eval_queue.sh   # other GPU, in tmux
```

The queue scores each finished checkpoint on `data/val.json` with `src.infer_feln`
(batch 8), writes `checkpoints/*/eval-val.json`, and ends with `checkpoint_selection.json`:
validation `exact_match`, earliest step on ties. Test is never scored by either job.

## Result — `runs/automodel-nemotron-20260914/`

Selected `checkpoints/epoch_1_step_443` (step 444): **441/444 exact (99.32%)**, raw 96.85%,
0 parse errors; all 19 checkpoints from step 100 on score 420–441 (`checkpoint_selection.json`).
Merged export `exports/nemotron-4b-step443` (`src.infer_feln --export`: FP32-accumulated
merge, FP16 stored, `inference_config.json` prompt prefix/suffix + JSON-schema grammar,
`Layers.json`, `export_manifest.json`) re-scored identically: 441/444
(`exports/nemotron-4b-step443/eval-val-merged.json`). Re-scoring the recorded predictions
with `feln.FELN.same` (this repo's comparator) also gives 441/444 with no flipped verdict.

## GGUF — `runs/nemotron-mac-20260915/gguf/`, RTX `runs/automodel-nemotron-20260914/gguf/`

llama.cpp `b29c606e2` (the pinned older build has the `NEMOTRON_H` arch but no converter
class). The transformers-5 export confuses `convert_hf_to_gguf.py` (MoE misdetection,
`TokenizersBackend`), so convert from a `stage/` directory = the base model's
`config.json`/tokenizer files + a symlink to the merged `model.safetensors`:

```bash
python convert_hf_to_gguf.py stage --outtype f16 --outfile gguf/nemotron-4b-step443-f16.gguf
llama-quantize gguf/nemotron-4b-step443-f16.gguf gguf/nemotron-4b-step443-q8_0.gguf Q8_0
llama-quantize gguf/nemotron-4b-step443-f16.gguf gguf/nemotron-4b-step443-q4_k_m.gguf Q4_K_M
```

Q8_0 `110fdab9…` (4.04 GB) is byte-identical on both machines and checksum-pinned in
`gpu_server.json`; Q4_K_M `301908f4…` on the Mac differs from the RTX one (NEON vs AVX
quantization) but scores the same. Validation through `src.edge_client --records` against
`llama-server -c 2048 -np 1 -ngl all` (`validate-gguf-*.sh`):

| GGUF | RTX CUDA | Mac Metal (Homebrew llama.cpp b10964) |
|---|---|---|
| F16 | 438/444 | — (pruned) |
| Q8_0 | 438/444 | 438/444, warm p50 1.13 s |
| Q4_K_M | 439/444 | 438/444 |

F16 already loses the same three questions as Q8_0, so the gap to the HF model is a
runtime (llama.cpp decoding/grammar) effect, not quantization. Challenge set
(`tests/challenge.json`, 40 hand-written questions): 37/40 on the Mac Q8_0.

## QLoRA — `runs/automodel-nemotron-qlora-20260916/`, Mac `runs/nemotron-mac-qlora-20260916/`

Same data, adapter shape and schedule as the LoRA run (`scripts/automodel_nemotron_feln_qlora.yaml`
diffs from the LoRA recipe only in its `quantization:` block and the absence of the custom-model
`backend:` block), base loaded 4-bit NF4 with double quantization (`bitsandbytes` 0.50.2,
compute and storage dtype bf16), trained on **both GPUs** (`--nproc-per-node 2`, FSDP2 dp=2,
local batch 2 → effective 16 unchanged, seed 42), then scored by two `automodel_eval_queue.sh`
workers (one per GPU; the script claims a checkpoint with an atomic `mkdir`).

Two facts about the AutoModel checkout (`4e00f6be0`) decide the recipe:

- With `quantization:` set, AutoModel abandons its custom `NemotronHForCausalLM` and loads
  the HF class (transformers 5.15.1), whose Mamba fast path hands `out_proj.weight` straight
  to the fused kernel — a 4-bit `out_proj` would break there. `llm_int8_skip_modules:
  ["out_proj", "lm_head"]` keeps them bf16, which needs
  `scripts/automodel_qlora_skip_modules.patch` (one line: `create_bnb_config` forwards
  `llm_int8_skip_modules`) applied to the checkout; the applied diff is kept in the run dir
  (`automodel-4e00f6be0-patched.diff`).
- The HF class takes no `backend:` kwarg; it attends with SDPA by default.

`uv pip install bitsandbytes` into the AutoModel venv also downgraded `nvidia-cublas`
13.4.1.1 → 13.1.0.3; it was pinned back with `--no-deps` (bf16 matmul, NF4 and
`Linear4bit` verified afterwards).

Trainable parameters 17,054,208 (0.85%) in both runs — the same target set. Selected
`epoch_1_step_443` (step 444, the LoRA selection too): **442/444 exact (99.55%)**, raw
96.4%, 0 parse errors; the merged FP16 export (adapter merged onto the *BF16* base, FP32
accumulation — the standard QLoRA deployment, not the NF4 weights it trained against)
re-scores 442/444. Per-checkpoint validation, LoRA vs QLoRA:

| step | 100 | 200 | 222 | 300 | 400 | 444 | 500 | 600 | 666 | 700 | 800 | 888 | 900 | 1000–1332 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| LoRA | 420 | 436 | 433 | 438 | 438 | **441** | 441 | 441 | 440 | 440 | 441 | 440 | 440 | 441 |
| QLoRA | 425 | 431 | 419 | 436 | 440 | **442** | 442 | 440 | 442 | 442 | 441 | 439 | 440 | 441 |

Both settle at 441/444 from step 1000; the ±1–2 differences are within a single question
and both selections land on the same checkpoint.

| | LoRA (1 GPU) | QLoRA NF4 (2 GPUs, dp=2) |
|---|---|---|
| Wall time, 1,332 steps incl. loss-only val | 28:24 | 27:15 (27:47 with load) |
| Step time | ~1.1 s | ~1.04 s |
| Peak memory per GPU | 15.5 GiB | 17.4 GiB |
| Final validation loss | 0.0030 | 0.0026 |
| Val exact, selected checkpoint (HF) | 441/444 | 442/444 |
| Val exact, merged FP16 export | 441/444 | 442/444 |
| Val exact, F16 GGUF (RTX CUDA) | 438/444 | 442/444 |
| Val exact, Q8_0 GGUF (RTX CUDA) | 438/444 | 442/444 |
| Val exact, Q8_0 GGUF (Mac Metal) | 438/444 | MAC_VAL |
| Challenge 40, Q8_0 GGUF (Mac Metal) | 37/40 | MAC_CHALLENGE |
| Warm p50 per question, Q8_0 (RTX / Mac) | 0.33 s / 1.13 s | 0.32 s / MAC_P50 |

Memory did not drop: on a 4B model the bf16 weights are ~8 GB and LoRA activations,
optimizer state and the FSDP2 all-gathers dominate, and the NF4 dequantization is not
free — two GPUs ran the same 1,332 steps in about the wall time one GPU took in bf16.
QLoRA's value here is that it costs nothing in accuracy (the fine-tuned adapter sits on
a 4-bit base at train time and merges cleanly into the full-precision base for the GGUF),
not that it is cheaper on this hardware. GGUF conversion is the same `stage/` recipe:
`gguf/nemotron-4b-qlora-step443-{f16,q8_0}.gguf` (Q8_0 `ec443cbb…`, 4.23 GB).

The Mac serves the QLoRA GGUF next to the LoRA one:

```bash
uv run --no-sync python -m src.studio \
  --llama "QLoRA NF4 · Q8_0 · 2026-09-16=runs/nemotron-mac-qlora-20260916/merged=http://127.0.0.1:8093"
```

## Serving

Mac: `src.studio` starts `llama-server` on 8092 from the Q8_0 GGUF. RTX: `src.gpu_server
server start` runs `llama-server -c 4096 -np 2 -ngl all` per GPU in tmux sessions
`feln-gpu-server-<gpu>` on ports 8090/8091 (localhost), reached from the Mac over an
`ssh -L` tunnel; two instances measured 14.5 s vs 23.1 s for the same batch on one, while
one instance split across both GPUs gained nothing. Stopping the EC2 instance kills the
tmux sessions; `server start` recreates them after a checksum gate on the GGUF.

Operational rules: never `pkill -f`/`pgrep -f` a pattern that appears in your own ssh
command line (it kills the remote shell); never commit or copy the SSH key; no HF tokens in
logs or docs.
