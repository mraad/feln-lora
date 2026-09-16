# Training, scoring, export — Nemotron-3-Nano-4B LoRA

Everything below was run once on 2026-09-14/15 and is complete. Do not restart the
training or scoring jobs; rerun only to reproduce or to train on new data.

## Machines

| | Path | Notes |
|---|---|---|
| Mac | `~/GWorkspace/feln-lora` | data generation, GGUF validation on Metal, Studio, MCP; `uv sync` here |
| RTX (EC2 box in `gpu_server.json`, 2× RTX PRO 6000) | `/home/ubuntu/feln-lora` (rsync of this repo, no `.venv`) | training, checkpoint scoring, merge/export, GGUF conversion, inference servers |

The RTX Python is `/home/ubuntu/Automodel/.venv` (NeMo AutoModel checkout `4e00f6be0`,
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
