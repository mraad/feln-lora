# feln-lora

## Why a small model

I believe the future belongs to small language models: a model that runs on the edge and
does one narrow job extremely well, because it was shaped for exactly that job. Retrieval
works, but it still leans on a large model to do the reasoning, and that model lives
somewhere else. A small model fine-tuned on the task lets you aim precisely at what you
are after — here, turning a North Sea question into one exact FELN query — and run it on
the machine in front of you, offline, in under a second. This repository does that with
NVIDIA's Nemotron-3-Nano-4B, fine-tuned with LoRA and QLoRA through
[NeMo AutoModel](https://github.com/NVIDIA-NeMo/Automodel) and served as a GGUF.

## What it is

LoRA (and QLoRA) fine-tuning of **Nemotron-3-Nano-4B** via
[NeMo AutoModel](https://github.com/NVIDIA-NeMo/Automodel) (the YAML recipes in
`scripts/`, run with its `automodel` CLI) to translate North Sea
questions into [FELN](../feln) — `{"layers": [...], "where": [...], "relations": [...]}` —
served locally as a GGUF (llama.cpp) with a JSON-schema grammar. The sibling of
[`feln-rag`](../feln-rag), which solves the same task by retrieval + frontier LLM; both
depend on [`feln`](../feln) (the FELN model, strict comparator, units), which in turn pins
the public [`layers-json`](https://github.com/mraad/layers-json) catalog model.

Measured (validation, 444 questions, `feln.FELN.same`):

| | LoRA (BF16 base, 1 GPU) | QLoRA (NF4 base, 2 GPUs) |
|---|---|---|
| Adapter / merged FP16 export | **441/444 (99.3%)** | **442/444 (99.55%)** |
| Q8_0 GGUF, RTX CUDA | 438/444 | 442/444 |
| Q8_0 GGUF, Mac Metal | 438/444 | 442/444 |
| Challenge 40 (`tests/challenge.json`), Mac Q8_0 | 37/40 | 37/40 (same three misses) |
| Train wall time, 1,332 steps | 28 min | 27 min |

Both select the same checkpoint (step 444) and settle at 441/444 from step 1000; the
differences are one or two questions. The LoRA GGUFs lose three questions to a llama.cpp
runtime effect (F16 misses the same ones as Q8_0), the QLoRA GGUFs none: on the Mac the
QLoRA Q8_0 misses two validation questions, both among the LoRA GGUF's six. QLoRA neither
saved memory nor time on this 4B model (see [TRAINING.md](TRAINING.md)); its result is
that a 4-bit base costs nothing in accuracy. No real-user log has been measured.

The catalog and gold changed on 2026-09-16 (new aliases/hints, 1,000 regenerated questions,
138 of them on geometry-less table layers that FELN excludes). Zero-shot on the 862
geometry-layer records both v1 models score **832/862 (96.5%)**. Retrained in parallel on
the regenerated data (`runs/regen-20260916-v2`, 3,463/410/419), one GPU each:

| v2 (validation, 410 questions) | LoRA | QLoRA NF4 (one GPU) |
|---|---|---|
| Adapter / merged FP16 export | **409/410 (99.76%)** | **409/410 (99.76%)** |
| Q8_0 GGUF, RTX CUDA / Mac Metal | 408 / 408 | 408 / 408 |
| Challenge 40, Mac Q8_0 | 37/40 | 36/40 |
| Train wall time, 1,302 steps | 28 min | 31 min |

Mac bundles and Q8_0 GGUFs: `runs/nemotron-mac-v2-20260916/{lora,qlora}`. See TRAINING.md
"v2 catalog" for the per-checkpoint table and misses.

## Layout

```
src/prompt.py         trained prompt, dataset rows, strict JSON parsing   (feln.FELN)
src/feln_data.py      Schema: validate/compile FELN against Layers.json, question generator, grouped split
src/edge_client.py    llama-server client: grammar, prompt bundle, benchmark   (feln.FELNCompare)
src/infer_feln.py     GPU box: score a checkpoint / export the merged bundle
src/spatial_query.py  read-only, parameterized DuckDB execution of a FELN   (feln.to_meters)
src/gpu_server.py     EC2 machine control, per-GPU llama-servers over SSH, local tunnel
src/mcp_server.py     MCP over stdio: feln / execute_feln / status / machine_* / server_stop
scripts/              prepare_northsea.py (data), automodel_nemotron_feln{,_qlora}.yaml (LoRA / QLoRA recipes),
                      automodel_feln.py + automodel_eval_queue.sh (train + score), execution_fidelity.py
runs/                 git-ignored artifacts: regen-20260914-ilike (data), automodel-nemotron-20260914 and
                      automodel-nemotron-qlora-20260916, automodel-nemotron{,-qlora}-v2-20260916 (RTX mirrors),
                      nemotron-mac-20260915 (v1 LoRA bundle + GGUFs), nemotron-mac-qlora-20260916 (v1 QLoRA),
                      nemotron-mac-v2-20260916/{lora,qlora} (v2 bundles + Q8_0; the Studio serves qlora)
```

`relations[i]` connects `layers[0]` to `layers[i+1]`. `where[i]` filters `layers[i]`; an
empty string means no filter, and `field = ''` differs from `field IS NULL`.

## Setup (Mac)

```bash
uv sync                                     # py3.13; ../feln editable, layers-json from GitHub
uv run --no-sync python -m unittest discover -s tests
uv run --no-sync ruff check src scripts tests && uv run --no-sync pyright src scripts tests
```

`--no-sync` keeps the environment as installed. `layers-json` is public, so no GitHub
credentials or local checkout are needed; to test a local `../layers-json` edit,
`uv pip install -e ../layers-json` and keep using `--no-sync`. Training/scoring deps (`torch`, `transformers`, `peft`) live only on the GPU box's
AutoModel venv; `uv sync --extra train` installs them elsewhere if ever needed.

## FELN Studio

The local playground moved to [`../feln-studio`](../feln-studio): one SPA over this GGUF,
the feln-liquid MLX adapter and feln-rag, with the same strict judge. From there:

```bash
uv run --no-sync python -m feln_studio.server --start        # http://127.0.0.1:8766/
```

It starts `llama-server` on 8092 from the v2 QLoRA GGUF
`runs/nemotron-mac-v2-20260916/qlora/gguf/nemotron-4b-v2-qlora-q8_0.gguf` and reads the prompt
and grammar from `runs/nemotron-mac-v2-20260916/qlora/merged/` (since 2026-09-16; the v1
model is `runs/nemotron-mac-20260915`). To compare another GGUF side by side, serve it on
another port and register it with `--llama LABEL=BUNDLE=URL`, e.g.
`--llama "LoRA v2 · Q8_0=runs/nemotron-mac-v2-20260916/lora/merged=http://127.0.0.1:8094"`;
`--gold tests/challenge.json` judges against the challenge set.

Benchmark any served GGUF through the same client:

```bash
uv run --no-sync python -m src.edge_client --bundle runs/nemotron-mac-v2-20260916/qlora/merged \
  --url http://127.0.0.1:8092 --records tests/challenge.json --output /tmp/challenge.json
```

## Remote GPU inference + MCP

`gpu_server.json` (git-ignored; copy `gpu_server.example.json` and fill in the AWS profile,
instance id, host and SSH key path) names the EC2 box, the remote `llama-server` build and
the checksum-pinned Q8_0 GGUF under
`/home/ubuntu/feln-lora/runs/automodel-nemotron-20260914/gguf/`, and the local bundle
whose `inference_config.json`/`Layers.json` define prompt and grammar. `server start` runs
one `llama-server` per listed GPU (ports 8090, 8091; two slots each) and the Mac
round-robins across them — 1.6× the throughput of one instance, whereas splitting one
instance across both GPUs gained nothing for a 4 GB model.

```bash
uv run --no-sync python -m src.gpu_server machine status|start|stop   # aws ec2; stop kills tmux jobs
uv run --no-sync python -m src.gpu_server server status|start|stop    # ssh + tmux llama-servers
uv run --no-sync python -m src.gpu_server feln 'Show gas wells in Norway'
claude mcp add feln-gpu -- $PWD/.venv/bin/python -m src.mcp_server    # MCP over stdio
```

MCP tools: `feln` (remote GPU; starts the server on demand, never the machine),
`execute_feln` (read-only local DuckDB at `src/spatial_query.DATABASE` or `$FELN_DATABASE`,
rows without geometry), `status`, `machine_start`, `machine_stop`, `server_stop`.
Nothing listens publicly on the GPU host; the Mac reaches it through an `ssh -L` tunnel.

## Data, training, export

See [TRAINING.md](TRAINING.md) for the exact recipe, environment fixes, GGUF conversion and
provenance. In short:

```bash
# Mac: regenerate the run inputs (gold FELN.json, LIKE→ILIKE on hinted columns, synthetic
# questions, execute-filtered, grouped train/val/test)
uv run --no-sync python -m scripts.prepare_northsea --output runs/<exp>

# RTX (AutoModel venv, see the YAML header for LD_LIBRARY_PATH), from the experiment root:
# train, then score every checkpoint on validation and select the earliest best
REPO=/home/ubuntu/feln-lora; cd $REPO/runs/<exp>; export PYTHONPATH=$REPO
automodel $REPO/scripts/automodel_nemotron_feln.yaml --nproc-per-node 1         # LoRA, one GPU
automodel $REPO/scripts/automodel_nemotron_feln_qlora.yaml --nproc-per-node 2   # QLoRA NF4, both GPUs (v2 ran with 1)
bash $REPO/scripts/automodel_eval_queue.sh                                      # one worker per free GPU
python -m src.infer_feln --model checkpoints/<best>/model --schema data/Layers.json --export exports/<name>
```

Scoring: `exact_match` is `feln.FELN.same` (sqlglot-normalized WHERE, literal casts
dropped, secondaries matched by name); `raw_exact_match` is byte-identical output;
`parse_error_rate` counts output that is not FELN or fails the catalog. Report raw and
schema-compiled (`--compile-sql`) accuracy separately; never relax operators, codes,
spatial direction, NULL semantics or the primary layer to improve a score. Validation
selects checkpoints; the test split and the challenge set are regression sets only. FELN
validity plus the strict match is the objective — whether a query returns rows (or none) is
diagnostic (`scripts/execution_fidelity.py`), never a tuning signal.

## Fine-tuning versus RAG (`../feln-rag`)

| | feln-rag (5 shots → gpt-5.5) | feln-lora (LoRA → local GGUF) |
|---|---|---|
| Strict exact | 0.94 (mpnet RAG, 50 q); 0.86–0.88 zero-shot | v1: 441–442/444 val (99.3–99.55%) HF, 438–442/444 Q8 GGUF, 37/40 challenge; v2: 409/410 HF, 408/410 Q8, 36–37/40 challenge |
| Latency | cloud round trip + ~10 ms retrieval | 0.3 s RTX GPU, ~1 s Mac Metal |
| Runtime cost / privacy | LLM tokens per query; query + catalog + shots leave the machine | none after training; fully offline |
| Catalog change | edit `Layers.json` / examples | regenerate data, retrain (~30 min), re-validate |
| Where conventions live | retrieved shots | the weights, plus a JSON-schema grammar |

Numbers are each project's recorded runs, not one shared benchmark (RAG scored on a
100-question holdout of the original `FELN.json`; the fine-tune on grouped splits of the
regenerated ILIKE set plus the challenge set). The two compose: pasting RAG's shots into
the fine-tuned prompt (feln-studio's prompt override) is the cheapest unexplored experiment.

## History

Imported from `gait-feln-finetuning@6869f11` on 2026-09-15 with the Qwen QLoRA/Optuna
trainer, ONNX/Jetson Orin edge path, SPA and hybrid router removed, and the local FELN
comparator replaced by `feln`'s (re-scoring the recorded predictions with it changed no
verdict: 441/444 and 438/444, zero flips). That repository remains the archive of the
Qwen-era experiments and their documents.
