# feln-lora

LoRA fine-tuning of **Nemotron-3-Nano-4B** to translate North Sea questions into
[FELN](../feln) — `{"layers": [...], "where": [...], "relations": [...]}` — served locally
as a GGUF (llama.cpp) with a JSON-schema grammar. The sibling of [`feln-rag`](../feln-rag),
which solves the same task by retrieval + frontier LLM; both depend on [`feln`](../feln)
(the FELN model, strict comparator, units) and [`layers-json`](../layers-json) (catalog model).

Measured (validation, 444 questions, `feln.FELN.same`): **441/444 (99.3%)** for the adapter
and the merged FP16 export, **438–439/444 (98.6%)** as F16, Q8_0 or Q4_K_M GGUF on CUDA and Metal
(a llama.cpp runtime effect, not quantization: F16 misses the same questions). The 40 hand-written challenge questions
(`tests/challenge.json`): 37/40 on the Mac Q8_0 GGUF. No real-user log has been measured.

## Layout

```
src/prompt.py         trained prompt, dataset rows, strict JSON parsing   (feln.FELN)
src/feln_data.py      Schema: validate/compile FELN against Layers.json, question generator, grouped split
src/edge_client.py    llama-server client: grammar, prompt bundle, benchmark   (feln.FELNCompare)
src/infer_feln.py     GPU box: score a checkpoint / export the merged bundle
src/studio.py         FELN Studio, local playground on the Mac GGUF          → web/studio/
src/spatial_query.py  read-only, parameterized DuckDB execution of a FELN   (feln.to_meters)
src/gpu_server.py     EC2 machine control, per-GPU llama-servers over SSH, local tunnel
src/mcp_server.py     MCP over stdio: feln / execute_feln / status / machine_* / server_stop
scripts/              prepare_northsea.py (data), automodel_*.{yaml,py,sh} (train + score), execution_fidelity.py
runs/                 git-ignored artifacts: regen-20260914-ilike (data), automodel-nemotron-20260914 (RTX mirror),
                      nemotron-mac-20260915 (merged bundle + GGUFs)
```

`relations[i]` connects `layers[0]` to `layers[i+1]`. `where[i]` filters `layers[i]`; an
empty string means no filter, and `field = ''` differs from `field IS NULL`.

## Setup (Mac)

```bash
uv sync                                     # py3.13; ../feln and ../layers-json editable
uv run --no-sync python -m unittest discover -s tests
uv run --no-sync ruff check src scripts tests && uv run --no-sync pyright src scripts tests
```

`--no-sync` keeps the environment as installed. The `layers-json` override in
`pyproject.toml` is an absolute path (as in `feln-rag`); adjust it if the workspace moves.
Training/scoring deps (`torch`, `transformers`, `peft`) live only on the GPU box's
AutoModel venv; `uv sync --extra train` installs them elsewhere if ever needed.

## FELN Studio

```bash
uv run --no-sync python -m src.studio       # http://127.0.0.1:8766/
```

Starts Homebrew's `llama-server` on port 8092 with
`runs/nemotron-mac-20260915/gguf/nemotron-4b-step443-q8_0.gguf` (or reuses a healthy one)
and stops it on exit; prompt and grammar come from `runs/nemotron-mac-20260915/merged/`.
Ask a question and inspect the FELN, raw output, timings, the trained prompt (editable as
an experiment) and the strict comparator's verdict when the question is a recorded one
(`--records`, default `tests/challenge.json`). Stdlib HTTP server, vanilla JS, localhost
only. `--llama LABEL=BUNDLE=URL` adds another served GGUF; `--no-nemotron` drops the default.

Benchmark any served GGUF through the same client:

```bash
uv run --no-sync python -m src.edge_client --bundle runs/nemotron-mac-20260915/merged \
  --url http://127.0.0.1:8092 --records tests/challenge.json --output /tmp/challenge.json
```

## Remote GPU inference + MCP

`gpu_server.json` names the EC2 box (`i-01c983f1111ea3b69`, GeoCowork profile, elastic IP
52.88.16.111), the remote `llama-server` build and the checksum-pinned Q8_0 GGUF under
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

# RTX (AutoModel venv, see the YAML header for LD_LIBRARY_PATH): train, then score every
# checkpoint on validation and select the earliest best
automodel scripts/automodel_nemotron_feln.yaml --nproc-per-node 1
bash scripts/automodel_eval_queue.sh
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
| Strict exact | 0.94 (mpnet RAG, 50 q); 0.86–0.88 zero-shot | 441/444 val (99.3%) HF, 438/444 Q8 GGUF; 37/40 challenge |
| Latency | cloud round trip + ~10 ms retrieval | 0.3 s RTX GPU, ~1 s Mac Metal |
| Runtime cost / privacy | LLM tokens per query; query + catalog + shots leave the machine | none after training; fully offline |
| Catalog change | edit `Layers.json` / examples | regenerate data, retrain (~30 min), re-validate |
| Where conventions live | retrieved shots | the weights, plus a JSON-schema grammar |

Numbers are each project's recorded runs, not one shared benchmark (RAG scored on a
100-question holdout of the original `FELN.json`; the fine-tune on grouped splits of the
regenerated ILIKE set plus the challenge set). The two compose: pasting RAG's shots into
the fine-tuned prompt (the Studio's prompt override) is the cheapest unexplored experiment.

## History

Imported from `gait-feln-finetuning@6869f11` on 2026-09-15 with the Qwen QLoRA/Optuna
trainer, ONNX/Jetson Orin edge path, SPA and hybrid router removed, and the local FELN
comparator replaced by `feln`'s (re-scoring the recorded predictions with it changed no
verdict: 441/444 and 438/444, zero flips). That repository remains the archive of the
Qwen-era experiments and their documents.
