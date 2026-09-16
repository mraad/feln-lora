#!/bin/bash
# Score every AutoModel checkpoint on validation with src.infer_feln, on a separate
# GPU, until training has exited and no checkpoint is left unscored.
# Usage (from the experiment root runs/<exp>/, which holds data/ and checkpoints/):
#   CUDA_VISIBLE_DEVICES=0 bash /home/ubuntu/feln-lora/scripts/automodel_eval_queue.sh
# Several workers (one per free GPU) may run at once: each claims a checkpoint with
# `flock -n` on $d/.scoring; the kernel releases the claim when the worker dies.
set -u
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONPATH=$(cd "$(dirname "$0")/.." && pwd)  # the feln-lora checkout, not the experiment root
# The AutoModel venv carries the mamba_ssm kernels; the shared training venv falls back
# to a naive chunk scan that needs ~96 GiB at batch 8. Same LD_LIBRARY_PATH rule as training.
VENV=/home/ubuntu/Automodel/.venv
export LD_LIBRARY_PATH=$(ls -d $VENV/lib/python3.12/site-packages/nvidia/*/lib | tr '\n' ':')
PY=$VENV/bin/python

pending() {
  for d in checkpoints/epoch_*_step_*/; do
    [ -f "$d/model/adapter_model.safetensors" ] && [ ! -f "$d/eval-val.json" ] && echo "$d"
  done
}

while true; do
  for d in $(pending); do
    # A checkpoint still being written is younger than two minutes.
    [ $(( $(date +%s) - $(stat -c %Y "$d") )) -lt 120 ] && continue
    # The log redirect lives inside the locked command so a busy lock never truncates a
    # sibling's live log; exit 75 = lock held elsewhere, the checkpoint stays pending.
    flock -n -E 75 "$d/.scoring" -c "$PY -m src.infer_feln --model '$d/model' --schema data/Layers.json \
      --records data/val.json --output '$d/eval-val.json' --batch-size 8 > '$d/eval-val.log' 2>&1"
    rc=$?
    [ $rc -eq 75 ] && continue
    echo "$(date -u +%FT%TZ) scored $d exit $rc"
  done
  if grep -q "^EXIT=" train.log 2>/dev/null && [ -z "$(pending)" ]; then
    break
  fi
  sleep 60
done

python3 - <<'EOF'
import json, glob, os
rows = []
for path in glob.glob("checkpoints/epoch_*_step_*/eval-val.json"):
    r = json.load(open(path))
    name = os.path.basename(os.path.dirname(path))
    rows.append({"checkpoint": name, "step": int(name.rsplit("_", 1)[1]) + 1,
                 "exact_match": r["exact_match"], "raw_exact_match": r["raw_exact_match"],
                 "parse_error_rate": r["parse_error_rate"], "n": r["n"]})
rows.sort(key=lambda r: r["step"])
# Ties go to the earliest checkpoint, as in the 2026-09-13 backbone trials.
best = max(rows, key=lambda r: (r["exact_match"], -r["step"])) if rows else None
json.dump({"selected_on": "validation exact_match, earliest step on ties", "best": best,
           "checkpoints": rows, "test_evaluated": False},
          open("checkpoint_selection.json", "w"), indent=2)
print(json.dumps(best, indent=2))
EOF
