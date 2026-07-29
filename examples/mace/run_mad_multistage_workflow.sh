#!/usr/bin/env bash

#set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)

if [[ -f "$REPO_ROOT/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$REPO_ROOT/.venv/bin/activate"
fi

export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="${TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD:-1}"

OUTPUT_DIR="${1:-$SCRIPT_DIR/runs/mad_multistage}"
PRECISION="${PRECISION:-32}"
ACCELERATOR="${ACCELERATOR:-auto}"
DEVICES="${DEVICES:-1}"
NUM_WORKERS="${NUM_WORKERS:-0}"
STAGE1_BATCH_SIZE="${STAGE1_BATCH_SIZE:-50}"
STAGE2_BATCH_SIZE="${STAGE2_BATCH_SIZE:-50}"
STAGE3_BATCH_SIZE="${STAGE3_BATCH_SIZE:-20}"
STAGE1_MAX_EPOCHS="${STAGE1_MAX_EPOCHS:-200}" 
STAGE2_MAX_EPOCHS="${STAGE2_MAX_EPOCHS:-500}" 
STAGE3_MAX_EPOCHS="${STAGE3_MAX_EPOCHS:-10}" 

python "$SCRIPT_DIR/mad_stage1_pretrain.py" \
  --output-dir "$OUTPUT_DIR/stage1" \
  --batch-size "$STAGE1_BATCH_SIZE" \
  --precision "$PRECISION" \
  --accelerator "$ACCELERATOR" \
  --devices "$DEVICES" \
  --num-workers "$NUM_WORKERS" \
  --max-epochs "$STAGE1_MAX_EPOCHS"

python "$SCRIPT_DIR/mad_stage2_fcl_finetune.py" \
  --foundation-model-path "$OUTPUT_DIR/stage1/foundation.model" \
  --output-dir "$OUTPUT_DIR/stage2" \
  --batch-size "$STAGE2_BATCH_SIZE" \
  --precision "$PRECISION" \
  --accelerator "$ACCELERATOR" \
  --devices "$DEVICES" \
  --num-workers "$NUM_WORKERS" \
  --max-epochs "$STAGE2_MAX_EPOCHS"

python "$SCRIPT_DIR/mad_stage3_calibrate.py" \
  --model-path "$OUTPUT_DIR/stage2/fcl_wrapper.pt" \
  --output-dir "$OUTPUT_DIR/stage3" \
  --batch-size "$STAGE3_BATCH_SIZE" \
  --precision "$PRECISION" \
  --accelerator "$ACCELERATOR" \
  --devices "$DEVICES" \
  --num-workers "$NUM_WORKERS" \
  --max-epochs "$STAGE3_MAX_EPOCHS"

printf '\nWorkflow artifacts:\n'
printf '  Stage 1 foundation model: %s\n' "$OUTPUT_DIR/stage1/foundation.model"
printf '  Stage 1 summary: %s\n' "$OUTPUT_DIR/stage1/summary.json"
printf '  Stage 2 flexible wrapper: %s\n' "$OUTPUT_DIR/stage2/fcl_wrapper.pt"
printf '  Stage 2 summary: %s\n' "$OUTPUT_DIR/stage2/summary.json"
printf '  Stage 3 sweep summary: %s\n' "$OUTPUT_DIR/stage3/summary.json"
