#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

EPOCHS="${EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-2}"
GPU_ID="${GPU_ID:-1}"
IMAGE_SIZE="${IMAGE_SIZE:-512}"
OUTPUT_DIR="${OUTPUT_DIR:-checkpoints}"
LR="${LR:-1e-4}"

if [[ -n "${DATA_ROOT:-}" ]]; then
  DATA_ARGS=(--data_root "$DATA_ROOT")
else
  DATA_ARGS=(--kaggle_download)
fi

exec conda run -n fastdiff --no-capture-output -- \
  python -m src.train \
    --backend md_paper_like \
    "${DATA_ARGS[@]}" \
    --image_size "$IMAGE_SIZE" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --gpu_id "$GPU_ID" \
    --lr "$LR" \
    --save_every_epochs 1 \
    --output_dir "$OUTPUT_DIR" \
    "$@"

