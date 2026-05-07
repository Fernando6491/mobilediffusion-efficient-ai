#!/usr/bin/env bash
set -euo pipefail

# "MD-UFO (no adversarial)" style one-step denoiser training.
# This does NOT implement the paper's GAN/discriminator part.
# It trains a UNet to denoise from a fixed timestep t in ONE step using an x0 reconstruction loss.
#
# Recommended starting point (aim ~<=6 hours):
#   STEPS=30000 BATCH_SIZE=2 GPU_ID=1 \
#   TEACHER=checkpoints/unet_md_paper_like_epoch2.pt \
#   ./scripts/train_md_ufo_noadv.sh
#

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

GPU_ID="${GPU_ID:-1}"
BATCH_SIZE="${BATCH_SIZE:-2}"
IMAGE_SIZE="${IMAGE_SIZE:-512}"
LR="${LR:-1e-4}"
STEPS="${STEPS:-30000}"
ONE_STEP_T="${ONE_STEP_T:-999}"
OUTPUT_DIR="${OUTPUT_DIR:-checkpoints}"
SAVE_EVERY_STEPS="${SAVE_EVERY_STEPS:-2000}"

TEACHER="${TEACHER:-}"
if [[ -z "$TEACHER" ]]; then
  echo "ERROR: set TEACHER=/path/to/init_unet.pt (e.g. md_paper_like_epoch2.pt)" >&2
  exit 1
fi

if [[ -n "${DATA_ROOT:-}" ]]; then
  DATA_ARGS=(--data_root "$DATA_ROOT")
else
  DATA_ARGS=(--kaggle_download)
fi

exec conda run -n fastdiff --no-capture-output -- \
  python -m src.train \
    --backend md_ufo_noadv \
    --teacher_checkpoint "$TEACHER" \
    "${DATA_ARGS[@]}" \
    --image_size "$IMAGE_SIZE" \
    --batch_size "$BATCH_SIZE" \
    --gpu_id "$GPU_ID" \
    --lr "$LR" \
    --max_steps "$STEPS" \
    --one_step_t "$ONE_STEP_T" \
    --save_every_steps "$SAVE_EVERY_STEPS" \
    --output_dir "$OUTPUT_DIR"

