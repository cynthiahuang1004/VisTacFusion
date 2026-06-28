#!/usr/bin/env bash
# Train fusion trunk + DPT head + pose head (encoders frozen).
#
# Single GPU:
#   bash scripts/train.sh
#
# Multi-GPU (2 GPUs):
#   NGPUS=2 bash scripts/train.sh
#
# Specific GPUs:
#   CUDA_VISIBLE_DEVICES=0,1 NGPUS=2 bash scripts/train.sh
#
# Resume:
#   bash scripts/train.sh --resume outputs/last.pt
#   NGPUS=2 bash scripts/train.sh --resume outputs/last.pt
set -euo pipefail

OUTDIR=${OUTDIR:-outputs/$(date +%Y%m%d_%H%M%S)}
NGPUS=${NGPUS:-1}

COMMON_ARGS=(
  -m vistacfusion.engine.train
  --model configs/model.yaml
  --train configs/train.yaml
  --data  configs/data.yaml
  --output-dir "$OUTDIR"
  "$@"
)

if [ "$NGPUS" -gt 1 ]; then
  torchrun --nproc_per_node="$NGPUS" "${COMMON_ARGS[@]}"
else
  python "${COMMON_ARGS[@]}"
fi
