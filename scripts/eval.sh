#!/usr/bin/env bash
# Inference / evaluation script.
#
# Single pair:
#   bash scripts/eval.sh --checkpoint outputs/best.pt \
#       --tactile /path/to/tactile.png --rgb /path/to/rgb.png
#
# Batch (full session):
#   bash scripts/eval.sh --checkpoint outputs/best.pt \
#       --session-dir /media/hdd2/ihsuan/gs_blender/renders/button/session_000/sensor_0000/
#
# Sim val set visualization:
#   bash scripts/eval.sh --checkpoint outputs/best.pt --eval-sim --num-vis 30
set -euo pipefail

python -m vistacfusion.engine.inference \
  --model configs/model.yaml \
  --train configs/train.yaml \
  --data  configs/data.yaml \
  "$@"
