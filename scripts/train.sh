#!/usr/bin/env bash
# Train the shared model with modality dropout. With configs/data.yaml dataset:synthetic and
# encoder.checkpoint:null this is the end-to-end smoke run (mock encoder, no DINOv3 weights).
set -euo pipefail
python -m vistacfusion.engine.train \
  --model configs/model.yaml \
  --train configs/train.yaml \
  --data  configs/data.yaml \
  "$@"
