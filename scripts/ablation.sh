#!/usr/bin/env bash
# Ablation sweep skeleton (CLAUDE.md 9). Each run overrides a config flag; stop on REAL-data
# gains, not sim val. Override flags by editing a copy of configs/model.yaml per run, or extend
# vistacfusion/engine/train.py with --override key=value parsing.
set -euo pipefail

echo "Ablation knobs to sweep (edit configs/model.yaml per run):"
echo "  fusion_trunk.num_bottleneck_tokens   m in {8, 16, 32}"
echo "  fusion_trunk.num_layers              L in {2, 3, 4}  (keep len(tap_layers)==4)"
echo "  fusion_trunk.bottleneck_continuity   {reset, carry}        (ablation A)"
echo "  fusion_trunk.fusion_variant          {asymmetric, symmetric_coattention}  (ablation B)"
echo "  heads.pose.pose_mode                 {regression, classification}         (ablation C)"
echo "  encoder.share_encoder_weights        {true, false}"
echo "  heads.dpt.tap_source                 {trunk (v1), encoder_multiscale (v2)}  (RETRAIN to switch)"
