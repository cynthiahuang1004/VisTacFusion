#!/usr/bin/env bash
# Ablation sweep skeleton. Each run overrides one config flag; judge on real-data gains, not
# sim val. Override by editing a copy of configs/model.yaml per run.
set -euo pipefail

echo "Ablation knobs to sweep (edit configs/model.yaml per run):"
echo "  fusion_trunk.num_bottleneck_tokens   m in {8, 16, 32}"
echo "  fusion_trunk.num_layers              L in {2, 3, 4}  (keep len(tap_layers)==4)"
echo "  fusion_trunk.bottleneck_continuity   {reset, carry}"
echo "  fusion_trunk.fusion_variant          {asymmetric, symmetric_coattention}"
echo "  heads.pose.pose_mode                 {regression, classification}"
echo "  encoder.share_encoder_weights        {true, false}"
echo "  heads.dpt.tap_source                 {trunk (v1), encoder_multiscale (v2)}  (retrain to switch)"
