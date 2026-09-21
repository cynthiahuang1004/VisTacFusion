#!/bin/bash
# usage: launch.sh <gpu> <name> [name...]   — single-GPU SITR tactile-only pilot runs
cd /media/hdd2/ihsuan/VisTacFusion
PY=/home/shared/miniconda3/envs/vistacfusion/bin/python
OUT=/media/hdd2/ihsuan/VisTacFusion_outputs_hdd2
gpu=$1; shift
for n in "$@"; do
  CUDA_VISIBLE_DEVICES=$gpu nohup $PY -m vistacfusion.engine.train \
    --model ablation/encoder/tac_sitr_single.yaml \
    --train ablation/pilot_robosoft/train_pilot_e50.yaml \
    --data ablation/pilot_robosoft/data_$n.yaml \
    --output-dir $OUT/pilot_$n > outputs/pilot_logs/$n.log 2>&1 &
done
