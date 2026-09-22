#!/bin/bash
# gated_launch.sh <run_name> <data_cfg> [model_cfg] [train_cfg] [extra args...]
# Waits until some GPU has < $MEMLIMIT MiB used (default 36000), then launches there via launch2.sh.
cd /media/hdd2/ihsuan/VisTacFusion
MEMLIMIT=${MEMLIMIT:-36000}
while :; do
  best=""; bestm=999999
  for g in 0 1 2 3; do m=$(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader,nounits); [ $m -lt $bestm ] && { bestm=$m; best=$g; }; done
  [ $bestm -lt $MEMLIMIT ] && break
  sleep 180
done
ablation/pilot_robosoft/launch2.sh $best "$@"; sleep 240
