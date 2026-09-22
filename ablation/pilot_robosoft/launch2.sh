#!/bin/bash
# launch2.sh <gpu> <run_name> <data_cfg_name> [model_cfg] [train_cfg] [extra train.py args...]
cd /media/hdd2/ihsuan/VisTacFusion
PY=/home/shared/miniconda3/envs/vistacfusion/bin/python
OUT=/media/hdd2/ihsuan/VisTacFusion_outputs_hdd2
gpu=$1; name=$2; data=$3; model=${4:-ablation/encoder/tac_sitr_single.yaml}; train=${5:-ablation/pilot_robosoft/train_pilot_e50.yaml}; shift 5 2>/dev/null || shift $#
CUDA_VISIBLE_DEVICES=$gpu nohup $PY -m vistacfusion.engine.train --model $model --train $train \
  --data ablation/pilot_robosoft/data_$data.yaml --output-dir $OUT/pilot_$name "$@" > outputs/pilot_logs/$name.log 2>&1 &
echo "launched $name on GPU$gpu"
