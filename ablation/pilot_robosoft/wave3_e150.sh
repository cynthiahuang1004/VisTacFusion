#!/bin/bash
# 150-epoch confirmation of the 4 key comparisons, after the 50-epoch pilot drains
cd /media/hdd2/ihsuan/VisTacFusion
PY=/home/shared/miniconda3/envs/vistacfusion/bin/python
OUT=/media/hdd2/ihsuan/VisTacFusion_outputs_hdd2
until [ $(ps -u shared -o cmd | grep -c '[e]ngine.train --model') -le 1 ]; do sleep 300; done
g=0
for n in A_realonly A_ganloo B_k100_realonly B_k100_gank; do
  CUDA_VISIBLE_DEVICES=$g nohup $PY -m vistacfusion.engine.train \
    --model ablation/encoder/tac_sitr_single.yaml \
    --train ablation/pilot_robosoft/train_pilot_e150.yaml \
    --data ablation/pilot_robosoft/data_$n.yaml \
    --output-dir $OUT/pilot150_$n > outputs/pilot_logs/e150_$n.log 2>&1 &
  g=$((g+1))
done
echo launched
