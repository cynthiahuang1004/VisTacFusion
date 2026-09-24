#!/bin/bash
# Sensor-B (GelSlim 4.0) K-shot pilot runs (50 ep): real-only / +Blender sim / +GAN-4.0-K. Memory-gated, <=3 of ours per GPU.
cd /media/hdd2/ihsuan/VisTacFusion; PY=/home/shared/miniconda3/envs/vistacfusion/bin/python; OUT=/media/hdd2/ihsuan/VisTacFusion_outputs_hdd2
RUNS="S_k10_realonly S_k25_realonly S_k100_realonly S_k10_blender S_k25_blender S_k100_blender S_k10_gan40 S_k25_gan40 S_k100_gan40"
count(){ ps -u shared -o pid,cmd 2>/dev/null | grep '[e]ngine.train --model' | while read p rest; do tr '\0' '\n' < /proc/$p/environ 2>/dev/null | grep -q "^CUDA_VISIBLE_DEVICES=$1$" && echo "$rest" | grep -o 'output-dir [^ ]*/pilot[0-9]*_[A-Za-z0-9_]*'; done | sort -u | wc -l; }
for n in $RUNS; do
  [ -f outputs/pilot_logs/$n.log ] && { echo "skip $n"; continue; }
  case $n in *gan40) K=${n#S_k}; K=${K%%_*}; until [ $(find /media/hdd2/ihsuan/gs_blender/renders_v3 -path "*samples_g40_k$K/*.png" | wc -l) -ge 69600 ]; do sleep 300; done;; esac
  while :; do
    for g in 0 1 2 3; do
      if [ $(count $g) -lt 3 ] && [ $(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader,nounits) -lt 40000 ]; then
        CUDA_VISIBLE_DEVICES=$g nohup $PY -m vistacfusion.engine.train --model ablation/encoder/tac_sitr_single.yaml --train ablation/pilot_robosoft/train_pilot_e50.yaml \
          --data ablation/pilot_robosoft/data_$n.yaml --output-dir $OUT/pilot_$n > outputs/pilot_logs/$n.log 2>&1 &
        echo "$(date +%m-%d\ %H:%M) launched $n on GPU$g"; sleep 150; break 2
      fi
    done; sleep 300
  done
done; echo "all launched"
