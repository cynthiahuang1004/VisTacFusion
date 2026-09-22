#!/bin/bash
# Formal 150-epoch runs: keep at most $MAXPER of our runs per GPU, launch from the list in order.
cd /media/hdd2/ihsuan/VisTacFusion
PY=/home/shared/miniconda3/envs/vistacfusion/bin/python
OUT=/media/hdd2/ihsuan/VisTacFusion_outputs_hdd2
MAXPER=${MAXPER:-3}
RUNS="A_blender A_ganloo_9obj A_ganfull A_blender_bgsub A_diffloo \
      B_k10_realonly B_k10_blender B_k10_gank B_k25_realonly B_k25_blender B_k25_gank \
      B_k100_blender B_k0_blender B_k0_ganfull B_k25_ganfull B_k100_ganfull"
# number of distinct runs (dataloader workers share the run name) on GPU $1
count(){ ps -u shared -o pid,cmd | grep '[e]ngine.train --model' | while read p rest; do tr '\0' '\n' < /proc/$p/environ 2>/dev/null | grep -q "^CUDA_VISIBLE_DEVICES=$1$" && echo "$rest" | grep -o 'pilot150_[A-Za-z0-9_]*'; done | sort -u | wc -l; }
for n in $RUNS; do
  [ -f outputs/pilot_logs/e150_$n.log ] && continue   # log is created at launch
  while :; do
    for g in 0 1 2 3; do
      if [ $(count $g) -lt $MAXPER ] && [ $(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader,nounits) -lt 40000 ]; then
        CUDA_VISIBLE_DEVICES=$g nohup $PY -m vistacfusion.engine.train --model ablation/encoder/tac_sitr_single.yaml \
          --train ablation/pilot_robosoft/train_pilot_e150.yaml --data ablation/pilot_robosoft/data_$n.yaml \
          --output-dir $OUT/pilot150_$n > outputs/pilot_logs/e150_$n.log 2>&1 &
        echo "$(date +%H:%M) launched $n on GPU$g"; sleep 150; break 2   # next run in RUNS
      fi
    done
    sleep 300
  done
done
echo "all launched"
