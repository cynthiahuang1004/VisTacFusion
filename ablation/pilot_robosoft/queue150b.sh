#!/bin/bash
# Formal 150-epoch runs for the paper's final method set (2026-09-23).
# Skips runs whose history already has 150 epochs, resumes runs that have latest.pt,
# keeps at most $MAXPER of our runs per GPU and only launches on GPUs with < 40 GB used.
cd /media/hdd2/ihsuan/VisTacFusion
PY=/home/shared/miniconda3/envs/vistacfusion/bin/python
OUT=/media/hdd2/ihsuan/VisTacFusion_outputs_hdd2
MAXPER=${MAXPER:-3}
RUNS="B_k25_realonly B_k25_gank B_k50_realonly B_k50_gank B_k10_gank \
      A_bgcondloo A_blender_cov1 A_blender_cov2 A_ganloo_gr \
      B_k25_blender B_k50_blender B_k100_blender B_k0_blender B_k0_ganfull \
      A_ganfull B_k25_ganfull B_k100_ganfull"
count(){ ps -u shared -o pid,cmd | grep '[e]ngine.train --model' | while read p rest; do tr '\0' '\n' < /proc/$p/environ 2>/dev/null | grep -q "^CUDA_VISIBLE_DEVICES=$1$" && echo "$rest" | grep -o 'pilot150_[A-Za-z0-9_]*'; done | sort -u | wc -l; }
epochs(){ $PY -c "import json,sys;h=json.load(open('$1/history.json'));print(len(h['train']) if isinstance(h,dict) and 'train' in h else len(h))" 2>/dev/null || echo 0; }
for n in $RUNS; do
  d=$OUT/pilot150_$n
  if [ "$(epochs $d)" -ge 150 ]; then echo "skip $n (done)"; continue; fi
  ps -u shared -o cmd | grep -q "[o]utput-dir $d\b" && { echo "skip $n (running)"; continue; }
  extra=""; [ -f $d/latest.pt ] && extra="--resume $d/latest.pt"
  while :; do
    for g in 0 1 2 3; do
      if [ $(count $g) -lt $MAXPER ] && [ $(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader,nounits) -lt 40000 ]; then
        CUDA_VISIBLE_DEVICES=$g nohup $PY -m vistacfusion.engine.train --model ablation/encoder/tac_sitr_single.yaml \
          --train ablation/pilot_robosoft/train_pilot_e150.yaml --data ablation/pilot_robosoft/data_$n.yaml \
          --output-dir $d $extra >> outputs/pilot_logs/e150_$n.log 2>&1 &
        echo "$(date +%m-%d\ %H:%M) launched $n on GPU$g $extra"; sleep 150; break 2
      fi
    done
    sleep 300
  done
done
echo "$(date +%m-%d\ %H:%M) all launched"
