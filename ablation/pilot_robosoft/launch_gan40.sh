#!/bin/bash
# Sensor-B (GelSlim 4.0) learned renderers: GAN-K trained only on the K-shot 4.0 real subset (strict, no leakage).
cd /media/hdd2/ihsuan/VisTacFusion; PY=/home/shared/miniconda3/envs/vistacfusion/bin/python
until grep -q "review grid" outputs/pilot_logs/build_gs40_train.log; do sleep 30; done
MS=$(grep -o "min_steps=[0-9]*" outputs/pilot_logs/gan_k50.log | head -1 | cut -d= -f2); MS=${MS:-20000}
echo "min_steps=$MS"
i=0
for K in 10 25 100; do
  g=$((1 + i % 3)); i=$((i+1))
  nohup $PY scripts/tactile_render/train_tactile_gan.py --real-root /media/hdd2/ihsuan/gs_blender/real_gelslim40_train \
    --per-object $K --min-steps $MS --device cuda:$g --out outputs/tactile_gan40_k$K > outputs/pilot_logs/gan40_k$K.log 2>&1 &
  echo "$(date +%H:%M) launched gan40_k$K on GPU$g"; sleep 60
done
