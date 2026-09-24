#!/bin/bash
# sensor-B learned renderers -> sim tactile in renders_v3/*/samples_g40_k{K} (all 12 objects)
cd /media/hdd2/ihsuan/VisTacFusion; PY=/home/shared/miniconda3/envs/vistacfusion/bin/python
for K in 10 25 100; do
  $PY scripts/tactile_render/generate_tactile.py --ckpt outputs/tactile_gan40_k$K/G_final.pt --device cuda:1 --out-subdir samples_g40_k$K > outputs/pilot_logs/gen_g40_k$K.log 2>&1
  echo "$(date +%H:%M) samples_g40_k$K done: $(ls /media/hdd2/ihsuan/gs_blender/renders_v3/*/session_*/sensor_0000/samples_g40_k$K/*.png 2>/dev/null | wc -l) files"
done
