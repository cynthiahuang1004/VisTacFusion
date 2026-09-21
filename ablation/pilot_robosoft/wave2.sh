#!/bin/bash
# wave2.sh <gan_tag> <gpu> <run names...>: wait for the GAN, render sim tactile, launch strict runs
cd /media/hdd2/ihsuan/VisTacFusion
PY=/home/shared/miniconda3/envs/vistacfusion/bin/python
tag=$1; gpu=$2; shift 2
OBJS=$($PY -c "import yaml;print(' '.join(yaml.safe_load(open('ablation/pilot_robosoft/data_A_blender.yaml'))['sim']['include_objects']))")
until [ -f outputs/tactile_gan_$tag/G_final.pt ]; do sleep 60; done
$PY scripts/tactile_render/generate_tactile.py --ckpt outputs/tactile_gan_$tag/G_final.pt \
  --device cuda:$gpu --out-subdir samples_g_$tag --objects $OBJS > outputs/pilot_logs/gen_$tag.log 2>&1
for n in "$@"; do   # launch one at a time, only when the shared GPU has room
  until [ $(nvidia-smi -i $gpu --query-gpu=memory.used --format=csv,noheader,nounits) -lt 34000 ]; do sleep 120; done
  ablation/pilot_robosoft/launch.sh $gpu $n; sleep 240
done
echo "wave2 $tag launched: $@"
