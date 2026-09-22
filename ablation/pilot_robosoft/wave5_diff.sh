#!/bin/bash
# diff-GAN chain: wait for G -> render sim (composited on ref) -> downstream with real bg normalization
cd /media/hdd2/ihsuan/VisTacFusion
PY=/home/shared/miniconda3/envs/vistacfusion/bin/python
tag=$1; gpu=$2; run=$3
OBJS=$($PY -c "import yaml;print(' '.join(yaml.safe_load(open('ablation/pilot_robosoft/data_A_blender.yaml'))['sim']['include_objects']))")
until [ -f outputs/tactile_gan_diff_$tag/D_final.pt ]; do sleep 60; done
$PY scripts/tactile_render/generate_tactile.py --diff --ckpt outputs/tactile_gan_diff_$tag/G_final.pt --device cuda:$gpu \
   --out-subdir samples_d_$tag --objects $OBJS > outputs/pilot_logs/gen_d_$tag.log 2>&1
until [ $(nvidia-smi -i $gpu --query-gpu=memory.used --format=csv,noheader,nounits) -lt 38000 ]; do sleep 120; done
ablation/pilot_robosoft/launch.sh $gpu $run
