#!/bin/bash
cd /media/hdd2/ihsuan/VisTacFusion; PY=/home/shared/miniconda3/envs/vistacfusion/bin/python
OBJS=$($PY -c "import yaml;print(' '.join(yaml.safe_load(open('ablation/pilot_robosoft/data_A_blender.yaml'))['sim']['include_objects']))")
until [ -f outputs/tactile_gan_k50/D_final.pt ]; do sleep 120; done
$PY scripts/tactile_render/generate_tactile.py --ckpt outputs/tactile_gan_k50/G_final.pt --device cuda:0 --out-subdir samples_g_k50 --objects $OBJS > outputs/pilot_logs/gen_k50.log 2>&1
ablation/pilot_robosoft/gated_launch.sh B_k50_gank B_k50_gank
