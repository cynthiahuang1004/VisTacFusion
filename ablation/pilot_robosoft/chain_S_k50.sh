#!/bin/bash
cd /media/hdd2/ihsuan/VisTacFusion; PY=/home/shared/miniconda3/envs/vistacfusion/bin/python
$PY scripts/tactile_render/train_tactile_gan.py --real-root /media/hdd2/ihsuan/gs_blender/real_gelslim40_train --per-object 50 --min-steps 30000 --device cuda:3 --out outputs/tactile_gan40_k50 > outputs/pilot_logs/gan40_k50.log 2>&1
$PY scripts/tactile_render/generate_tactile.py --ckpt outputs/tactile_gan40_k50/G_final.pt --device cuda:3 --out-subdir samples_g40_k50 > outputs/pilot_logs/gen_g40_k50.log 2>&1
ablation/pilot_robosoft/launch2.sh 3 S_k50_realonly S_k50_realonly; sleep 5
ablation/pilot_robosoft/launch2.sh 3 S_k50_blender S_k50_blender; sleep 5
ablation/pilot_robosoft/launch2.sh 1 S_k50_gan40 S_k50_gan40
echo "$(date +%H:%M) S_k50 launched"
