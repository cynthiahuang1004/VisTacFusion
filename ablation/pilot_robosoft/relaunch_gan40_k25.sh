#!/bin/bash
cd /media/hdd2/ihsuan/VisTacFusion; PY=/home/shared/miniconda3/envs/vistacfusion/bin/python
while :; do for g in 2 1 3 0; do m=$(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader,nounits); [ $m -lt 40000 ] && { echo "$(date +%H:%M) GPU$g ${m}MiB -> launch"; nohup $PY scripts/tactile_render/train_tactile_gan.py --real-root /media/hdd2/ihsuan/gs_blender/real_gelslim40_train --per-object 25 --min-steps 30000 --device cuda:$g --out outputs/tactile_gan40_k25 > outputs/pilot_logs/gan40_k25.log 2>&1 & exit 0; }; done; sleep 120; done
